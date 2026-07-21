#!/usr/bin/env python3
"""Standalone qwen-asr worker process for the main API."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner


QWEN_SAMPLE_RATE_HZ = 16_000
MAX_SEGMENT_CHARS = 84
MAX_SEGMENT_SECONDS = 6.0
HARD_MAX_SEGMENT_SECONDS = 8.0
SPLIT_GAP_SECONDS = 0.65
SENTENCE_ENDINGS = (".", "!", "?", ";", ":")


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def encode_json(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def torch_dtype_from_name(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def normalize_device_map(device: str) -> str:
    normalized = device.strip().lower()
    if normalized.startswith("cuda"):
        return "cuda"
    if normalized.startswith("cpu"):
        return "cpu"
    return device


@dataclass(frozen=True)
class WordPayload:
    start: float
    end: float
    word: str
    probability: float | None = None


def audio_duration_seconds(audio_path: str) -> float:
    info = sf.info(audio_path)
    return float(info.frames) / float(info.samplerate)


def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def align_items_to_words(items: list[Any]) -> list[dict[str, Any]]:
    """Normalize exact aligner items without applying subtitle heuristics."""
    return [
        asdict(
            WordPayload(
                start=float(item.start_time),
                end=float(item.end_time),
                word=str(item.text),
            )
        )
        for item in items
    ]


def word_text(word: dict[str, Any]) -> str:
    """Return display text for one aligned word payload."""
    return str(word.get("word", "")).strip()


def joined_word_text(words: list[dict[str, Any]]) -> str:
    """Build subtitle text from aligned words without changing timing payloads."""
    return " ".join(token for token in (word_text(word) for word in words) if token)


def should_start_new_segment(
    *,
    current_words: list[dict[str, Any]],
    next_word: dict[str, Any],
) -> bool:
    """Decide whether a word-level alignment should become a subtitle boundary.

    Qwen's forced aligner gives precise word timings. SRT/VTT clients, however,
    need human-readable segments rather than one huge block. These heuristics
    keep word timestamps intact while splitting on long pauses, sentence-like
    punctuation, and practical subtitle size limits.
    """
    if not current_words:
        return False

    current_start = float(current_words[0]["start"])
    current_end = float(current_words[-1]["end"])
    next_end = float(next_word["end"])
    next_text = word_text(next_word)
    current_text = joined_word_text(current_words)
    previous_text = word_text(current_words[-1])
    gap_seconds = float(next_word["start"]) - current_end
    candidate_chars = len(f"{current_text} {next_text}".strip())
    candidate_seconds = next_end - current_start
    previous_ends_sentence = previous_text.endswith(SENTENCE_ENDINGS)

    if gap_seconds >= SPLIT_GAP_SECONDS:
        return True
    if candidate_chars > MAX_SEGMENT_CHARS:
        return True
    if candidate_seconds > HARD_MAX_SEGMENT_SECONDS:
        return True
    if previous_ends_sentence and candidate_seconds >= MAX_SEGMENT_SECONDS:
        return True
    return False


def words_to_timestamp_segments(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group aligned words into timestamped segments suitable for JSON/SRT/VTT."""
    segments: list[dict[str, Any]] = []
    current_words: list[dict[str, Any]] = []

    for word in words:
        if should_start_new_segment(current_words=current_words, next_word=word):
            segment_words = current_words
            segments.append(
                {
                    "id": len(segments),
                    "start": float(segment_words[0]["start"]),
                    "end": float(segment_words[-1]["end"]),
                    "text": joined_word_text(segment_words),
                    "words": segment_words,
                }
            )
            current_words = []
        current_words.append(word)

    if current_words:
        segments.append(
            {
                "id": len(segments),
                "start": float(current_words[0]["start"]),
                "end": float(current_words[-1]["end"]),
                "text": joined_word_text(current_words),
                "words": current_words,
            }
        )
    return segments


class Worker:
    def __init__(
        self,
        *,
        model_name: str,
        model_source: str,
        device: str,
        dtype_name: str,
        aligner_source: str | None,
        aligner_device: str,
        aligner_dtype_name: str,
    ) -> None:
        self.model_name = model_name
        self.model_source = model_source
        self.device = device
        self.dtype = torch_dtype_from_name(dtype_name)
        self.aligner_source = aligner_source or None
        self.aligner_device = aligner_device
        self.aligner_dtype = torch_dtype_from_name(aligner_dtype_name)
        self.model = self._load_model()
        self.aligner: Qwen3ForcedAligner | None = None

    def _load_model(self) -> Qwen3ASRModel:
        log(
            f"Loading qwen-asr model={self.model_name} source={self.model_source} "
            f"device={self.device} dtype={self.dtype}"
        )
        return Qwen3ASRModel.from_pretrained(
            self.model_source,
            torch_dtype=self.dtype,
            device_map=normalize_device_map(self.device),
            max_inference_batch_size=1,
        )

    def _load_aligner(self) -> Qwen3ForcedAligner:
        if self.aligner is not None:
            return self.aligner
        if not self.aligner_source:
            raise RuntimeError(
                f"Model '{self.model_name}' was asked for timestamps but no aligner was configured."
            )
        log(
            f"Loading qwen aligner source={self.aligner_source} "
            f"device={self.aligner_device} dtype={self.aligner_dtype}"
        )
        self.aligner = Qwen3ForcedAligner.from_pretrained(
            self.aligner_source,
            torch_dtype=self.aligner_dtype,
            device_map=normalize_device_map(self.aligner_device),
        )
        return self.aligner

    def _transcribe(
        self,
        *,
        audio: Any,
        duration: float,
        language: str | None,
        prompt: str | None,
    ) -> dict[str, Any]:
        result = self.model.transcribe(
            audio=audio,
            context=prompt or "",
            language=language,
            return_time_stamps=False,
        )[0]
        return {
            "text": str(result.text).strip(),
            "language": str(result.language),
            "duration": duration,
            "language_probability": None,
            "segments": [],
        }

    def transcribe_file(
        self,
        *,
        audio_path: str,
        language: str | None,
        prompt: str | None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        return self._transcribe(
            audio=audio_path,
            duration=audio_duration_seconds(audio_path),
            language=language,
            prompt=prompt,
        )

    def transcribe_pcm(
        self,
        *,
        pcm_base64: str,
        sample_rate_hz: int,
        language: str | None,
        prompt: str | None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        pcm_bytes = base64.b64decode(pcm_base64)
        waveform = pcm16_to_float32(pcm_bytes)
        duration = len(waveform) / float(sample_rate_hz)
        return self._transcribe(
            audio=(waveform, sample_rate_hz),
            duration=duration,
            language=language,
            prompt=prompt,
        )

    def align_file(
        self,
        *,
        audio_path: str,
        text: str,
        language: str,
    ) -> dict[str, Any]:
        words = self._align_words(
            audio=audio_path,
            text=text,
            language=language,
        )
        return {
            "duration": audio_duration_seconds(audio_path),
            "segments": words_to_timestamp_segments(words),
        }

    def _align_words(
        self,
        *,
        audio: Any,
        text: str,
        language: str,
    ) -> list[dict[str, Any]]:
        """Run the one cached forced aligner and return its native word view."""
        aligner = self._load_aligner()
        result = aligner.align(
            audio=audio,
            text=text,
            language=language,
        )[0]
        return align_items_to_words(list(result.items))

    def align_words_file(
        self,
        *,
        audio_path: str,
        text: str,
        language: str,
    ) -> dict[str, Any]:
        return {
            "duration": audio_duration_seconds(audio_path),
            "words": self._align_words(
                audio=audio_path,
                text=text,
                language=language,
            ),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="qwen-asr sidecar worker")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--aligner-source", default="")
    parser.add_argument("--aligner-device", default="cpu")
    parser.add_argument("--aligner-dtype", default="float32")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        worker = Worker(
            model_name=args.model_name,
            model_source=args.model_source,
            device=args.device,
            dtype_name=args.dtype,
            aligner_source=args.aligner_source or None,
            aligner_device=args.aligner_device,
            aligner_dtype_name=args.aligner_dtype,
        )
    except Exception as exc:
        encode_json(
            {
                "type": "error",
                "error": {"message": str(exc)},
            }
        )
        return 1

    encode_json({"type": "ready", "pid": os.getpid(), "device": args.device})

    handlers = {
        "transcribe_file": worker.transcribe_file,
        "transcribe_pcm": worker.transcribe_pcm,
        "align_file": worker.align_file,
        "align_words_file": worker.align_words_file,
    }

    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue
        request = json.loads(raw)
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        try:
            if method == "shutdown":
                encode_json({"id": request_id, "ok": True, "result": {"shutdown": True}})
                return 0
            if method not in handlers:
                raise ValueError(f"Unsupported method: {method}")
            result = handlers[method](**params)
            encode_json({"id": request_id, "ok": True, "result": result})
        except Exception as exc:  # pragma: no cover - exercised in live smoke tests
            encode_json(
                {
                    "id": request_id,
                    "ok": False,
                    "error": {"message": str(exc)},
                }
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
