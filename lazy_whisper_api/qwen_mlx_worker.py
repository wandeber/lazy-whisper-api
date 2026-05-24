#!/usr/bin/env python3
"""Standalone mlx-qwen3-asr worker process for Apple Silicon.

This worker intentionally mirrors the JSON-RPC protocol used by the CUDA
`qwen_worker.py`. The main API can therefore keep one OpenAI-compatible surface
while selecting either the PyTorch/CUDA or MLX/Metal runtime from configuration.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PCM16_SAMPLE_WIDTH_BYTES = 2
PCM16_CHANNELS = 1
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


def write_pcm16_wav(
    *,
    pcm_bytes: bytes,
    sample_rate_hz: int,
    destination: Path,
) -> None:
    """Write realtime PCM snapshots to a WAV file accepted by mlx-qwen3-asr."""
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(PCM16_CHANNELS)
        handle.setsampwidth(PCM16_SAMPLE_WIDTH_BYTES)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(pcm_bytes)


def mlx_dtype_from_name(name: str) -> Any:
    """Resolve the configured dtype after importing MLX inside the worker venv."""
    import mlx.core as mx

    normalized = name.strip().lower()
    if normalized in {"float16", "fp16", "half"}:
        return mx.float16
    if normalized in {"bfloat16", "bf16"}:
        return mx.bfloat16
    if normalized in {"float32", "fp32", "float"}:
        return mx.float32
    raise ValueError(f"Unsupported MLX dtype: {name}")


def value_from(item: Any, key: str, default: Any = None) -> Any:
    """Read from either dict-like MLX payloads or lightweight result objects."""
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


@dataclass(frozen=True)
class WordPayload:
    start: float
    end: float
    word: str
    probability: float | None = None


def duration_from_timing_items(items: list[Any], fallback: float = 0.0) -> float:
    """Estimate duration from returned chunks/segments without extra audio deps."""
    ends = [float(value_from(item, "end", 0.0) or 0.0) for item in items]
    return max(ends, default=fallback)


def normalize_word_items(items: list[Any]) -> list[dict[str, Any]]:
    """Normalize MLX timestamp items into the existing worker word payload shape."""
    words: list[dict[str, Any]] = []
    for item in items:
        text = str(value_from(item, "text", value_from(item, "word", ""))).strip()
        if not text:
            continue
        words.append(
            asdict(
                WordPayload(
                    start=float(value_from(item, "start", 0.0) or 0.0),
                    end=float(value_from(item, "end", 0.0) or 0.0),
                    word=text,
                    probability=value_from(item, "probability"),
                )
            )
        )
    return words


def chunks_to_segments(chunks: list[Any]) -> list[dict[str, Any]]:
    """Normalize MLX chunk dictionaries into this API's segment schema."""
    segments: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        text = str(value_from(chunk, "text", "")).strip()
        if not text:
            continue
        segments.append(
            {
                "id": int(value_from(chunk, "id", value_from(chunk, "chunk_index", index)) or index),
                "start": float(value_from(chunk, "start", 0.0) or 0.0),
                "end": float(value_from(chunk, "end", 0.0) or 0.0),
                "text": text,
            }
        )
    return segments


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

    mlx-qwen3-asr returns precise aligned timestamp items, which are ideal for
    verbose JSON word timings but too granular for SRT/VTT on their own. These
    conservative boundaries keep the word timings intact while producing
    readable subtitle-sized segments from pauses, punctuation, and length caps.
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
    ) -> None:
        self.model_name = model_name
        self.model_source = model_source
        self.device = device
        self.dtype_name = dtype_name
        self.aligner_source = aligner_source or None
        self.session = self._load_session()

    def _load_session(self) -> Any:
        from mlx_qwen3_asr import Session

        log(
            f"Loading mlx-qwen3-asr model={self.model_name} source={self.model_source} "
            f"device={self.device} dtype={self.dtype_name}"
        )
        return Session(
            model=self.model_source,
            dtype=mlx_dtype_from_name(self.dtype_name),
        )

    def _transcribe(
        self,
        *,
        audio: str,
        language: str | None,
        prompt: str | None,
        return_timestamps: bool,
    ) -> Any:
        # `forced_aligner` is only provided for timestamped requests. Keeping it
        # off the normal path avoids loading the aligner until the public API
        # actually needs verbose_json/SRT/VTT/segment data.
        return self.session.transcribe(
            audio,
            context=prompt or "",
            language=language,
            return_timestamps=return_timestamps,
            return_chunks=True,
            forced_aligner=self.aligner_source if return_timestamps else None,
        )

    def transcribe_file(
        self,
        *,
        audio_path: str,
        language: str | None,
        prompt: str | None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        result = self._transcribe(
            audio=audio_path,
            language=language,
            prompt=prompt,
            return_timestamps=False,
        )
        chunks = list(value_from(result, "chunks", []) or [])
        return {
            "text": str(value_from(result, "text", "")).strip(),
            "language": str(value_from(result, "language", language or "")),
            "duration": duration_from_timing_items(chunks),
            "language_probability": None,
            # Match the CUDA Qwen worker: timestamps are populated through
            # align_file only when the public request needs them.
            "segments": [],
        }

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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = Path(tmp.name)
        try:
            write_pcm16_wav(
                pcm_bytes=pcm_bytes,
                sample_rate_hz=sample_rate_hz,
                destination=tmp_path,
            )
            result = self.transcribe_file(
                audio_path=str(tmp_path),
                language=language,
                prompt=prompt,
                temperature=temperature,
            )
            result["duration"] = len(pcm_bytes) / float(sample_rate_hz * PCM16_SAMPLE_WIDTH_BYTES)
            return result
        finally:
            tmp_path.unlink(missing_ok=True)

    def align_file(
        self,
        *,
        audio_path: str,
        text: str,
        language: str,
    ) -> dict[str, Any]:
        result = self._transcribe(
            audio=audio_path,
            language=language,
            prompt=text,
            return_timestamps=True,
        )
        timestamp_items = list(value_from(result, "segments", []) or [])
        chunks = list(value_from(result, "chunks", []) or [])
        words = normalize_word_items(timestamp_items)
        segments = words_to_timestamp_segments(words) or chunks_to_segments(chunks)
        return {
            "duration": duration_from_timing_items(timestamp_items or chunks),
            "segments": segments,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mlx-qwen3-asr sidecar worker")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--aligner-source", default="")
    parser.add_argument("--aligner-device", default="")
    parser.add_argument("--aligner-dtype", default="")
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
