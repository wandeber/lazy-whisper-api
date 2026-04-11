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


def align_items_to_segments(text: str, items: list[Any]) -> list[dict[str, Any]]:
    if not items:
        return []
    words = [
        asdict(
            WordPayload(
                start=float(item.start_time),
                end=float(item.end_time),
                word=str(item.text),
            )
        )
        for item in items
    ]
    return [
        {
            "id": 0,
            "start": words[0]["start"],
            "end": words[-1]["end"],
            "text": text,
            "words": words,
        }
    ]


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
        aligner = self._load_aligner()
        result = aligner.align(
            audio=audio_path,
            text=text,
            language=language,
        )[0]
        segments = align_items_to_segments(text, list(result.items))
        return {
            "duration": audio_duration_seconds(audio_path),
            "segments": segments,
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
