#!/usr/bin/env python3
"""Standalone pyannote.audio diarization worker for the main API process."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def encode_json(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def turn_bounds(turn: Any) -> tuple[float, float]:
    """Extract start/end seconds from a pyannote Segment-like object."""
    return float(getattr(turn, "start", 0.0)), float(getattr(turn, "end", 0.0))


def iter_annotation_turns(annotation: Any) -> list[dict[str, Any]]:
    """Normalize pyannote Annotation variants into JSON-safe speaker turns."""
    turns: list[dict[str, Any]] = []
    if annotation is None:
        return turns

    if hasattr(annotation, "itertracks"):
        iterator = annotation.itertracks(yield_label=True)
    else:
        iterator = iter(annotation)

    for item in iterator:
        if isinstance(item, tuple) and len(item) == 3:
            turn, _track, speaker = item
        elif isinstance(item, tuple) and len(item) == 2:
            turn, speaker = item
        else:
            continue

        start, end = turn_bounds(turn)
        if end <= start:
            continue
        turns.append(
            {
                "start": start,
                "end": end,
                "speaker": str(speaker),
            }
        )

    turns.sort(key=lambda entry: (entry["start"], entry["end"], entry["speaker"]))
    return turns


class Worker:
    """Loaded pyannote pipeline wrapper."""

    def __init__(
        self,
        *,
        model_id: str,
        model_path: str,
        device: str,
    ) -> None:
        self.model_id = model_id
        self.model_path = Path(model_path).resolve()
        self.device = device
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self) -> Any:
        # Import pyannote only inside the sidecar. This keeps the main ASR API
        # runtime free of large diarization dependencies and makes unit tests
        # able to exercise the worker protocol without installing pyannote.
        from pyannote.audio import Pipeline

        if not self.model_path.is_dir() or not (self.model_path / "config.yaml").is_file():
            raise FileNotFoundError(
                f"Local pyannote pipeline is incomplete or missing: {self.model_path}"
            )

        log(
            "Loading local pyannote diarization "
            f"model={self.model_id} path={self.model_path} device={self.device}"
        )
        # A local directory plus the offline environment enforced by the parent
        # process guarantees that serving a request cannot contact Hugging Face.
        pipeline = Pipeline.from_pretrained(str(self.model_path))

        normalized_device = self.device.strip().lower()
        if normalized_device and normalized_device != "cpu":
            import torch

            pipeline.to(torch.device(normalized_device))
        return pipeline

    def diarize_file(
        self,
        *,
        audio_path: str,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if num_speakers is not None:
            params["num_speakers"] = int(num_speakers)
        if min_speakers is not None:
            params["min_speakers"] = int(min_speakers)
        if max_speakers is not None:
            params["max_speakers"] = int(max_speakers)

        started = time.perf_counter()
        output = self.pipeline(audio_path, **params)
        processing_seconds = time.perf_counter() - started
        annotation = getattr(output, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = getattr(output, "speaker_diarization", None)
        if annotation is None:
            annotation = output

        return {
            "model": self.model_id,
            "device": self.device,
            "turns": iter_annotation_turns(annotation),
            "processing_seconds": processing_seconds,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pyannote diarization sidecar worker")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", required=True)
    return parser.parse_args()


def main() -> int:
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args = parse_args()
    try:
        worker = Worker(
            model_id=args.model_id,
            model_path=args.model_path,
            device=args.device,
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
        "diarize_file": worker.diarize_file,
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


def cli() -> int:
    """Run the worker and treat a parent process interrupt as a clean shutdown.

    The development launcher sends SIGINT to the complete foreground process
    group. The API then performs its normal lifespan cleanup, while the worker
    can receive the same signal before the parent sends the JSON shutdown
    request. Suppressing that expected interrupt keeps shutdown logs free of a
    misleading traceback without hiding ordinary worker exceptions.
    """
    try:
        return main()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(cli())
