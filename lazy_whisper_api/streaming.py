"""SSE helpers for OpenAI-like streaming audio transcription."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from .config import Settings
from .model_manager import LoadedModel, ModelManager
from .transcription import (
    TranscriptionRequest,
    iter_transcribe_sync,
    normalize_timestamp_granularities,
    validate_request,
    write_upload_to_tempfile,
)


def encode_sse_event(event_name: str, payload: dict[str, Any]) -> bytes:
    """Encode one SSE event payload."""
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event_name}\ndata: {data}\n\n".encode("utf-8")


def segment_payload(segment: Any) -> dict[str, Any]:
    """Serialize segment timestamps for streaming clients."""
    return {
        "id": segment.id,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text.strip(),
    }


def error_payload(exc: Exception) -> dict[str, Any]:
    """Normalize worker errors into stream-safe error payloads."""
    if isinstance(exc, HTTPException) and isinstance(exc.detail, dict):
        return {
            "type": "error",
            "error": {
                "type": exc.detail.get("type", "server_error"),
                "message": exc.detail.get("message", ""),
            },
        }
    return {
        "type": "error",
        "error": {
            "type": "server_error",
            "message": str(exc),
        },
    }


async def create_transcription_stream_response(
    *,
    settings: Settings,
    model_manager: ModelManager,
    payload: TranscriptionRequest,
) -> StreamingResponse:
    """Create an SSE response for a completed uploaded audio file."""
    canonical_model = validate_request(settings, payload)
    granularity_set = normalize_timestamp_granularities(payload.timestamp_granularities)

    suffix = Path(payload.file.filename or "upload.bin").suffix
    tmp_path: Path | None = None
    with_model = model_manager.lease(canonical_model)
    lease: LoadedModel | None = None

    try:
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)

        await write_upload_to_tempfile(
            upload=payload.file,
            destination=tmp_path,
            chunk_size=settings.upload_chunk_size,
        )
        await payload.file.close()
        lease = with_model.__enter__()
    except Exception:
        await payload.file.close()
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise

    event_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    worker_done = threading.Event()

    def push_event(event_name: str, payload: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(
            event_queue.put_nowait,
            encode_sse_event(event_name, payload),
        )

    def worker() -> None:
        assert lease is not None
        assert tmp_path is not None
        try:
            segments_iter, info = iter_transcribe_sync(
                model=lease.model,
                audio_path=tmp_path,
                language=payload.language,
                task=payload.task,
                prompt=payload.prompt,
                temperature=payload.temperature,
                word_timestamps=False,
                vad_filter=lease.spec.vad_filter,
            )
            text_parts: list[str] = []
            for segment in segments_iter:
                segment_text = segment.text.strip()
                text_parts.append(segment.text)
                if not segment_text:
                    continue
                event_payload = {
                    "type": "transcript.text.delta",
                    "delta": segment_text,
                    "text": "".join(text_parts).strip(),
                }
                if "segment" in granularity_set:
                    event_payload["segment"] = segment_payload(segment)
                push_event("transcript.text.delta", event_payload)

            push_event(
                "transcript.text.done",
                {
                    "type": "transcript.text.done",
                    "text": "".join(text_parts).strip(),
                    "model": canonical_model,
                    "device": lease.actual_device,
                    "language": info.language,
                    "duration": info.duration,
                },
            )
        except Exception as exc:  # pragma: no cover - exercised in integration flow
            push_event("error", error_payload(exc))
        finally:
            worker_done.set()
            loop.call_soon_threadsafe(event_queue.put_nowait, None)

    worker_thread = threading.Thread(target=worker, daemon=True, name="transcription-sse")
    worker_thread.start()

    async def event_stream():
        try:
            while True:
                item = await event_queue.get()
                if item is None:
                    break
                yield item
        finally:
            await run_in_threadpool(worker_thread.join)
            if lease is not None:
                with_model.__exit__(None, None, None)
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
