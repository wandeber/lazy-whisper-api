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

from .backends import BackendTranscription, SegmentTiming, segments_to_text
from .config import Settings
from .errors import api_error
from .model_manager import LoadedModel, ModelManager
from .transcription import (
    TranscriptionRequest,
    ensure_timestamp_segments,
    iter_transcribe_sync,
    load_audio_file_as_pcm16,
    normalize_timestamp_granularities,
    requires_timestamps,
    transcribe_pcm16_sync,
    validate_request,
    write_upload_to_tempfile,
)


PCM_SAMPLE_WIDTH_BYTES = 2
SYNTHETIC_STREAM_MIN_SECONDS = 8
SYNTHETIC_STREAM_STEP_SECONDS = 6


def common_prefix_length(left: str, right: str) -> int:
    """Return the length of the shared prefix between two strings."""
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


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


def build_done_payload(
    *,
    transcription: BackendTranscription,
    canonical_model: str,
    lease: LoadedModel,
    include_segments: bool,
) -> dict[str, Any]:
    """Render the final SSE done event payload."""
    payload: dict[str, Any] = {
        "type": "transcript.text.done",
        "text": transcription.text or segments_to_text(transcription.segments),
        "model": canonical_model,
        "device": lease.actual_device,
        "language": transcription.info.language,
        "duration": transcription.info.duration,
    }
    if include_segments:
        payload["segments"] = [segment_payload(segment) for segment in transcription.segments]
    return payload


def iter_synthetic_stream_events(
    *,
    lease: LoadedModel,
    audio_path: Path,
    language: str | None,
    task: str,
    prompt: str | None,
    temperature: float,
    include_segments: bool,
) -> tuple[list[dict[str, Any]], BackendTranscription]:
    """Generate progressive transcription deltas for backends without native streaming."""
    if lease.runtime is None:
        raise RuntimeError(f"Model '{lease.spec.name}' is not loaded.")

    sample_rate_hz = lease.runtime.preferred_stream_sample_rate_hz
    pcm_bytes = load_audio_file_as_pcm16(
        audio_path=audio_path,
        sample_rate_hz=sample_rate_hz,
    )
    if not pcm_bytes:
        empty = BackendTranscription(
            text="",
            info=lease.runtime.transcribe_file(
                audio_path=audio_path,
                language=language,
                task=task,
                prompt=prompt,
                temperature=temperature,
                word_timestamps=False,
            ).info,
            segments=[],
        )
        return [], empty

    bytes_per_second = max(1, sample_rate_hz * PCM_SAMPLE_WIDTH_BYTES)
    min_chunk_bytes = bytes_per_second * SYNTHETIC_STREAM_MIN_SECONDS
    step_bytes = bytes_per_second * SYNTHETIC_STREAM_STEP_SECONDS
    total_bytes = len(pcm_bytes)

    emitted_text = ""
    event_segments: list[dict[str, Any]] = []
    latest = None

    prefix_end = min(total_bytes, max(min_chunk_bytes, step_bytes))
    while True:
        latest = lease.runtime.transcribe_pcm(
            pcm_bytes=pcm_bytes[:prefix_end],
            sample_rate_hz=sample_rate_hz,
            language=language,
            task=task,
            prompt=prompt,
            temperature=temperature,
            word_timestamps=False,
        )
        candidate_text = latest.text.strip()
        if candidate_text and candidate_text != emitted_text:
            prefix_length = common_prefix_length(emitted_text, candidate_text)
            delta = candidate_text[prefix_length:].strip()
            if delta:
                segment = SegmentTiming(
                    id=len(event_segments),
                    start=max(0.0, (prefix_end - step_bytes) / bytes_per_second),
                    end=prefix_end / bytes_per_second,
                    text=delta,
                )
                event_payload = {
                    "type": "transcript.text.delta",
                    "delta": delta,
                    "text": candidate_text,
                }
                if include_segments:
                    event_payload["segment"] = segment_payload(segment)
                event_segments.append(event_payload)
            emitted_text = candidate_text

        if prefix_end >= total_bytes:
            break
        prefix_end = min(total_bytes, prefix_end + step_bytes)

    if latest is None:
        latest = lease.runtime.transcribe_file(
            audio_path=audio_path,
            language=language,
            task=task,
            prompt=prompt,
            temperature=temperature,
            word_timestamps=False,
        )
    elif not latest.text.strip():
        latest = lease.runtime.transcribe_file(
            audio_path=audio_path,
            language=language,
            task=task,
            prompt=prompt,
            temperature=temperature,
            word_timestamps=False,
        )

    return event_segments, latest


async def create_transcription_stream_response(
    *,
    settings: Settings,
    model_manager: ModelManager,
    payload: TranscriptionRequest,
) -> StreamingResponse:
    """Create an SSE response for a completed uploaded audio file."""
    canonical_model = validate_request(settings, payload)
    spec = settings.model_settings[canonical_model]
    if not spec.supports("stream"):
        raise api_error(
            400,
            f"Model '{payload.model}' does not support streaming responses.",
            error_type="invalid_request_error",
        )

    granularity_set = normalize_timestamp_granularities(payload.timestamp_granularities)
    wants_timestamps = requires_timestamps(
        response_format=payload.response_format,
        granularity_set=granularity_set,
    )

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
        if lease.runtime is None:
            raise RuntimeError(f"Model '{canonical_model}' is not loaded.")
    except Exception:
        await payload.file.close()
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise

    event_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def push_event(event_name: str, payload: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(
            event_queue.put_nowait,
            encode_sse_event(event_name, payload),
        )

    def worker() -> None:
        assert lease is not None
        assert tmp_path is not None
        try:
            if lease.runtime.supports_native_streaming:
                native = iter_transcribe_sync(
                    runtime=lease.runtime,
                    audio_path=tmp_path,
                    language=payload.language,
                    task=payload.task,
                    prompt=payload.prompt,
                    temperature=payload.temperature,
                    word_timestamps=False,
                )
                if native is None:
                    raise RuntimeError(
                        f"Backend '{lease.spec.backend}' declared native streaming but returned none."
                    )
                segments_iter, info = native
                text_parts: list[str] = []
                done_segments: list[SegmentTiming] = []
                for segment in segments_iter:
                    segment_text = segment.text.strip()
                    text_parts.append(segment.text)
                    done_segments.append(segment)
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

                final_transcription = BackendTranscription(
                    text="".join(text_parts).strip(),
                    info=info,
                    segments=done_segments,
                )
            else:
                synthetic_events, final_transcription = iter_synthetic_stream_events(
                    lease=lease,
                    audio_path=tmp_path,
                    language=payload.language,
                    task=payload.task,
                    prompt=payload.prompt,
                    temperature=payload.temperature,
                    include_segments="segment" in granularity_set,
                )
                for event_payload in synthetic_events:
                    push_event("transcript.text.delta", event_payload)

            if wants_timestamps and not final_transcription.segments:
                final_transcription = BackendTranscription(
                    text=final_transcription.text,
                    info=final_transcription.info,
                    segments=ensure_timestamp_segments(
                        lease=lease,
                        audio_path=tmp_path,
                        transcription=final_transcription,
                    ),
                )

            push_event(
                "transcript.text.done",
                build_done_payload(
                    transcription=final_transcription,
                    canonical_model=canonical_model,
                    lease=lease,
                    include_segments="segment" in granularity_set,
                ),
            )
        except Exception as exc:  # pragma: no cover - exercised in integration flow
            push_event("error", error_payload(exc))
        finally:
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
