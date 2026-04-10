"""Request validation and transcription orchestration."""

from __future__ import annotations

import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from .config import Settings
from .errors import api_error
from .model_manager import ModelManager


SUPPORTED_RESPONSE_FORMATS = {"json", "text", "srt", "verbose_json", "vtt"}
SUPPORTED_TIMESTAMP_GRANULARITIES = {"segment", "word"}
PCM16_SAMPLE_WIDTH_BYTES = 2
PCM16_CHANNELS = 1


@dataclass(frozen=True)
class TranscriptionRequest:
    """Normalized payload for a transcription or translation request."""

    file: UploadFile
    model: str
    task: str
    language: str | None
    prompt: str | None
    response_format: str
    temperature: float
    timestamp_granularities: list[str] | None


@dataclass(frozen=True)
class TranscriptionResult:
    """Raw output returned by Faster Whisper plus request metadata."""

    model_name: str
    device: str
    response_format: str
    text: str
    info: Any
    segments: list[Any]


def validate_request(settings: Settings, payload: TranscriptionRequest) -> str:
    """Validate public request parameters and resolve model aliases."""
    try:
        canonical_model = settings.resolve_model_name(payload.model)
    except KeyError as exc:
        supported = ", ".join(settings.supported_model_ids)
        raise api_error(
            400,
            f"Unsupported model '{payload.model}'. Supported values: {supported}.",
            error_type="invalid_request_error",
        ) from exc

    if payload.response_format not in SUPPORTED_RESPONSE_FORMATS:
        raise api_error(400, "Unsupported response_format.", error_type="invalid_request_error")

    granularity_set = set(payload.timestamp_granularities or [])
    invalid_granularities = granularity_set - SUPPORTED_TIMESTAMP_GRANULARITIES
    if invalid_granularities:
        raise api_error(
            400,
            "Unsupported timestamp_granularities value.",
            error_type="invalid_request_error",
        )

    return canonical_model


def segments_to_text(segments: list[Any]) -> str:
    """Join transcribed segments into the plain-text response body."""
    return "".join(segment.text for segment in segments).strip()


def normalize_timestamp_granularities(values: list[str] | None) -> set[str]:
    """Return a normalized timestamp granularity set."""
    return set(values or [])


async def write_upload_to_tempfile(
    *,
    upload: UploadFile,
    destination: Path,
    chunk_size: int,
) -> None:
    """Copy the uploaded file to disk in chunks to avoid RAM spikes."""
    with destination.open("wb") as handle:
        while True:
            chunk = await upload.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)


def transcribe_sync(
    *,
    model: Any,
    audio_path: Path,
    language: str | None,
    task: str,
    prompt: str | None,
    temperature: float,
    word_timestamps: bool,
    vad_filter: bool,
) -> tuple[list[Any], Any]:
    """Run Faster Whisper off the event loop."""
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language or None,
        task=task,
        initial_prompt=prompt or None,
        temperature=temperature,
        word_timestamps=word_timestamps,
        vad_filter=vad_filter,
    )
    return list(segments_iter), info


def iter_transcribe_sync(
    *,
    model: Any,
    audio_path: Path,
    language: str | None,
    task: str,
    prompt: str | None,
    temperature: float,
    word_timestamps: bool,
    vad_filter: bool,
) -> tuple[Any, Any]:
    """Start a Faster Whisper transcription and return its segment iterator."""
    return model.transcribe(
        str(audio_path),
        language=language or None,
        task=task,
        initial_prompt=prompt or None,
        temperature=temperature,
        word_timestamps=word_timestamps,
        vad_filter=vad_filter,
    )


def write_pcm16_wav(
    *,
    pcm_bytes: bytes,
    sample_rate_hz: int,
    destination: Path,
) -> None:
    """Wrap raw PCM16 mono bytes in a WAV container for Faster Whisper."""
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(PCM16_CHANNELS)
        handle.setsampwidth(PCM16_SAMPLE_WIDTH_BYTES)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(pcm_bytes)


def transcribe_pcm16_sync(
    *,
    model: Any,
    pcm_bytes: bytes,
    sample_rate_hz: int,
    language: str | None,
    task: str,
    prompt: str | None,
    temperature: float,
    word_timestamps: bool,
    vad_filter: bool,
) -> tuple[list[Any], Any]:
    """Transcribe raw PCM16 mono audio by wrapping it in a temporary WAV file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = Path(tmp.name)

    try:
        write_pcm16_wav(
            pcm_bytes=pcm_bytes,
            sample_rate_hz=sample_rate_hz,
            destination=tmp_path,
        )
        return transcribe_sync(
            model=model,
            audio_path=tmp_path,
            language=language,
            task=task,
            prompt=prompt,
            temperature=temperature,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


async def transcribe_upload(
    *,
    settings: Settings,
    model_manager: ModelManager,
    payload: TranscriptionRequest,
) -> TranscriptionResult:
    """Persist the upload temporarily, run Faster Whisper, and clean up."""
    canonical_model = validate_request(settings, payload)

    suffix = Path(payload.file.filename or "upload.bin").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)

    try:
        await write_upload_to_tempfile(
            upload=payload.file,
            destination=tmp_path,
            chunk_size=settings.upload_chunk_size,
        )

        granularity_set = normalize_timestamp_granularities(payload.timestamp_granularities)
        with model_manager.lease(canonical_model) as lease:
            if lease.model is None:
                raise RuntimeError(f"Model '{canonical_model}' is not loaded.")
            segments, info = await run_in_threadpool(
                transcribe_sync,
                model=lease.model,
                audio_path=tmp_path,
                language=payload.language or None,
                task=payload.task,
                prompt=payload.prompt or None,
                temperature=payload.temperature,
                word_timestamps="word" in granularity_set,
                vad_filter=lease.spec.vad_filter,
            )
            text = segments_to_text(segments)
            device = lease.actual_device
    except HTTPException:
        raise
    except Exception as exc:
        raise api_error(500, str(exc), error_type="server_error") from exc
    finally:
        await payload.file.close()
        tmp_path.unlink(missing_ok=True)

    return TranscriptionResult(
        model_name=canonical_model,
        device=device,
        response_format=payload.response_format,
        text=text,
        info=info,
        segments=segments,
    )
