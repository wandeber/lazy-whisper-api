"""Request validation and transcription orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from tempfile import NamedTemporaryFile

import av
from fastapi import HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from .backends import BackendTranscription, SegmentTiming, segments_to_text, write_pcm16_wav
from .config import Settings
from .errors import api_error
from .model_manager import LoadedModel, ModelManager


SUPPORTED_RESPONSE_FORMATS = {"json", "text", "srt", "verbose_json", "vtt"}
SUPPORTED_TIMESTAMP_GRANULARITIES = {"segment", "word"}


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
    """Raw output returned by one ASR backend plus request metadata."""

    model_name: str
    device: str
    response_format: str
    text: str
    info: Any
    segments: list[Any]


def normalize_timestamp_granularities(values: list[str] | None) -> set[str]:
    """Return a normalized timestamp granularity set."""
    return set(values or [])


def requires_timestamps(*, response_format: str, granularity_set: set[str]) -> bool:
    """Whether this request needs timestamps in the final response."""
    return response_format in {"srt", "vtt", "verbose_json"} or bool(granularity_set)


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

    spec = settings.model_settings[canonical_model]
    if payload.task == "translate" and not spec.supports("translate"):
        raise api_error(
            400,
            f"Model '{payload.model}' does not support translation.",
            error_type="invalid_request_error",
        )
    if payload.task == "transcribe" and not spec.supports("transcribe"):
        raise api_error(
            400,
            f"Model '{payload.model}' does not support transcription.",
            error_type="invalid_request_error",
        )

    if payload.response_format not in SUPPORTED_RESPONSE_FORMATS:
        raise api_error(400, "Unsupported response_format.", error_type="invalid_request_error")

    granularity_set = normalize_timestamp_granularities(payload.timestamp_granularities)
    invalid_granularities = granularity_set - SUPPORTED_TIMESTAMP_GRANULARITIES
    if invalid_granularities:
        raise api_error(
            400,
            "Unsupported timestamp_granularities value.",
            error_type="invalid_request_error",
        )
    if granularity_set and not spec.supports("timestamps"):
        raise api_error(
            400,
            f"Model '{payload.model}' does not support timestamps.",
            error_type="invalid_request_error",
        )
    if requires_timestamps(
        response_format=payload.response_format,
        granularity_set=granularity_set,
    ) and not spec.supports("timestamps"):
        raise api_error(
            400,
            f"Model '{payload.model}' does not support timestamped responses.",
            error_type="invalid_request_error",
        )

    return canonical_model


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


def load_audio_file_as_pcm16(
    *,
    audio_path: Path,
    sample_rate_hz: int,
) -> bytes:
    """Decode audio with PyAV and return mono PCM16 bytes at the requested sample rate."""
    container = av.open(str(audio_path))
    stream = container.streams.audio[0]
    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=sample_rate_hz,
    )

    chunks = bytearray()
    try:
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                array = out.to_ndarray()
                chunks.extend(array.astype("int16").tobytes())
    finally:
        container.close()
    return bytes(chunks)


def transcribe_sync(
    *,
    runtime: Any,
    audio_path: Path,
    language: str | None,
    task: str,
    prompt: str | None,
    temperature: float,
    word_timestamps: bool,
) -> BackendTranscription:
    """Run one backend file transcription off the event loop."""
    return runtime.transcribe_file(
        audio_path=audio_path,
        language=language,
        task=task,
        prompt=prompt,
        temperature=temperature,
        word_timestamps=word_timestamps,
    )


def iter_transcribe_sync(
    *,
    runtime: Any,
    audio_path: Path,
    language: str | None,
    task: str,
    prompt: str | None,
    temperature: float,
    word_timestamps: bool,
) -> tuple[Any, Any] | None:
    """Start a native backend streaming transcription when available."""
    return runtime.iter_transcribe_file(
        audio_path=audio_path,
        language=language,
        task=task,
        prompt=prompt,
        temperature=temperature,
        word_timestamps=word_timestamps,
    )


def transcribe_pcm16_sync(
    *,
    runtime: Any,
    pcm_bytes: bytes,
    sample_rate_hz: int,
    language: str | None,
    task: str,
    prompt: str | None,
    temperature: float,
    word_timestamps: bool,
) -> BackendTranscription:
    """Transcribe raw PCM16 mono audio via the selected backend."""
    return runtime.transcribe_pcm(
        pcm_bytes=pcm_bytes,
        sample_rate_hz=sample_rate_hz,
        language=language,
        task=task,
        prompt=prompt,
        temperature=temperature,
        word_timestamps=word_timestamps,
    )


def ensure_timestamp_segments(
    *,
    lease: LoadedModel,
    audio_path: Path,
    transcription: BackendTranscription,
) -> list[SegmentTiming]:
    """Return timestamped segments, using the backend aligner when needed."""
    if transcription.segments:
        return transcription.segments
    if not transcription.text.strip():
        return []
    if not transcription.info.language:
        return []
    return lease.runtime.align_file(
        audio_path=audio_path,
        text=transcription.text,
        language=transcription.info.language,
    )


def ensure_timestamp_segments_for_pcm(
    *,
    lease: LoadedModel,
    pcm_bytes: bytes,
    sample_rate_hz: int,
    transcription: BackendTranscription,
) -> list[SegmentTiming]:
    """Align PCM audio when the backend only returns final text."""
    if transcription.segments:
        return transcription.segments
    if not transcription.text.strip():
        return []
    if not transcription.info.language:
        return []

    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = Path(tmp.name)

    try:
        write_pcm16_wav(
            pcm_bytes=pcm_bytes,
            sample_rate_hz=sample_rate_hz,
            destination=tmp_path,
        )
        return lease.runtime.align_file(
            audio_path=tmp_path,
            text=transcription.text,
            language=transcription.info.language,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


async def transcribe_upload(
    *,
    settings: Settings,
    model_manager: ModelManager,
    payload: TranscriptionRequest,
) -> TranscriptionResult:
    """Persist the upload temporarily, run one backend, and clean up."""
    canonical_model = validate_request(settings, payload)

    suffix = Path(payload.file.filename or "upload.bin").suffix
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)

    try:
        await write_upload_to_tempfile(
            upload=payload.file,
            destination=tmp_path,
            chunk_size=settings.upload_chunk_size,
        )

        granularity_set = normalize_timestamp_granularities(payload.timestamp_granularities)
        wants_timestamps = requires_timestamps(
            response_format=payload.response_format,
            granularity_set=granularity_set,
        )
        with model_manager.lease(canonical_model) as lease:
            if lease.runtime is None:
                raise RuntimeError(f"Model '{canonical_model}' is not loaded.")
            transcription = await run_in_threadpool(
                transcribe_sync,
                runtime=lease.runtime,
                audio_path=tmp_path,
                language=payload.language or None,
                task=payload.task,
                prompt=payload.prompt or None,
                temperature=payload.temperature,
                word_timestamps=("word" in granularity_set) and lease.spec.family == "whisper",
            )
            segments = transcription.segments
            if wants_timestamps:
                segments = await run_in_threadpool(
                    ensure_timestamp_segments,
                    lease=lease,
                    audio_path=tmp_path,
                    transcription=transcription,
                )
            text = transcription.text or segments_to_text(segments)
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
        info=transcription.info,
        segments=segments,
    )
