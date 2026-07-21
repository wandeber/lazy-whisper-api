"""Request validation and transcription orchestration."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import av
from fastapi import HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from .audio_timeline import DecodedPcm16Timeline, decode_audio_timeline
from .backends import (
    BackendTranscription,
    SegmentTiming,
    TranscriptionInfo,
    WordTiming,
    segments_to_text,
    write_pcm16_wav,
)
from .config import ModelRoute, Settings
from .diarization import DiarizationManager, validate_diarization_request
from .diarization_types import DiarizationResult
from .editing import build_edit_transcript
from .editing_types import EditingResult
from .errors import api_error
from .model_manager import LoadedModel, ModelManager
from .silero_vad import analyze_speech
from .speaker_attribution import enrich_segments_with_speakers


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
    diarize: bool
    num_speakers: int | None
    min_speakers: int | None
    max_speakers: int | None


@dataclass(frozen=True)
class TranscriptionResult:
    """Raw output returned by one ASR backend plus request metadata."""

    model_name: str
    device: str
    response_format: str
    text: str
    info: Any
    segments: list[Any]
    diarization: DiarizationResult | None = None
    editing: EditingResult | None = None


def normalize_timestamp_granularities(values: list[str] | None) -> set[str]:
    """Return a normalized timestamp granularity set."""
    return set(values or [])


def requires_timestamps(*, response_format: str, granularity_set: set[str]) -> bool:
    """Whether this request needs timestamps in the final response."""
    return response_format in {"srt", "vtt", "verbose_json"} or bool(granularity_set)


def validate_request(
    settings: Settings,
    payload: TranscriptionRequest,
    *,
    surface: str = "batch",
) -> ModelRoute:
    """Validate public request parameters and preserve the resolved profile."""
    if not payload.diarize and any(
        value is not None
        for value in (payload.num_speakers, payload.min_speakers, payload.max_speakers)
    ):
        raise api_error(
            400,
            "Speaker-count options require diarize=true.",
            error_type="invalid_request_error",
        )

    try:
        route = settings.resolve_model_route(payload.model)
    except KeyError as exc:
        supported = ", ".join(settings.supported_model_ids)
        raise api_error(
            400,
            f"Unsupported model '{payload.model}'. Supported values: {supported}.",
            error_type="invalid_request_error",
        ) from exc

    spec = settings.model_settings[route.canonical_model]
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

    if route.profile.is_edit_max:
        if surface != "batch":
            raise api_error(
                400,
                f"Model '{payload.model}' is available only for non-streaming batch transcription.",
                error_type="invalid_request_error",
            )
        if payload.task != "transcribe":
            raise api_error(
                400,
                f"Model '{payload.model}' supports transcription only.",
                error_type="invalid_request_error",
            )
        if payload.response_format != "verbose_json":
            raise api_error(
                400,
                f"Model '{payload.model}' requires response_format='verbose_json'.",
                error_type="invalid_request_error",
            )

    granularity_set = normalize_timestamp_granularities(payload.timestamp_granularities)
    invalid_granularities = granularity_set - SUPPORTED_TIMESTAMP_GRANULARITIES
    if invalid_granularities:
        raise api_error(
            400,
            "Unsupported timestamp_granularities value.",
            error_type="invalid_request_error",
        )
    if payload.diarize:
        validate_diarization_request(
            settings=settings,
            response_format=payload.response_format,
            task=payload.task,
            num_speakers=payload.num_speakers,
            min_speakers=payload.min_speakers,
            max_speakers=payload.max_speakers,
        )
    if granularity_set and not spec.supports("timestamps"):
        raise api_error(
            400,
            f"Model '{payload.model}' does not support timestamps.",
            error_type="invalid_request_error",
        )
    needs_timestamped_response = payload.diarize or requires_timestamps(
        response_format=payload.response_format,
        granularity_set=granularity_set,
    )
    if needs_timestamped_response and not spec.supports("timestamps"):
        raise api_error(
            400,
            f"Model '{payload.model}' does not support timestamped responses.",
            error_type="invalid_request_error",
        )

    return route


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


def align_words_sync(
    *,
    runtime: Any,
    audio_path: Path,
    text: str,
    language: str,
) -> list[WordTiming]:
    """Run the edit-only exact-text aligner off the event loop."""
    return runtime.align_words_file(
        audio_path=audio_path,
        text=text,
        language=language,
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
    diarization_manager: DiarizationManager,
    payload: TranscriptionRequest,
) -> TranscriptionResult:
    """Persist the upload temporarily, run one backend, and clean up."""
    route = validate_request(settings, payload)
    canonical_model = route.canonical_model
    reservation = diarization_manager.reserve() if payload.diarize else nullcontext()
    tmp_path: Path | None = None
    canonical_audio_path: Path | None = None
    editing: EditingResult | None = None

    try:
        with reservation:
            suffix = Path(payload.file.filename or "upload.bin").suffix
            with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = Path(tmp.name)

            await write_upload_to_tempfile(
                upload=payload.file,
                destination=tmp_path,
                chunk_size=settings.upload_chunk_size,
            )

            granularity_set = normalize_timestamp_granularities(payload.timestamp_granularities)
            if route.profile.is_edit_max:
                edit_settings = route.profile.edit_max
                if edit_settings is None:  # pragma: no cover - protected by config validation
                    raise RuntimeError("The edit-max profile has no acoustic settings.")
                timeline: DecodedPcm16Timeline = await run_in_threadpool(
                    decode_audio_timeline,
                    audio_path=tmp_path,
                    sample_rate_hz=edit_settings.sample_rate_hz,
                )
                with NamedTemporaryFile(delete=False, suffix=".wav") as canonical_tmp:
                    canonical_audio_path = Path(canonical_tmp.name)
                await run_in_threadpool(timeline.write_wav, canonical_audio_path)

                with model_manager.lease(canonical_model) as lease:
                    if lease.runtime is None:
                        raise RuntimeError(f"Model '{canonical_model}' is not loaded.")
                    transcription = await run_in_threadpool(
                        transcribe_sync,
                        runtime=lease.runtime,
                        audio_path=canonical_audio_path,
                        language=payload.language or None,
                        task=payload.task,
                        prompt=payload.prompt or None,
                        temperature=payload.temperature,
                        word_timestamps=False,
                    )
                    text = transcription.text.strip()
                    resolved_language = (
                        transcription.info.language or payload.language or ""
                    ).strip()
                    aligned_words: list[WordTiming] = []
                    if text:
                        if not resolved_language:
                            raise RuntimeError(
                                "Edit-max forced alignment requires a detected or "
                                "requested language."
                            )
                        aligned_words = await run_in_threadpool(
                            align_words_sync,
                            runtime=lease.runtime,
                            audio_path=canonical_audio_path,
                            text=text,
                            language=resolved_language,
                        )
                        if not any(word.word.strip() for word in aligned_words):
                            # The proxy enforces this too. Repeating the gate here
                            # prevents a future backend from silently degrading a
                            # precision profile into an acoustic-only transcript.
                            raise RuntimeError(
                                "Edit-max forced alignment returned no words for a "
                                "non-empty transcript."
                            )
                    device = lease.actual_device

                vad = await run_in_threadpool(
                    analyze_speech,
                    pcm_bytes=timeline.pcm_bytes,
                    settings=edit_settings,
                )
                edit_transcript = await run_in_threadpool(
                    build_edit_transcript,
                    aligned_words=aligned_words,
                    vad=vad,
                    settings=edit_settings,
                    requested_model=route.requested_model,
                    canonical_model=canonical_model,
                    profile_name=route.profile.name,
                )
                segments = list(edit_transcript.segments)
                editing = edit_transcript.editing
                info = TranscriptionInfo(
                    language=resolved_language,
                    duration=timeline.duration,
                    language_probability=transcription.info.language_probability,
                )
                diarization_audio_path = canonical_audio_path
            else:
                wants_timestamps = payload.diarize or requires_timestamps(
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
                        # Diarization needs the smallest practical ASR intervals. A single
                        # Whisper segment often spans a speaker change, while word timings
                        # let the reconciliation step preserve that change accurately.
                        word_timestamps=(payload.diarize or "word" in granularity_set)
                        and lease.spec.family == "whisper",
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
                info = transcription.info
                diarization_audio_path = tmp_path

            diarization: DiarizationResult | None = None
            if payload.diarize:
                diarization = await run_in_threadpool(
                    diarization_manager.diarize_reserved,
                    audio_path=diarization_audio_path,
                    num_speakers=payload.num_speakers,
                    min_speakers=payload.min_speakers,
                    max_speakers=payload.max_speakers,
                )
                segments = enrich_segments_with_speakers(
                    segments=segments,
                    turns=diarization.turns,
                )
    except HTTPException:
        raise
    except Exception as exc:
        raise api_error(500, str(exc), error_type="server_error") from exc
    finally:
        await payload.file.close()
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        if canonical_audio_path is not None:
            canonical_audio_path.unlink(missing_ok=True)

    return TranscriptionResult(
        model_name=canonical_model,
        device=device,
        response_format=payload.response_format,
        text=text,
        info=info,
        segments=segments,
        diarization=diarization,
        editing=editing,
    )
