"""Backend runtimes for local ASR model families."""

from __future__ import annotations

import base64
import gc
import json
import logging
import os
import subprocess
import tempfile
import threading
import uuid
import wave
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from faster_whisper import WhisperModel

from .config import ModelSettings, Settings


LOGGER = logging.getLogger("whisper_api")
QWEN_WORKER_PATH = Path(__file__).resolve().parent / "qwen_worker.py"
QWEN_LANGUAGE_MAP = {
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
}
QWEN_LANGUAGE_REVERSE_MAP = {value.lower(): key for key, value in QWEN_LANGUAGE_MAP.items()}
QWEN_DEFAULT_SAMPLE_RATE_HZ = 16_000
PCM16_SAMPLE_WIDTH_BYTES = 2
PCM16_CHANNELS = 1


@dataclass(frozen=True)
class WordTiming:
    """Normalized word timing."""

    start: float
    end: float
    word: str
    probability: float | None = None


@dataclass(frozen=True)
class SegmentTiming:
    """Normalized segment timing."""

    id: int
    start: float
    end: float
    text: str
    seek: int = 0
    tokens: list[int] = field(default_factory=list)
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0
    words: list[WordTiming] = field(default_factory=list)


@dataclass(frozen=True)
class TranscriptionInfo:
    """Normalized transcription metadata."""

    language: str
    duration: float
    language_probability: float | None = None


@dataclass(frozen=True)
class BackendTranscription:
    """Normalized backend transcription payload."""

    text: str
    info: TranscriptionInfo
    segments: list[SegmentTiming]


class RuntimeHandle(ABC):
    """A loaded runtime instance for one model."""

    supports_native_streaming: bool = False
    worker_pid: int | None = None
    preferred_stream_sample_rate_hz: int = 24_000

    def __init__(self, *, spec: ModelSettings, settings: Settings, device: str) -> None:
        self.spec = spec
        self.settings = settings
        self.device = device

    @abstractmethod
    def transcribe_file(
        self,
        *,
        audio_path: Path,
        language: str | None,
        task: str,
        prompt: str | None,
        temperature: float,
        word_timestamps: bool,
    ) -> BackendTranscription:
        """Transcribe one audio file."""

    def iter_transcribe_file(
        self,
        *,
        audio_path: Path,
        language: str | None,
        task: str,
        prompt: str | None,
        temperature: float,
        word_timestamps: bool,
    ) -> tuple[Iterable[SegmentTiming], TranscriptionInfo] | None:
        """Return a native streaming iterator when the backend supports it."""
        return None

    @abstractmethod
    def transcribe_pcm(
        self,
        *,
        pcm_bytes: bytes,
        sample_rate_hz: int,
        language: str | None,
        task: str,
        prompt: str | None,
        temperature: float,
        word_timestamps: bool,
    ) -> BackendTranscription:
        """Transcribe raw PCM16 mono audio."""

    def align_file(
        self,
        *,
        audio_path: Path,
        text: str,
        language: str,
    ) -> list[SegmentTiming]:
        """Return aligned timestamps when supported by the backend."""
        raise NotImplementedError(
            f"Backend '{self.spec.backend}' for model '{self.spec.name}' does not support alignment."
        )

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""


def normalize_qwen_language(language: str | None) -> str | None:
    """Translate ISO-ish API language codes into Qwen language names."""
    if language is None:
        return None
    stripped = language.strip()
    if not stripped:
        return None
    return QWEN_LANGUAGE_MAP.get(stripped.lower(), stripped)


def segments_to_text(segments: Iterable[SegmentTiming]) -> str:
    """Join segment text preserving natural spacing."""
    return "".join(segment.text for segment in segments).strip()


def build_segment_from_aligned_words(
    *,
    text: str,
    words: list[WordTiming],
) -> list[SegmentTiming]:
    """Convert aligned words into one coarse segment for OpenAI-compatible payloads."""
    if not words:
        return []
    return [
        SegmentTiming(
            id=0,
            start=words[0].start,
            end=words[-1].end,
            text=text,
            words=words,
        )
    ]


def write_pcm16_wav(
    *,
    pcm_bytes: bytes,
    sample_rate_hz: int,
    destination: Path,
) -> None:
    """Write PCM16 mono bytes to a temporary WAV file."""
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(PCM16_CHANNELS)
        handle.setsampwidth(PCM16_SAMPLE_WIDTH_BYTES)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(pcm_bytes)


class WhisperRuntime(RuntimeHandle):
    """Direct Faster Whisper runtime inside the main API process."""

    supports_native_streaming = True
    preferred_stream_sample_rate_hz = 24_000

    def __init__(self, *, spec: ModelSettings, settings: Settings, device: str) -> None:
        super().__init__(spec=spec, settings=settings, device=device)
        self.model = WhisperModel(
            spec.source,
            device=device,
            compute_type=spec.compute_type,
            download_root=settings.download_root,
            cpu_threads=settings.cpu_threads,
            use_auth_token=settings.hf_token,
        )

    def _normalize_segment(self, segment: Any) -> SegmentTiming:
        return SegmentTiming(
            id=getattr(segment, "id", 0),
            seek=getattr(segment, "seek", 0),
            start=float(getattr(segment, "start", 0.0)),
            end=float(getattr(segment, "end", 0.0)),
            text=getattr(segment, "text", "").strip(),
            tokens=list(getattr(segment, "tokens", [])),
            temperature=float(getattr(segment, "temperature", 0.0)),
            avg_logprob=float(getattr(segment, "avg_logprob", 0.0)),
            compression_ratio=float(getattr(segment, "compression_ratio", 0.0)),
            no_speech_prob=float(getattr(segment, "no_speech_prob", 0.0)),
            words=[
                WordTiming(
                    start=float(word.start),
                    end=float(word.end),
                    word=str(word.word),
                    probability=float(word.probability),
                )
                for word in (getattr(segment, "words", None) or [])
            ],
        )

    def _normalize_info(self, info: Any) -> TranscriptionInfo:
        return TranscriptionInfo(
            language=str(getattr(info, "language", "")),
            duration=float(getattr(info, "duration", 0.0)),
            language_probability=(
                None
                if getattr(info, "language_probability", None) is None
                else float(getattr(info, "language_probability"))
            ),
        )

    def _transcribe(
        self,
        *,
        audio_path: Path,
        language: str | None,
        task: str,
        prompt: str | None,
        temperature: float,
        word_timestamps: bool,
    ) -> tuple[Any, Any]:
        return self.model.transcribe(
            str(audio_path),
            language=language or None,
            task=task,
            initial_prompt=prompt or None,
            temperature=temperature,
            word_timestamps=word_timestamps,
            vad_filter=self.spec.vad_filter,
        )

    def transcribe_file(
        self,
        *,
        audio_path: Path,
        language: str | None,
        task: str,
        prompt: str | None,
        temperature: float,
        word_timestamps: bool,
    ) -> BackendTranscription:
        segments_iter, info = self._transcribe(
            audio_path=audio_path,
            language=language,
            task=task,
            prompt=prompt,
            temperature=temperature,
            word_timestamps=word_timestamps,
        )
        segments = [self._normalize_segment(segment) for segment in segments_iter]
        text = segments_to_text(segments)
        return BackendTranscription(
            text=text,
            info=self._normalize_info(info),
            segments=segments,
        )

    def iter_transcribe_file(
        self,
        *,
        audio_path: Path,
        language: str | None,
        task: str,
        prompt: str | None,
        temperature: float,
        word_timestamps: bool,
    ) -> tuple[Iterable[SegmentTiming], TranscriptionInfo] | None:
        segments_iter, info = self._transcribe(
            audio_path=audio_path,
            language=language,
            task=task,
            prompt=prompt,
            temperature=temperature,
            word_timestamps=word_timestamps,
        )
        normalized_info = self._normalize_info(info)

        def generator() -> Iterator[SegmentTiming]:
            for segment in segments_iter:
                yield self._normalize_segment(segment)

        return generator(), normalized_info

    def transcribe_pcm(
        self,
        *,
        pcm_bytes: bytes,
        sample_rate_hz: int,
        language: str | None,
        task: str,
        prompt: str | None,
        temperature: float,
        word_timestamps: bool,
    ) -> BackendTranscription:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = Path(tmp.name)
        try:
            write_pcm16_wav(
                pcm_bytes=pcm_bytes,
                sample_rate_hz=sample_rate_hz,
                destination=tmp_path,
            )
            return self.transcribe_file(
                audio_path=tmp_path,
                language=language,
                task=task,
                prompt=prompt,
                temperature=temperature,
                word_timestamps=word_timestamps,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    def close(self) -> None:
        model = self.model
        self.model = None
        del model
        gc.collect()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


class QwenWorkerProxy(RuntimeHandle):
    """Proxy object that talks to a dedicated qwen-asr worker process."""

    preferred_stream_sample_rate_hz = QWEN_DEFAULT_SAMPLE_RATE_HZ

    def __init__(self, *, spec: ModelSettings, settings: Settings, device: str) -> None:
        super().__init__(spec=spec, settings=settings, device=device)
        self._io_lock = threading.RLock()
        self._stderr_thread: threading.Thread | None = None
        python_path = spec.runtime_python
        if not python_path:
            raise RuntimeError(
                f"Model '{spec.name}' requires a separate runtime_python but none was configured."
            )

        runtime_python = Path(python_path)
        if not runtime_python.exists():
            raise RuntimeError(
                f"Configured runtime_python for model '{spec.name}' does not exist: {runtime_python}"
            )

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = (
            f"{settings.project_root}:{env['PYTHONPATH']}"
            if env.get("PYTHONPATH")
            else str(settings.project_root)
        )

        args = [
            str(runtime_python),
            str(QWEN_WORKER_PATH),
            "--model-name",
            spec.name,
            "--model-source",
            spec.source,
            "--device",
            device,
            "--dtype",
            spec.compute_type,
            "--aligner-source",
            spec.aligner_source or "",
            "--aligner-device",
            spec.aligner_device or "cpu",
            "--aligner-dtype",
            spec.aligner_dtype or "float32",
        ]
        self.process = subprocess.Popen(
            args,
            cwd=str(settings.project_root),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._start_stderr_drain()
        ready = self._read_startup_message()
        self.worker_pid = int(ready.get("pid", self.process.pid or 0)) or self.process.pid

    def _start_stderr_drain(self) -> None:
        if self.process.stderr is None:
            return

        def drain() -> None:
            assert self.process.stderr is not None
            for line in self.process.stderr:
                LOGGER.info("[qwen-worker:%s] %s", self.spec.name, line.rstrip())

        self._stderr_thread = threading.Thread(
            target=drain,
            daemon=True,
            name=f"qwen-worker-stderr-{self.spec.name}",
        )
        self._stderr_thread.start()

    def _read_startup_message(self) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError(f"Worker for '{self.spec.name}' has no stdout pipe.")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(
                f"Worker for '{self.spec.name}' exited before signalling readiness."
            )
        payload = json.loads(line)
        if payload.get("type") != "ready":
            raise RuntimeError(
                payload.get("error", {}).get("message")
                or f"Worker for '{self.spec.name}' failed to initialize."
            )
        return payload

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise RuntimeError(
                f"Worker for '{self.spec.name}' is not running (exit={self.process.returncode})."
            )
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError(f"Worker for '{self.spec.name}' has broken pipes.")

        request_id = uuid.uuid4().hex
        payload = {"id": request_id, "method": method, "params": params}
        encoded = json.dumps(payload, ensure_ascii=False) + "\n"

        with self._io_lock:
            self.process.stdin.write(encoded)
            self.process.stdin.flush()
            line = self.process.stdout.readline()

        if not line:
            raise RuntimeError(f"Worker for '{self.spec.name}' closed the connection.")

        response = json.loads(line)
        if response.get("id") != request_id:
            raise RuntimeError(
                f"Worker for '{self.spec.name}' returned a mismatched response id."
            )
        if not response.get("ok", False):
            error = response.get("error", {})
            raise RuntimeError(error.get("message", f"Worker error in {method}."))
        return response["result"]

    def _normalize_transcription(self, result: dict[str, Any]) -> BackendTranscription:
        raw_language = str(result.get("language", ""))
        normalized_language = QWEN_LANGUAGE_REVERSE_MAP.get(raw_language.lower(), raw_language)
        segments = [
            SegmentTiming(
                id=int(segment.get("id", 0)),
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=str(segment.get("text", "")),
                seek=int(segment.get("seek", 0)),
                tokens=list(segment.get("tokens", [])),
                temperature=float(segment.get("temperature", 0.0)),
                avg_logprob=float(segment.get("avg_logprob", 0.0)),
                compression_ratio=float(segment.get("compression_ratio", 0.0)),
                no_speech_prob=float(segment.get("no_speech_prob", 0.0)),
                words=[
                    WordTiming(
                        start=float(word.get("start", 0.0)),
                        end=float(word.get("end", 0.0)),
                        word=str(word.get("word", "")),
                        probability=(
                            None
                            if word.get("probability") is None
                            else float(word.get("probability"))
                        ),
                    )
                    for word in segment.get("words", [])
                ],
            )
            for segment in result.get("segments", [])
        ]
        info = TranscriptionInfo(
            language=normalized_language,
            duration=float(result.get("duration", 0.0)),
            language_probability=(
                None
                if result.get("language_probability") is None
                else float(result["language_probability"])
            ),
        )
        return BackendTranscription(
            text=str(result.get("text", "")).strip(),
            info=info,
            segments=segments,
        )

    def transcribe_file(
        self,
        *,
        audio_path: Path,
        language: str | None,
        task: str,
        prompt: str | None,
        temperature: float,
        word_timestamps: bool,
    ) -> BackendTranscription:
        if task != "transcribe":
            raise RuntimeError(
                f"Model '{self.spec.name}' only supports task='transcribe' in the Qwen backend."
            )
        result = self._request(
            "transcribe_file",
            {
                "audio_path": str(audio_path),
                "language": normalize_qwen_language(language),
                "prompt": prompt,
                "temperature": temperature,
            },
        )
        return self._normalize_transcription(result)

    def transcribe_pcm(
        self,
        *,
        pcm_bytes: bytes,
        sample_rate_hz: int,
        language: str | None,
        task: str,
        prompt: str | None,
        temperature: float,
        word_timestamps: bool,
    ) -> BackendTranscription:
        if task != "transcribe":
            raise RuntimeError(
                f"Model '{self.spec.name}' only supports task='transcribe' in the Qwen backend."
            )
        result = self._request(
            "transcribe_pcm",
            {
                "pcm_base64": base64.b64encode(pcm_bytes).decode("ascii"),
                "sample_rate_hz": int(sample_rate_hz),
                "language": normalize_qwen_language(language),
                "prompt": prompt,
                "temperature": temperature,
            },
        )
        return self._normalize_transcription(result)

    def align_file(
        self,
        *,
        audio_path: Path,
        text: str,
        language: str,
    ) -> list[SegmentTiming]:
        result = self._request(
            "align_file",
            {
                "audio_path": str(audio_path),
                "text": text,
                "language": normalize_qwen_language(language) or language,
            },
        )
        return self._normalize_transcription(
            {
                "text": text,
                "language": language,
                "duration": result.get("duration", 0.0),
                "segments": result.get("segments", []),
            }
        ).segments

    def close(self) -> None:
        with self._io_lock:
            try:
                self._request("shutdown", {})
            except Exception:
                pass

        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


def build_runtime(
    *,
    spec: ModelSettings,
    settings: Settings,
    device: str,
) -> RuntimeHandle:
    """Build one runtime handle for a model spec."""
    if spec.family == "whisper":
        return WhisperRuntime(spec=spec, settings=settings, device=device)
    if spec.family == "qwen":
        return QwenWorkerProxy(spec=spec, settings=settings, device=device)
    raise RuntimeError(
        f"Unsupported model family '{spec.family}' for model '{spec.name}'."
    )
