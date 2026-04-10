#!/usr/bin/env python3
import atexit
import gc
import io
import logging
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from faster_whisper import WhisperModel
from starlette.requests import Request


APP_DIR = Path(__file__).resolve().parent
DOWNLOAD_ROOT = str(APP_DIR / ".cache" / "faster-whisper")
DEFAULT_MODEL = os.environ.get("WHISPER_DEFAULT_MODEL", "turbo")
DEFAULT_DEVICE = os.environ.get("WHISPER_DEFAULT_DEVICE", "cpu")
API_KEY = os.environ.get("WHISPER_API_KEY", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN") or None
CPU_THREADS = int(os.environ.get("WHISPER_CPU_THREADS", "0"))
LOG_LEVEL = os.environ.get("WHISPER_LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=LOG_LEVEL)
LOGGER = logging.getLogger("whisper_api")


def parse_mapping(raw_value: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in raw_value.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"Invalid mapping entry: {entry}")
        key, value = entry.split("=", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def parse_int_mapping(raw_value: str) -> dict[str, int]:
    mapping = parse_mapping(raw_value)
    return {key: int(value) for key, value in mapping.items()}


def parse_bool(raw_value: str, default: bool = False) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


MODEL_ALIAS_MAP = parse_mapping(
    os.environ.get(
        "WHISPER_MODEL_ALIAS_MAP",
        "whisper-1=turbo,turbo=turbo,large-v3=large-v3,distil=distil-multi4,distil-multi4=distil-multi4",
    )
)
MODEL_SOURCE_MAP = parse_mapping(
    os.environ.get(
        "WHISPER_MODEL_SOURCE_MAP",
        f"turbo=turbo,large-v3=large-v3,distil-multi4={APP_DIR / 'models' / 'distil-multi4-ct2'}",
    )
)
MODEL_DEVICE_MAP = parse_mapping(
    os.environ.get(
        "WHISPER_MODEL_DEVICE_MAP",
        f"turbo=cuda,large-v3=cpu,distil-multi4=cuda",
    )
)
MODEL_COMPUTE_TYPE_MAP = parse_mapping(
    os.environ.get(
        "WHISPER_MODEL_COMPUTE_TYPE_MAP",
        "turbo=default,large-v3=default,distil-multi4=default",
    )
)
MODEL_IDLE_SECONDS_MAP = parse_int_mapping(
    os.environ.get(
        "WHISPER_MODEL_IDLE_SECONDS_MAP",
        "turbo=5400,large-v3=0,distil-multi4=5400",
    )
)
MODEL_VAD_MAP = {
    key: parse_bool(value, default=False)
    for key, value in parse_mapping(
        os.environ.get(
            "WHISPER_MODEL_VAD_MAP",
            "turbo=false,large-v3=false,distil-multi4=false",
        )
    ).items()
}
CANONICAL_MODELS = sorted(set(MODEL_SOURCE_MAP) | set(MODEL_ALIAS_MAP.values()) | {DEFAULT_MODEL})

app = FastAPI(title="Local Whisper API", version="2.0.0")


@dataclass
class ModelSpec:
    name: str
    source: str
    preferred_device: str
    compute_type: str
    idle_seconds: int
    vad_filter: bool


@dataclass
class LoadedModel:
    spec: ModelSpec
    model: WhisperModel | None
    actual_device: str
    actual_compute_type: str
    loaded_at: float
    last_used: float
    use_count: int = 0
    unload_at: float | None = None
    timer: threading.Timer | None = None


def resolve_model_name(requested_model: str) -> str:
    canonical = MODEL_ALIAS_MAP.get(requested_model, requested_model)
    if canonical not in CANONICAL_MODELS:
        supported = ", ".join(sorted(set(MODEL_ALIAS_MAP) | set(CANONICAL_MODELS)))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{requested_model}'. Supported values: {supported}.",
        )
    return canonical


def build_model_spec(model_name: str) -> ModelSpec:
    if model_name not in MODEL_SOURCE_MAP:
        raise HTTPException(status_code=500, detail=f"Model '{model_name}' is not configured.")
    return ModelSpec(
        name=model_name,
        source=MODEL_SOURCE_MAP[model_name],
        preferred_device=MODEL_DEVICE_MAP.get(model_name, DEFAULT_DEVICE),
        compute_type=MODEL_COMPUTE_TYPE_MAP.get(model_name, "default"),
        idle_seconds=MODEL_IDLE_SECONDS_MAP.get(model_name, 5400),
        vad_filter=MODEL_VAD_MAP.get(model_name, False),
    )


class ModelManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._loaded: dict[str, LoadedModel] = {}

    def _cancel_timer_locked(self, entry: LoadedModel) -> None:
        if entry.timer is not None:
            entry.timer.cancel()
            entry.timer = None
        entry.unload_at = None

    def _unload_entry_locked(self, model_name: str) -> bool:
        entry = self._loaded.get(model_name)
        if entry is None or entry.use_count > 0:
            return False

        self._cancel_timer_locked(entry)
        del self._loaded[model_name]
        model = entry.model
        entry.model = None
        device = entry.actual_device
        del model
        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        LOGGER.info("Unloaded model '%s' from %s", model_name, device)
        return True

    def unload(self, model_name: str) -> bool:
        with self._lock:
            return self._unload_entry_locked(model_name)

    def unload_all(self) -> None:
        with self._lock:
            names = list(self._loaded)
        for model_name in names:
            self.unload(model_name)

    def _schedule_unload_locked(self, entry: LoadedModel) -> None:
        self._cancel_timer_locked(entry)
        if entry.spec.idle_seconds <= 0:
            self._unload_entry_locked(entry.spec.name)
            return

        entry.unload_at = time.time() + entry.spec.idle_seconds
        timer = threading.Timer(entry.spec.idle_seconds, self.unload, args=(entry.spec.name,))
        timer.daemon = True
        entry.timer = timer
        timer.start()

    def _evict_idle_gpu_models_locked(self, exclude: set[str]) -> None:
        candidates = sorted(
            (
                loaded
                for loaded in self._loaded.values()
                if loaded.actual_device.startswith("cuda")
                and loaded.use_count == 0
                and loaded.spec.name not in exclude
            ),
            key=lambda loaded: loaded.last_used,
        )
        for loaded in candidates:
            self._unload_entry_locked(loaded.spec.name)

    def _preferred_device(self, spec: ModelSpec) -> str:
        if spec.preferred_device.startswith("cuda") and not torch.cuda.is_available():
            LOGGER.warning("CUDA is not available for %s; using CPU instead", spec.name)
            return "cpu"
        return spec.preferred_device

    def _load_model(self, spec: ModelSpec) -> LoadedModel:
        requested_device = self._preferred_device(spec)
        candidate_devices = [requested_device]
        if requested_device.startswith("cuda"):
            candidate_devices.append("cpu")

        last_error: Exception | None = None
        for device in candidate_devices:
            try:
                LOGGER.info(
                    "Loading model '%s' from '%s' on %s with compute_type=%s",
                    spec.name,
                    spec.source,
                    device,
                    spec.compute_type,
                )
                model = WhisperModel(
                    spec.source,
                    device=device,
                    compute_type=spec.compute_type,
                    download_root=DOWNLOAD_ROOT,
                    cpu_threads=CPU_THREADS,
                    use_auth_token=HF_TOKEN,
                )
                return LoadedModel(
                    spec=spec,
                    model=model,
                    actual_device=device,
                    actual_compute_type=spec.compute_type,
                    loaded_at=time.time(),
                    last_used=time.time(),
                )
            except Exception as exc:
                last_error = exc
                if device.startswith("cuda"):
                    LOGGER.warning(
                        "Could not load model '%s' on %s: %s",
                        spec.name,
                        device,
                        exc,
                    )
                    with self._lock:
                        self._evict_idle_gpu_models_locked(exclude={spec.name})
                    try:
                        LOGGER.info("Retrying model '%s' on %s after evicting idle GPU models", spec.name, device)
                        model = WhisperModel(
                            spec.source,
                            device=device,
                            compute_type=spec.compute_type,
                            download_root=DOWNLOAD_ROOT,
                            cpu_threads=CPU_THREADS,
                            use_auth_token=HF_TOKEN,
                        )
                        return LoadedModel(
                            spec=spec,
                            model=model,
                            actual_device=device,
                            actual_compute_type=spec.compute_type,
                            loaded_at=time.time(),
                            last_used=time.time(),
                        )
                    except Exception as retry_exc:
                        last_error = retry_exc
                        LOGGER.warning(
                            "Retry on %s for model '%s' also failed: %s",
                            device,
                            spec.name,
                            retry_exc,
                        )
                        continue
                continue

        raise RuntimeError(f"Could not load model '{spec.name}': {last_error}") from last_error

    @contextmanager
    def lease(self, model_name: str) -> LoadedModel:
        spec = build_model_spec(model_name)
        with self._lock:
            entry = self._loaded.get(model_name)
            if entry is not None:
                self._cancel_timer_locked(entry)
                entry.use_count += 1
            else:
                entry = self._load_model(spec)
                entry.use_count = 1
                self._loaded[model_name] = entry

        try:
            yield entry
        finally:
            with self._lock:
                current = self._loaded.get(model_name)
                if current is None:
                    return
                current.use_count = max(0, current.use_count - 1)
                current.last_used = time.time()
                if current.use_count == 0:
                    self._schedule_unload_locked(current)

    def snapshot(self) -> list[dict[str, Any]]:
        now = time.time()
        with self._lock:
            loaded = []
            for model_name, entry in sorted(self._loaded.items()):
                loaded.append(
                    {
                        "id": model_name,
                        "device": entry.actual_device,
                        "compute_type": entry.actual_compute_type,
                        "busy": entry.use_count > 0,
                        "idle_seconds": entry.spec.idle_seconds,
                        "seconds_until_unload": (
                            None
                            if entry.use_count > 0 or entry.unload_at is None
                            else max(0, int(entry.unload_at - now))
                        ),
                    }
                )
            return loaded


MODEL_MANAGER = ModelManager()
atexit.register(MODEL_MANAGER.unload_all)


def format_timestamp(seconds: float, *, always_include_hours: bool = False, decimal_marker: str = ".") -> str:
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    secs = milliseconds // 1000
    milliseconds -= secs * 1000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{secs:02d}{decimal_marker}{milliseconds:03d}"


def segments_to_text(segments: list[Any]) -> str:
    return "".join(segment.text for segment in segments).strip()


def write_srt(segments: list[Any]) -> str:
    output = io.StringIO()
    for index, segment in enumerate(segments, start=1):
        output.write(f"{index}\n")
        output.write(
            f"{format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')} --> "
            f"{format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')}\n"
        )
        output.write(f"{segment.text.strip()}\n\n")
    return output.getvalue()


def write_vtt(segments: list[Any]) -> str:
    output = io.StringIO()
    output.write("WEBVTT\n\n")
    for segment in segments:
        output.write(
            f"{format_timestamp(segment.start, always_include_hours=True)} --> "
            f"{format_timestamp(segment.end, always_include_hours=True)}\n"
        )
        output.write(f"{segment.text.strip()}\n\n")
    return output.getvalue()


def build_verbose_json(
    *,
    model_name: str,
    device: str,
    text: str,
    info: Any,
    segments: list[Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_name,
        "device": device,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "text": text,
        "segments": [],
    }

    words: list[dict[str, Any]] = []
    for segment in segments:
        payload["segments"].append(
            {
                "id": segment.id,
                "seek": segment.seek,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "tokens": segment.tokens,
                "temperature": segment.temperature,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob,
                "words": [
                    {
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "probability": word.probability,
                    }
                    for word in (segment.words or [])
                ],
            }
        )
        for word in segment.words or []:
            words.append(
                {
                    "start": word.start,
                    "end": word.end,
                    "word": word.word,
                    "probability": word.probability,
                }
            )
    if words:
        payload["words"] = words
    return payload


def api_error(status_code: int, message: str, *, error_type: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "message": message,
            "type": error_type,
        },
    )


def require_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    if not API_KEY:
        return

    provided_key: str | None = None
    if authorization and authorization.lower().startswith("bearer "):
        provided_key = authorization[7:].strip()
    elif x_api_key:
        provided_key = x_api_key.strip()

    if provided_key != API_KEY:
        raise api_error(
            401,
            "Invalid API key provided.",
            error_type="invalid_api_key",
        )


async def transcribe_request(
    *,
    file: UploadFile,
    model: str,
    task: str,
    language: str | None,
    prompt: str | None,
    response_format: str,
    temperature: float,
    timestamp_granularities: list[str] | None,
) -> JSONResponse | PlainTextResponse:
    canonical_model = resolve_model_name(model)

    if response_format not in {"json", "text", "srt", "verbose_json", "vtt"}:
        raise api_error(400, "Unsupported response_format.", error_type="invalid_request_error")

    granularity_set = set(timestamp_granularities or [])
    invalid_granularities = granularity_set - {"segment", "word"}
    if invalid_granularities:
        raise api_error(
            400,
            "Unsupported timestamp_granularities value.",
            error_type="invalid_request_error",
        )

    suffix = Path(file.filename or "upload.bin").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=APP_DIR) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with tmp_path.open("wb") as handle:
            handle.write(await file.read())

        with MODEL_MANAGER.lease(canonical_model) as lease:
            if lease.model is None:
                raise RuntimeError(f"Model '{canonical_model}' is not loaded.")
            segments_iter, info = lease.model.transcribe(
                str(tmp_path),
                language=language or None,
                task=task,
                initial_prompt=prompt or None,
                temperature=temperature,
                word_timestamps="word" in granularity_set,
                vad_filter=lease.spec.vad_filter,
            )
            segments = list(segments_iter)
            text = segments_to_text(segments)
            device = lease.actual_device
    except HTTPException:
        raise
    except Exception as exc:
        raise api_error(500, str(exc), error_type="server_error") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    if response_format == "json":
        return JSONResponse({"text": text, "model": canonical_model})
    if response_format == "verbose_json":
        return JSONResponse(
            build_verbose_json(
                model_name=canonical_model,
                device=device,
                text=text,
                info=info,
                segments=segments,
            )
        )
    if response_format == "text":
        return PlainTextResponse(text)
    if response_format == "srt":
        return PlainTextResponse(write_srt(segments))
    if response_format == "vtt":
        return PlainTextResponse(write_vtt(segments))

    raise api_error(500, "Unexpected response format handling.", error_type="server_error")


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
    error_type = detail.get("type")
    if error_type is None:
        if exc.status_code == 401:
            error_type = "invalid_api_key"
        elif exc.status_code >= 500:
            error_type = "server_error"
        else:
            error_type = "invalid_request_error"

    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": detail.get("message", ""), "type": error_type}},
    )


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL,
        "cuda_available": torch.cuda.is_available(),
        "loaded_models": MODEL_MANAGER.snapshot(),
    }


@app.get("/v1/models", dependencies=[Depends(require_api_key)])
def list_models() -> dict[str, Any]:
    model_ids = sorted(set(MODEL_ALIAS_MAP) | set(CANONICAL_MODELS))
    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "owned_by": "local-whisper"}
            for model_id in model_ids
        ],
    }


@app.post("/v1/audio/transcriptions", response_model=None, dependencies=[Depends(require_api_key)])
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    timestamp_granularities: list[str] | None = Form(default=None, alias="timestamp_granularities[]"),
):
    return await transcribe_request(
        file=file,
        model=model,
        task="transcribe",
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        timestamp_granularities=timestamp_granularities,
    )


@app.post("/v1/audio/translations", response_model=None, dependencies=[Depends(require_api_key)])
async def create_translation(
    file: UploadFile = File(...),
    model: str = Form(...),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    timestamp_granularities: list[str] | None = Form(default=None, alias="timestamp_granularities[]"),
):
    return await transcribe_request(
        file=file,
        model=model,
        task="translate",
        language=None,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        timestamp_granularities=timestamp_granularities,
    )
