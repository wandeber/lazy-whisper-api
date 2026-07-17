"""Local speaker diarization configuration, validation, and worker lifecycle."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import DiarizationSettings, Settings
from .diarization_types import DiarizationResult, DiarizationTurn
from .errors import api_error
from .worker_protocol import JsonLineWorkerClient


LOGGER = logging.getLogger("whisper_api")
DIARIZATION_WORKER_PATH = Path(__file__).resolve().parent / "diarization_worker.py"
DIARIZATION_MODEL_READY_MARKER = ".lazy-whisper-offline-ready"
HF_TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]+")
WORKER_INHERITED_ENV_NAMES = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
)


def redact_secrets(message: str) -> str:
    """Remove Hugging Face token-looking values before surfacing worker errors."""
    redacted = HF_TOKEN_PATTERN.sub("<redacted>", message)
    token = os.environ.get("HF_TOKEN")
    if token:
        redacted = redacted.replace(token, "<redacted>")
    return redacted


def diarization_installation_status(
    diarization: DiarizationSettings,
) -> dict[str, bool]:
    """Return cheap filesystem readiness checks without loading pyannote."""
    runtime_available = Path(diarization.runtime_python).is_file()
    model_root = Path(diarization.model_path)
    marker_matches_model = False
    try:
        marker = json.loads(
            (model_root / DIARIZATION_MODEL_READY_MARKER).read_text(encoding="utf-8")
        )
        marker_matches_model = (
            marker.get("format_version") == 1
            and marker.get("model_id") == diarization.model_id
        )
    except (OSError, json.JSONDecodeError, AttributeError):
        # Missing, stale and pre-metadata marker files are all treated as not
        # ready. Rerunning the setup command revalidates or resumes the model.
        pass
    model_available = (
        model_root.is_dir()
        and (model_root / "config.yaml").is_file()
        and marker_matches_model
    )
    return {
        "runtime_available": runtime_available,
        "model_available": model_available,
        "ready": diarization.enabled and runtime_available and model_available,
    }


def build_diarization_worker_env(
    *,
    settings: Settings,
    source_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build an air-gapped worker environment with no inherited HF credential.

    The setup command needs a Hugging Face token once to place the gated model
    on disk. Runtime requests do not: the worker loads only that local directory,
    and these offline flags turn an accidental remote reference into a hard error
    instead of a network call.
    """
    source = os.environ if source_env is None else source_env
    # Start from an allowlist instead of inheriting the API process environment.
    # This prevents unrelated credentials such as the API key or cloud-provider
    # tokens from crossing the sidecar boundary while retaining only locale,
    # temporary-directory, accelerator and dynamic-library settings needed by
    # Python, Torch and TorchCodec on the supported platforms.
    env = {
        name: source[name]
        for name in WORKER_INHERITED_ENV_NAMES
        if source.get(name)
    }
    env["PYTHONUNBUFFERED"] = "1"
    env["PYANNOTE_METRICS_ENABLED"] = "0"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    cache_root = settings.project_root / ".cache" / "diarization"
    runtime_home = cache_root / "home"
    hugging_face_home = cache_root / "huggingface"
    matplotlib_cache = cache_root / "matplotlib"
    torch_cache = cache_root / "torch"
    xdg_cache = cache_root / "xdg"
    for directory in (
        runtime_home,
        hugging_face_home,
        matplotlib_cache,
        torch_cache,
        xdg_cache,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    # HOME and all library caches are isolated from the user's real home. In
    # particular, huggingface_hub cannot discover a token saved by `hf auth`.
    env["HOME"] = str(runtime_home)
    env["HF_HOME"] = str(hugging_face_home)
    env["HF_TOKEN_PATH"] = str(hugging_face_home / "token-disabled")
    env["MPLCONFIGDIR"] = str(matplotlib_cache)
    env["TORCH_HOME"] = str(torch_cache)
    env["XDG_CACHE_HOME"] = str(xdg_cache)
    # Do not inherit PYTHONPATH: arbitrary parent search paths would allow the
    # sidecar to import code outside this repository and its isolated venv.
    env["PYTHONPATH"] = str(settings.project_root)
    return env


def normalize_speaker_count(value: int | None, field_name: str) -> int | None:
    """Validate one optional speaker-count request parameter."""
    if value is None:
        return None
    if value < 1:
        raise api_error(
            400,
            f"{field_name} must be greater than or equal to 1.",
            error_type="invalid_request_error",
        )
    return value


def validate_diarization_request(
    *,
    settings: Settings,
    response_format: str,
    task: str,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> tuple[int | None, int | None, int | None]:
    """Validate public diarization options before any expensive backend work."""
    if task != "transcribe":
        raise api_error(
            400,
            "Diarization is only supported for /v1/audio/transcriptions.",
            error_type="invalid_request_error",
        )
    if not settings.diarization.enabled:
        raise api_error(
            400,
            "Diarization is not enabled. Set ASR_DIARIZATION_ENABLED=true after installing the diarization runtime.",
            error_type="invalid_request_error",
        )
    if response_format != "verbose_json":
        raise api_error(
            400,
            "Diarization requires response_format=verbose_json so speaker labels are visible.",
            error_type="invalid_request_error",
        )

    normalized_num = normalize_speaker_count(num_speakers, "num_speakers")
    normalized_min = normalize_speaker_count(min_speakers, "min_speakers")
    normalized_max = normalize_speaker_count(max_speakers, "max_speakers")

    if normalized_num is not None and (
        normalized_min is not None or normalized_max is not None
    ):
        raise api_error(
            400,
            "num_speakers cannot be combined with min_speakers or max_speakers.",
            error_type="invalid_request_error",
        )
    if (
        normalized_min is not None
        and normalized_max is not None
        and normalized_min > normalized_max
    ):
        raise api_error(
            400,
            "min_speakers cannot be greater than max_speakers.",
            error_type="invalid_request_error",
        )

    installation = diarization_installation_status(settings.diarization)
    if not installation["runtime_available"]:
        raise api_error(
            503,
            "The local diarization runtime is not installed. Run make install-diarization-runtime.",
            error_type="server_error",
        )
    if not installation["model_available"]:
        raise api_error(
            503,
            "The local diarization model is not installed. Configure the setup token and run make install-diarization-runtime.",
            error_type="server_error",
        )
    return normalized_num, normalized_min, normalized_max


class DiarizationWorkerProxy:
    """JSON-RPC proxy for the isolated pyannote diarization runtime."""

    worker_pid: int | None = None

    def __init__(
        self,
        *,
        settings: Settings,
        diarization: DiarizationSettings,
    ) -> None:
        self.settings = settings
        self.diarization = diarization

        runtime_python = Path(diarization.runtime_python)
        if not runtime_python.is_file():
            raise RuntimeError(
                "Configured ASR_DIARIZATION_RUNTIME_PYTHON does not exist: "
                f"{runtime_python}"
            )
        model_path = Path(diarization.model_path)
        installation = diarization_installation_status(diarization)
        if not installation["model_available"]:
            raise RuntimeError(
                "Configured ASR_DIARIZATION_MODEL_PATH is not a complete local pipeline: "
                f"{model_path}"
            )

        env = build_diarization_worker_env(settings=settings)

        args = [
            str(runtime_python),
            str(DIARIZATION_WORKER_PATH),
            "--model-id",
            diarization.model_id,
            "--model-path",
            diarization.model_path,
            "--device",
            diarization.device,
        ]
        self._client = JsonLineWorkerClient(
            args=args,
            cwd=settings.project_root,
            env=env,
            label=f"diarization:{diarization.model_id}",
            startup_timeout_seconds=diarization.startup_timeout_seconds,
            request_timeout_seconds=diarization.request_timeout_seconds,
            sanitize_message=redact_secrets,
        )
        self.process = self._client.process
        self.worker_pid = self._client.worker_pid

    def is_running(self) -> bool:
        """Whether the sidecar process is still available for new requests."""
        return self._client.is_running()

    def diarize_file(
        self,
        *,
        audio_path: Path,
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> DiarizationResult:
        """Run speaker diarization in the sidecar process."""
        result = self._client.request(
            "diarize_file",
            {
                "audio_path": str(audio_path),
                "num_speakers": num_speakers,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            },
        )
        turns = [
            DiarizationTurn(
                start=float(turn.get("start", 0.0)),
                end=float(turn.get("end", 0.0)),
                speaker=str(turn.get("speaker", "")),
            )
            for turn in result.get("turns", [])
        ]
        return DiarizationResult(
            model=str(result.get("model", self.diarization.model_id)),
            device=str(result.get("device", self.diarization.device)),
            turns=turns,
            processing_seconds=(
                None
                if result.get("processing_seconds") is None
                else float(result["processing_seconds"])
            ),
        )

    def close(self) -> None:
        self._client.close()


class DiarizationManager:
    """Lazy-load one local diarization sidecar and serialize expensive runs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._run_lock = threading.Lock()
        self._runtime: DiarizationWorkerProxy | None = None
        self._loaded_at: float | None = None
        self._last_used: float | None = None
        self._unload_at: float | None = None
        self._timer: threading.Timer | None = None
        self._active_runs = 0
        self._generation = 0
        self._last_error: str | None = None
        self._loading = False

    def _cancel_timer_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._unload_at = None

    def _runtime_for_run(self) -> DiarizationWorkerProxy:
        """Return a live runtime, loading it without holding the state lock.

        Loading pyannote can take several minutes on the first request. The
        manager state lock must remain available during that wait so `/healthz`
        can report `loading` instead of blocking for the complete startup
        timeout. `_run_lock` guarantees that only one caller can reach this
        method, so no second worker can be created concurrently.
        """
        runtime_to_close: DiarizationWorkerProxy | None = None
        with self._lock:
            if self._runtime is not None:
                if self._runtime.is_running():
                    self._cancel_timer_locked()
                    return self._runtime
                runtime_to_close = self._unload_locked()
            if self.settings.diarization.backend != "pyannote":
                raise RuntimeError(
                    f"Unsupported diarization backend '{self.settings.diarization.backend}'."
                )
            self._loading = True

        try:
            if runtime_to_close is not None:
                runtime_to_close.close()
            runtime = DiarizationWorkerProxy(
                settings=self.settings,
                diarization=self.settings.diarization,
            )
        except Exception:
            with self._lock:
                self._loading = False
            raise

        with self._lock:
            self._runtime = runtime
            self._loading = False
            self._generation += 1
            now = time.time()
            self._loaded_at = now
            self._last_used = now
        LOGGER.info(
            "Loaded diarization backend '%s' on %s",
            self.settings.diarization.model_id,
            self.settings.diarization.device,
        )
        return runtime

    def _schedule_unload_locked(self) -> None:
        self._cancel_timer_locked()
        idle_seconds = self.settings.diarization.idle_seconds
        self._unload_at = time.time() + idle_seconds
        generation = self._generation
        # Use the same generation-checked path for immediate unloads. Starting
        # a zero-delay timer avoids closing the process while this lock is held
        # and keeps the stale-timer protection identical for every idle policy.
        timer = threading.Timer(max(0, idle_seconds), self.unload_if_idle, args=(generation,))
        timer.daemon = True
        self._timer = timer
        timer.start()

    def _unload_locked(self) -> DiarizationWorkerProxy | None:
        self._cancel_timer_locked()
        runtime = self._runtime
        self._runtime = None
        self._loaded_at = None
        self._last_used = None
        if runtime is not None:
            self._generation += 1
        return runtime

    def unload_if_idle(self, generation: int) -> None:
        """Unload only when the timer still matches the idle runtime generation."""
        # A timer must not detach a runtime and continue closing it in its daemon
        # thread while FastAPI's lifespan concurrently concludes that there is
        # nothing left to unload. Sharing the run lock makes shutdown wait for a
        # close already in progress. A timer that races with a new reservation
        # simply gives up; the reservation will schedule a fresh idle deadline.
        if not self._run_lock.acquire(blocking=False):
            return
        try:
            with self._lock:
                if self._active_runs > 0 or generation != self._generation:
                    return
                runtime = self._unload_locked()
            if runtime is not None:
                runtime.close()
                LOGGER.info("Unloaded diarization backend")
        finally:
            self._run_lock.release()

    @contextmanager
    def reserve(self) -> Iterator[None]:
        """Reserve the single diarization slot before upstream ASR work starts.

        A diarized request performs ASR first and pyannote second. Reserving here
        prevents a concurrent request from spending minutes transcribing only to
        discover at the final stage that the diarization worker is busy. The lock
        is acquired non-blockingly, so rejected requests still receive an immediate
        `429` and no event-loop thread waits for the current job to finish.
        """
        if not self.settings.diarization.enabled:
            raise api_error(
                400,
                "Diarization is not enabled.",
                error_type="invalid_request_error",
            )
        if not self._run_lock.acquire(blocking=False):
            raise api_error(
                429,
                "Diarization backend is busy.",
                error_type="server_busy",
            )

        try:
            with self._lock:
                self._active_runs += 1
                self._last_error = None
                # A warm worker must remain alive while the reserved request performs
                # ASR. The reservation exit schedules a fresh idle deadline.
                self._cancel_timer_locked()
        except Exception:
            self._run_lock.release()
            raise

        try:
            yield
        finally:
            try:
                runtime_to_close: DiarizationWorkerProxy | None = None
                with self._lock:
                    self._active_runs = max(0, self._active_runs - 1)
                    if self._runtime is not None and not self._runtime.is_running():
                        runtime_to_close = self._unload_locked()
                    elif self._runtime is not None:
                        self._schedule_unload_locked()
                if runtime_to_close is not None:
                    runtime_to_close.close()
            finally:
                self._run_lock.release()

    def diarize_reserved(
        self,
        *,
        audio_path: Path,
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> DiarizationResult:
        """Run pyannote while the caller owns a reservation."""
        with self._lock:
            if self._active_runs < 1:
                raise RuntimeError("Diarization requires an active reservation.")

        try:
            runtime = self._runtime_for_run()
            result = runtime.diarize_file(
                audio_path=audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        except Exception as exc:
            with self._lock:
                self._last_error = redact_secrets(str(exc))
            raise

        with self._lock:
            self._last_used = time.time()
            self._last_error = None
        return result

    def diarize(
        self,
        *,
        audio_path: Path,
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> DiarizationResult:
        """Reserve and run a standalone diarization job."""
        with self.reserve():
            return self.diarize_reserved(
                audio_path=audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

    def unload(self) -> None:
        # Serialize explicit application shutdown with an in-flight load or run;
        # otherwise a worker created just after `_unload_locked` could be leaked.
        with self._run_lock:
            with self._lock:
                runtime = self._unload_locked()
        if runtime is not None:
            runtime.close()
            LOGGER.info("Unloaded diarization backend")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            installation = diarization_installation_status(self.settings.diarization)
            loaded = self._runtime is not None and self._runtime.is_running()
            if not self.settings.diarization.enabled:
                state = "disabled"
            elif not installation["runtime_available"]:
                state = "runtime_missing"
            elif not installation["model_available"]:
                state = "model_missing"
            elif self._loading:
                state = "loading"
            elif self._active_runs:
                state = "busy"
            elif self._last_error is not None:
                state = "error"
            elif loaded:
                state = "loaded"
            else:
                state = "ready"
            return {
                "enabled": self.settings.diarization.enabled,
                "backend": self.settings.diarization.backend,
                "model": self.settings.diarization.model_id,
                "model_path": self.settings.diarization.model_path,
                "device": self.settings.diarization.device,
                "offline": True,
                **installation,
                "state": state,
                "loaded": loaded,
                "worker_pid": self._runtime.worker_pid if self._runtime is not None else None,
                "loaded_at": self._loaded_at,
                "last_used": self._last_used,
                "unload_at": self._unload_at,
                "active_runs": self._active_runs,
                "last_error": self._last_error,
            }
