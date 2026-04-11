"""Lazy loading and lifecycle management for multi-backend ASR models."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from fastapi import HTTPException

from .backends import RuntimeHandle, build_runtime
from .config import ModelSettings, Settings
from .errors import api_error


LOGGER = logging.getLogger("whisper_api")


@dataclass
class LoadedModel:
    """Runtime state for a model currently loaded in memory."""

    spec: ModelSettings
    runtime: RuntimeHandle | None
    actual_device: str
    actual_compute_type: str
    loaded_at: float
    last_used: float
    use_count: int = 0
    unload_at: float | None = None
    timer: threading.Timer | None = None


class ModelManager:
    """Load models on demand and unload them according to idle policy."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._loaded: dict[str, LoadedModel] = {}

    def _device_family(self, device: str) -> str:
        return "cuda" if device.startswith("cuda") else "cpu"

    def _loaded_for_family_locked(self, device_family: str) -> list[LoadedModel]:
        return [
            entry
            for entry in self._loaded.values()
            if self._device_family(entry.actual_device) == device_family
        ]

    def _gpu_reserved_mb_locked(self) -> int:
        return sum(
            entry.spec.gpu_memory_reservation_mb
            for entry in self._loaded.values()
            if self._device_family(entry.actual_device) == "cuda"
        )

    def _evict_oldest_idle_model_locked(
        self,
        *,
        device_family: str,
        exclude: set[str] | None = None,
    ) -> bool:
        excluded = exclude or set()
        candidates = sorted(
            (
                entry
                for entry in self._loaded.values()
                if self._device_family(entry.actual_device) == device_family
                and entry.use_count == 0
                and entry.spec.name not in excluded
            ),
            key=lambda entry: entry.loaded_at,
        )
        if not candidates:
            return False
        return self._unload_entry_locked(candidates[0].spec.name)

    def _ensure_capacity_for_new_model_locked(self, spec: ModelSettings, device: str) -> None:
        device_family = self._device_family(device)
        if device_family == "cuda":
            while (
                self._gpu_reserved_mb_locked() + spec.gpu_memory_reservation_mb
                > self.settings.gpu_memory_budget_mb
            ):
                if not self._evict_oldest_idle_model_locked(device_family="cuda"):
                    raise api_error(
                        503,
                        (
                            f"No GPU capacity available to load model '{spec.name}'. "
                            f"Reservation requested: {spec.gpu_memory_reservation_mb} MiB; "
                            f"budget: {self.settings.gpu_memory_budget_mb} MiB."
                        ),
                        error_type="server_busy",
                    )
            return

        loaded_cpu_models = self._loaded_for_family_locked("cpu")
        while len(loaded_cpu_models) >= self.settings.max_loaded_models_cpu:
            if not self._evict_oldest_idle_model_locked(device_family="cpu"):
                raise api_error(
                    503,
                    (
                        f"No CPU capacity available to load model '{spec.name}'. "
                        f"Limit: {self.settings.max_loaded_models_cpu} loaded model(s) on CPU."
                    ),
                    error_type="server_busy",
                )
            loaded_cpu_models = self._loaded_for_family_locked("cpu")

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
        runtime = entry.runtime
        entry.runtime = None
        if runtime is not None:
            runtime.close()
        LOGGER.info("Unloaded model '%s' from %s", model_name, entry.actual_device)
        return True

    def unload(self, model_name: str) -> bool:
        with self._lock:
            return self._unload_entry_locked(model_name)

    def unload_all(self) -> None:
        with self._lock:
            model_names = list(self._loaded)
        for model_name in model_names:
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

    def _preferred_device(self, spec: ModelSettings) -> str:
        if spec.preferred_device.startswith("cuda") and not torch.cuda.is_available():
            LOGGER.warning("CUDA is not available for %s; using CPU instead", spec.name)
            return "cpu"
        if spec.preferred_device.startswith("cuda"):
            return "cuda"
        return spec.preferred_device

    def _build_loaded_model(
        self,
        *,
        spec: ModelSettings,
        device: str,
    ) -> LoadedModel:
        with self._lock:
            self._ensure_capacity_for_new_model_locked(spec, device)
        runtime = build_runtime(spec=spec, settings=self.settings, device=device)
        now = time.time()
        return LoadedModel(
            spec=spec,
            runtime=runtime,
            actual_device=device,
            actual_compute_type=spec.compute_type,
            loaded_at=now,
            last_used=now,
        )

    def _load_model(self, spec: ModelSettings) -> LoadedModel:
        requested_device = self._preferred_device(spec)
        candidate_devices = [requested_device]

        last_error: Exception | None = None
        for device in candidate_devices:
            try:
                LOGGER.info(
                    "Loading model '%s' family=%s backend=%s source='%s' on %s dtype=%s",
                    spec.name,
                    spec.family,
                    spec.backend,
                    spec.source,
                    device,
                    spec.compute_type,
                )
                return self._build_loaded_model(spec=spec, device=device)
            except HTTPException:
                raise
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
                        evicted = self._evict_oldest_idle_model_locked(
                            device_family="cuda",
                            exclude={spec.name},
                        )
                    if not evicted:
                        break
                    try:
                        LOGGER.info(
                            "Retrying model '%s' on %s after evicting idle GPU models",
                            spec.name,
                            device,
                        )
                        return self._build_loaded_model(spec=spec, device=device)
                    except HTTPException:
                        raise
                    except Exception as retry_exc:
                        last_error = retry_exc
                        LOGGER.warning(
                            "Retry on %s for model '%s' also failed: %s",
                            device,
                            spec.name,
                            retry_exc,
                        )

        raise RuntimeError(f"Could not load model '{spec.name}': {last_error}") from last_error

    @contextmanager
    def lease(self, model_name: str) -> LoadedModel:
        """Lease a model for one request, loading it if needed."""
        spec = self.settings.model_settings[model_name]
        with self._lock:
            entry = self._loaded.get(model_name)
            if entry is not None:
                if entry.use_count >= spec.max_concurrent_requests:
                    raise api_error(
                        429,
                        (
                            f"Model '{model_name}' is already handling "
                            f"{spec.max_concurrent_requests} transcription(s)."
                        ),
                        error_type="rate_limit_error",
                    )
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
        """Return runtime state for health and debugging endpoints."""
        now = time.time()
        with self._lock:
            loaded = []
            for model_name, entry in sorted(self._loaded.items()):
                loaded.append(
                    {
                        "id": model_name,
                        "family": entry.spec.family,
                        "backend": entry.spec.backend,
                        "device": entry.actual_device,
                        "device_family": self._device_family(entry.actual_device),
                        "compute_type": entry.actual_compute_type,
                        "gpu_memory_reservation_mb": entry.spec.gpu_memory_reservation_mb,
                        "busy": entry.use_count > 0,
                        "active_requests": entry.use_count,
                        "max_concurrent_requests": entry.spec.max_concurrent_requests,
                        "idle_seconds": entry.spec.idle_seconds,
                        "worker_pid": None if entry.runtime is None else entry.runtime.worker_pid,
                        "seconds_until_unload": (
                            None
                            if entry.use_count > 0 or entry.unload_at is None
                            else max(0, int(entry.unload_at - now))
                        ),
                    }
                )
            return loaded
