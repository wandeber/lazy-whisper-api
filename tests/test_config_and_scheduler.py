from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import lazy_whisper_api.model_manager as model_manager_module
from lazy_whisper_api.config import load_settings
from lazy_whisper_api.model_manager import ModelManager

from .conftest import TEST_API_KEY


def test_settings_resolve_qwen_alias_and_capabilities(app) -> None:
    settings = app.state.settings

    assert settings.resolve_model_name("qwen-0.6b") == "qwen3-asr-0.6b"
    spec = settings.model_settings["qwen3-asr-0.6b"]
    assert spec.family == "qwen"
    assert spec.backend == "qwen-worker"
    assert spec.supports("stream")
    assert spec.supports("realtime")
    assert spec.runtime_python.endswith(".venv-qwen/bin/python")


def test_legacy_whisper_env_aliases_still_load(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {
        "WHISPER_API_KEY": TEST_API_KEY,
        "WHISPER_DEFAULT_MODEL": "large-v3",
        "WHISPER_DEFAULT_DEVICE": "cpu",
        "WHISPER_MODEL_ALIAS_MAP": "whisper-1=turbo,turbo=turbo,large-v3=large-v3",
        "WHISPER_MODEL_SOURCE_MAP": "turbo=turbo,large-v3=large-v3",
        "WHISPER_MODEL_DEVICE_MAP": "turbo=cpu,large-v3=cpu",
        "WHISPER_MODEL_COMPUTE_TYPE_MAP": "turbo=int8,large-v3=int8",
        "WHISPER_MODEL_IDLE_SECONDS_MAP": "turbo=60,large-v3=60",
        "WHISPER_MODEL_VAD_MAP": "turbo=false,large-v3=false",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    settings = load_settings()

    assert settings.api_key == TEST_API_KEY
    assert settings.default_model == "large-v3"
    assert settings.resolve_model_name("whisper-1") == "turbo"


def test_cuda_preferred_device_uses_backend_safe_name(app) -> None:
    manager = ModelManager(app.state.settings)
    spec = replace(
        app.state.settings.model_settings["turbo"],
        preferred_device="cuda",
    )

    assert manager._preferred_device(spec) == "cuda"


def test_gpu_scheduler_evicts_idle_model_to_make_room(
    app,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = app.state.settings
    model_settings = dict(settings.model_settings)
    model_settings["turbo"] = replace(
        model_settings["turbo"],
        preferred_device="cuda",
        gpu_memory_reservation_mb=5200,
    )
    model_settings["qwen3-asr-1.7b"] = replace(
        model_settings["qwen3-asr-1.7b"],
        preferred_device="cuda",
        gpu_memory_reservation_mb=7800,
    )
    manager = ModelManager(
        replace(
            settings,
            gpu_memory_budget_mb=8192,
            model_settings=model_settings,
        )
    )

    def fake_build_runtime(*, spec, settings, device):
        return SimpleNamespace(
            close=lambda: None,
            worker_pid=9999 if spec.family == "qwen" else None,
            supports_native_streaming=(spec.family == "whisper"),
            preferred_stream_sample_rate_hz=24_000 if spec.family == "whisper" else 16_000,
        )

    monkeypatch.setattr(model_manager_module, "build_runtime", fake_build_runtime)

    with manager.lease("turbo"):
        pass

    with manager.lease("qwen3-asr-1.7b"):
        snapshot = manager.snapshot()

    assert [entry["id"] for entry in snapshot] == ["qwen3-asr-1.7b"]
    assert snapshot[0]["worker_pid"] == 9999


def test_gpu_scheduler_returns_503_when_busy_model_blocks_capacity(
    app,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = app.state.settings
    model_settings = dict(settings.model_settings)
    model_settings["turbo"] = replace(
        model_settings["turbo"],
        preferred_device="cuda",
        gpu_memory_reservation_mb=5200,
    )
    model_settings["qwen3-asr-1.7b"] = replace(
        model_settings["qwen3-asr-1.7b"],
        preferred_device="cuda",
        gpu_memory_reservation_mb=7800,
    )
    manager = ModelManager(
        replace(
            settings,
            gpu_memory_budget_mb=8192,
            model_settings=model_settings,
        )
    )

    monkeypatch.setattr(
        model_manager_module,
        "build_runtime",
        lambda **kwargs: SimpleNamespace(
            close=lambda: None,
            worker_pid=None,
            supports_native_streaming=False,
            preferred_stream_sample_rate_hz=16_000,
        ),
    )

    lease_cm = manager.lease("turbo")
    lease_cm.__enter__()
    try:
        with pytest.raises(HTTPException) as exc_info:
            with manager.lease("qwen3-asr-1.7b"):
                pass
    finally:
        lease_cm.__exit__(None, None, None)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["type"] == "server_busy"
