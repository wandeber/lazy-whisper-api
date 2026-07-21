from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import lazy_whisper_api.backends as backends_module
import lazy_whisper_api.model_manager as model_manager_module
from lazy_whisper_api.backends import QwenWorkerProxy
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


def test_settings_resolve_edit_profile_without_new_canonical_model(app) -> None:
    settings = app.state.settings

    route = settings.resolve_model_route("qwen-1.7b-edit-max")

    assert route.requested_model == "qwen-1.7b-edit-max"
    assert route.canonical_model == "qwen3-asr-1.7b"
    assert route.profile.name == "edit-max-v1"
    assert route.profile.is_edit_max is True
    assert "qwen-1.7b-edit-max" not in settings.model_settings
    assert settings.resolve_model_name("qwen-1.7b-edit-max") == "qwen3-asr-1.7b"


def test_subtitle_and_edit_routes_share_one_loaded_qwen_runtime(
    app,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = app.state.settings
    manager = ModelManager(settings)
    builds = []

    def fake_build_runtime(*, spec, settings, device):
        builds.append(spec.name)
        return SimpleNamespace(
            close=lambda: None,
            worker_pid=123,
            supports_native_streaming=False,
            preferred_stream_sample_rate_hz=16_000,
        )

    monkeypatch.setattr(model_manager_module, "build_runtime", fake_build_runtime)
    normal = settings.resolve_model_route("qwen-1.7b")
    editing = settings.resolve_model_route("qwen-1.7b-edit-max")

    with manager.lease(normal.canonical_model):
        pass
    with manager.lease(editing.canonical_model):
        pass

    assert builds == ["qwen3-asr-1.7b"]
    assert [entry["id"] for entry in manager.snapshot()] == ["qwen3-asr-1.7b"]


def test_old_explicit_alias_map_can_omit_edit_profile(
    app,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ASR_MODEL_ALIAS_MAP",
        "whisper-1=turbo,turbo=turbo,qwen-1.7b=qwen3-asr-1.7b",
    )
    monkeypatch.delenv("ASR_MODEL_PROFILE_MAP", raising=False)

    settings = load_settings()

    assert "qwen-1.7b-edit-max" not in settings.supported_model_ids
    with pytest.raises(KeyError):
        settings.resolve_model_route("qwen-1.7b-edit-max")


def test_reserved_edit_id_requires_exact_profile_binding(
    app,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ASR_MODEL_PROFILE_MAP", raising=False)

    with pytest.raises(ValueError, match="Reserved model ID 'qwen-1.7b-edit-max'"):
        load_settings()


def test_reserved_edit_id_cannot_redirect_to_another_model(
    app,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aliases = app.state.settings.model_alias_map.copy()
    aliases["qwen-1.7b-edit-max"] = "qwen3-asr-0.6b"
    monkeypatch.setenv(
        "ASR_MODEL_ALIAS_MAP",
        ",".join(f"{key}={value}" for key, value in aliases.items()),
    )

    with pytest.raises(ValueError, match="Reserved model ID 'qwen-1.7b-edit-max'"):
        load_settings()


def test_invalid_edit_threshold_fails_at_startup(
    app,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASR_EDIT_MAX_VAD_START_THRESHOLD", "0.2")
    monkeypatch.setenv("ASR_EDIT_MAX_VAD_END_THRESHOLD", "0.3")

    with pytest.raises(ValueError, match="ASR_EDIT_MAX_VAD_START_THRESHOLD"):
        load_settings()


def test_settings_load_diarization_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASR_DIARIZATION_ENABLED", "true")
    monkeypatch.setenv("ASR_DIARIZATION_MODEL_ID", "pyannote/custom")
    monkeypatch.setenv("ASR_DIARIZATION_MODEL_PATH", "./models/pyannote-custom")
    monkeypatch.setenv("ASR_DIARIZATION_DEVICE", "cpu")
    monkeypatch.setenv("ASR_DIARIZATION_IDLE_SECONDS", "120")
    monkeypatch.setenv("ASR_DIARIZATION_STARTUP_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("ASR_DIARIZATION_REQUEST_TIMEOUT_SECONDS", "900")
    monkeypatch.setenv(
        "ASR_DIARIZATION_RUNTIME_PYTHON",
        "./.venv-diarization/bin/python",
    )

    settings = load_settings()

    assert settings.diarization.enabled is True
    assert settings.diarization.backend == "pyannote"
    assert settings.diarization.model_id == "pyannote/custom"
    assert settings.diarization.model_path.endswith("models/pyannote-custom")
    assert settings.diarization.device == "cpu"
    assert settings.diarization.idle_seconds == 120
    assert settings.diarization.runtime_python.endswith(".venv-diarization/bin/python")
    assert settings.diarization.startup_timeout_seconds == 45
    assert settings.diarization.request_timeout_seconds == 900


def test_model_runtime_python_map_allows_apple_silicon_qwen_backend(
    app,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ASR_MODEL_BACKEND_MAP",
        (
            "turbo=faster-whisper,"
            "large-v3=faster-whisper,"
            "distil-multi4=faster-whisper,"
            "qwen3-asr-0.6b=qwen-mlx-worker,"
            "qwen3-asr-1.7b=qwen-mlx-worker"
        ),
    )
    monkeypatch.setenv(
        "ASR_MODEL_DEVICE_MAP",
        "turbo=cpu,large-v3=cpu,distil-multi4=cpu,qwen3-asr-0.6b=mlx,qwen3-asr-1.7b=mlx",
    )
    monkeypatch.setenv(
        "ASR_MODEL_RUNTIME_PYTHON_MAP",
        (
            "qwen3-asr-0.6b=./.venv-qwen-mlx/bin/python,"
            "qwen3-asr-1.7b=./.venv-qwen-mlx/bin/python"
        ),
    )

    settings = load_settings()

    assert settings.resolve_model_name("qwen-0.6b") == "qwen3-asr-0.6b"
    spec = settings.model_settings["qwen3-asr-0.6b"]
    assert spec.backend == "qwen-mlx-worker"
    assert spec.preferred_device == "mlx"
    assert spec.runtime_python.endswith(".venv-qwen-mlx/bin/python")


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


def test_cuda_preferred_device_uses_backend_safe_name(
    app,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_manager_module.torch.cuda, "is_available", lambda: True)
    manager = ModelManager(app.state.settings)
    spec = replace(
        app.state.settings.model_settings["turbo"],
        preferred_device="cuda",
    )

    assert manager._preferred_device(spec) == "cuda"


def test_mlx_device_family_does_not_share_cpu_capacity(app) -> None:
    manager = ModelManager(app.state.settings)

    assert manager._device_family("mlx") == "mlx"
    assert manager._device_family("mlx:0") == "mlx"


def test_build_runtime_dispatches_qwen_workers_by_backend(
    app,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class FakeProxy:
        def __init__(self, *, spec, settings, device, worker_path, worker_label):
            calls.append(
                {
                    "backend": spec.backend,
                    "device": device,
                    "worker_path": Path(worker_path).name,
                    "worker_label": worker_label,
                }
            )

    monkeypatch.setattr(backends_module, "QwenWorkerProxy", FakeProxy)
    settings = app.state.settings
    cuda_spec = replace(settings.model_settings["qwen3-asr-0.6b"], backend="qwen-worker")
    mlx_spec = replace(settings.model_settings["qwen3-asr-0.6b"], backend="qwen-mlx-worker")

    backends_module.build_runtime(spec=cuda_spec, settings=settings, device="cuda")
    backends_module.build_runtime(spec=mlx_spec, settings=settings, device="mlx")

    assert calls == [
        {
            "backend": "qwen-worker",
            "device": "cuda",
            "worker_path": "qwen_worker.py",
            "worker_label": "qwen-worker",
        },
        {
            "backend": "qwen-mlx-worker",
            "device": "mlx",
            "worker_path": "qwen_mlx_worker.py",
            "worker_label": "qwen-mlx-worker",
        },
    ]


def test_qwen_proxy_normalizes_exact_word_alignment() -> None:
    proxy = object.__new__(QwenWorkerProxy)
    proxy._client = SimpleNamespace(
        request=lambda method, params: {
            "words": [
                {"start": 0.1, "end": 0.4, "word": "hola", "probability": None},
                {"start": 0.5, "end": 0.9, "word": "mundo", "probability": 0.8},
            ]
        }
    )

    words = proxy.align_words_file(
        audio_path=Path("/tmp/audio.wav"),
        text="hola mundo",
        language="es",
    )

    assert [(word.start, word.end, word.word) for word in words] == [
        (0.1, 0.4, "hola"),
        (0.5, 0.9, "mundo"),
    ]
    assert words[1].probability == 0.8


def test_qwen_proxy_rejects_empty_alignment_for_nonempty_text() -> None:
    proxy = object.__new__(QwenWorkerProxy)
    proxy._client = SimpleNamespace(request=lambda method, params: {"words": []})

    with pytest.raises(RuntimeError, match="returned no words"):
        proxy.align_words_file(
            audio_path=Path("/tmp/audio.wav"),
            text="hola",
            language="es",
        )


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
