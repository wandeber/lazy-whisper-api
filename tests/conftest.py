from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lazy_whisper_api.app import create_app
from lazy_whisper_api.model_manager import LoadedModel


TEST_API_KEY = "test-key"


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator:
    model_dir = tmp_path / "models" / "distil-multi4-ct2"
    model_dir.mkdir(parents=True)

    env = {
        "ASR_API_HOST": "127.0.0.1",
        "ASR_API_PORT": "43556",
        "ASR_API_KEY": TEST_API_KEY,
        "ASR_DEFAULT_MODEL": "turbo",
        "ASR_DEFAULT_DEVICE": "cpu",
        "ASR_MODEL_ALIAS_MAP": (
            "whisper-1=turbo,"
            "turbo=turbo,"
            "large-v3=large-v3,"
            "distil=distil-multi4,"
            "distil-multi4=distil-multi4,"
            "qwen-0.6b=qwen3-asr-0.6b,"
            "qwen-1.7b=qwen3-asr-1.7b,"
            "qwen3-asr-0.6b=qwen3-asr-0.6b,"
            "qwen3-asr-1.7b=qwen3-asr-1.7b"
        ),
        "ASR_MODEL_SOURCE_MAP": (
            f"turbo=turbo,"
            f"large-v3=large-v3,"
            f"distil-multi4={model_dir},"
            "qwen3-asr-0.6b=Qwen/Qwen3-ASR-0.6B,"
            "qwen3-asr-1.7b=Qwen/Qwen3-ASR-1.7B"
        ),
        "ASR_MODEL_FAMILY_MAP": (
            "turbo=whisper,"
            "large-v3=whisper,"
            "distil-multi4=whisper,"
            "qwen3-asr-0.6b=qwen,"
            "qwen3-asr-1.7b=qwen"
        ),
        "ASR_MODEL_BACKEND_MAP": (
            "turbo=faster-whisper,"
            "large-v3=faster-whisper,"
            "distil-multi4=faster-whisper,"
            "qwen3-asr-0.6b=qwen-worker,"
            "qwen3-asr-1.7b=qwen-worker"
        ),
        "ASR_MODEL_DEVICE_MAP": (
            "turbo=cpu,large-v3=cpu,distil-multi4=cpu,qwen3-asr-0.6b=cpu,qwen3-asr-1.7b=cpu"
        ),
        "ASR_MODEL_COMPUTE_TYPE_MAP": (
            "turbo=int8,large-v3=int8,distil-multi4=int8,qwen3-asr-0.6b=float16,qwen3-asr-1.7b=float16"
        ),
        "ASR_MODEL_IDLE_SECONDS_MAP": (
            "turbo=60,large-v3=60,distil-multi4=60,qwen3-asr-0.6b=60,qwen3-asr-1.7b=60"
        ),
        "ASR_MODEL_CAPABILITIES_MAP": (
            "turbo=transcribe|translate|timestamps|stream|realtime,"
            "large-v3=transcribe|translate|timestamps|stream|realtime,"
            "distil-multi4=transcribe|translate|timestamps|stream|realtime,"
            "qwen3-asr-0.6b=transcribe|timestamps|stream|realtime,"
            "qwen3-asr-1.7b=transcribe|timestamps|stream|realtime"
        ),
        "ASR_MODEL_VAD_MAP": (
            "turbo=false,large-v3=false,distil-multi4=false,qwen3-asr-0.6b=false,qwen3-asr-1.7b=false"
        ),
        "ASR_MODEL_GPU_MEMORY_RESERVATION_MB_MAP": (
            "turbo=5200,large-v3=0,distil-multi4=4200,qwen3-asr-0.6b=6500,qwen3-asr-1.7b=7800"
        ),
        "ASR_MODEL_MAX_CONCURRENT_REQUESTS_MAP": (
            "turbo=2,large-v3=2,distil-multi4=2,qwen3-asr-0.6b=1,qwen3-asr-1.7b=1"
        ),
        "ASR_MODEL_ALIGNER_SOURCE_MAP": (
            "qwen3-asr-0.6b=Qwen/Qwen3-ForcedAligner-0.6B,qwen3-asr-1.7b=Qwen/Qwen3-ForcedAligner-0.6B"
        ),
        "ASR_MODEL_ALIGNER_DEVICE_MAP": "qwen3-asr-0.6b=cpu,qwen3-asr-1.7b=cpu",
        "ASR_MODEL_ALIGNER_DTYPE_MAP": "qwen3-asr-0.6b=float32,qwen3-asr-1.7b=float32",
        "ASR_GPU_MEMORY_BUDGET_MB": "8192",
        "ASR_MAX_LOADED_MODELS_CPU": "1",
        "ASR_MAX_CONCURRENT_REQUESTS_PER_MODEL": "2",
        "ASR_UPLOAD_CHUNK_SIZE": "1024",
        "ASR_CPU_THREADS": "0",
        "ASR_LOG_LEVEL": "INFO",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    yield create_app()


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TEST_API_KEY}"}


@pytest.fixture
def sample_upload() -> dict[str, tuple[str, bytes, str]]:
    return {"file": ("audio.wav", b"not-a-real-audio-file", "audio/wav")}


def make_loaded_model(app, model_name: str = "turbo", device: str = "cpu") -> LoadedModel:
    spec = app.state.settings.model_settings[model_name]
    return LoadedModel(
        spec=spec,
        runtime=SimpleNamespace(
            supports_native_streaming=True,
            preferred_stream_sample_rate_hz=24_000,
            worker_pid=None,
        ),
        actual_device=device,
        actual_compute_type=spec.compute_type,
        loaded_at=time.time(),
        last_used=time.time(),
    )


def install_fake_lease(
    monkeypatch: pytest.MonkeyPatch,
    app,
    *,
    model_name: str = "turbo",
    device: str = "cpu",
) -> None:
    @contextmanager
    def fake_lease(requested_model_name: str):
        requested = requested_model_name or model_name
        yield make_loaded_model(app, model_name=requested, device=device)

    monkeypatch.setattr(app.state.model_manager, "lease", fake_lease)
