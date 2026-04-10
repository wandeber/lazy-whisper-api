from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

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
        "WHISPER_API_HOST": "127.0.0.1",
        "WHISPER_API_PORT": "43556",
        "WHISPER_API_KEY": TEST_API_KEY,
        "WHISPER_DEFAULT_MODEL": "turbo",
        "WHISPER_DEFAULT_DEVICE": "cpu",
        "WHISPER_MODEL_ALIAS_MAP": (
            "whisper-1=turbo,"
            "turbo=turbo,"
            "large-v3=large-v3,"
            "distil=distil-multi4,"
            "distil-multi4=distil-multi4"
        ),
        "WHISPER_MODEL_SOURCE_MAP": (
            f"turbo=turbo,large-v3=large-v3,distil-multi4={model_dir}"
        ),
        "WHISPER_MODEL_DEVICE_MAP": "turbo=cpu,large-v3=cpu,distil-multi4=cpu",
        "WHISPER_MODEL_COMPUTE_TYPE_MAP": "turbo=int8,large-v3=int8,distil-multi4=int8",
        "WHISPER_MODEL_IDLE_SECONDS_MAP": "turbo=60,large-v3=60,distil-multi4=60",
        "WHISPER_MODEL_VAD_MAP": "turbo=false,large-v3=false,distil-multi4=false",
        "WHISPER_MAX_LOADED_MODELS_CPU": "1",
        "WHISPER_MAX_LOADED_MODELS_GPU": "2",
        "WHISPER_MAX_CONCURRENT_REQUESTS_PER_MODEL": "2",
        "WHISPER_UPLOAD_CHUNK_SIZE": "1024",
        "WHISPER_CPU_THREADS": "0",
        "WHISPER_LOG_LEVEL": "INFO",
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
        model=object(),
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
