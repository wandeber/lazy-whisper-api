from __future__ import annotations

import importlib
import json
import time
from contextlib import contextmanager
from types import SimpleNamespace

import lazy_whisper_api.streaming as streaming_module
from lazy_whisper_api.model_manager import LoadedModel
from lazy_whisper_api.transcription import TranscriptionResult

from .conftest import install_fake_lease


app_module = importlib.import_module("lazy_whisper_api.app")


def parse_sse(payload: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in payload.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = ""
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ").strip()
            elif line.startswith("data: "):
                data = json.loads(line.removeprefix("data: ").strip())
        if event_name and data is not None:
            events.append((event_name, data))
    return events


def test_healthz_requires_api_key(client) -> None:
    response = client.get("/healthz")

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "invalid_api_key"


def test_models_endpoint_includes_qwen_aliases(client, auth_headers) -> None:
    response = client.get("/v1/models", headers=auth_headers)

    assert response.status_code == 200
    model_ids = {item["id"] for item in response.json()["data"]}
    assert "whisper-1" in model_ids
    assert "qwen3-asr-0.6b" in model_ids
    assert "qwen-0.6b" in model_ids


def test_transcription_without_stream_returns_json(
    client,
    auth_headers,
    sample_upload,
    monkeypatch,
) -> None:
    async def fake_transcribe_upload(*, settings, model_manager, payload):
        assert settings.default_model == "turbo"
        assert payload.model == "whisper-1"
        return TranscriptionResult(
            model_name="turbo",
            device="cpu",
            response_format="json",
            text="hola desde test",
            info=SimpleNamespace(language="es", duration=1.0, language_probability=1.0),
            segments=[],
        )

    monkeypatch.setattr(app_module, "transcribe_upload", fake_transcribe_upload)

    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        files=sample_upload,
        data={"model": "whisper-1", "response_format": "json"},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "hola desde test", "model": "turbo"}


def test_transcription_stream_true_emits_sse_events(
    app,
    client,
    auth_headers,
    sample_upload,
    monkeypatch,
) -> None:
    install_fake_lease(monkeypatch, app, model_name="turbo", device="cuda")

    def fake_iter_transcribe_sync(**kwargs):
        segments = iter(
            [
                SimpleNamespace(id=1, start=0.0, end=0.4, text="hola"),
                SimpleNamespace(id=2, start=0.4, end=0.8, text=" mundo"),
            ]
        )
        info = SimpleNamespace(language="es", duration=0.8)
        return segments, info

    monkeypatch.setattr(streaming_module, "iter_transcribe_sync", fake_iter_transcribe_sync)

    with client.stream(
        "POST",
        "/v1/audio/transcriptions",
        headers=auth_headers,
        files=sample_upload,
        data={
            "model": "whisper-1",
            "stream": "true",
            "timestamp_granularities[]": "segment",
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(body)
    assert [event_name for event_name, _ in events] == [
        "transcript.text.delta",
        "transcript.text.delta",
        "transcript.text.done",
    ]
    assert events[0][1]["delta"] == "hola"
    assert events[0][1]["segment"]["start"] == 0.0
    assert events[1][1]["delta"] == "mundo"
    assert events[2][1]["text"] == "hola mundo"
    assert events[2][1]["model"] == "turbo"
    assert events[2][1]["device"] == "cuda"


def test_translation_stream_true_is_rejected(
    client,
    auth_headers,
    sample_upload,
) -> None:
    response = client.post(
        "/v1/audio/translations",
        headers=auth_headers,
        files=sample_upload,
        data={"model": "whisper-1", "stream": "true"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "Streaming is only supported for /v1/audio/transcriptions.",
            "type": "invalid_request_error",
        }
    }


def test_transcription_stream_true_uses_synthetic_qwen_path(
    app,
    client,
    auth_headers,
    sample_upload,
    monkeypatch,
) -> None:
    class FakeQwenRuntime:
        supports_native_streaming = False
        preferred_stream_sample_rate_hz = 16_000
        worker_pid = 1234

        def transcribe_pcm(self, **kwargs):
            seconds = len(kwargs["pcm_bytes"]) / (self.preferred_stream_sample_rate_hz * 2)
            text = "hola" if seconds < 12 else "hola mundo"
            return SimpleNamespace(
                text=text,
                info=SimpleNamespace(language="es", duration=seconds, language_probability=None),
                segments=[],
            )

        def transcribe_file(self, **kwargs):
            return SimpleNamespace(
                text="hola mundo",
                info=SimpleNamespace(language="es", duration=14.0, language_probability=None),
                segments=[],
            )

        def align_file(self, **kwargs):
            return [
                SimpleNamespace(id=0, start=0.0, end=1.2, text="hola"),
                SimpleNamespace(id=1, start=1.2, end=2.4, text="mundo"),
            ]

    @contextmanager
    def fake_lease(_model_name: str):
        spec = app.state.settings.model_settings["qwen3-asr-0.6b"]
        yield LoadedModel(
            spec=spec,
            runtime=FakeQwenRuntime(),
            actual_device="cuda:0",
            actual_compute_type=spec.compute_type,
            loaded_at=time.time(),
            last_used=time.time(),
        )

    monkeypatch.setattr(app.state.model_manager, "lease", fake_lease)
    monkeypatch.setattr(
        streaming_module,
        "load_audio_file_as_pcm16",
        lambda **kwargs: (b"\x01\x00" * 16_000 * 14),
    )

    with client.stream(
        "POST",
        "/v1/audio/transcriptions",
        headers=auth_headers,
        files=sample_upload,
        data={
            "model": "qwen-0.6b",
            "stream": "true",
            "timestamp_granularities[]": "segment",
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    events = parse_sse(body)
    assert [event_name for event_name, _ in events] == [
        "transcript.text.delta",
        "transcript.text.delta",
        "transcript.text.done",
    ]
    assert events[0][1]["delta"] == "hola"
    assert events[1][1]["delta"] == "mundo"
    assert events[2][1]["model"] == "qwen3-asr-0.6b"
    assert events[2][1]["segments"][0]["start"] == 0.0
