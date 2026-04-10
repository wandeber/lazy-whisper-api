from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import lazy_whisper_api.streaming as streaming_module
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
