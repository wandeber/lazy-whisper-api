from __future__ import annotations

import base64
import struct
import time
from types import SimpleNamespace

import pytest
from starlette.websockets import WebSocketDisconnect

import lazy_whisper_api.realtime as realtime_module
from lazy_whisper_api.errors import api_error

from .conftest import TEST_API_KEY, install_fake_lease


def b64_pcm(pcm_bytes: bytes) -> str:
    return base64.b64encode(pcm_bytes).decode("ascii")


def collect_until_completed(websocket, *, deadline_seconds: float = 2.0) -> list[dict]:
    events: list[dict] = []
    deadline = time.time() + deadline_seconds
    while time.time() < deadline:
        event = websocket.receive_json()
        events.append(event)
        if event["type"] == "conversation.item.input_audio_transcription.completed":
            return events
    raise AssertionError(f"Timed out waiting for completed event: {events}")


def test_realtime_requires_auth(client) -> None:
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()


def test_realtime_rejects_unsupported_session_update(client) -> None:
    with client.websocket_connect(f"/v1/realtime?api_key={TEST_API_KEY}") as websocket:
        created = websocket.receive_json()
        assert created["type"] == "session.created"

        websocket.send_json(
            {
                "type": "session.update",
                "event_id": "evt_bad",
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 16000},
                        }
                    },
                },
            }
        )
        error = websocket.receive_json()

    assert error["type"] == "error"
    assert error["error"]["code"] == "unsupported_field"


def test_realtime_manual_commit_emits_delta_and_completed(
    app,
    client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_lease(monkeypatch, app, model_name="turbo")

    def fake_transcribe_pcm16_sync(**kwargs):
        pcm_bytes = kwargs["pcm_bytes"]
        text = "hola" if len(pcm_bytes) < 2400 else "hola mundo"
        segments = [SimpleNamespace(text=text)]
        info = SimpleNamespace(language=kwargs["language"] or "es")
        return segments, info

    monkeypatch.setattr(realtime_module, "transcribe_pcm16_sync", fake_transcribe_pcm16_sync)

    first_chunk = b"\x01\x00" * 400
    second_chunk = b"\x02\x00" * 1000

    with client.websocket_connect(f"/v1/realtime?api_key={TEST_API_KEY}") as websocket:
        created = websocket.receive_json()
        assert created["type"] == "session.created"

        websocket.send_json(
            {
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "transcription": {"model": "whisper-1", "language": "es"},
                            "turn_detection": None,
                        }
                    },
                },
            }
        )
        updated = websocket.receive_json()
        assert updated["type"] == "session.updated"

        websocket.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": b64_pcm(first_chunk),
            }
        )
        partial = websocket.receive_json()
        assert partial["type"] == "conversation.item.input_audio_transcription.delta"
        item_id = partial["item_id"]
        assert partial["text"] == "hola"

        websocket.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": b64_pcm(second_chunk),
            }
        )
        websocket.send_json({"type": "input_audio_buffer.commit"})

        events = collect_until_completed(websocket)

    event_types = [event["type"] for event in events]
    assert "input_audio_buffer.committed" in event_types
    completed = events[-1]
    assert completed["type"] == "conversation.item.input_audio_transcription.completed"
    assert completed["item_id"] == item_id
    assert completed["transcript"] == "hola mundo"
    assert completed["model"] == "turbo"


def test_realtime_server_vad_auto_commits(
    app,
    client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_lease(monkeypatch, app, model_name="turbo")

    def fake_transcribe_pcm16_sync(**kwargs):
        return [SimpleNamespace(text="voz detectada")], SimpleNamespace(language="es")

    monkeypatch.setattr(realtime_module, "transcribe_pcm16_sync", fake_transcribe_pcm16_sync)

    voice_frame = struct.pack("<h", 20_000) * 480
    silence_frame = b"\x00\x00" * 480
    payload = (voice_frame * 4) + (silence_frame * 6)

    with client.websocket_connect(f"/v1/realtime?api_key={TEST_API_KEY}") as websocket:
        websocket.receive_json()
        websocket.send_json(
            {
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "transcription": {"model": "whisper-1", "language": "es"},
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": 0.05,
                                "prefix_padding_ms": 20,
                                "silence_duration_ms": 60,
                            },
                        }
                    },
                },
            }
        )
        websocket.receive_json()
        websocket.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": b64_pcm(payload),
            }
        )

        events = collect_until_completed(websocket)

    committed = next(
        event for event in events if event["type"] == "input_audio_buffer.committed"
    )
    completed = events[-1]
    assert committed["commit_mode"] == "server_vad"
    assert completed["transcript"] == "voz detectada"


def test_realtime_turn_unavailable_emits_error_and_keeps_socket_open(
    app,
    client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RejectingLease:
        def __enter__(self):
            raise api_error(
                429,
                "Model 'turbo' is already handling 2 transcription(s).",
                error_type="rate_limit_error",
            )

        def __exit__(self, exc_type, exc, tb):
            return False

    def reject_lease(_model_name: str):
        return RejectingLease()

    monkeypatch.setattr(app.state.model_manager, "lease", reject_lease)

    with client.websocket_connect(f"/v1/realtime?api_key={TEST_API_KEY}") as websocket:
        websocket.receive_json()
        websocket.send_json(
            {
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "transcription": {"model": "whisper-1"},
                            "turn_detection": None,
                        }
                    },
                },
            }
        )
        updated = websocket.receive_json()
        assert updated["type"] == "session.updated"

        websocket.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": b64_pcm(b"\x01\x00" * 400),
            }
        )
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["error"]["type"] == "rate_limit_error"
        assert error["error"]["code"] == "turn_unavailable"

        websocket.send_json({"type": "session.update", "session": {"type": "transcription"}})
        follow_up = websocket.receive_json()

    assert follow_up["type"] == "session.updated"
