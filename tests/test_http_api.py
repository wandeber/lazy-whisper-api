from __future__ import annotations

import importlib
import json
import time
from contextlib import contextmanager
from types import SimpleNamespace

import lazy_whisper_api.streaming as streaming_module
import lazy_whisper_api.transcription as transcription_module
from fastapi.testclient import TestClient
from lazy_whisper_api.audio_timeline import DecodedPcm16Timeline
from lazy_whisper_api.backends import (
    BackendTranscription,
    SegmentTiming,
    TranscriptionInfo,
    WordTiming,
)
from lazy_whisper_api.diarization_types import DiarizationResult, DiarizationTurn
from lazy_whisper_api.editing_types import AcousticSpeechSpan, VadAnalysis
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


def test_healthz_reports_diarization_installation_state(client, auth_headers) -> None:
    response = client.get("/healthz", headers=auth_headers)

    assert response.status_code == 200
    diarization = response.json()["diarization"]
    assert diarization["state"] == "disabled"
    assert diarization["runtime_available"] is True
    assert diarization["model_available"] is True
    assert diarization["ready"] is False
    assert diarization["offline"] is True


def test_app_lifespan_unloads_all_runtime_managers(app, monkeypatch) -> None:
    unloaded = []
    monkeypatch.setattr(app.state.diarization_manager, "unload", lambda: unloaded.append("diarization"))
    monkeypatch.setattr(app.state.model_manager, "unload_all", lambda: unloaded.append("models"))

    with TestClient(app):
        pass

    assert unloaded == ["diarization", "models"]


def test_models_endpoint_includes_qwen_aliases(client, auth_headers) -> None:
    response = client.get("/v1/models", headers=auth_headers)

    assert response.status_code == 200
    models = {item["id"]: item for item in response.json()["data"]}
    model_ids = set(models)
    assert "whisper-1" in model_ids
    assert "qwen3-asr-0.6b" in model_ids
    assert "qwen-0.6b" in model_ids
    assert models["qwen-1.7b-edit-max"]["profile"] == "edit-max-v1"
    assert models["qwen-1.7b-edit-max"]["canonical_model"] == "qwen3-asr-1.7b"
    assert "profile" not in models["qwen-1.7b"]


def test_edit_max_rejects_incompatible_format_before_model_lease(
    app,
    client,
    auth_headers,
    sample_upload,
    monkeypatch,
) -> None:
    def reject_lease(_model_name: str):
        raise AssertionError("Validation must fail before a model lease.")

    monkeypatch.setattr(app.state.model_manager, "lease", reject_lease)

    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        files=sample_upload,
        data={"model": "qwen-1.7b-edit-max", "response_format": "json"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        "Model 'qwen-1.7b-edit-max' requires response_format='verbose_json'."
    )


def test_edit_max_rejects_streaming_before_model_lease(
    app,
    client,
    auth_headers,
    sample_upload,
    monkeypatch,
) -> None:
    def reject_lease(_model_name: str):
        raise AssertionError("Validation must fail before a model lease.")

    monkeypatch.setattr(app.state.model_manager, "lease", reject_lease)

    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        files=sample_upload,
        data={
            "model": "qwen-1.7b-edit-max",
            "response_format": "verbose_json",
            "stream": "true",
        },
    )

    assert response.status_code == 400
    assert "non-streaming batch transcription" in response.json()["error"]["message"]


def test_edit_max_returns_words_and_sample_native_editing_metadata(
    app,
    client,
    auth_headers,
    sample_upload,
    monkeypatch,
) -> None:
    timeline = DecodedPcm16Timeline(pcm_bytes=b"\x00\x00" * 16_000, sample_rate_hz=16_000)
    monkeypatch.setattr(
        transcription_module,
        "decode_audio_timeline",
        lambda **_kwargs: timeline,
    )
    monkeypatch.setattr(
        transcription_module,
        "analyze_speech",
        lambda **_kwargs: VadAnalysis(
            sample_rate_hz=16_000,
            sample_count=16_000,
            frames=tuple(),
            spans=(
                AcousticSpeechSpan(
                    start_sample=2_880,
                    end_sample=13_120,
                    peak_probability=0.96,
                    mean_probability=0.90,
                    start_energy_confirmed=True,
                    end_energy_confirmed=True,
                ),
            ),
        ),
    )

    class FakeEditRuntime:
        worker_pid = 123

        def transcribe_file(self, **_kwargs):
            return BackendTranscription(
                text="hola mundo",
                info=TranscriptionInfo(language="es", duration=1.0),
                segments=[],
            )

        def align_words_file(self, **_kwargs):
            return [
                WordTiming(start=0.2, end=0.4, word="hola"),
                WordTiming(start=0.5, end=0.8, word="mundo"),
            ]

    leased_names = []

    @contextmanager
    def fake_lease(model_name: str):
        leased_names.append(model_name)
        spec = app.state.settings.model_settings[model_name]
        yield LoadedModel(
            spec=spec,
            runtime=FakeEditRuntime(),
            actual_device="mlx",
            actual_compute_type=spec.compute_type,
            loaded_at=time.time(),
            last_used=time.time(),
        )

    monkeypatch.setattr(app.state.model_manager, "lease", fake_lease)

    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        files=sample_upload,
        data={
            "model": "qwen-1.7b-edit-max",
            "response_format": "verbose_json",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert leased_names == ["qwen3-asr-1.7b"]
    assert payload["model"] == "qwen3-asr-1.7b"
    assert [word["word"] for word in payload["words"]] == ["hola", "mundo"]
    assert payload["words"][0]["start"] == 0.18
    assert payload["words"][1]["end"] == 0.82
    assert payload["editing"]["requested_model"] == "qwen-1.7b-edit-max"
    assert payload["editing"]["profile"] == "edit-max-v1"
    assert payload["editing"]["timeline"] == {
        "sample_rate_hz": 16_000,
        "sample_count": 16_000,
        "duration": 1.0,
        "time_origin": "decoded_audio_start",
    }
    assert payload["editing"]["speech_regions"][0]["evidence"] == (
        "alignment_acoustic"
    )
    assert payload["editing"]["edit_boundaries"][0]["sample"] == 2_880


def test_edit_max_fails_closed_when_nonempty_text_has_no_aligned_words(
    app,
    client,
    auth_headers,
    sample_upload,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        transcription_module,
        "decode_audio_timeline",
        lambda **_kwargs: DecodedPcm16Timeline(
            pcm_bytes=b"\x00\x00" * 16_000,
            sample_rate_hz=16_000,
        ),
    )
    vad_called = []
    monkeypatch.setattr(
        transcription_module,
        "analyze_speech",
        lambda **_kwargs: vad_called.append(True),
    )

    class EmptyAlignmentRuntime:
        worker_pid = 123

        def transcribe_file(self, **_kwargs):
            return BackendTranscription(
                text="hola",
                info=TranscriptionInfo(language="es", duration=1.0),
                segments=[],
            )

        def align_words_file(self, **_kwargs):
            return []

    @contextmanager
    def fake_lease(model_name: str):
        spec = app.state.settings.model_settings[model_name]
        yield LoadedModel(
            spec=spec,
            runtime=EmptyAlignmentRuntime(),
            actual_device="mlx",
            actual_compute_type=spec.compute_type,
            loaded_at=time.time(),
            last_used=time.time(),
        )

    monkeypatch.setattr(app.state.model_manager, "lease", fake_lease)

    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        files=sample_upload,
        data={
            "model": "qwen-1.7b-edit-max",
            "response_format": "verbose_json",
        },
    )

    assert response.status_code == 500
    assert response.json()["error"]["message"] == (
        "Edit-max forced alignment returned no words for a non-empty transcript."
    )
    assert vad_called == []


def test_transcription_without_stream_returns_json(
    client,
    auth_headers,
    sample_upload,
    monkeypatch,
) -> None:
    async def fake_transcribe_upload(*, settings, model_manager, diarization_manager, payload):
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


def test_transcription_diarize_verbose_json_returns_speaker_payload(
    client,
    auth_headers,
    sample_upload,
    monkeypatch,
) -> None:
    async def fake_transcribe_upload(*, settings, model_manager, diarization_manager, payload):
        assert payload.diarize is True
        assert payload.num_speakers == 2
        return TranscriptionResult(
            model_name="turbo",
            device="cpu",
            response_format="verbose_json",
            text="hola mundo",
            info=SimpleNamespace(language="es", duration=1.0, language_probability=1.0),
            segments=[
                SegmentTiming(
                    id=0,
                    start=0.0,
                    end=1.0,
                    text="hola mundo",
                    speaker="SPEAKER_00",
                    words=[
                        WordTiming(
                            start=0.0,
                            end=0.4,
                            word="hola",
                            probability=0.9,
                            speaker="SPEAKER_00",
                        ),
                        WordTiming(
                            start=0.6,
                            end=1.0,
                            word="mundo",
                            probability=0.9,
                            speaker="SPEAKER_01",
                        ),
                    ],
                )
            ],
            diarization=DiarizationResult(
                model="pyannote/speaker-diarization-community-1",
                device="cpu",
                turns=[
                    DiarizationTurn(start=0.0, end=0.5, speaker="SPEAKER_00"),
                    DiarizationTurn(start=0.5, end=1.0, speaker="SPEAKER_01"),
                ],
            ),
        )

    monkeypatch.setattr(app_module, "transcribe_upload", fake_transcribe_upload)

    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        files=sample_upload,
        data={
            "model": "whisper-1",
            "response_format": "verbose_json",
            "diarize": "true",
            "num_speakers": "2",
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["segments"][0]["speaker"] == "SPEAKER_00"
    assert payload["segments"][0]["words"][1]["speaker"] == "SPEAKER_01"
    assert payload["words"][0]["speaker"] == "SPEAKER_00"
    assert payload["diarization"]["num_speakers"] == 2
    assert payload["diarization"]["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert payload["diarization"]["segments"][1] == {
        "start": 0.5,
        "end": 1.0,
        "speaker": "SPEAKER_01",
    }
    assert payload["diarization"]["speaker_segments"] == [
        {
            "start": 0.0,
            "end": 0.4,
            "speaker": "SPEAKER_00",
            "text": "hola",
        },
        {
            "start": 0.6,
            "end": 1.0,
            "speaker": "SPEAKER_01",
            "text": "mundo",
        },
    ]


def test_speaker_count_options_require_diarization(
    client,
    auth_headers,
    sample_upload,
) -> None:
    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        files=sample_upload,
        data={
            "model": "whisper-1",
            "response_format": "verbose_json",
            "num_speakers": "2",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        "Speaker-count options require diarize=true."
    )


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


def test_transcription_stream_true_with_diarization_is_rejected(
    client,
    auth_headers,
    sample_upload,
) -> None:
    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        files=sample_upload,
        data={
            "model": "whisper-1",
            "stream": "true",
            "diarize": "true",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "Diarization is not supported for streaming responses.",
        "type": "invalid_request_error",
    }


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


def test_qwen_translation_is_rejected(
    client,
    auth_headers,
    sample_upload,
) -> None:
    response = client.post(
        "/v1/audio/translations",
        headers=auth_headers,
        files=sample_upload,
        data={"model": "qwen-0.6b"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "does not support translation" in response.json()["error"]["message"]


def test_translation_with_diarization_is_rejected(
    client,
    auth_headers,
    sample_upload,
) -> None:
    response = client.post(
        "/v1/audio/translations",
        headers=auth_headers,
        files=sample_upload,
        data={
            "model": "whisper-1",
            "response_format": "verbose_json",
            "diarize": "true",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "Diarization is only supported for /v1/audio/transcriptions.",
        "type": "invalid_request_error",
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
