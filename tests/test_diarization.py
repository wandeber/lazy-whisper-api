from __future__ import annotations

import asyncio
import io
import threading
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi import UploadFile

from lazy_whisper_api.backends import (
    BackendTranscription,
    SegmentTiming,
    TranscriptionInfo,
    WordTiming,
)
from lazy_whisper_api.diarization import (
    DiarizationManager,
    DiarizationWorkerProxy,
    build_diarization_worker_env,
    diarization_installation_status,
    redact_secrets,
    validate_diarization_request,
)
from lazy_whisper_api.diarization_types import DiarizationResult, DiarizationTurn
from lazy_whisper_api.diarization_worker import Worker, cli, iter_annotation_turns
from lazy_whisper_api.speaker_attribution import (
    build_speaker_transcript_segments,
    choose_speaker_for_interval,
    enrich_segments_with_speakers,
)
from lazy_whisper_api.transcription import TranscriptionRequest, transcribe_upload


def test_choose_speaker_prefers_largest_overlap() -> None:
    turns = [
        DiarizationTurn(start=0.0, end=0.7, speaker="SPEAKER_00"),
        DiarizationTurn(start=0.7, end=1.4, speaker="SPEAKER_01"),
    ]

    assert (
        choose_speaker_for_interval(start=0.2, end=1.2, turns=turns)
        == "SPEAKER_01"
    )


def test_enrich_segments_with_speakers_keeps_original_timings_immutable() -> None:
    source = [
        SegmentTiming(
            id=0,
            start=0.0,
            end=1.2,
            text="hola mundo",
            words=[
                WordTiming(start=0.0, end=0.4, word="hola"),
                WordTiming(start=0.8, end=1.2, word="mundo"),
            ],
        )
    ]
    turns = [
        DiarizationTurn(start=0.0, end=0.6, speaker="SPEAKER_00"),
        DiarizationTurn(start=0.6, end=1.3, speaker="SPEAKER_01"),
    ]

    enriched = enrich_segments_with_speakers(segments=source, turns=turns)

    assert source[0].speaker is None
    assert source[0].words[0].speaker is None
    assert enriched[0].speaker == "SPEAKER_00"
    assert [word.speaker for word in enriched[0].words] == ["SPEAKER_00", "SPEAKER_01"]


def test_build_speaker_transcript_segments_splits_on_speaker_changes() -> None:
    segments = [
        SegmentTiming(
            id=0,
            start=0.0,
            end=2.0,
            text="Hello, there. General Kenobi.",
            words=[
                WordTiming(0.0, 0.4, "Hello", speaker="SPEAKER_00"),
                WordTiming(0.4, 0.7, ",", speaker="SPEAKER_00"),
                WordTiming(0.8, 1.1, "there", speaker="SPEAKER_00"),
                WordTiming(1.2, 1.5, "General", speaker="SPEAKER_01"),
                WordTiming(1.5, 1.9, "Kenobi", speaker="SPEAKER_01"),
                WordTiming(1.9, 2.0, ".", speaker="SPEAKER_01"),
            ],
        )
    ]

    grouped = build_speaker_transcript_segments(segments=segments)

    assert [(item.speaker, item.text) for item in grouped] == [
        ("SPEAKER_00", "Hello, there"),
        ("SPEAKER_01", "General Kenobi."),
    ]


def test_redact_secrets_removes_hugging_face_tokens(monkeypatch) -> None:
    explicit_token = "hf_" + "explicitSecret123"
    other_token = "hf_" + "otherSecret456"
    monkeypatch.setenv("HF_TOKEN", explicit_token)

    assert redact_secrets(f"bad {other_token} and {explicit_token}") == (
        "bad <redacted> and <redacted>"
    )


def test_worker_env_enforces_offline_mode_and_removes_credentials(app) -> None:
    env = build_diarization_worker_env(
        settings=app.state.settings,
        source_env={
            "HF_TOKEN": "hf_secret",
            "HUGGING_FACE_HUB_TOKEN": "second-secret",
            "ASR_DIARIZATION_SETUP_HF_TOKEN": "setup-secret",
            "ASR_API_KEY": "api-secret",
            "AWS_SECRET_ACCESS_KEY": "aws-secret",
            "HOME": "/Users/example",
            "PATH": "/usr/bin",
            "PYTHONPATH": "/existing/path",
        },
    )

    assert "HF_TOKEN" not in env
    assert "HUGGING_FACE_HUB_TOKEN" not in env
    assert "ASR_DIARIZATION_SETUP_HF_TOKEN" not in env
    assert "ASR_API_KEY" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert env["HOME"] != "/Users/example"
    assert env["PATH"] == "/usr/bin"
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["PYANNOTE_METRICS_ENABLED"] == "0"
    assert env["HF_HOME"].endswith(".cache/diarization/huggingface")
    assert env["MPLCONFIGDIR"].endswith(".cache/diarization/matplotlib")
    assert env["PYTHONPATH"] == str(app.state.settings.project_root)
    assert "/existing/path" not in env["PYTHONPATH"]


def test_installation_status_checks_runtime_and_local_pipeline(app) -> None:
    status = diarization_installation_status(app.state.settings.diarization)

    assert status == {
        "runtime_available": True,
        "model_available": True,
        "ready": False,
    }


def test_installation_status_rejects_marker_for_another_model(app) -> None:
    diarization = replace(
        app.state.settings.diarization,
        model_id="pyannote/a-different-local-pipeline",
    )

    status = diarization_installation_status(diarization)

    assert status["model_available"] is False
    assert status["ready"] is False


def test_diarized_whisper_transcription_requests_word_timestamps(app) -> None:
    settings = replace(
        app.state.settings,
        diarization=replace(app.state.settings.diarization, enabled=True),
    )
    calls = []
    events = []

    class FakeRuntime:
        def transcribe_file(self, **kwargs):
            calls.append(kwargs)
            return BackendTranscription(
                text="hello world",
                info=TranscriptionInfo(language="en", duration=1.0),
                segments=[
                    SegmentTiming(
                        id=0,
                        start=0.0,
                        end=1.0,
                        text="hello world",
                        words=[
                            WordTiming(0.0, 0.4, "hello"),
                            WordTiming(0.6, 1.0, "world"),
                        ],
                    )
                ],
            )

    runtime = FakeRuntime()

    class FakeModelManager:
        @contextmanager
        def lease(self, model_name):
            events.append("asr")
            yield SimpleNamespace(
                runtime=runtime,
                spec=settings.model_settings[model_name],
                actual_device="cpu",
            )

    class FakeDiarizationManager:
        @contextmanager
        def reserve(self):
            events.append("reserved")
            try:
                yield
            finally:
                events.append("released")

        def diarize_reserved(self, **kwargs):
            events.append("diarized")
            return DiarizationResult(
                model="pyannote/speaker-diarization-community-1",
                device="cpu",
                turns=[DiarizationTurn(0.0, 1.0, "SPEAKER_00")],
            )

    payload = TranscriptionRequest(
        file=UploadFile(filename="test.wav", file=io.BytesIO(b"fake audio")),
        model="whisper-1",
        task="transcribe",
        language="en",
        prompt=None,
        response_format="verbose_json",
        temperature=0.0,
        timestamp_granularities=None,
        diarize=True,
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
    )

    result = asyncio.run(
        transcribe_upload(
            settings=settings,
            model_manager=FakeModelManager(),
            diarization_manager=FakeDiarizationManager(),
            payload=payload,
        )
    )

    assert calls[0]["word_timestamps"] is True
    assert result.segments[0].words[0].speaker == "SPEAKER_00"
    assert events == ["reserved", "asr", "diarized", "released"]


def test_diarization_manager_unload_if_idle_ignores_active_or_stale_timer(app) -> None:
    manager = DiarizationManager(app.state.settings)
    closed = []

    class FakeRuntime:
        worker_pid = 1234

        def close(self):
            closed.append(True)

    manager._runtime = FakeRuntime()
    manager._generation = 1
    manager._active_runs = 1

    manager.unload_if_idle(1)

    assert closed == []
    assert manager._runtime is not None

    manager._active_runs = 0
    manager._generation = 2
    manager.unload_if_idle(1)

    assert closed == []
    assert manager._runtime is not None

    manager.unload_if_idle(2)

    assert closed == [True]
    assert manager._runtime is None


def test_diarization_shutdown_waits_for_idle_worker_cleanup(app) -> None:
    manager = DiarizationManager(app.state.settings)
    close_started = threading.Event()
    allow_close = threading.Event()
    idle_cleanup_finished = threading.Event()
    shutdown_finished = threading.Event()

    class SlowRuntime:
        worker_pid = 1234

        def close(self):
            close_started.set()
            assert allow_close.wait(timeout=2)

    manager._runtime = SlowRuntime()
    manager._generation = 1

    def run_idle_cleanup() -> None:
        manager.unload_if_idle(1)
        idle_cleanup_finished.set()

    def run_shutdown() -> None:
        manager.unload()
        shutdown_finished.set()

    idle_thread = threading.Thread(target=run_idle_cleanup, daemon=True)
    idle_thread.start()
    assert close_started.wait(timeout=1)

    shutdown_thread = threading.Thread(target=run_shutdown, daemon=True)
    shutdown_thread.start()
    assert not shutdown_finished.wait(timeout=0.05)

    allow_close.set()
    assert idle_cleanup_finished.wait(timeout=1)
    assert shutdown_finished.wait(timeout=1)
    idle_thread.join(timeout=1)
    shutdown_thread.join(timeout=1)


def test_diarization_reservation_rejects_concurrent_work_before_loading(app) -> None:
    settings = replace(
        app.state.settings,
        diarization=replace(app.state.settings.diarization, enabled=True),
    )
    manager = DiarizationManager(settings)

    with manager.reserve():
        assert manager.snapshot()["state"] == "busy"
        with pytest.raises(Exception) as exc_info:
            with manager.reserve():
                pass

    assert getattr(exc_info.value, "status_code", None) == 429
    assert manager.snapshot()["state"] == "ready"


def test_diarization_manager_replaces_dead_runtime_before_reuse(
    app,
    monkeypatch,
) -> None:
    settings = replace(
        app.state.settings,
        diarization=replace(app.state.settings.diarization, enabled=True),
    )
    manager = DiarizationManager(settings)
    created = []

    class FakeRuntime:
        worker_pid = 1234

        def __init__(self, *, settings, diarization):
            self.running = True
            self.closed = False
            created.append(self)

        def is_running(self):
            return self.running

        def diarize_file(self, **kwargs):
            return DiarizationResult(model="fake", device="cpu", turns=[])

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        "lazy_whisper_api.diarization.DiarizationWorkerProxy",
        FakeRuntime,
    )
    manager._runtime = FakeRuntime(settings=settings, diarization=settings.diarization)
    manager._runtime.running = False

    result = manager.diarize(
        audio_path=SimpleNamespace(),
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
    )

    assert result.turns == []
    assert len(created) == 2
    assert created[0].closed is True
    assert manager._runtime is created[1]


def test_diarization_manager_reports_loading_without_blocking_health(
    app,
    monkeypatch,
) -> None:
    settings = replace(
        app.state.settings,
        diarization=replace(app.state.settings.diarization, enabled=True),
    )
    load_started = threading.Event()
    allow_load = threading.Event()
    run_finished = threading.Event()
    errors: list[BaseException] = []

    class SlowRuntime:
        worker_pid = 1234

        def __init__(self, *, settings, diarization):
            load_started.set()
            if not allow_load.wait(timeout=2):
                raise RuntimeError("Test did not release the simulated model load.")

        def is_running(self):
            return True

        def diarize_file(self, **kwargs):
            return DiarizationResult(model="fake", device="cpu", turns=[])

        def close(self):
            return None

    monkeypatch.setattr(
        "lazy_whisper_api.diarization.DiarizationWorkerProxy",
        SlowRuntime,
    )
    manager = DiarizationManager(settings)

    def run_diarization() -> None:
        try:
            manager.diarize(
                audio_path=SimpleNamespace(),
                num_speakers=None,
                min_speakers=None,
                max_speakers=None,
            )
        except BaseException as exc:  # pragma: no cover - failure assertion below
            errors.append(exc)
        finally:
            run_finished.set()

    worker_thread = threading.Thread(target=run_diarization, daemon=True)
    worker_thread.start()
    assert load_started.wait(timeout=1)

    try:
        snapshot = manager.snapshot()
        assert snapshot["state"] == "loading"
        assert snapshot["active_runs"] == 1
    finally:
        allow_load.set()

    assert run_finished.wait(timeout=2)
    worker_thread.join(timeout=1)
    manager.unload()
    assert errors == []


def test_validate_diarization_rejects_invisible_response_formats(app) -> None:
    settings = replace(
        app.state.settings,
        diarization=replace(app.state.settings.diarization, enabled=True),
    )

    with pytest.raises(Exception) as exc_info:
        validate_diarization_request(
            settings=settings,
            response_format="json",
            task="transcribe",
            num_speakers=None,
            min_speakers=None,
            max_speakers=None,
        )

    assert "response_format=verbose_json" in str(exc_info.value)


def test_validate_diarization_rejects_invalid_speaker_counts(app) -> None:
    settings = replace(
        app.state.settings,
        diarization=replace(app.state.settings.diarization, enabled=True),
    )

    with pytest.raises(Exception) as exc_info:
        validate_diarization_request(
            settings=settings,
            response_format="verbose_json",
            task="transcribe",
            num_speakers=2,
            min_speakers=1,
            max_speakers=None,
        )

    assert "num_speakers cannot be combined" in str(exc_info.value)


class FakeAnnotation:
    def itertracks(self, *, yield_label: bool):
        assert yield_label is True
        yield SimpleNamespace(start=1.0, end=2.0), "track-b", "SPEAKER_01"
        yield SimpleNamespace(start=0.0, end=1.0), "track-a", "SPEAKER_00"


class FakePipeline:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, audio_path: str, **kwargs):
        self.calls.append({"audio_path": audio_path, **kwargs})
        return SimpleNamespace(exclusive_speaker_diarization=FakeAnnotation())


def test_worker_diarize_file_normalizes_exclusive_speaker_turns(monkeypatch) -> None:
    fake_pipeline = FakePipeline()
    monkeypatch.setattr(Worker, "_load_pipeline", lambda self: fake_pipeline)
    worker = Worker(
        model_id="pyannote/speaker-diarization-community-1",
        model_path="/tmp/fake-pyannote-model",
        device="cpu",
    )

    result = worker.diarize_file(
        audio_path="/tmp/audio.wav",
        num_speakers=2,
        min_speakers=None,
        max_speakers=None,
    )

    assert fake_pipeline.calls == [
        {
            "audio_path": "/tmp/audio.wav",
            "num_speakers": 2,
        }
    ]
    assert result["turns"] == [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
    ]


def test_iter_annotation_turns_accepts_pair_iteration() -> None:
    annotation = [
        (SimpleNamespace(start=0.0, end=0.5), "A"),
        (SimpleNamespace(start=0.5, end=1.0), "B"),
    ]

    assert iter_annotation_turns(annotation) == [
        {"start": 0.0, "end": 0.5, "speaker": "A"},
        {"start": 0.5, "end": 1.0, "speaker": "B"},
    ]


def test_worker_cli_treats_sigint_as_clean_shutdown(monkeypatch) -> None:
    def interrupt_main() -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr("lazy_whisper_api.diarization_worker.main", interrupt_main)

    assert cli() == 0
