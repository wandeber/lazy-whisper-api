from __future__ import annotations

import base64
from types import SimpleNamespace

from lazy_whisper_api.qwen_mlx_worker import Worker


class FakeSession:
    def __init__(self) -> None:
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({"audio": audio, **kwargs})
        if kwargs["return_timestamps"]:
            return SimpleNamespace(
                text="hola mundo adios casa",
                language="Spanish",
                segments=[
                    {"start": 0.0, "end": 0.4, "text": "hola"},
                    {"start": 0.4, "end": 0.9, "text": "mundo"},
                    {"start": 1.8, "end": 2.2, "text": "adios"},
                    {"start": 2.2, "end": 2.7, "text": "casa"},
                ],
                chunks=[],
            )
        return SimpleNamespace(
            text="hola mundo",
            language="Spanish",
            segments=[],
            chunks=[
                {"chunk_index": 0, "start": 0.0, "end": 0.9, "text": "hola mundo"},
            ],
        )


def make_worker(monkeypatch) -> tuple[Worker, FakeSession]:
    fake_session = FakeSession()
    monkeypatch.setattr(Worker, "_load_session", lambda self: fake_session)
    worker = Worker(
        model_name="qwen3-asr-0.6b",
        model_source="Qwen/Qwen3-ASR-0.6B",
        device="mlx",
        dtype_name="float16",
        aligner_source="Qwen/Qwen3-ForcedAligner-0.6B",
    )
    return worker, fake_session


def test_mlx_worker_transcribe_file_returns_text_without_segments(monkeypatch) -> None:
    worker, fake_session = make_worker(monkeypatch)

    result = worker.transcribe_file(
        audio_path="/tmp/audio.wav",
        language="Spanish",
        prompt="domain words",
    )

    assert result["text"] == "hola mundo"
    assert result["language"] == "Spanish"
    assert result["duration"] == 0.9
    assert result["segments"] == []
    assert fake_session.calls[0]["return_timestamps"] is False
    assert fake_session.calls[0]["forced_aligner"] is None


def test_mlx_worker_transcribe_pcm_writes_wav_snapshot(monkeypatch) -> None:
    worker, fake_session = make_worker(monkeypatch)
    pcm_bytes = b"\x01\x00" * 16_000

    result = worker.transcribe_pcm(
        pcm_base64=base64.b64encode(pcm_bytes).decode("ascii"),
        sample_rate_hz=16_000,
        language="Spanish",
        prompt=None,
    )

    assert result["text"] == "hola mundo"
    assert result["duration"] == len(pcm_bytes) / (16_000 * 2)
    assert fake_session.calls[0]["audio"].endswith(".wav")


def test_mlx_worker_align_file_groups_word_timings_into_segments(monkeypatch) -> None:
    worker, fake_session = make_worker(monkeypatch)

    result = worker.align_file(
        audio_path="/tmp/audio.wav",
        text="hola mundo adios casa",
        language="Spanish",
    )

    assert result["duration"] == 2.7
    assert result["segments"] == [
        {
            "id": 0,
            "start": 0.0,
            "end": 0.9,
            "text": "hola mundo",
            "words": [
                {"start": 0.0, "end": 0.4, "word": "hola", "probability": None},
                {"start": 0.4, "end": 0.9, "word": "mundo", "probability": None},
            ],
        },
        {
            "id": 1,
            "start": 1.8,
            "end": 2.7,
            "text": "adios casa",
            "words": [
                {"start": 1.8, "end": 2.2, "word": "adios", "probability": None},
                {"start": 2.2, "end": 2.7, "word": "casa", "probability": None},
            ],
        },
    ]
    assert fake_session.calls[0]["return_timestamps"] is True
    assert fake_session.calls[0]["forced_aligner"] == "Qwen/Qwen3-ForcedAligner-0.6B"
