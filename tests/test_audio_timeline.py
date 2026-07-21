from __future__ import annotations

import wave

import numpy as np

from lazy_whisper_api.audio_timeline import decode_audio_timeline


def test_canonical_decoder_resamples_stereo_and_flushes_tail(tmp_path) -> None:
    source = tmp_path / "stereo-8khz.wav"
    frame_count = 8_000
    left = np.full(frame_count, 1_000, dtype="<i2")
    right = np.full(frame_count, -500, dtype="<i2")
    interleaved = np.column_stack((left, right)).reshape(-1).astype("<i2")
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(8_000)
        handle.writeframes(interleaved.tobytes())

    timeline = decode_audio_timeline(audio_path=source, sample_rate_hz=16_000)

    assert timeline.sample_rate_hz == 16_000
    assert timeline.sample_count == 16_000
    assert timeline.duration == 1.0
    assert len(timeline.pcm_bytes) == timeline.sample_count * 2


def test_canonical_decoder_rejects_invalid_media(tmp_path) -> None:
    source = tmp_path / "empty.bin"
    source.write_bytes(b"not media")

    try:
        decode_audio_timeline(audio_path=source, sample_rate_hz=16_000)
    except Exception as exc:
        assert str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Invalid media unexpectedly decoded as audio.")
