from __future__ import annotations

import numpy as np

from lazy_whisper_api.backends import WordTiming
from lazy_whisper_api.editing import build_edit_transcript
from lazy_whisper_api.editing_types import AcousticSpeechSpan, VadAnalysis, VadFrame
from lazy_whisper_api.silero_vad import analyze_speech


def pcm16_bytes(samples: np.ndarray) -> bytes:
    return np.clip(samples * 32768.0, -32768, 32767).astype("<i2").tobytes()


def test_silero_hysteresis_uses_energy_to_refine_outer_edges(app) -> None:
    settings = app.state.settings.model_profiles["edit-max-v1"].edit_max
    assert settings is not None
    sample_rate = settings.sample_rate_hz
    audio = np.zeros(round(1.5 * sample_rate), dtype=np.float32)
    audio[round(0.5 * sample_rate) : round(1.0 * sample_rate)] = 0.2

    def fake_model(padded_audio: np.ndarray) -> np.ndarray:
        frame_count = len(padded_audio) // 512
        probabilities = np.full(frame_count, 0.05, dtype=np.float32)
        # Deliberately make the neural edges coarser than the known waveform
        # transition so the local energy pass has useful refinement work.
        probabilities[15:32] = 0.95
        return probabilities[:, None]

    analysis = analyze_speech(
        pcm_bytes=pcm16_bytes(audio),
        settings=settings,
        model=fake_model,
    )

    assert len(analysis.spans) == 1
    span = analysis.spans[0]
    assert abs(span.start_sample - round(0.5 * sample_rate)) <= round(0.01 * sample_rate)
    assert abs(span.end_sample - round(1.0 * sample_rate)) <= round(0.01 * sample_rate)
    assert span.start_energy_confirmed is True
    assert span.end_energy_confirmed is True


def test_edit_fusion_snaps_only_region_outer_words(app) -> None:
    settings = app.state.settings.model_profiles["edit-max-v1"].edit_max
    assert settings is not None
    sample_rate = settings.sample_rate_hz
    vad = VadAnalysis(
        sample_rate_hz=sample_rate,
        sample_count=sample_rate * 2,
        frames=tuple(),
        spans=(
            AcousticSpeechSpan(
                start_sample=round(0.5 * sample_rate),
                end_sample=round(1.0 * sample_rate),
                peak_probability=0.95,
                mean_probability=0.90,
                start_energy_confirmed=True,
                end_energy_confirmed=True,
            ),
        ),
    )
    source_words = [
        WordTiming(start=0.48, end=0.70, word="hola"),
        WordTiming(start=0.72, end=0.98, word="mundo"),
    ]

    result = build_edit_transcript(
        aligned_words=source_words,
        vad=vad,
        settings=settings,
        requested_model="qwen-1.7b-edit-max",
        canonical_model="qwen3-asr-1.7b",
        profile_name="edit-max-v1",
    )

    assert [word.start for word in result.words] == [0.5, 0.72]
    assert [word.end for word in result.words] == [0.70, 1.0]
    assert result.editing.speech_regions[0].evidence == "alignment_acoustic"
    assert result.editing.speech_regions[0].word_start_index == 0
    assert result.editing.speech_regions[0].word_end_index == 2
    assert [boundary.type for boundary in result.editing.edit_boundaries] == [
        "speech_start",
        "speech_end",
    ]


def test_edit_fusion_keeps_alignment_only_and_strong_acoustic_only_regions(app) -> None:
    settings = app.state.settings.model_profiles["edit-max-v1"].edit_max
    assert settings is not None
    sample_rate = settings.sample_rate_hz
    frames = tuple(
        VadFrame(
            start_sample=index * 512,
            end_sample=(index + 1) * 512,
            probability=0.05,
        )
        for index in range(100)
    )
    vad = VadAnalysis(
        sample_rate_hz=sample_rate,
        sample_count=sample_rate * 4,
        frames=frames,
        spans=(
            AcousticSpeechSpan(
                start_sample=round(2.0 * sample_rate),
                end_sample=round(2.4 * sample_rate),
                peak_probability=0.96,
                mean_probability=0.85,
            ),
        ),
    )

    result = build_edit_transcript(
        aligned_words=[WordTiming(start=0.2, end=0.5, word="quiet")],
        vad=vad,
        settings=settings,
        requested_model="qwen-1.7b-edit-max",
        canonical_model="qwen3-asr-1.7b",
        profile_name="edit-max-v1",
    )

    assert [region.evidence for region in result.editing.speech_regions] == [
        "alignment_only",
        "acoustic_only",
    ]
    assert result.editing.speech_regions[1].word_start_index is None


def test_edit_fusion_never_splits_a_word_crossing_two_acoustic_islands(app) -> None:
    settings = app.state.settings.model_profiles["edit-max-v1"].edit_max
    assert settings is not None
    sample_rate = settings.sample_rate_hz
    vad = VadAnalysis(
        sample_rate_hz=sample_rate,
        sample_count=sample_rate * 2,
        frames=tuple(),
        spans=(
            AcousticSpeechSpan(
                start_sample=round(0.5 * sample_rate),
                end_sample=round(0.8 * sample_rate),
                peak_probability=0.95,
                mean_probability=0.90,
            ),
            AcousticSpeechSpan(
                start_sample=round(1.2 * sample_rate),
                end_sample=round(1.5 * sample_rate),
                peak_probability=0.95,
                mean_probability=0.90,
            ),
        ),
    )

    result = build_edit_transcript(
        aligned_words=[WordTiming(start=0.4, end=1.6, word="continuous")],
        vad=vad,
        settings=settings,
        requested_model="qwen-1.7b-edit-max",
        canonical_model="qwen3-asr-1.7b",
        profile_name="edit-max-v1",
    )

    assert len(result.words) == 1
    assert result.words[0].start == 0.4
    assert result.words[0].end == 1.6
    assert len(result.editing.speech_regions) == 1
    assert result.editing.speech_regions[0].evidence == "alignment_acoustic"
