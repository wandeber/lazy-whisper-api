"""Bundled Silero VAD adapter and bounded acoustic edge refinement."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from .config import EditMaxSettings
from .editing_types import AcousticSpeechSpan, VadAnalysis, VadFrame


SILERO_FRAME_SAMPLES = 512


def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert little-endian mono PCM16 to Silero's normalized float input."""
    if len(pcm_bytes) % 2:
        raise ValueError("PCM16 input contains an incomplete sample.")
    samples = np.frombuffer(pcm_bytes, dtype="<i2")
    return samples.astype(np.float32) / 32768.0


def infer_frame_probabilities(
    audio: np.ndarray,
    *,
    model: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Run the pinned Faster Whisper Silero model without chunk padding rules."""
    if audio.ndim != 1:
        raise ValueError("Silero input must be one-dimensional.")
    if audio.size == 0:
        return np.empty(0, dtype=np.float32)

    if model is None:
        # This is the only coupling to Faster Whisper's bundled MIT-licensed
        # Silero asset. Keeping it lazy avoids ONNX startup for normal profiles.
        from faster_whisper.vad import get_vad_model

        model = get_vad_model()

    frame_count = math.ceil(audio.size / SILERO_FRAME_SAMPLES)
    padding = frame_count * SILERO_FRAME_SAMPLES - audio.size
    padded = np.pad(audio, (0, padding)) if padding else audio
    raw = np.asarray(model(padded), dtype=np.float32).reshape(-1)
    if raw.size < frame_count:
        raise RuntimeError(
            "Silero VAD returned fewer probabilities than canonical audio frames."
        )
    finite = np.nan_to_num(raw[:frame_count], nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(finite, 0.0, 1.0)


def _frames_from_probabilities(
    probabilities: np.ndarray,
    *,
    sample_count: int,
) -> tuple[VadFrame, ...]:
    return tuple(
        VadFrame(
            start_sample=index * SILERO_FRAME_SAMPLES,
            end_sample=min((index + 1) * SILERO_FRAME_SAMPLES, sample_count),
            probability=float(probability),
        )
        for index, probability in enumerate(probabilities)
        if index * SILERO_FRAME_SAMPLES < sample_count
    )


def _coarse_spans(
    frames: tuple[VadFrame, ...],
    *,
    sample_count: int,
    settings: EditMaxSettings,
) -> list[tuple[int, int, float, float]]:
    """Apply deterministic entry/exit hysteresis to raw Silero frames."""
    min_speech_samples = round(
        settings.sample_rate_hz * settings.min_speech_ms / 1000
    )
    min_silence_samples = round(
        settings.sample_rate_hz * settings.min_silence_ms / 1000
    )
    spans: list[tuple[int, int, float, float]] = []
    speech_start: int | None = None
    possible_end: int | None = None

    def append_span(start: int, end: int) -> None:
        if end - start < min_speech_samples:
            return
        supporting = [
            frame.probability
            for frame in frames
            if frame.end_sample > start and frame.start_sample < end
        ]
        if not supporting:
            return
        spans.append((start, end, max(supporting), sum(supporting) / len(supporting)))

    for frame in frames:
        probability = frame.probability
        if speech_start is None:
            if probability >= settings.vad_start_threshold:
                speech_start = frame.start_sample
            continue

        if probability < settings.vad_end_threshold:
            if possible_end is None:
                possible_end = frame.start_sample
            if frame.end_sample - possible_end >= min_silence_samples:
                append_span(speech_start, possible_end)
                speech_start = None
                possible_end = None
            continue

        # A value at or above the lower hysteresis threshold breaks the
        # consecutive-silence run even when it is below the entry threshold.
        possible_end = None

    if speech_start is not None:
        append_span(speech_start, sample_count)
    return spans


def _energy_rms_frames(
    audio: np.ndarray,
    window_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return frame starts and RMS values without padding the public timeline."""
    starts = np.arange(0, audio.size, window_samples, dtype=np.int64)
    values = np.empty(starts.size, dtype=np.float64)
    for index, start in enumerate(starts):
        frame = audio[int(start) : min(int(start) + window_samples, audio.size)]
        values[index] = (
            math.sqrt(float(np.mean(frame.astype(np.float64) ** 2)))
            if frame.size
            else 0.0
        )
    return starts, values


def _local_energy_threshold(
    *,
    rms: np.ndarray,
    starts: np.ndarray,
    edge_sample: int,
    search_samples: int,
    settings: EditMaxSettings,
) -> float:
    left = max(0, edge_sample - search_samples)
    right = edge_sample + search_samples
    local = rms[(starts >= left) & (starts <= right)]
    if local.size == 0:
        local = rms
    noise_floor = float(np.percentile(local, settings.energy_noise_percentile))
    lower = 10.0 ** (settings.energy_min_dbfs / 20.0)
    upper = 10.0 ** (settings.energy_max_dbfs / 20.0)
    return min(upper, max(lower, noise_floor * settings.energy_noise_multiplier))


def _find_energy_transition(
    *,
    starts: np.ndarray,
    rms: np.ndarray,
    edge_sample: int,
    sample_count: int,
    settings: EditMaxSettings,
    onset: bool,
) -> int | None:
    """Find the nearest silence/speech RMS transition around one VAD edge.

    Energy is deliberately only a bounded edge refiner. It cannot determine
    lexical boundaries inside continuous speech, so this function is used only
    for the outside edges of VAD islands.
    """
    if rms.size == 0:
        return None
    sample_rate = settings.sample_rate_hz
    search_samples = round(sample_rate * settings.energy_search_ms / 1000)
    silence_frames = max(
        1,
        math.ceil(settings.energy_silence_run_ms / settings.energy_window_ms),
    )
    speech_frames = max(
        1,
        math.ceil(settings.energy_speech_run_ms / settings.energy_window_ms),
    )
    threshold = _local_energy_threshold(
        rms=rms,
        starts=starts,
        edge_sample=edge_sample,
        search_samples=search_samples,
        settings=settings,
    )

    candidates: list[int] = []
    for index in range(silence_frames, len(rms) - speech_frames + 1):
        candidate = int(starts[index])
        if abs(candidate - edge_sample) > search_samples or candidate > sample_count:
            continue
        if onset:
            quiet = rms[index - silence_frames : index]
            active = rms[index : index + speech_frames]
        else:
            if index < speech_frames or index + silence_frames > len(rms):
                continue
            active = rms[index - speech_frames : index]
            quiet = rms[index : index + silence_frames]
        if np.all(quiet < threshold) and np.all(active >= threshold):
            # Use the exact energy-frame boundary. Its maximum quantization
            # error is the configured window (10 ms by default), substantially
            # finer than Silero's native 32 ms frame.
            candidates.append(min(sample_count, max(0, candidate)))

    if not candidates:
        return None
    if onset:
        return min(candidates, key=lambda value: (abs(value - edge_sample), value))
    return min(candidates, key=lambda value: (abs(value - edge_sample), -value))


def analyze_speech(
    *,
    pcm_bytes: bytes,
    settings: EditMaxSettings,
    model: Callable[[np.ndarray], np.ndarray] | None = None,
) -> VadAnalysis:
    """Produce sample-native speech spans from Silero plus local energy evidence."""
    audio = pcm16_to_float32(pcm_bytes)
    probabilities = infer_frame_probabilities(audio, model=model)
    frames = _frames_from_probabilities(probabilities, sample_count=int(audio.size))
    coarse = _coarse_spans(
        frames,
        sample_count=int(audio.size),
        settings=settings,
    )

    window_samples = max(
        1,
        round(settings.sample_rate_hz * settings.energy_window_ms / 1000),
    )
    starts, rms = _energy_rms_frames(audio, window_samples)
    refined: list[AcousticSpeechSpan] = []
    for coarse_start, coarse_end, peak, mean in coarse:
        refined_start = _find_energy_transition(
            starts=starts,
            rms=rms,
            edge_sample=coarse_start,
            sample_count=int(audio.size),
            settings=settings,
            onset=True,
        )
        refined_end = _find_energy_transition(
            starts=starts,
            rms=rms,
            edge_sample=coarse_end,
            sample_count=int(audio.size),
            settings=settings,
            onset=False,
        )
        start = coarse_start if refined_start is None else refined_start
        end = coarse_end if refined_end is None else refined_end
        if start >= end:
            # Conflicting local evidence must never erase a VAD region. Keeping
            # the coarse bounds is safer for an automatic editor than emitting
            # a zero/negative interval or silently dropping possible speech.
            start, end = coarse_start, coarse_end
            refined_start = refined_end = None
        refined.append(
            AcousticSpeechSpan(
                start_sample=max(0, start),
                end_sample=min(int(audio.size), end),
                peak_probability=peak,
                mean_probability=mean,
                start_energy_confirmed=refined_start is not None,
                end_energy_confirmed=refined_end is not None,
            )
        )

    merged: list[AcousticSpeechSpan] = []
    for span in refined:
        if not merged or span.start_sample >= merged[-1].end_sample:
            merged.append(span)
            continue
        previous = merged[-1]
        merged[-1] = AcousticSpeechSpan(
            start_sample=previous.start_sample,
            end_sample=max(previous.end_sample, span.end_sample),
            peak_probability=max(previous.peak_probability, span.peak_probability),
            mean_probability=max(previous.mean_probability, span.mean_probability),
            start_energy_confirmed=previous.start_energy_confirmed,
            end_energy_confirmed=(
                span.end_energy_confirmed
                if span.end_sample > previous.end_sample
                else previous.end_energy_confirmed
            ),
        )

    rescored: list[AcousticSpeechSpan] = []
    for span in merged:
        supporting = [
            frame.probability
            for frame in frames
            if frame.end_sample > span.start_sample and frame.start_sample < span.end_sample
        ]
        rescored.append(
            AcousticSpeechSpan(
                start_sample=span.start_sample,
                end_sample=span.end_sample,
                peak_probability=max(supporting, default=span.peak_probability),
                mean_probability=(
                    sum(supporting) / len(supporting)
                    if supporting
                    else span.mean_probability
                ),
                start_energy_confirmed=span.start_energy_confirmed,
                end_energy_confirmed=span.end_energy_confirmed,
            )
        )

    return VadAnalysis(
        sample_rate_hz=settings.sample_rate_hz,
        sample_count=int(audio.size),
        frames=frames,
        spans=tuple(rescored),
    )
