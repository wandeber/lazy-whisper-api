"""Canonical audio decoding for sample-exact edit-max analysis."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import av


PCM16_SAMPLE_WIDTH_BYTES = 2


@dataclass(frozen=True)
class DecodedPcm16Timeline:
    """Mono PCM16 whose sample zero is the first sequential decoded sample."""

    pcm_bytes: bytes
    sample_rate_hz: int

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("The decoded sample rate must be positive.")
        if len(self.pcm_bytes) % PCM16_SAMPLE_WIDTH_BYTES:
            raise ValueError("Decoded PCM16 contains an incomplete sample.")

    @property
    def sample_count(self) -> int:
        return len(self.pcm_bytes) // PCM16_SAMPLE_WIDTH_BYTES

    @property
    def duration(self) -> float:
        return self.sample_count / float(self.sample_rate_hz)

    def write_wav(self, destination: Path) -> None:
        """Persist the canonical timeline without another decode/resample pass."""
        with wave.open(str(destination), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(PCM16_SAMPLE_WIDTH_BYTES)
            handle.setframerate(self.sample_rate_hz)
            handle.writeframes(self.pcm_bytes)


def _append_resampled_frames(chunks: bytearray, frames: list[av.AudioFrame]) -> None:
    """Append PyAV output in timeline order as little-endian signed PCM16."""
    for frame in frames:
        # The resampler is configured as mono/s16. Flattening handles PyAV's
        # one-plane ndarray shape without changing the sequential sample order.
        samples = frame.to_ndarray().astype("<i2", copy=False).reshape(-1)
        chunks.extend(samples.tobytes())


def decode_audio_timeline(
    *,
    audio_path: Path,
    sample_rate_hz: int = 16_000,
) -> DecodedPcm16Timeline:
    """Decode the first audio stream to a zero-based mono PCM16 timeline.

    Container presentation timestamps are intentionally not synthesized into
    gaps here. Every downstream edit-max component receives this exact sample
    sequence, so word alignment, VAD, diarization, and returned sample indices
    all share one unambiguous origin.
    """
    container = av.open(str(audio_path))
    chunks = bytearray()
    try:
        stream = next(iter(container.streams.audio), None)
        if stream is None:
            raise ValueError("Uploaded media does not contain an audio stream.")
        resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=sample_rate_hz,
        )
        for frame in container.decode(stream):
            _append_resampled_frames(chunks, resampler.resample(frame))

        # Codecs and resamplers may retain a short tail. Flushing is essential:
        # dropping it would make the returned duration and final speech edge
        # systematically shorter than the audio given to the aligner.
        _append_resampled_frames(chunks, resampler.resample(None))
    finally:
        container.close()

    if len(chunks) % PCM16_SAMPLE_WIDTH_BYTES:
        del chunks[-1]
    return DecodedPcm16Timeline(
        pcm_bytes=bytes(chunks),
        sample_rate_hz=sample_rate_hz,
    )
