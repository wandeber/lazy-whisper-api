"""Sample-native value objects for edit-oriented transcription output."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VadFrame:
    """One Silero probability over a half-open PCM sample interval."""

    start_sample: int
    end_sample: int
    probability: float


@dataclass(frozen=True)
class AcousticSpeechSpan:
    """One VAD speech island after bounded local-energy edge refinement."""

    start_sample: int
    end_sample: int
    peak_probability: float
    mean_probability: float
    start_energy_confirmed: bool = False
    end_energy_confirmed: bool = False


@dataclass(frozen=True)
class VadAnalysis:
    """Complete deterministic acoustic evidence for one decoded timeline."""

    sample_rate_hz: int
    sample_count: int
    frames: tuple[VadFrame, ...]
    spans: tuple[AcousticSpeechSpan, ...]


@dataclass(frozen=True)
class EditingTimeline:
    """Canonical decoded-audio timeline shared by every edit-max stage."""

    sample_rate_hz: int
    sample_count: int

    @property
    def duration(self) -> float:
        return self.sample_count / float(self.sample_rate_hz)


@dataclass(frozen=True)
class EditingSpeechRegion:
    """One final speech region suitable for clip-boundary decisions."""

    id: int
    start_sample: int
    end_sample: int
    evidence: str
    word_start_index: int | None
    word_end_index: int | None
    vad_peak_probability: float | None = None
    vad_mean_probability: float | None = None
    start_energy_confirmed: bool = False
    end_energy_confirmed: bool = False


@dataclass(frozen=True)
class EditBoundary:
    """A speech start or end derived directly from a final region edge."""

    region_id: int
    type: str
    sample: int
    evidence: str
    energy_confirmed: bool = False


@dataclass(frozen=True)
class EditingResult:
    """Versioned edit-max metadata serialized only for the opt-in profile."""

    schema_version: int
    profile: str
    requested_model: str
    canonical_model: str
    timeline: EditingTimeline
    speech_regions: tuple[EditingSpeechRegion, ...]
    edit_boundaries: tuple[EditBoundary, ...]
