"""Value objects shared by diarization orchestration and response rendering."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiarizationTurn:
    """One diarization speaker turn on the uploaded audio timeline."""

    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class DiarizationResult:
    """Normalized diarization result returned by a backend worker."""

    model: str
    device: str
    turns: list[DiarizationTurn]
    processing_seconds: float | None = None


@dataclass(frozen=True)
class SpeakerTranscriptSegment:
    """Contiguous transcript text attributed to one local speaker label."""

    start: float
    end: float
    speaker: str
    text: str
