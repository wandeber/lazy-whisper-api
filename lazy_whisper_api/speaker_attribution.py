"""Reconcile ASR timestamps with diarization turns and build readable output."""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import replace

from .backends import SegmentTiming
from .diarization_types import DiarizationTurn, SpeakerTranscriptSegment


def overlap_seconds(
    *,
    start: float,
    end: float,
    turn: DiarizationTurn,
) -> float:
    """Return the overlap in seconds between an ASR interval and a speaker turn."""
    return max(0.0, min(end, turn.end) - max(start, turn.start))


class SpeakerTurnIndex:
    """Search speaker turns without sorting the full timeline for every word.

    pyannote's exclusive diarization is normally non-overlapping, but the worker
    deliberately falls back to the regular annotation when an exclusive result
    is unavailable. The prefix maximum keeps lookups correct for both shapes: it
    lets a query walk backwards only while an earlier interval can still overlap.
    """

    def __init__(self, turns: list[DiarizationTurn]) -> None:
        self._turns = sorted(turns, key=lambda turn: (turn.start, turn.end, turn.speaker))
        self._starts = [turn.start for turn in self._turns]

        max_end = float("-inf")
        self._prefix_max_ends: list[float] = []
        for turn in self._turns:
            max_end = max(max_end, turn.end)
            self._prefix_max_ends.append(max_end)

        centered = sorted(
            (((turn.start + turn.end) / 2.0, turn) for turn in self._turns),
            key=lambda item: (item[0], item[1].start, item[1].end, item[1].speaker),
        )
        self._centers = [center for center, _turn in centered]
        self._center_turns = [turn for _center, turn in centered]

    def choose(self, *, start: float, end: float) -> str | None:
        """Choose the speaker with the strongest temporal evidence."""
        if not self._turns:
            return None

        midpoint = (start + end) / 2.0
        right = bisect_left(self._starts, end)
        best: tuple[float, float, str] | None = None
        index = right - 1
        while index >= 0 and self._prefix_max_ends[index] > start:
            turn = self._turns[index]
            overlap = overlap_seconds(start=start, end=end, turn=turn)
            if overlap > 0.0:
                rank = (
                    overlap,
                    -abs(midpoint - ((turn.start + turn.end) / 2.0)),
                    turn.speaker,
                )
                if best is None or rank > best:
                    best = rank
            index -= 1
        if best is not None:
            return best[2]

        # Zero-length backend artifacts can sit exactly on a turn boundary. Use
        # inclusive midpoint containment before falling back to the nearest turn.
        right = bisect_right(self._starts, midpoint)
        index = right - 1
        containing: list[DiarizationTurn] = []
        while index >= 0 and self._prefix_max_ends[index] >= midpoint:
            turn = self._turns[index]
            if turn.start <= midpoint <= turn.end:
                containing.append(turn)
            index -= 1
        if containing:
            return min(
                containing,
                key=lambda turn: (
                    abs(midpoint - ((turn.start + turn.end) / 2.0)),
                    turn.start,
                    turn.end,
                    turn.speaker,
                ),
            ).speaker

        position = bisect_left(self._centers, midpoint)
        nearby = self._center_turns[max(0, position - 1) : min(len(self._turns), position + 1)]
        return min(
            nearby,
            key=lambda turn: (
                abs(midpoint - ((turn.start + turn.end) / 2.0)),
                turn.start,
                turn.end,
                turn.speaker,
            ),
        ).speaker


def choose_speaker_for_interval(
    *,
    start: float,
    end: float,
    turns: list[DiarizationTurn],
) -> str | None:
    """Choose a speaker for a standalone interval.

    Batch reconciliation uses one shared index for all words. This convenience
    wrapper preserves the small public helper used by tests and local callers.
    """
    return SpeakerTurnIndex(turns).choose(start=start, end=end)


def enrich_segments_with_speakers(
    *,
    segments: list[SegmentTiming],
    turns: list[DiarizationTurn],
) -> list[SegmentTiming]:
    """Return immutable ASR segments with speaker labels copied onto timings."""
    turn_index = SpeakerTurnIndex(turns)
    enriched_segments: list[SegmentTiming] = []
    for segment in segments:
        enriched_words = [
            replace(
                word,
                speaker=turn_index.choose(start=word.start, end=word.end),
            )
            for word in (segment.words or [])
        ]
        enriched_segments.append(
            replace(
                segment,
                speaker=turn_index.choose(start=segment.start, end=segment.end),
                words=enriched_words,
            )
        )
    return enriched_segments


def _join_transcript_pieces(pieces: list[str]) -> str:
    """Join backend word tokens into readable text across backend variants."""
    text = " ".join(piece.strip() for piece in pieces if piece.strip())
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"([¿¡(\[])\s+", r"\1", text)
    return text.strip()


def build_speaker_transcript_segments(
    *,
    segments: list[SegmentTiming],
    max_gap_seconds: float = 1.5,
) -> list[SpeakerTranscriptSegment]:
    """Group timestamped ASR units into readable, speaker-attributed turns.

    Word timings are preferred because one ASR segment can span a real speaker
    change. Backends that only expose segment timings still get a useful fallback.
    A modest gap limit avoids joining distant utterances merely because the same
    speaker happened to talk again later.
    """
    units: list[tuple[float, float, str, str]] = []
    for segment in segments:
        if segment.words:
            units.extend(
                (word.start, word.end, word.speaker, word.word)
                for word in segment.words
                if word.speaker is not None and word.word.strip()
            )
        elif segment.speaker is not None and segment.text.strip():
            units.append((segment.start, segment.end, segment.speaker, segment.text))

    grouped: list[SpeakerTranscriptSegment] = []
    current_speaker: str | None = None
    current_start = 0.0
    current_end = 0.0
    current_pieces: list[str] = []

    def flush() -> None:
        nonlocal current_speaker, current_start, current_end, current_pieces
        if current_speaker is not None and current_pieces:
            grouped.append(
                SpeakerTranscriptSegment(
                    start=current_start,
                    end=current_end,
                    speaker=current_speaker,
                    text=_join_transcript_pieces(current_pieces),
                )
            )
        current_speaker = None
        current_pieces = []

    for start, end, speaker, text in units:
        can_extend = (
            current_speaker == speaker
            and start >= current_end
            and start - current_end <= max_gap_seconds
        )
        if not can_extend:
            flush()
            current_speaker = speaker
            current_start = start
        current_end = max(current_end, end) if can_extend else end
        current_pieces.append(text)
    flush()
    return grouped
