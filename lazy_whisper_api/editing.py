"""Deterministic fusion of forced alignment and acoustic speech evidence."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .backends import SegmentTiming, WordTiming
from .config import EditMaxSettings
from .editing_types import (
    AcousticSpeechSpan,
    EditBoundary,
    EditingResult,
    EditingSpeechRegion,
    EditingTimeline,
    VadAnalysis,
)


MAX_SEGMENT_CHARS = 84
MAX_SEGMENT_SECONDS = 6.0
HARD_MAX_SEGMENT_SECONDS = 8.0
SPLIT_GAP_SECONDS = 0.65
SENTENCE_ENDINGS = (".", "!", "?", ";", ":")


@dataclass
class _SampleWord:
    index: int
    start: int
    end: int
    timing: WordTiming


@dataclass
class _RegionCandidate:
    start: int
    end: int
    word_indices: set[int] = field(default_factory=set)
    has_alignment: bool = False
    has_acoustic: bool = False
    vad_peak: float | None = None
    vad_mean: float | None = None
    start_energy_confirmed: bool = False
    end_energy_confirmed: bool = False


@dataclass(frozen=True)
class EditTranscript:
    """Final word/segment views plus the independent editing timeline."""

    words: tuple[WordTiming, ...]
    segments: tuple[SegmentTiming, ...]
    editing: EditingResult


def _seconds_to_sample(seconds: float, sample_rate_hz: int, sample_count: int) -> int:
    return min(sample_count, max(0, round(float(seconds) * sample_rate_hz)))


def _normalize_words(
    words: list[WordTiming],
    *,
    sample_rate_hz: int,
    sample_count: int,
) -> list[_SampleWord]:
    normalized: list[_SampleWord] = []
    previous_raw_start: float | None = None
    for timing in words:
        if not timing.word.strip():
            continue
        if previous_raw_start is not None and timing.start < previous_raw_start:
            raise RuntimeError("Forced alignment returned decreasing word start times.")
        previous_raw_start = timing.start
        start = _seconds_to_sample(timing.start, sample_rate_hz, sample_count)
        end = _seconds_to_sample(timing.end, sample_rate_hz, sample_count)
        normalized.append(
            _SampleWord(
                index=len(normalized),
                start=start,
                end=max(start, end),
                timing=timing,
            )
        )
    return normalized


def _distance_to_span(word: _SampleWord, span: AcousticSpeechSpan) -> tuple[int, int]:
    overlap = max(0, min(word.end, span.end_sample) - max(word.start, span.start_sample))
    if overlap:
        return overlap, 0
    if word.end <= span.start_sample:
        return 0, span.start_sample - word.end
    if span.end_sample <= word.start:
        return 0, word.start - span.end_sample
    # A zero-duration backend artifact can sit inside a non-empty acoustic
    # span. It has no overlap length but is still at zero temporal distance.
    return 0, 0


def _associate_words(
    words: list[_SampleWord],
    spans: tuple[AcousticSpeechSpan, ...],
    *,
    max_gap_samples: int,
) -> tuple[list[int | None], list[list[int]]]:
    word_to_span: list[int | None] = [None] * len(words)
    span_to_words: list[list[int]] = [[] for _span in spans]
    for word in words:
        ranks: list[tuple[int, int, int]] = []
        for span_index, span in enumerate(spans):
            overlap, gap = _distance_to_span(word, span)
            if overlap or gap <= max_gap_samples:
                # Greatest overlap wins. With no overlap, the shortest gap wins;
                # the earlier acoustic island makes the final tie deterministic.
                ranks.append((-overlap, gap, span_index))
        if not ranks:
            continue
        span_index = min(ranks)[2]
        word_to_span[word.index] = span_index
        span_to_words[span_index].append(word.index)
    return word_to_span, span_to_words


def _snap_outer_word_edges(
    words: list[_SampleWord],
    spans: tuple[AcousticSpeechSpan, ...],
    span_to_words: list[list[int]],
    *,
    max_snap_samples: int,
) -> None:
    """Use confirmed acoustic transitions only on a region's outside words.

    There is no reliable energy valley between many words in continuous speech.
    Consequently, internal forced-alignment edges are never touched here. This
    is the central safety invariant that keeps the edit profile from inventing
    lexical cut points merely because the waveform changes amplitude.
    """
    for span, indices in zip(spans, span_to_words, strict=True):
        if not indices:
            continue
        first = words[indices[0]]
        last = words[indices[-1]]
        if (
            first.end > first.start
            and span.start_energy_confirmed
            and abs(first.start - span.start_sample) <= max_snap_samples
            and span.start_sample <= first.end
            and (
                first.index == 0
                or span.start_sample >= words[first.index - 1].end
            )
        ):
            first.start = span.start_sample
        if (
            last.end > last.start
            and span.end_energy_confirmed
            and abs(last.end - span.end_sample) <= max_snap_samples
            and span.end_sample >= last.start
            and (
                last.index == len(words) - 1
                or span.end_sample <= words[last.index + 1].start
            )
        ):
            last.end = span.end_sample


def _gap_is_confirmed_silence(
    start: int,
    end: int,
    *,
    vad: VadAnalysis,
    settings: EditMaxSettings,
) -> bool:
    min_silence = round(settings.sample_rate_hz * settings.min_silence_ms / 1000)
    if end - start < min_silence:
        return False
    complete_frames = [
        frame
        for frame in vad.frames
        if frame.start_sample >= start and frame.end_sample <= end
    ]
    return bool(complete_frames) and all(
        frame.probability < settings.vad_end_threshold for frame in complete_frames
    )


def _merge_candidates(candidates: list[_RegionCandidate]) -> list[_RegionCandidate]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: (item.start, item.end))
    merged: list[_RegionCandidate] = []
    for candidate in ordered:
        if candidate.end <= candidate.start:
            continue
        if not merged or candidate.start >= merged[-1].end:
            merged.append(candidate)
            continue

        current = merged[-1]
        old_start = current.start
        old_end = current.end
        current.start = min(current.start, candidate.start)
        current.end = max(current.end, candidate.end)
        current.word_indices.update(candidate.word_indices)
        current.has_alignment = current.has_alignment or candidate.has_alignment
        current.has_acoustic = current.has_acoustic or candidate.has_acoustic
        if candidate.vad_peak is not None:
            current.vad_peak = max(current.vad_peak or 0.0, candidate.vad_peak)
        if candidate.vad_mean is not None:
            current.vad_mean = max(current.vad_mean or 0.0, candidate.vad_mean)
        if candidate.start < old_start:
            current.start_energy_confirmed = candidate.start_energy_confirmed
        elif candidate.start == old_start:
            current.start_energy_confirmed = (
                current.start_energy_confirmed or candidate.start_energy_confirmed
            )
        if candidate.end > old_end:
            current.end_energy_confirmed = candidate.end_energy_confirmed
        elif candidate.end == old_end:
            current.end_energy_confirmed = (
                current.end_energy_confirmed or candidate.end_energy_confirmed
            )
    return merged


def _candidate_evidence(candidate: _RegionCandidate) -> str:
    if candidate.has_alignment and candidate.has_acoustic:
        return "alignment_acoustic"
    if candidate.has_acoustic:
        return "acoustic_only"
    return "alignment_only"


def _word_text(word: WordTiming) -> str:
    return word.word.strip()


def _join_words(words: list[WordTiming]) -> str:
    return " ".join(token for token in (_word_text(word) for word in words) if token)


def _should_split_segment(current: list[WordTiming], next_word: WordTiming) -> bool:
    if not current:
        return False
    current_start = current[0].start
    current_end = max(word.end for word in current)
    candidate_text = f"{_join_words(current)} {_word_text(next_word)}".strip()
    candidate_duration = next_word.end - current_start
    if next_word.start - current_end >= SPLIT_GAP_SECONDS:
        return True
    if len(candidate_text) > MAX_SEGMENT_CHARS:
        return True
    if candidate_duration > HARD_MAX_SEGMENT_SECONDS:
        return True
    return (
        _word_text(current[-1]).endswith(SENTENCE_ENDINGS)
        and candidate_duration >= MAX_SEGMENT_SECONDS
    )


def words_to_readable_segments(words: list[WordTiming]) -> list[SegmentTiming]:
    """Apply the existing Qwen subtitle grouping without defining edit regions."""
    segments: list[SegmentTiming] = []
    current: list[WordTiming] = []

    def flush() -> None:
        if not current:
            return
        segments.append(
            SegmentTiming(
                id=len(segments),
                start=current[0].start,
                end=max(word.end for word in current),
                text=_join_words(current),
                words=list(current),
            )
        )

    for word in words:
        if _should_split_segment(current, word):
            flush()
            current = []
        current.append(word)
    flush()
    return segments


def build_edit_transcript(
    *,
    aligned_words: list[WordTiming],
    vad: VadAnalysis,
    settings: EditMaxSettings,
    requested_model: str,
    canonical_model: str,
    profile_name: str,
) -> EditTranscript:
    """Fuse word and acoustic evidence into safe automatic editing regions."""
    if vad.sample_rate_hz != settings.sample_rate_hz:
        raise RuntimeError("VAD and edit profile sample rates do not match.")
    if vad.sample_count < 0:
        raise RuntimeError("VAD returned a negative timeline length.")
    previous_span_end = 0
    for span in vad.spans:
        if not (
            previous_span_end <= span.start_sample < span.end_sample <= vad.sample_count
        ):
            raise RuntimeError("VAD returned invalid or overlapping speech spans.")
        previous_span_end = span.end_sample
    words = _normalize_words(
        aligned_words,
        sample_rate_hz=vad.sample_rate_hz,
        sample_count=vad.sample_count,
    )
    association_samples = round(
        settings.sample_rate_hz * settings.word_association_ms / 1000
    )
    word_to_span, span_to_words = _associate_words(
        words,
        vad.spans,
        max_gap_samples=association_samples,
    )
    _snap_outer_word_edges(
        words,
        vad.spans,
        span_to_words,
        max_snap_samples=round(
            settings.sample_rate_hz * settings.outer_word_snap_ms / 1000
        ),
    )

    candidates: list[_RegionCandidate] = []
    for span_index, span in enumerate(vad.spans):
        indices = span_to_words[span_index]
        if indices:
            candidate_start = min(
                span.start_sample,
                *(words[index].start for index in indices),
            )
            candidate_end = max(
                span.end_sample,
                *(words[index].end for index in indices),
            )
            candidates.append(
                _RegionCandidate(
                    start=candidate_start,
                    end=candidate_end,
                    word_indices=set(indices),
                    has_alignment=True,
                    has_acoustic=True,
                    vad_peak=span.peak_probability,
                    vad_mean=span.mean_probability,
                    start_energy_confirmed=(
                        span.start_energy_confirmed
                        and candidate_start == span.start_sample
                    ),
                    end_energy_confirmed=(
                        span.end_energy_confirmed
                        and candidate_end == span.end_sample
                    ),
                )
            )
        elif (
            span.peak_probability >= settings.vad_only_min_peak
            and span.mean_probability >= settings.vad_only_min_mean
        ):
            candidates.append(
                _RegionCandidate(
                    start=span.start_sample,
                    end=span.end_sample,
                    has_acoustic=True,
                    vad_peak=span.peak_probability,
                    vad_mean=span.mean_probability,
                    start_energy_confirmed=span.start_energy_confirmed,
                    end_energy_confirmed=span.end_energy_confirmed,
                )
            )

    unmatched = [index for index, span_index in enumerate(word_to_span) if span_index is None]
    fallback_group: list[int] = []

    def flush_fallback() -> None:
        if not fallback_group:
            return
        candidates.append(
            _RegionCandidate(
                start=min(words[index].start for index in fallback_group),
                end=max(words[index].end for index in fallback_group),
                word_indices=set(fallback_group),
                has_alignment=True,
            )
        )

    for index in unmatched:
        if fallback_group:
            previous = fallback_group[-1]
            not_consecutive = index != previous + 1
            confirmed_gap = _gap_is_confirmed_silence(
                words[previous].end,
                words[index].start,
                vad=vad,
                settings=settings,
            )
            if not_consecutive or confirmed_gap:
                flush_fallback()
                fallback_group = []
        fallback_group.append(index)
    flush_fallback()

    merged = _merge_candidates(candidates)
    final_words = [
        replace(
            word.timing,
            start=word.start / float(vad.sample_rate_hz),
            end=word.end / float(vad.sample_rate_hz),
        )
        for word in words
    ]

    regions: list[EditingSpeechRegion] = []
    boundaries: list[EditBoundary] = []
    for candidate in merged:
        start = min(vad.sample_count, max(0, candidate.start))
        end = min(vad.sample_count, max(start, candidate.end))
        if end <= start:
            continue
        indices = sorted(candidate.word_indices)
        evidence = _candidate_evidence(candidate)
        region_probabilities = [
            frame.probability
            for frame in vad.frames
            if candidate.has_acoustic
            and frame.end_sample > start
            and frame.start_sample < end
        ]
        vad_peak = (
            max(region_probabilities)
            if region_probabilities
            else candidate.vad_peak
        )
        vad_mean = (
            sum(region_probabilities) / len(region_probabilities)
            if region_probabilities
            else candidate.vad_mean
        )
        region = EditingSpeechRegion(
            id=len(regions),
            start_sample=start,
            end_sample=end,
            evidence=evidence,
            word_start_index=indices[0] if indices else None,
            word_end_index=(indices[-1] + 1) if indices else None,
            vad_peak_probability=vad_peak,
            vad_mean_probability=vad_mean,
            start_energy_confirmed=candidate.start_energy_confirmed,
            end_energy_confirmed=candidate.end_energy_confirmed,
        )
        regions.append(region)
        boundaries.extend(
            [
                EditBoundary(
                    region_id=region.id,
                    type="speech_start",
                    sample=region.start_sample,
                    evidence=evidence,
                    energy_confirmed=region.start_energy_confirmed,
                ),
                EditBoundary(
                    region_id=region.id,
                    type="speech_end",
                    sample=region.end_sample,
                    evidence=evidence,
                    energy_confirmed=region.end_energy_confirmed,
                ),
            ]
        )

    editing = EditingResult(
        schema_version=1,
        profile=profile_name,
        requested_model=requested_model,
        canonical_model=canonical_model,
        timeline=EditingTimeline(
            sample_rate_hz=vad.sample_rate_hz,
            sample_count=vad.sample_count,
        ),
        speech_regions=tuple(regions),
        edit_boundaries=tuple(boundaries),
    )
    return EditTranscript(
        words=tuple(final_words),
        segments=tuple(words_to_readable_segments(final_words)),
        editing=editing,
    )
