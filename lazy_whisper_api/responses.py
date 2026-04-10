"""Response rendering for plain text, subtitles, and verbose JSON."""

from __future__ import annotations

import io
from typing import Any

from fastapi.responses import JSONResponse, PlainTextResponse

from .transcription import TranscriptionResult


def format_timestamp(
    seconds: float,
    *,
    always_include_hours: bool = False,
    decimal_marker: str = ".",
) -> str:
    """Format subtitle timestamps in SRT/VTT-compatible forms."""
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    secs = milliseconds // 1000
    milliseconds -= secs * 1000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{secs:02d}{decimal_marker}{milliseconds:03d}"


def write_srt(segments: list[Any]) -> str:
    """Serialize segments into SRT subtitle format."""
    output = io.StringIO()
    for index, segment in enumerate(segments, start=1):
        output.write(f"{index}\n")
        output.write(
            f"{format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')} --> "
            f"{format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')}\n"
        )
        output.write(f"{segment.text.strip()}\n\n")
    return output.getvalue()


def write_vtt(segments: list[Any]) -> str:
    """Serialize segments into WebVTT format."""
    output = io.StringIO()
    output.write("WEBVTT\n\n")
    for segment in segments:
        output.write(
            f"{format_timestamp(segment.start, always_include_hours=True)} --> "
            f"{format_timestamp(segment.end, always_include_hours=True)}\n"
        )
        output.write(f"{segment.text.strip()}\n\n")
    return output.getvalue()


def build_verbose_json(result: TranscriptionResult) -> dict[str, Any]:
    """Build the OpenAI-style verbose JSON payload."""
    payload: dict[str, Any] = {
        "model": result.model_name,
        "device": result.device,
        "language": result.info.language,
        "language_probability": result.info.language_probability,
        "duration": result.info.duration,
        "text": result.text,
        "segments": [],
    }

    words: list[dict[str, Any]] = []
    for segment in result.segments:
        payload["segments"].append(
            {
                "id": segment.id,
                "seek": segment.seek,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "tokens": segment.tokens,
                "temperature": segment.temperature,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob,
                "words": [
                    {
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "probability": word.probability,
                    }
                    for word in (segment.words or [])
                ],
            }
        )
        for word in segment.words or []:
            words.append(
                {
                    "start": word.start,
                    "end": word.end,
                    "word": word.word,
                    "probability": word.probability,
                }
            )
    if words:
        payload["words"] = words
    return payload


def build_transcription_response(result: TranscriptionResult) -> JSONResponse | PlainTextResponse:
    """Render the final HTTP response in the requested format."""
    if result.response_format == "json":
        return JSONResponse({"text": result.text, "model": result.model_name})
    if result.response_format == "verbose_json":
        return JSONResponse(build_verbose_json(result))
    if result.response_format == "text":
        return PlainTextResponse(result.text)
    if result.response_format == "srt":
        return PlainTextResponse(write_srt(result.segments))
    if result.response_format == "vtt":
        return PlainTextResponse(write_vtt(result.segments))

    raise RuntimeError(f"Unexpected response format: {result.response_format}")
