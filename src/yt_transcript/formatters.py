"""Transcript output formatters."""

from __future__ import annotations

import json

from yt_transcript.models import TranscriptResult


def format_output(result: TranscriptResult, fmt: str, timestamps: bool) -> str:
    if fmt == "txt":
        return to_txt(result, timestamps=timestamps)
    if fmt == "srt":
        return to_srt(result)
    if fmt == "json":
        return to_json(result, timestamps=timestamps)
    raise ValueError(f"Unsupported format: {fmt}")


def to_txt(result: TranscriptResult, timestamps: bool) -> str:
    lines: list[str] = []
    for segment in result.segments:
        if timestamps:
            lines.append(f"[{_format_time_txt(segment.start)}] {segment.text}")
        else:
            lines.append(segment.text)
    return "\n".join(lines).strip() + "\n"


def to_srt(result: TranscriptResult) -> str:
    lines: list[str] = []
    for index, segment in enumerate(result.segments, start=1):
        start = _format_time_srt(segment.start)
        end = _format_time_srt(segment.end)
        lines.extend([str(index), f"{start} --> {end}", segment.text, ""])
    return "\n".join(lines).strip() + "\n"


def to_json(result: TranscriptResult, timestamps: bool) -> str:
    payload: dict[str, object] = {
        "video_id": result.video_id,
        "source": result.source,
        "language": result.language,
        "segments": [],
    }
    segments = []
    for segment in result.segments:
        item: dict[str, object] = {"text": segment.text}
        if timestamps:
            item["start"] = round(segment.start, 3)
            item["end"] = round(segment.end, 3)
        segments.append(item)
    payload["segments"] = segments
    return json.dumps(payload, indent=2, ensure_ascii=True) + "\n"


def _format_time_txt(seconds: float) -> str:
    total = int(max(seconds, 0))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_time_srt(seconds: float) -> str:
    ms = int(round(max(seconds, 0) * 1000))
    hours = ms // 3_600_000
    minutes = (ms % 3_600_000) // 60_000
    secs = (ms % 60_000) // 1000
    millis = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
