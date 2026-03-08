"""Input classification and normalization for YouTube URLs and local media."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from audio_transcribe_translate.errors import InvalidInputError
from audio_transcribe_translate.url_parser import extract_video_id, normalize_video_url

InputType = Literal["youtube", "local"]

_AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}
_VIDEO_EXTENSIONS = {
    ".3gp",
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ts",
    ".webm",
    ".wmv",
}
_SUPPORTED_EXTENSIONS = _AUDIO_EXTENSIONS | _VIDEO_EXTENSIONS
_INVALID_ID_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_SEPARATOR_RUNS = re.compile(r"[-_.]{2,}")


@dataclass(frozen=True, slots=True)
class ResolvedInput:
    input_type: InputType
    input_id: str
    input_reference: str
    source: str
    local_path: Path | None = None
    video_id: str | None = None
    canonical_url: str | None = None


def resolve_input(raw_input: str) -> ResolvedInput:
    candidate = raw_input.strip()
    if not candidate:
        raise InvalidInputError("Input must not be empty")

    if "://" in candidate:
        return _resolve_youtube_input(candidate)
    return _resolve_local_input(candidate)


def _resolve_youtube_input(raw_input: str) -> ResolvedInput:
    video_id = extract_video_id(raw_input)
    canonical_url = normalize_video_url(video_id)
    return ResolvedInput(
        input_type="youtube",
        input_id=video_id,
        input_reference=canonical_url,
        source=raw_input,
        video_id=video_id,
        canonical_url=canonical_url,
    )


def _resolve_local_input(raw_input: str) -> ResolvedInput:
    local_path = Path(raw_input).expanduser()

    if not local_path.exists():
        raise InvalidInputError(f"Local media file does not exist: {local_path}")
    if not local_path.is_file():
        raise InvalidInputError(f"Local media input must be a file: {local_path}")
    if local_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(_SUPPORTED_EXTENSIONS))
        raise InvalidInputError(
            f"Unsupported local media type '{local_path.suffix or '<none>'}'. "
            f"Supported extensions: {supported}"
        )
    if not os.access(local_path, os.R_OK):
        raise InvalidInputError(f"Local media file is not readable: {local_path}")

    resolved_path = local_path.resolve()
    return ResolvedInput(
        input_type="local",
        input_id=_sanitize_input_id(resolved_path.stem),
        input_reference=str(resolved_path),
        source=raw_input,
        local_path=resolved_path,
    )


def _sanitize_input_id(stem: str) -> str:
    cleaned = _INVALID_ID_CHARS.sub("-", stem.strip())
    cleaned = _SEPARATOR_RUNS.sub("-", cleaned)
    cleaned = cleaned.strip("._-")
    return cleaned or "input"
