"""Filesystem helpers for transcript output."""

from __future__ import annotations

from pathlib import Path

from yt_transcript.errors import OutputWriteError


def ensure_out_dir(out_dir: Path) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise OutputWriteError(f"Unable to create output directory '{out_dir}': {exc}") from exc


def build_output_path(out_dir: Path, video_id: str, fmt: str) -> Path:
    extension = {"txt": "txt", "srt": "srt", "json": "json"}[fmt]
    return out_dir / f"{video_id}.{extension}"


def write_text(path: Path, content: str) -> None:
    try:
        path.write_text(content, encoding="utf-8")
    except Exception as exc:
        raise OutputWriteError(f"Unable to write transcript file '{path}': {exc}") from exc
