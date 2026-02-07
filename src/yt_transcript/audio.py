"""Audio download and preprocessing for STT fallback."""

from __future__ import annotations

import subprocess
from pathlib import Path

from yt_dlp import YoutubeDL

from yt_transcript.errors import RetrievalError, TranscriptionError


def download_audio(video_url: str, video_id: str, temp_dir: Path) -> Path:
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(temp_dir / f"{video_id}.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            downloaded = ydl.prepare_filename(info)
    except Exception as exc:
        raise RetrievalError(f"Failed to download audio: {exc}") from exc

    path = Path(downloaded)
    if not path.exists():
        raise RetrievalError("yt-dlp did not produce an audio file")
    return path


def preprocess_audio_to_wav(input_path: Path, temp_dir: Path, video_id: str) -> Path:
    output_path = temp_dir / f"{video_id}.wav"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]

    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise TranscriptionError("ffmpeg is not installed or not on PATH") from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "Unknown ffmpeg error"
        raise TranscriptionError(f"Audio preprocessing failed: {detail}")
    if not output_path.exists():
        raise TranscriptionError("ffmpeg completed but output WAV file is missing")
    return output_path
