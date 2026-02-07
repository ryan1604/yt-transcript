"""YouTube URL parsing and validation."""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from yt_transcript.errors import InvalidInputError


YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "www.youtu.be",
}


def extract_video_id(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise InvalidInputError("URL must start with http:// or https://")
    if parsed.netloc.lower() not in YOUTUBE_HOSTS:
        raise InvalidInputError("Only YouTube video URLs are supported")

    host = parsed.netloc.lower()
    if "youtu.be" in host:
        video_id = parsed.path.strip("/").split("/")[0]
    else:
        if parsed.path.startswith("/shorts/"):
            video_id = parsed.path.split("/shorts/")[-1].split("/")[0]
        else:
            query = parse_qs(parsed.query)
            video_id = query.get("v", [""])[0]

    if not _is_valid_video_id(video_id):
        raise InvalidInputError("Could not extract a valid YouTube video ID from URL")
    return video_id


def normalize_video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _is_valid_video_id(video_id: str) -> bool:
    # YouTube IDs are typically 11 chars; this keeps validation strict for v1.
    return len(video_id) == 11 and all(ch.isalnum() or ch in {"-", "_"} for ch in video_id)
