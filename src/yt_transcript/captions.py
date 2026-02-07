"""Captions retrieval from YouTube."""

from __future__ import annotations

from yt_transcript.errors import RetrievalError
from yt_transcript.models import TranscriptSegment

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except Exception as exc:  # pragma: no cover - import guarded for runtime clarity
    raise RuntimeError("youtube-transcript-api is required") from exc


def fetch_captions(video_id: str, language: str) -> list[TranscriptSegment]:
    transcript_data = _get_transcript(video_id, language)
    segments: list[TranscriptSegment] = []
    for item in transcript_data:
        start = float(item.get("start", 0.0))
        duration = float(item.get("duration", 0.0))
        end = start + max(duration, 0.0)
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        segments.append(TranscriptSegment(start=start, end=end, text=text))

    if not segments:
        raise RetrievalError("Captions were found but no usable text segments were returned")
    return segments


def _get_transcript(video_id: str, language: str) -> list[dict]:
    candidates = [[language], [language, "en"], ["en"]]
    api = YouTubeTranscriptApi()
    transcript_list = None

    try:
        transcript_list = api.list(video_id)
    except Exception as exc:
        raise RetrievalError(f"Unable to list captions for video '{video_id}': {exc}") from exc

    last_error: Exception | None = None
    for langs in candidates:
        try:
            transcript = transcript_list.find_manually_created_transcript(langs)
            data = transcript.fetch()
            # NormalizedTranscriptSnippet behaves like an iterable of snippets.
            return [{"start": item.start, "duration": item.duration, "text": item.text} for item in data]
        except Exception as exc:
            last_error = exc

    raise RetrievalError(
        f"No manually created captions available for video '{video_id}' in languages "
        f"{candidates}: {last_error}"
    )
