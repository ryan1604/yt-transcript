"""Local speech-to-text transcription using faster-whisper."""

from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel

from yt_transcript.errors import TranscriptionError
from yt_transcript.models import TranscriptSegment


def transcribe_audio(
    wav_path: Path,
    model_name: str,
    device: str,
    language: str | None,
) -> tuple[list[TranscriptSegment], str | None]:
    try:
        model = WhisperModel(model_name, device=device)
    except Exception as exc:
        raise TranscriptionError(f"Unable to load faster-whisper model '{model_name}': {exc}") from exc

    try:
        segments_iter, info = model.transcribe(str(wav_path), language=language, vad_filter=True)
    except Exception as exc:
        raise TranscriptionError(f"Transcription failed: {exc}") from exc

    segments: list[TranscriptSegment] = []
    for segment in segments_iter:
        text = segment.text.strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                start=max(float(segment.start), 0.0),
                end=max(float(segment.end), 0.0),
                text=text,
            )
        )

    if not segments:
        raise TranscriptionError("faster-whisper returned no transcript segments")
    detected_language = getattr(info, "language", None)
    return segments, detected_language
