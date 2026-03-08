"""Local speech-to-text transcription using faster-whisper."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from audio_transcribe_translate.runtime import ensure_windows_cuda_dlls

ensure_windows_cuda_dlls()

from faster_whisper import WhisperModel

from audio_transcribe_translate.errors import TranscriptionError
from audio_transcribe_translate.models import TranscriptSegment

_MODEL_CACHE: dict[tuple[str, str], WhisperModel] = {}


def transcribe_audio(
    wav_path: Path,
    model_name: str,
    device: str,
    language: str | None,
    task: Literal["transcribe", "translate"],
) -> tuple[list[TranscriptSegment], str | None]:
    try:
        model = _get_whisper_model(model_name=model_name, device=device)
    except Exception as exc:
        raise TranscriptionError(f"Unable to load faster-whisper model '{model_name}': {exc}") from exc

    try:
        segments_iter, info = model.transcribe(
            str(wav_path),
            language=language,
            task=task,
            # vad_filter=True,
            beam_size=3,
        )
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


def _get_whisper_model(model_name: str, device: str) -> WhisperModel:
    cache_key = (model_name, device)
    model = _MODEL_CACHE.get(cache_key)
    if model is not None:
        return model

    # Keep models alive for the process lifetime. On some Windows CUDA setups,
    # releasing the model immediately after translation can terminate the process
    # before the CLI reaches output writing.
    model = WhisperModel(model_name, device=device)
    _MODEL_CACHE[cache_key] = model
    return model
