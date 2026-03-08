from pathlib import Path
from types import SimpleNamespace

from audio_transcribe_translate.models import TranscriptSegment
from audio_transcribe_translate.transcribe import transcribe_audio


def test_transcribe_audio_passes_task_through_to_whisper(monkeypatch, tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"wav")
    calls: list[tuple[str, str, str | None, str, bool]] = []

    class FakeWhisperModel:
        def __init__(self, model_name: str, device: str):
            assert model_name == "large-v3"
            assert device == "cpu"

        def transcribe(self, audio_path: str, language: str | None, task: str, vad_filter: bool):
            calls.append((audio_path, task, language, "cpu", vad_filter))
            segments = [SimpleNamespace(start=0.0, end=1.25, text=" translated text ")]
            info = SimpleNamespace(language="de")
            return iter(segments), info

    monkeypatch.setattr("audio_transcribe_translate.transcribe.WhisperModel", FakeWhisperModel)
    monkeypatch.setattr("audio_transcribe_translate.transcribe._MODEL_CACHE", {})

    segments, detected_language = transcribe_audio(
        wav_path=wav_path,
        model_name="large-v3",
        device="cpu",
        language="de",
        task="translate",
    )

    assert calls == [(str(wav_path), "translate", "de", "cpu", True)]
    assert segments == [TranscriptSegment(start=0.0, end=1.25, text="translated text")]
    assert detected_language == "de"


def test_transcribe_audio_reuses_cached_model(monkeypatch, tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"wav")
    init_calls: list[tuple[str, str]] = []

    class FakeWhisperModel:
        def __init__(self, model_name: str, device: str):
            init_calls.append((model_name, device))

        def transcribe(self, audio_path: str, language: str | None, task: str, vad_filter: bool):
            segments = [SimpleNamespace(start=0.0, end=1.0, text="hello")]
            info = SimpleNamespace(language=language)
            return iter(segments), info

    monkeypatch.setattr("audio_transcribe_translate.transcribe.WhisperModel", FakeWhisperModel)
    monkeypatch.setattr("audio_transcribe_translate.transcribe._MODEL_CACHE", {})

    first_segments, _ = transcribe_audio(
        wav_path=wav_path,
        model_name="large-v3",
        device="cuda",
        language=None,
        task="translate",
    )
    second_segments, _ = transcribe_audio(
        wav_path=wav_path,
        model_name="large-v3",
        device="cuda",
        language=None,
        task="translate",
    )

    assert init_calls == [("large-v3", "cuda")]
    assert first_segments == [TranscriptSegment(start=0.0, end=1.0, text="hello")]
    assert second_segments == [TranscriptSegment(start=0.0, end=1.0, text="hello")]
