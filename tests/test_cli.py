from pathlib import Path

from typer.testing import CliRunner

from audio_transcribe_translate.cli import app
from audio_transcribe_translate.errors import RetrievalError
from audio_transcribe_translate.models import TranscriptSegment


runner = CliRunner()


def test_cli_uses_local_media_without_download_or_captions(monkeypatch, tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp3"
    media_path.write_bytes(b"media")
    output_path = tmp_path / "clip.txt"
    calls: list[str] = []

    def fake_fetch_captions(*args, **kwargs):
        calls.append("captions")
        raise AssertionError("local input should not fetch captions")

    def fake_download_audio(*args, **kwargs):
        calls.append("download")
        raise AssertionError("local input should not download media")

    def fake_preprocess_audio_to_wav(input_path: Path, temp_dir: Path, input_id: str) -> Path:
        calls.append(f"preprocess:{input_path.name}:{input_id}")
        wav_path = temp_dir / f"{input_id}.wav"
        wav_path.write_bytes(b"wav")
        return wav_path

    def fake_transcribe_audio(
        wav_path: Path, model_name: str, device: str, language: str | None, task: str
    ):
        calls.append(f"transcribe:{wav_path.name}:{task}:{language}")
        return ([TranscriptSegment(start=0.0, end=1.0, text="hello local world")], "en")

    monkeypatch.setattr("audio_transcribe_translate.cli.fetch_captions", fake_fetch_captions)
    monkeypatch.setattr("audio_transcribe_translate.cli.download_audio", fake_download_audio)
    monkeypatch.setattr("audio_transcribe_translate.cli.preprocess_audio_to_wav", fake_preprocess_audio_to_wav)
    monkeypatch.setattr("audio_transcribe_translate.cli.transcribe_audio", fake_transcribe_audio)

    result = runner.invoke(app, [str(media_path), "--out-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert calls == ["preprocess:clip.mp3:clip", "transcribe:clip.wav:transcribe:en"]
    assert output_path.read_text(encoding="utf-8") == "hello local world\n"


def test_cli_translation_bypasses_captions_for_youtube(monkeypatch, tmp_path: Path) -> None:
    output_path = tmp_path / "dQw4w9WgXcQ.txt"
    calls: list[str] = []

    def fake_fetch_captions(*args, **kwargs):
        calls.append("captions")
        raise AssertionError("translation mode should bypass captions")

    def fake_download_audio(video_url: str, input_id: str, temp_dir: Path) -> Path:
        calls.append(f"download:{input_id}")
        media_path = temp_dir / f"{input_id}.mp3"
        media_path.write_bytes(b"media")
        return media_path

    def fake_preprocess_audio_to_wav(input_path: Path, temp_dir: Path, input_id: str) -> Path:
        calls.append(f"preprocess:{input_path.name}:{input_id}")
        wav_path = temp_dir / f"{input_id}.wav"
        wav_path.write_bytes(b"wav")
        return wav_path

    def fake_transcribe_audio(
        wav_path: Path, model_name: str, device: str, language: str | None, task: str
    ):
        calls.append(f"transcribe:{wav_path.name}:{task}:{language}")
        return ([TranscriptSegment(start=0.0, end=1.0, text="hello world")], "es")

    monkeypatch.setattr("audio_transcribe_translate.cli.fetch_captions", fake_fetch_captions)
    monkeypatch.setattr("audio_transcribe_translate.cli.download_audio", fake_download_audio)
    monkeypatch.setattr("audio_transcribe_translate.cli.preprocess_audio_to_wav", fake_preprocess_audio_to_wav)
    monkeypatch.setattr("audio_transcribe_translate.cli.transcribe_audio", fake_transcribe_audio)

    result = runner.invoke(
        app,
        [
            "https://youtu.be/dQw4w9WgXcQ",
            "--task",
            "translate",
            "--model",
            "large-v3",
            "--out-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        "download:dQw4w9WgXcQ",
        "preprocess:dQw4w9WgXcQ.mp3:dQw4w9WgXcQ",
        "transcribe:dQw4w9WgXcQ.wav:translate:None",
    ]
    assert "task=translate" in result.stdout
    assert "language=en" in result.stdout
    assert "source_language=es" in result.stdout
    assert output_path.read_text(encoding="utf-8") == "hello world\n"


def test_cli_uses_youtube_captions_for_transcription(monkeypatch, tmp_path: Path) -> None:
    output_path = tmp_path / "dQw4w9WgXcQ.txt"
    calls: list[str] = []

    def fake_fetch_captions(video_id: str, language: str):
        calls.append(f"captions:{video_id}:{language}")
        return [TranscriptSegment(start=0.0, end=1.0, text="caption transcript")]

    def fake_download_audio(*args, **kwargs):
        calls.append("download")
        raise AssertionError("caption success should not download media")

    def fake_preprocess_audio_to_wav(*args, **kwargs):
        calls.append("preprocess")
        raise AssertionError("caption success should not preprocess audio")

    def fake_transcribe_audio(*args, **kwargs):
        calls.append("transcribe")
        raise AssertionError("caption success should not run STT")

    monkeypatch.setattr("audio_transcribe_translate.cli.fetch_captions", fake_fetch_captions)
    monkeypatch.setattr("audio_transcribe_translate.cli.download_audio", fake_download_audio)
    monkeypatch.setattr("audio_transcribe_translate.cli.preprocess_audio_to_wav", fake_preprocess_audio_to_wav)
    monkeypatch.setattr("audio_transcribe_translate.cli.transcribe_audio", fake_transcribe_audio)

    result = runner.invoke(app, ["https://youtu.be/dQw4w9WgXcQ", "--out-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert calls == ["captions:dQw4w9WgXcQ:en"]
    assert "source=captions" in result.stdout
    assert output_path.read_text(encoding="utf-8") == "caption transcript\n"


def test_cli_falls_back_to_stt_when_youtube_captions_unavailable(monkeypatch, tmp_path: Path) -> None:
    output_path = tmp_path / "dQw4w9WgXcQ.txt"
    calls: list[str] = []

    def fake_fetch_captions(*args, **kwargs):
        calls.append("captions")
        raise RetrievalError("no captions")

    def fake_download_audio(video_url: str, input_id: str, temp_dir: Path) -> Path:
        calls.append(f"download:{input_id}")
        media_path = temp_dir / f"{input_id}.mp3"
        media_path.write_bytes(b"media")
        return media_path

    def fake_preprocess_audio_to_wav(input_path: Path, temp_dir: Path, input_id: str) -> Path:
        calls.append(f"preprocess:{input_path.name}:{input_id}")
        wav_path = temp_dir / f"{input_id}.wav"
        wav_path.write_bytes(b"wav")
        return wav_path

    def fake_transcribe_audio(
        wav_path: Path, model_name: str, device: str, language: str | None, task: str
    ):
        calls.append(f"transcribe:{wav_path.name}:{task}:{language}")
        return ([TranscriptSegment(start=0.0, end=1.0, text="fallback transcript")], "en")

    monkeypatch.setattr("audio_transcribe_translate.cli.fetch_captions", fake_fetch_captions)
    monkeypatch.setattr("audio_transcribe_translate.cli.download_audio", fake_download_audio)
    monkeypatch.setattr("audio_transcribe_translate.cli.preprocess_audio_to_wav", fake_preprocess_audio_to_wav)
    monkeypatch.setattr("audio_transcribe_translate.cli.transcribe_audio", fake_transcribe_audio)

    result = runner.invoke(app, ["https://youtu.be/dQw4w9WgXcQ", "--out-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert calls == [
        "captions",
        "download:dQw4w9WgXcQ",
        "preprocess:dQw4w9WgXcQ.mp3:dQw4w9WgXcQ",
        "transcribe:dQw4w9WgXcQ.wav:transcribe:en",
    ]
    assert "source=stt" in result.stdout
    assert output_path.read_text(encoding="utf-8") == "fallback transcript\n"


def test_cli_uses_local_media_for_translation(monkeypatch, tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp3"
    media_path.write_bytes(b"media")
    output_path = tmp_path / "clip.txt"
    calls: list[str] = []

    def fake_fetch_captions(*args, **kwargs):
        calls.append("captions")
        raise AssertionError("local translation should not fetch captions")

    def fake_download_audio(*args, **kwargs):
        calls.append("download")
        raise AssertionError("local translation should not download media")

    def fake_preprocess_audio_to_wav(input_path: Path, temp_dir: Path, input_id: str) -> Path:
        calls.append(f"preprocess:{input_path.name}:{input_id}")
        wav_path = temp_dir / f"{input_id}.wav"
        wav_path.write_bytes(b"wav")
        return wav_path

    def fake_transcribe_audio(
        wav_path: Path, model_name: str, device: str, language: str | None, task: str
    ):
        calls.append(f"transcribe:{wav_path.name}:{task}:{language}")
        return ([TranscriptSegment(start=0.0, end=1.0, text="english translation")], "ja")

    monkeypatch.setattr("audio_transcribe_translate.cli.fetch_captions", fake_fetch_captions)
    monkeypatch.setattr("audio_transcribe_translate.cli.download_audio", fake_download_audio)
    monkeypatch.setattr("audio_transcribe_translate.cli.preprocess_audio_to_wav", fake_preprocess_audio_to_wav)
    monkeypatch.setattr("audio_transcribe_translate.cli.transcribe_audio", fake_transcribe_audio)

    result = runner.invoke(
        app,
        [str(media_path), "--task", "translate", "--model", "large-v3", "--out-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert calls == ["preprocess:clip.mp3:clip", "transcribe:clip.wav:translate:None"]
    assert "task=translate" in result.stdout
    assert "language=en" in result.stdout
    assert "source_language=ja" in result.stdout
    assert output_path.read_text(encoding="utf-8") == "english translation\n"


def test_cli_rejects_force_stt_for_local_input(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp3"
    media_path.write_bytes(b"media")

    result = runner.invoke(app, [str(media_path), "--force-stt"])

    assert result.exit_code == 1
    assert "--force-stt is only supported for YouTube transcription runs" in result.stdout


def test_cli_reports_missing_local_file_with_invalid_input_exit_code(tmp_path: Path) -> None:
    result = runner.invoke(app, [str(tmp_path / "missing.mp3")])

    assert result.exit_code == 1
    assert "Local media file does not exist" in result.stdout


def test_cli_reports_retrieval_failures_from_youtube_stt_path(monkeypatch, tmp_path: Path) -> None:
    def fake_download_audio(*args, **kwargs):
        raise RetrievalError("download failed")

    monkeypatch.setattr("audio_transcribe_translate.cli.download_audio", fake_download_audio)

    result = runner.invoke(
        app,
        [
            "https://youtu.be/dQw4w9WgXcQ",
            "--task",
            "translate",
            "--model",
            "large-v3",
            "--out-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 2
    assert "download failed" in result.stdout
