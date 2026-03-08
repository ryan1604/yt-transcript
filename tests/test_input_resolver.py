from pathlib import Path

import pytest

from audio_transcribe_translate.errors import InvalidInputError
from audio_transcribe_translate.input_resolver import resolve_input


def test_resolve_youtube_input_uses_canonical_metadata() -> None:
    resolved = resolve_input("https://youtu.be/dQw4w9WgXcQ")

    assert resolved.input_type == "youtube"
    assert resolved.input_id == "dQw4w9WgXcQ"
    assert resolved.input_reference == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert resolved.video_id == "dQw4w9WgXcQ"
    assert resolved.local_path is None


def test_resolve_local_input_uses_resolved_path_and_sanitized_stem(tmp_path: Path) -> None:
    media_path = tmp_path / "My sample clip!.mp3"
    media_path.write_bytes(b"audio")

    resolved = resolve_input(str(media_path))

    assert resolved.input_type == "local"
    assert resolved.input_id == "My-sample-clip"
    assert resolved.input_reference == str(media_path.resolve())
    assert resolved.local_path == media_path.resolve()
    assert resolved.video_id is None


def test_resolve_non_http_url_preserves_youtube_validation_rules() -> None:
    with pytest.raises(InvalidInputError, match="URL must start with http:// or https://"):
        resolve_input("ftp://www.youtube.com/watch?v=dQw4w9WgXcQ")


def test_resolve_local_input_rejects_missing_files(tmp_path: Path) -> None:
    with pytest.raises(InvalidInputError, match="does not exist"):
        resolve_input(str(tmp_path / "missing.mp3"))


def test_resolve_local_input_rejects_directories(tmp_path: Path) -> None:
    with pytest.raises(InvalidInputError, match="must be a file"):
        resolve_input(str(tmp_path))


def test_resolve_local_input_rejects_unsupported_extensions(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("not media", encoding="utf-8")

    with pytest.raises(InvalidInputError, match="Unsupported local media type"):
        resolve_input(str(file_path))
