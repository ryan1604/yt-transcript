from pathlib import Path

from audio_transcribe_translate.io import build_output_path


def test_build_output_path_uses_generic_input_id(tmp_path: Path) -> None:
    output_path = build_output_path(tmp_path, "source-file-01", fmt="json")

    assert output_path == tmp_path / "source-file-01.json"


def test_build_output_path_appends_incrementing_suffix_on_collision(tmp_path: Path) -> None:
    (tmp_path / "source-file-01.json").write_text("existing", encoding="utf-8")
    (tmp_path / "source-file-01-2.json").write_text("existing", encoding="utf-8")

    output_path = build_output_path(tmp_path, "source-file-01", fmt="json")

    assert output_path == tmp_path / "source-file-01-3.json"
