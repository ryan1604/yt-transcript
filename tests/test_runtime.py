import os
from pathlib import Path

from audio_transcribe_translate import runtime


def test_candidate_cuda_dirs_include_repo_src_cuda_bin() -> None:
    candidate_dirs = runtime._candidate_cuda_dirs()
    expected_dir = Path(runtime.__file__).resolve().parent.parent / "cuda" / "bin"

    assert expected_dir in candidate_dirs


def test_ensure_windows_cuda_dlls_prefers_existing_candidate(monkeypatch, tmp_path: Path) -> None:
    first_dir = tmp_path / "missing"
    second_dir = tmp_path / "cuda" / "bin"
    second_dir.mkdir(parents=True)
    handles: list[str] = []

    monkeypatch.setattr(runtime.sys, "platform", "win32")
    monkeypatch.setattr(runtime, "_DLL_HANDLES", [])
    monkeypatch.setattr(runtime, "_candidate_cuda_dirs", lambda: (first_dir, second_dir))
    monkeypatch.setattr(runtime.os, "add_dll_directory", lambda path: handles.append(path) or path)
    monkeypatch.setenv("PATH", "C:\\existing")

    runtime.ensure_windows_cuda_dlls()

    second_dir_str = str(second_dir)
    assert handles == [second_dir_str]
    assert runtime._DLL_HANDLES == [second_dir_str]
    assert os.environ["PATH"].split(os.pathsep)[0] == second_dir_str
