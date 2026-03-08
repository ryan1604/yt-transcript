"""Runtime helpers for native dependencies."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DLL_HANDLES: list[object] = []


def ensure_windows_cuda_dlls() -> None:
    """Expose bundled CUDA DLLs before importing GPU-backed libraries."""
    if sys.platform != "win32":
        return

    for dll_dir in _candidate_cuda_dirs():
        if not dll_dir.is_dir():
            continue

        dll_dir_str = str(dll_dir)
        if hasattr(os, "add_dll_directory"):
            _DLL_HANDLES.append(os.add_dll_directory(dll_dir_str))

        path_value = os.environ.get("PATH", "")
        path_parts = path_value.split(os.pathsep) if path_value else []
        if dll_dir_str not in path_parts:
            os.environ["PATH"] = dll_dir_str if not path_value else dll_dir_str + os.pathsep + path_value
        return


def _candidate_cuda_dirs() -> tuple[Path, ...]:
    package_dir = Path(__file__).resolve().parent
    return (
        package_dir.parent / "cuda" / "bin",
        package_dir / "cuda" / "bin",
    )
