# audio-transcribe-translate

`audio-transcribe-translate` is a CLI for one-shot speech workflows. It accepts either a YouTube URL or a local audio/video file and writes either a transcript or an English translation via the `att` command.

## What It Does

- Runs exactly one task per command: `transcribe` or `translate`.
- Accepts either a YouTube video URL or a local media file path.
- Uses YouTube captions only for YouTube transcription.
- Falls back to local `faster-whisper` speech processing when captions are unavailable.
- Uses local `faster-whisper` for all translation work.
- Writes output as `txt`, `srt`, or `json`.

## Input Types

`att <input>` accepts:

- A YouTube watch URL or short URL such as `https://www.youtube.com/watch?v=...` or `https://youtu.be/...`
- A local media file such as `.mp3`, `.wav`, `.m4a`, `.mp4`, `.mkv`, `.webm`, and similar supported audio/video formats

YouTube inputs use the canonical video ID as the output stem. Local files use a sanitized file stem. If an output name already exists, the CLI writes `-2`, `-3`, and so on instead of overwriting the existing file.

## Tasks

`transcribe`

- Default task
- Preserves the spoken language in the output text
- For YouTube inputs, tries manual captions first unless `--force-stt` is used

`translate`

- Always produces English output
- Always uses local speech processing
- Bypasses the YouTube captions subsystem

`--language` behavior:

- In `transcribe` mode, it is the preferred captions language for YouTube and the source-language hint for STT
- In `translate` mode, it is treated as the source-language hint for STT

## Installation

Install dependencies and make sure `ffmpeg` is available on `PATH` or present in the repository root.

```powershell
git clone <repo_url>
cd <repo_dir>
uv sync
uv run att --help
```

On Windows, if you want CUDA inference without installing PyTorch, place the required CUDA DLLs in `src/cuda/bin`. The runtime adds that directory to the DLL search path before importing `faster-whisper`.

If `faster-whisper` still reports a missing CUDA DLL, add the missing file to `src/cuda/bin`. Common CUDA 12 files include `cublas64_12.dll`, `cublasLt64_12.dll`, and `cudart64_12.dll`; some setups also need `cudnn64_9.dll`.

## Usage

```text
uv run att <input> [OPTIONS]
```

Examples:

```powershell
uv run att "https://youtu.be/dQw4w9WgXcQ"
uv run att "https://youtu.be/dQw4w9WgXcQ" --task translate --model large-v3
uv run att .\media\interview.mp4 --task transcribe --format json --timestamps
uv run att .\media\podcast.mp3 --task translate --language es --out-dir .\outputs
```

Key options:

- `--task transcribe|translate`
- `--format txt|srt|json`
- `--timestamps`
- `--language <code>`
- `--model tiny|base|small|medium|large-v3|large-v3-turbo|distil-large-v3`
- `--device auto|cpu|cuda`
- `--force-stt` for YouTube transcription only
- `--out-dir <path>`
- `--keep-temp`
- `--verbose`

Notes:

- `large-v3` is the default model because it supports both transcription and translation.
- `large-v3-turbo` is not accepted for translation mode.
- `txt` and `srt` stay text-focused; `json` includes input metadata, task, source, language, and segments.

## Output Contract

JSON output includes:

- `input_id`
- `input_type`
- `input_reference`
- `source`
- `task`
- `language`
- `source_language`
- `segments`

## Developer Notes

- CLI entrypoint: `att`
- Python package path: `audio_transcribe_translate`
- Module entrypoint: `python -m audio_transcribe_translate`

Run tests with:

```powershell
uv run --extra dev pytest
```
