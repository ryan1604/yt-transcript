# YT Transcript

`YT Transcript` is a Python CLI program that takes a YouTube video URL and outputs the transcript to a file.

Behavior:
- It tries manually created YouTube captions first.
- If manual captions are unavailable, it falls back to local speech-to-text using `faster-whisper`.

## Features

- Accepts a YouTube video URL as input.
- Validates and normalizes the URL before processing.
- Prioritizes manually created captions.
- Falls back to local transcription with `faster-whisper` when manual captions are missing.
- Supports transcript output formats: `txt`, `srt`, and `json`.
- Supports optional timestamp output.
- Provides CLI options for language, model size, and device (`cpu`/`cuda`).

## Workflow

1. Input: user runs `uv run yt-transcript <youtube_url> [OPTIONS]`.
2. URL parsing: the program validates the URL and extracts the YouTube video ID.
3. Captions attempt: it tries to fetch manually created captions for the requested language.
4. Fallback path: if manual captions are unavailable, it downloads audio and runs local STT with `faster-whisper`.
5. Formatting: transcript segments are converted into the selected output format (`txt`, `srt`, or `json`).
6. Output: transcript is written to the output directory as a file named by video ID.

## Quick start

Download `ffmpeg` and put it in the root directory.

```powershell
git clone <repo_url>
cd yt_transcript
uv sync
uv run yt-transcript <youtube_url> [OPTIONS]
```

## Command

```text
uv run yt-transcript <youtube_url> [OPTIONS]
```

Run `uv run yt-transcript --help` for all options.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
