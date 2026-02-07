"""CLI entrypoint for yt-transcript."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console

from yt_transcript import config
from yt_transcript.audio import download_audio, preprocess_audio_to_wav
from yt_transcript.captions import fetch_captions
from yt_transcript.errors import RetrievalError, YtTranscriptError
from yt_transcript.formatters import format_output
from yt_transcript.io import build_output_path, ensure_out_dir, write_text
from yt_transcript.models import TranscriptResult
from yt_transcript.transcribe import transcribe_audio
from yt_transcript.url_parser import extract_video_id, normalize_video_url

FormatOption = Literal["txt", "srt", "json"]
DeviceOption = Literal["auto", "cpu", "cuda"]
ModelOption = Literal["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"]

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Fetch transcript from a YouTube URL (captions first, local STT fallback).",
)
console = Console()


@app.command()
def main(
    youtube_url: str = typer.Argument(..., help="YouTube video URL"),
    out_dir: Path = typer.Option(
        Path(config.DEFAULT_OUT_DIR), "--out-dir", help="Directory to write transcript output."
    ),
    fmt: FormatOption = typer.Option(
        config.DEFAULT_FORMAT, "--format", help="Output format."
    ),
    timestamps: bool = typer.Option(
        False, "--timestamps", help="Include timestamps in txt/json outputs."
    ),
    language: str = typer.Option(
        config.DEFAULT_LANGUAGE, "--language", help="Preferred language code (e.g. en)."
    ),
    model: ModelOption = typer.Option(
        config.DEFAULT_MODEL, "--model", help="faster-whisper model for STT fallback."
    ),
    device: DeviceOption = typer.Option(
        config.DEFAULT_DEVICE, "--device", help="Inference device for faster-whisper."
    ),
    force_stt: bool = typer.Option(
        False, "--force-stt", help="Skip captions and always run STT fallback."
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Print extra progress logs."),
    keep_temp: bool = typer.Option(False, "--keep-temp", help="Keep temporary audio files."),
) -> None:
    try:
        video_id = extract_video_id(youtube_url)
        canonical_url = normalize_video_url(video_id)
        ensure_out_dir(out_dir)

        if not force_stt:
            try:
                if verbose:
                    console.print("[cyan]Trying captions...[/cyan]")
                segments = fetch_captions(video_id=video_id, language=language)
                result = TranscriptResult(
                    video_id=video_id,
                    source="captions",
                    language=language,
                    segments=segments,
                )
                _write_result(result, fmt, timestamps, out_dir)
                return
            except RetrievalError as exc:
                if verbose:
                    console.print(f"[yellow]Captions unavailable: {exc}. Falling back to STT.[/yellow]")

        temp_dir = Path(tempfile.mkdtemp(prefix=f"yt_transcript_{video_id}_"))
        try:
            if verbose:
                console.print("[cyan]Downloading audio...[/cyan]")
            downloaded = download_audio(canonical_url, video_id=video_id, temp_dir=temp_dir)
            if verbose:
                console.print("[cyan]Preprocessing audio...[/cyan]")
            wav_path = preprocess_audio_to_wav(downloaded, temp_dir=temp_dir, video_id=video_id)
            if verbose:
                console.print("[cyan]Transcribing audio...[/cyan]")
            segments, detected_language = transcribe_audio(
                wav_path=wav_path,
                model_name=model,
                device=device,
                language=language or None,
            )
            result = TranscriptResult(
                video_id=video_id,
                source="stt",
                language=detected_language or language,
                segments=segments,
            )
            _write_result(result, fmt, timestamps, out_dir)
        finally:
            if keep_temp or verbose:
                console.print(f"[dim]Temporary files kept at: {temp_dir}[/dim]")
            else:
                shutil.rmtree(temp_dir, ignore_errors=True)

    except YtTranscriptError as exc:
        console.print(f"[red]Error:[/red] {exc.message}")
        raise typer.Exit(code=exc.exit_code)
    except Exception as exc:
        console.print(f"[red]Unexpected error:[/red] {exc}")
        raise typer.Exit(code=1)


def _write_result(
    result: TranscriptResult,
    fmt: FormatOption,
    timestamps: bool,
    out_dir: Path,
) -> None:
    content = format_output(result, fmt=fmt, timestamps=timestamps)
    output_path = build_output_path(out_dir, result.video_id, fmt=fmt)
    write_text(output_path, content)
    console.print(
        f"[green]Transcript written:[/green] {output_path} "
        f"(source={result.source}, segments={len(result.segments)})"
    )


if __name__ == "__main__":
    app()
