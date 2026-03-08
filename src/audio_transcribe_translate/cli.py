"""CLI entrypoint for audio-transcribe-translate."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console

from audio_transcribe_translate import config
from audio_transcribe_translate.audio import download_audio, preprocess_audio_to_wav
from audio_transcribe_translate.captions import fetch_captions
from audio_transcribe_translate.errors import InvalidInputError, RetrievalError, YtTranscriptError
from audio_transcribe_translate.formatters import format_output
from audio_transcribe_translate.input_resolver import ResolvedInput, resolve_input
from audio_transcribe_translate.io import build_output_path, ensure_out_dir, write_text
from audio_transcribe_translate.models import TranscriptResult
from audio_transcribe_translate.transcribe import transcribe_audio

FormatOption = Literal["txt", "srt", "json"]
DeviceOption = Literal["auto", "cpu", "cuda"]
ModelOption = Literal["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo", "distil-large-v3"]
TaskOption = Literal["transcribe", "translate"]

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Fetch a transcript or English translation from a YouTube URL or local media file.",
)
console = Console()


@dataclass(frozen=True, slots=True)
class CliOptions:
    out_dir: Path
    fmt: FormatOption
    timestamps: bool
    language: str | None
    task: TaskOption
    model: ModelOption
    device: DeviceOption
    force_stt: bool
    verbose: bool
    keep_temp: bool


@app.command()
def main(
    input_value: str = typer.Argument(..., metavar="INPUT", help="YouTube URL or local media file path."),
    out_dir: Path = typer.Option(
        Path(config.DEFAULT_OUT_DIR), "--out-dir", help="Directory to write transcript output."
    ),
    fmt: FormatOption = typer.Option(
        config.DEFAULT_FORMAT, "--format", help="Output format."
    ),
    timestamps: bool = typer.Option(
        False, "--timestamps", help="Include timestamps in txt/json outputs."
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        help="Source language hint for STT, or captions language for transcription mode.",
    ),
    task: TaskOption = typer.Option(
        config.DEFAULT_TASK,
        "--task",
        help="Run one task: transcribe in the source language or translate speech to English.",
    ),
    model: ModelOption = typer.Option(
        config.DEFAULT_MODEL, "--model", help="faster-whisper model for STT transcription or translation."
    ),
    device: DeviceOption = typer.Option(
        config.DEFAULT_DEVICE, "--device", help="Inference device for faster-whisper."
    ),
    force_stt: bool = typer.Option(
        False, "--force-stt", help="Skip captions and always run STT fallback for YouTube transcription."
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Print extra progress logs."),
    keep_temp: bool = typer.Option(False, "--keep-temp", help="Keep temporary audio files."),
) -> None:
    options = CliOptions(
        out_dir=out_dir,
        fmt=fmt,
        timestamps=timestamps,
        language=language,
        task=task,
        model=model,
        device=device,
        force_stt=force_stt,
        verbose=verbose,
        keep_temp=keep_temp,
    )
    try:
        _run_command(input_value, options)
    except YtTranscriptError as exc:
        console.print(f"[red]Error:[/red] {exc.message}")
        raise typer.Exit(code=exc.exit_code)
    except Exception as exc:
        console.print(f"[red]Unexpected audio-transcribe-translate error:[/red] {exc}")
        raise typer.Exit(code=1)


def _run_command(input_value: str, options: CliOptions) -> None:
    resolved_input = _resolve_input(input_value)
    _validate_runtime_options(
        resolved_input=resolved_input,
        task=options.task,
        force_stt=options.force_stt,
        model=options.model,
    )
    ensure_out_dir(options.out_dir)

    result = _maybe_fetch_youtube_captions(
        resolved_input=resolved_input,
        language=_resolve_caption_language(options.language),
        task=options.task,
        force_stt=options.force_stt,
        fmt=options.fmt,
        timestamps=options.timestamps,
        out_dir=options.out_dir,
        verbose=options.verbose,
    )
    if result is not None:
        return

    _run_stt_pipeline(resolved_input=resolved_input, options=options)


def _resolve_input(input_value: str) -> ResolvedInput:
    return resolve_input(input_value)


def _run_stt_pipeline(resolved_input: ResolvedInput, options: CliOptions) -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix=f"audio_transcribe_translate_{resolved_input.input_id}_"))
    try:
        media_path = _acquire_media(
            resolved_input=resolved_input,
            temp_dir=temp_dir,
            verbose=options.verbose,
        )
        wav_path = _preprocess_media(
            media_path=media_path,
            resolved_input=resolved_input,
            temp_dir=temp_dir,
            verbose=options.verbose,
        )
        result = _transcribe_media(
            wav_path=wav_path,
            resolved_input=resolved_input,
            options=options,
        )
        _write_result(result, options.fmt, options.timestamps, options.out_dir)
    finally:
        _cleanup_temp_dir(temp_dir=temp_dir, keep_temp=options.keep_temp)


def _write_result(
    result: TranscriptResult,
    fmt: FormatOption,
    timestamps: bool,
    out_dir: Path,
) -> None:
    content = format_output(result, fmt=fmt, timestamps=timestamps)
    output_path = build_output_path(out_dir, result.input_id, fmt=fmt)
    write_text(output_path, content)
    language_summary = result.language or "unknown"
    source_language_summary = result.source_language or "unknown"
    console.print(
        f"[green]Result written:[/green] {output_path} "
        f"(task={result.task}, source={result.source}, language={language_summary}, "
        f"source_language={source_language_summary}, segments={len(result.segments)})"
    )


def _maybe_fetch_youtube_captions(
    resolved_input: ResolvedInput,
    language: str,
    task: TaskOption,
    force_stt: bool,
    fmt: FormatOption,
    timestamps: bool,
    out_dir: Path,
    verbose: bool,
) -> TranscriptResult | None:
    if resolved_input.input_type != "youtube" or force_stt or task != "transcribe":
        return None

    try:
        if verbose:
            console.print("[cyan]Trying captions...[/cyan]")
        assert resolved_input.video_id is not None
        segments = fetch_captions(video_id=resolved_input.video_id, language=language)
        result = TranscriptResult(
            input_id=resolved_input.input_id,
            input_type=resolved_input.input_type,
            input_reference=resolved_input.input_reference,
            source="captions",
            task=task,
            language=language,
            source_language=language,
            segments=segments,
        )
        _write_result(result, fmt, timestamps, out_dir)
        return result
    except RetrievalError as exc:
        if verbose:
            console.print(f"[yellow]Captions unavailable: {exc}. Falling back to STT.[/yellow]")
        return None


def _acquire_media(
    resolved_input: ResolvedInput,
    temp_dir: Path,
    verbose: bool,
) -> Path:
    if resolved_input.input_type == "local":
        assert resolved_input.local_path is not None
        return resolved_input.local_path

    if verbose:
        console.print("[cyan]Downloading audio...[/cyan]")
    assert resolved_input.canonical_url is not None
    return download_audio(resolved_input.canonical_url, input_id=resolved_input.input_id, temp_dir=temp_dir)


def _preprocess_media(
    media_path: Path,
    resolved_input: ResolvedInput,
    temp_dir: Path,
    verbose: bool,
) -> Path:
    if verbose:
        console.print("[cyan]Preprocessing audio...[/cyan]")
    return preprocess_audio_to_wav(media_path, temp_dir=temp_dir, input_id=resolved_input.input_id)


def _transcribe_media(
    wav_path: Path,
    resolved_input: ResolvedInput,
    options: CliOptions,
) -> TranscriptResult:
    if options.verbose:
        console.print(f"[cyan]Running {options.task} task...[/cyan]")
    stt_language = _resolve_stt_language(task=options.task, language=options.language)
    segments, detected_language = transcribe_audio(
        wav_path=wav_path,
        model_name=options.model,
        device=options.device,
        language=stt_language,
        task=options.task,
    )
    return TranscriptResult(
        input_id=resolved_input.input_id,
        input_type=resolved_input.input_type,
        input_reference=resolved_input.input_reference,
        source="stt",
        task=options.task,
        language=_resolve_output_language(
            task=options.task,
            detected_language=detected_language,
            language=stt_language,
        ),
        source_language=detected_language or stt_language,
        segments=segments,
    )


def _cleanup_temp_dir(temp_dir: Path, keep_temp: bool) -> None:
    if keep_temp:
        console.print(f"[dim]Temporary files kept at: {temp_dir}[/dim]")
        return
    shutil.rmtree(temp_dir, ignore_errors=True)


def _validate_runtime_options(
    resolved_input: ResolvedInput,
    task: TaskOption,
    force_stt: bool,
    model: ModelOption,
) -> None:
    if force_stt and (resolved_input.input_type != "youtube" or task != "transcribe"):
        raise InvalidInputError("--force-stt is only supported for YouTube transcription runs")
    # if task == "translate" and model == "large-v3-turbo":
    #     raise InvalidInputError(
    #         "Model 'large-v3-turbo' is not supported for translation. Use 'large-v3' or 'distil-large-v3'."
    #     )


def _resolve_caption_language(language: str | None) -> str:
    return language or config.DEFAULT_LANGUAGE


def _resolve_stt_language(task: TaskOption, language: str | None) -> str | None:
    if task == "translate":
        return language or None
    return language or config.DEFAULT_LANGUAGE


def _resolve_output_language(
    task: TaskOption,
    detected_language: str | None,
    language: str | None,
) -> str | None:
    if task == "translate":
        return "en"
    return detected_language or language


if __name__ == "__main__":
    app()
