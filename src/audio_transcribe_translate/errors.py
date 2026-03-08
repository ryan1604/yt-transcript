"""Typed errors with deterministic CLI exit codes."""


class YtTranscriptError(Exception):
    """Base error with an exit code for CLI handling."""

    exit_code = 1

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidInputError(YtTranscriptError):
    exit_code = 1


class RetrievalError(YtTranscriptError):
    exit_code = 2


class TranscriptionError(YtTranscriptError):
    exit_code = 3


class OutputWriteError(YtTranscriptError):
    exit_code = 4
