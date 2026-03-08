"""Core data models for transcript processing."""

from typing import Literal

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str


class TranscriptResult(BaseModel):
    input_id: str
    input_type: Literal["youtube", "local"]
    input_reference: str
    source: str
    task: Literal["transcribe", "translate"] = "transcribe"
    language: str | None = None
    source_language: str | None = None
    segments: list[TranscriptSegment]
