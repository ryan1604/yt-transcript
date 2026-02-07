"""Core data models for transcript processing."""

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str


class TranscriptResult(BaseModel):
    video_id: str
    source: str
    language: str | None = None
    segments: list[TranscriptSegment]
