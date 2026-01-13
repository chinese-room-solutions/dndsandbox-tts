"""Pydantic models for API request/response schemas."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class AudioFormat(str, Enum):
    """Supported audio output formats."""

    MP3 = "mp3"
    WAV = "wav"
    OPUS = "opus"
    FLAC = "flac"


class SpeechRequest(BaseModel):
    """Request schema for speech synthesis (OpenAI-compatible)."""

    model: str = Field(
        default="bark",
        description="TTS model to use",
        examples=["bark"],
    )
    input: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Text to synthesize into speech",
        examples=["Hello, adventurer! Welcome to the dungeon."],
    )
    voice: str = Field(
        default="v2/en_speaker_6",
        description="Bark voice preset to use",
        examples=["v2/en_speaker_6", "v2/en_speaker_0", "v2/de_speaker_0"],
    )
    response_format: AudioFormat = Field(
        default=AudioFormat.MP3,
        description="Audio output format",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Speech speed multiplier (0.25 to 4.0)",
    )


class ErrorDetail(BaseModel):
    """Error detail schema (OpenAI-compatible)."""

    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    """Error response schema (OpenAI-compatible)."""

    error: ErrorDetail


class ModelInfo(BaseModel):
    """Model information schema (OpenAI-compatible)."""

    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "local"


class ModelList(BaseModel):
    """List of models response schema (OpenAI-compatible)."""

    object: Literal["list"] = "list"
    data: list[ModelInfo]


class VoiceInfo(BaseModel):
    """Voice preset information."""

    id: str
    name: str
    language: str = "en"
    description: str | None = None


class VoiceList(BaseModel):
    """List of available voices."""

    voices: list[VoiceInfo]


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "unhealthy"]
    model_loaded: bool = False
    device: str | None = None


class ServiceInfo(BaseModel):
    """Service information response."""

    service: str = "dndsandbox-tts"
    version: str
    device: str
    models_available: list[str]
