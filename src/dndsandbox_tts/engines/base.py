"""Base TTS engine interface."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AudioResult:
    """Result of audio synthesis."""

    audio: np.ndarray
    sample_rate: int


@dataclass
class VoicePreset:
    """Voice preset information."""

    id: str
    name: str
    language: str = "en"
    description: str | None = None


@dataclass
class EngineInfo:
    """Information about a TTS engine."""

    name: str
    display_name: str
    description: str
    version: str = "1.0.0"
    supported_formats: list[str] = field(default_factory=lambda: ["mp3", "wav"])
    supports_streaming: bool = False
    requires_gpu: bool = False
    min_vram_gb: float = 0.0
    dependencies: list[str] = field(default_factory=list)


class BaseTTSEngine(ABC):
    """Abstract base class for TTS engines.

    To create a new TTS engine:
    1. Subclass BaseTTSEngine
    2. Set the `name` class attribute
    3. Implement all abstract methods
    4. Register in the engine registry
    """

    name: str

    @classmethod
    @abstractmethod
    def get_engine_info(cls) -> EngineInfo:
        """Return information about this engine.

        This is called before instantiation to check compatibility.
        """
        pass

    @classmethod
    def is_available(cls) -> bool:
        """Check if the engine's dependencies are available.

        Override this to check for required packages/hardware.
        """
        return True

    @classmethod
    def get_unavailable_reason(cls) -> str | None:
        """Return reason why engine is unavailable, or None if available."""
        if cls.is_available():
            return None
        return "Required dependencies not installed"

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the engine with configuration.

        Args:
            **kwargs: Engine-specific configuration options
        """
        pass

    @abstractmethod
    def load(self) -> None:
        """Load model into memory."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model from memory."""
        pass

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
    ) -> AudioResult:
        """Generate speech from text.

        Args:
            text: Text to synthesize
            voice: Voice preset ID
            speed: Speech speed multiplier

        Returns:
            AudioResult containing audio data and sample rate
        """
        pass

    @abstractmethod
    def get_voices(self) -> list[VoicePreset]:
        """Return available voice presets for this engine."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        pass

    def validate_voice(self, voice: str) -> str:
        """Validate voice preset and return valid voice ID.

        Args:
            voice: Voice preset to validate

        Returns:
            Valid voice preset ID (fallback to default if invalid)
        """
        valid_voices = {v.id for v in self.get_voices()}
        if voice in valid_voices:
            return voice
        # Return default voice if invalid
        return self.get_default_voice()

    @abstractmethod
    def get_default_voice(self) -> str:
        """Return the default voice preset ID."""
        pass

    def get_config_schema(self) -> dict[str, Any]:
        """Return JSON schema for engine-specific configuration.

        Override to provide configuration options for the engine.
        """
        return {}

    def synthesize_stream(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
    ) -> Iterator[AudioResult]:
        """Generate speech from text as a stream of audio chunks.

        Default implementation falls back to non-streaming synthesize.
        Override to provide true streaming support.

        Args:
            text: Text to synthesize
            voice: Voice preset ID
            speed: Speech speed multiplier

        Yields:
            AudioResult for each chunk of audio
        """
        # Default: just yield the full result
        yield self.synthesize(text, voice, speed)
