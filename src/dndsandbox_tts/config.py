"""Configuration management using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BarkEngineSettings(BaseSettings):
    """Bark engine-specific settings."""

    model_config = SettingsConfigDict(
        env_prefix="BARK_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    use_small_models: bool = Field(
        default=False,
        description="Use smaller Bark models for faster inference (lower quality)",
    )
    chunk_size: int = Field(
        default=250,
        ge=100,
        le=500,
        description="Max characters per text chunk (larger = faster but may reduce quality)",
    )
    semantic_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Temperature for semantic token generation",
    )
    coarse_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Temperature for coarse acoustic token generation",
    )
    fine_temperature: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Temperature for fine acoustic token generation",
    )


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server bind address")
    port: int = Field(default=8100, description="Server port")

    # Authentication (optional)
    api_key: str | None = Field(default=None, description="Optional API key for authentication")
    require_auth: bool = Field(default=False, description="Require API key for requests")

    # Model settings
    default_model: str = Field(default="bark", description="Default TTS model to use")
    models_cache_dir: Path = Field(
        default=Path.home() / ".cache" / "dndsandbox-tts",
        description="Directory for caching model weights",
    )

    # Audio settings
    default_sample_rate: int = Field(default=24000, description="Default audio sample rate")
    output_format: str = Field(default="mp3", description="Default output audio format")

    # Performance settings
    device: str = Field(
        default="auto",
        description="Compute device: 'auto', 'cuda', 'cuda:0', 'cpu'",
    )
    preload_models: bool = Field(
        default=True,
        description="Preload models on startup for faster first request",
    )
    use_fp16: bool = Field(
        default=True,
        description="Use half-precision (FP16) for faster inference on GPU",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    json_logs: bool = Field(
        default=False,
        description="Output logs in JSON format (for production)",
    )

    # Caching settings
    cache_enabled: bool = Field(
        default=True,
        description="Enable audio caching for repeated requests",
    )
    cache_max_entries: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum number of cached audio entries",
    )
    cache_max_memory_mb: float = Field(
        default=500.0,
        ge=10.0,
        le=10000.0,
        description="Maximum cache memory in MB",
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        description="Cache entry time-to-live in seconds",
    )

    # Engine-specific settings
    bark: BarkEngineSettings = Field(default_factory=BarkEngineSettings)

    def get_device(self) -> str:
        """Resolve 'auto' device to actual device based on CUDA availability."""
        if self.device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device

    def get_engine_settings(self, engine_name: str) -> dict[str, Any]:
        """Get settings for a specific engine.

        Args:
            engine_name: Name of the engine

        Returns:
            Dict of engine-specific settings
        """
        engine_settings: dict[str, BaseSettings | None] = {
            "bark": self.bark,
        }
        settings = engine_settings.get(engine_name)
        if settings:
            return settings.model_dump()
        return {}


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


@lru_cache
def get_bark_settings() -> BarkEngineSettings:
    """Get cached Bark engine settings."""
    return BarkEngineSettings()
