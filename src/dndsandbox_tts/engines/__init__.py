"""TTS engine implementations."""

from dndsandbox_tts.engines.base import (
    AudioResult,
    BaseTTSEngine,
    EngineInfo,
    VoicePreset,
)
from dndsandbox_tts.engines.manager import (
    ModelManager,
    ModelNotFoundError,
    get_model_manager,
    init_model_manager,
)
from dndsandbox_tts.engines.registry import (
    EngineRegistry,
    discover_engines,
    get_registry,
    register_engine,
)

__all__ = [
    # Base types
    "AudioResult",
    "BaseTTSEngine",
    "EngineInfo",
    "VoicePreset",
    # Manager
    "ModelManager",
    "ModelNotFoundError",
    "get_model_manager",
    "init_model_manager",
    # Registry
    "EngineRegistry",
    "discover_engines",
    "get_registry",
    "register_engine",
]
