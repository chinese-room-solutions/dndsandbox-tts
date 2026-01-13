"""Model manager for TTS engines."""

import structlog
from typing import TYPE_CHECKING, Any

from dndsandbox_tts.engines.base import (
    AudioResult,
    BaseTTSEngine,
    EngineInfo,
    VoicePreset,
)
from dndsandbox_tts.engines.registry import discover_engines, get_registry

if TYPE_CHECKING:
    from dndsandbox_tts.config import Settings

logger = structlog.get_logger(__name__)


class ModelNotFoundError(Exception):
    """Raised when requested model is not found."""

    def __init__(self, model: str):
        self.model = model
        super().__init__(f"Model '{model}' not found")


class ModelManager:
    """Manages TTS engines and model loading.

    The ModelManager uses the engine registry to discover available engines
    and create instances on demand. It handles:
    - Engine discovery and availability checking
    - Engine instance lifecycle (creation, loading, unloading)
    - Synthesis requests routing to appropriate engines
    """

    def __init__(self, settings: "Settings"):
        """Initialize model manager.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._engines: dict[str, BaseTTSEngine] = {}
        self._registry = get_registry()

    def register_engine(self, engine: BaseTTSEngine) -> None:
        """Register a TTS engine instance.

        Args:
            engine: Engine instance to register
        """
        logger.info("Registering engine instance", engine=engine.name)
        self._engines[engine.name] = engine

    def _create_engine(self, name: str, **kwargs: Any) -> BaseTTSEngine:
        """Create an engine instance from the registry.

        Args:
            name: Engine name
            **kwargs: Engine-specific configuration

        Returns:
            Created engine instance

        Raises:
            ModelNotFoundError: If engine not found or unavailable
        """
        try:
            return self._registry.create_engine(name, **kwargs)
        except ValueError as e:
            raise ModelNotFoundError(str(e)) from e

    def get_engine(self, model: str) -> BaseTTSEngine:
        """Get engine by model name, creating it if needed.

        If the engine is not yet instantiated but is available in the registry,
        it will be created with default settings.

        Args:
            model: Model name (engine name)

        Returns:
            TTS engine instance

        Raises:
            ModelNotFoundError: If model is not registered or unavailable
        """
        # Return existing instance if available
        if model in self._engines:
            return self._engines[model]

        # Try to create from registry if available
        if self._registry.is_available(model):
            # Get engine-specific settings from config
            engine_settings = self.settings.get_engine_settings(model)
            engine = self._create_engine(
                model,
                device=self.settings.device,
                use_fp16=self.settings.use_fp16,
                **engine_settings,
            )
            self._engines[model] = engine
            return engine

        # Check if registered but unavailable
        reason = self._registry.get_unavailable_reason(model)
        if reason:
            raise ModelNotFoundError(f"Model '{model}' is unavailable: {reason}")

        # Not found at all
        available = self._registry.list_available()
        raise ModelNotFoundError(
            f"Model '{model}' not found. Available models: {available}"
        )

    def load_model(self, model: str) -> None:
        """Load a model into memory.

        Args:
            model: Model name to load
        """
        engine = self.get_engine(model)
        if not engine.is_loaded():
            logger.info("Loading model", model=model)
            engine.load()
        else:
            logger.debug("Model already loaded", model=model)

    def unload_model(self, model: str) -> None:
        """Unload a model from memory.

        Args:
            model: Model name to unload
        """
        engine = self.get_engine(model)
        if engine.is_loaded():
            logger.info("Unloading model", model=model)
            engine.unload()

    def synthesize(
        self,
        model: str,
        text: str,
        voice: str,
        speed: float = 1.0,
    ) -> AudioResult:
        """Synthesize speech using specified model.

        Args:
            model: Model name to use
            text: Text to synthesize
            voice: Voice preset
            speed: Speech speed multiplier

        Returns:
            AudioResult with generated audio
        """
        engine = self.get_engine(model)

        # Auto-load if not loaded
        if not engine.is_loaded():
            logger.info("Auto-loading model", model=model)
            engine.load()

        return engine.synthesize(text, voice, speed)

    def get_voices(self, model: str) -> list[VoicePreset]:
        """Get available voices for a model.

        Args:
            model: Model name

        Returns:
            List of voice presets
        """
        engine = self.get_engine(model)
        return engine.get_voices()

    def list_models(self) -> list[dict]:
        """List all available models from the registry.

        Returns:
            List of model info dicts with availability and load status
        """
        models = []
        for name in self._registry.list_registered():
            info = self._registry.get_engine_info(name)
            is_available = self._registry.is_available(name)
            is_loaded = name in self._engines and self._engines[name].is_loaded()

            model_info = {
                "id": name,
                "loaded": is_loaded,
                "available": is_available,
            }

            if info:
                model_info.update({
                    "display_name": info.display_name,
                    "description": info.description,
                    "requires_gpu": info.requires_gpu,
                })

            if not is_available:
                model_info["unavailable_reason"] = self._registry.get_unavailable_reason(name)

            models.append(model_info)

        return models

    def get_engine_info(self, model: str) -> EngineInfo | None:
        """Get detailed information about an engine.

        Args:
            model: Model name

        Returns:
            EngineInfo or None if not found
        """
        return self._registry.get_engine_info(model)

    def list_available_engines(self) -> list[str]:
        """List engines that are available for use.

        Returns:
            List of available engine names
        """
        return self._registry.list_available()

    def preload_all(self) -> None:
        """Preload all available models.

        Creates and loads all engines that are available in the registry.
        """
        for name in self._registry.list_available():
            try:
                engine = self.get_engine(name)
                if not engine.is_loaded():
                    logger.info("Preloading model", model=name)
                    engine.load()
            except ModelNotFoundError as e:
                logger.warning("Could not preload model", model=name, error=str(e))

    def unload_all(self) -> None:
        """Unload all models."""
        for name, engine in self._engines.items():
            if engine.is_loaded():
                logger.info("Unloading model", model=name)
                engine.unload()


# Global model manager instance
_model_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance.

    Returns:
        ModelManager instance

    Raises:
        RuntimeError: If manager not initialized
    """
    if _model_manager is None:
        raise RuntimeError("Model manager not initialized. Call init_model_manager() first.")
    return _model_manager


def init_model_manager(settings: "Settings") -> ModelManager:
    """Initialize the global model manager.

    This discovers all available engines via the registry and creates
    the model manager. Engines are instantiated on-demand when first
    requested.

    Args:
        settings: Application settings

    Returns:
        Initialized ModelManager instance
    """
    global _model_manager

    logger.info("Initializing model manager")

    # Discover all available engines
    discover_engines()

    _model_manager = ModelManager(settings)

    # Log discovered engines
    registry = get_registry()
    available = registry.list_available()
    logger.info("Available engines discovered", engines=available)

    return _model_manager
