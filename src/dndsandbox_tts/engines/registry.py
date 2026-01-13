"""Engine registry for discovering and managing TTS engines."""

import structlog
from typing import TYPE_CHECKING, Any

from dndsandbox_tts.engines.base import BaseTTSEngine, EngineInfo

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


class EngineRegistry:
    """Registry for TTS engines.

    Handles engine discovery, availability checking, and instantiation.
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._engine_classes: dict[str, type[BaseTTSEngine]] = {}

    def register(self, engine_class: type[BaseTTSEngine]) -> None:
        """Register an engine class.

        Args:
            engine_class: Engine class to register
        """
        name = engine_class.name
        if name in self._engine_classes:
            logger.warning("Engine already registered, overwriting", engine=name)
        self._engine_classes[name] = engine_class
        logger.debug("Registered engine", engine=name)

    def unregister(self, name: str) -> None:
        """Unregister an engine.

        Args:
            name: Name of engine to unregister
        """
        if name in self._engine_classes:
            del self._engine_classes[name]
            logger.debug("Unregistered engine", engine=name)

    def get_engine_class(self, name: str) -> type[BaseTTSEngine] | None:
        """Get an engine class by name.

        Args:
            name: Engine name

        Returns:
            Engine class or None if not found
        """
        return self._engine_classes.get(name)

    def list_registered(self) -> list[str]:
        """List all registered engine names.

        Returns:
            List of engine names
        """
        return list(self._engine_classes.keys())

    def list_available(self) -> list[str]:
        """List engines that are available (dependencies satisfied).

        Returns:
            List of available engine names
        """
        return [
            name
            for name, cls in self._engine_classes.items()
            if cls.is_available()
        ]

    def get_engine_info(self, name: str) -> EngineInfo | None:
        """Get information about an engine.

        Args:
            name: Engine name

        Returns:
            EngineInfo or None if not found
        """
        cls = self._engine_classes.get(name)
        if cls is None:
            return None
        return cls.get_engine_info()

    def get_all_engine_info(self) -> dict[str, EngineInfo]:
        """Get information about all registered engines.

        Returns:
            Dict mapping engine name to EngineInfo
        """
        return {
            name: cls.get_engine_info()
            for name, cls in self._engine_classes.items()
        }

    def is_available(self, name: str) -> bool:
        """Check if an engine is available.

        Args:
            name: Engine name

        Returns:
            True if engine exists and is available
        """
        cls = self._engine_classes.get(name)
        if cls is None:
            return False
        return cls.is_available()

    def get_unavailable_reason(self, name: str) -> str | None:
        """Get reason why an engine is unavailable.

        Args:
            name: Engine name

        Returns:
            Reason string or None if available
        """
        cls = self._engine_classes.get(name)
        if cls is None:
            return f"Engine '{name}' not registered"
        return cls.get_unavailable_reason()

    def create_engine(self, name: str, **kwargs: Any) -> BaseTTSEngine:
        """Create an engine instance.

        Args:
            name: Engine name
            **kwargs: Engine-specific configuration

        Returns:
            Engine instance

        Raises:
            ValueError: If engine not found or unavailable
        """
        cls = self._engine_classes.get(name)
        if cls is None:
            available = self.list_registered()
            raise ValueError(
                f"Engine '{name}' not found. Available: {available}"
            )

        if not cls.is_available():
            reason = cls.get_unavailable_reason()
            raise ValueError(
                f"Engine '{name}' is not available: {reason}"
            )

        logger.info("Creating engine instance", engine=name)
        return cls(**kwargs)


# Global registry instance
_registry = EngineRegistry()


def get_registry() -> EngineRegistry:
    """Get the global engine registry.

    Returns:
        Global EngineRegistry instance
    """
    return _registry


def register_engine(engine_class: type[BaseTTSEngine]) -> type[BaseTTSEngine]:
    """Decorator to register an engine class.

    Usage:
        @register_engine
        class MyEngine(BaseTTSEngine):
            name = "my-engine"
            ...

    Args:
        engine_class: Engine class to register

    Returns:
        The same engine class (for decorator chaining)
    """
    _registry.register(engine_class)
    return engine_class


def discover_engines() -> None:
    """Discover and register all available engines.

    This imports engine modules which triggers their registration
    via the @register_engine decorator.
    """
    logger.info("Discovering TTS engines")

    # Import built-in engines
    # Each module should use @register_engine decorator
    try:
        from dndsandbox_tts.engines import bark  # noqa: F401

        logger.debug("Bark engine module loaded")
    except ImportError as e:
        logger.warning("Could not load Bark engine", error=str(e))

    # Future: Add more engine imports here
    # try:
    #     from dndsandbox_tts.engines import piper  # noqa: F401
    # except ImportError as e:
    #     logger.warning("Could not load Piper engine", error=str(e))

    available = _registry.list_available()
    logger.info("Engine discovery complete", count=len(available), engines=available)
