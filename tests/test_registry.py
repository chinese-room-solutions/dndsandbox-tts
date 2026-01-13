"""Tests for engine registry."""

import pytest

from dndsandbox_tts.engines.base import (
    AudioResult,
    BaseTTSEngine,
    EngineInfo,
    VoicePreset,
)
from dndsandbox_tts.engines.registry import EngineRegistry, register_engine


class TestEngineRegistry:
    """Tests for EngineRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for testing."""
        return EngineRegistry()

    @pytest.fixture
    def mock_engine_class(self):
        """Create a mock engine class for testing."""

        class MockEngine(BaseTTSEngine):
            name = "mock-engine"
            _available = True

            @classmethod
            def get_engine_info(cls) -> EngineInfo:
                return EngineInfo(
                    name="mock-engine",
                    display_name="Mock Engine",
                    description="A mock TTS engine for testing",
                    version="1.0.0",
                )

            @classmethod
            def is_available(cls) -> bool:
                return cls._available

            @classmethod
            def get_unavailable_reason(cls) -> str | None:
                if cls.is_available():
                    return None
                return "Mock unavailable reason"

            def __init__(self, **kwargs):
                self._loaded = False

            def load(self) -> None:
                self._loaded = True

            def unload(self) -> None:
                self._loaded = False

            def synthesize(self, text: str, voice: str, speed: float = 1.0) -> AudioResult:
                import numpy as np
                return AudioResult(
                    audio=np.zeros(1000, dtype=np.float32),
                    sample_rate=24000,
                )

            def get_voices(self) -> list[VoicePreset]:
                return [VoicePreset(id="mock-voice", name="Mock Voice", language="en")]

            def is_loaded(self) -> bool:
                return self._loaded

            def get_default_voice(self) -> str:
                return "mock-voice"

        return MockEngine

    def test_register_engine(self, registry, mock_engine_class):
        """Test registering an engine class."""
        registry.register(mock_engine_class)

        assert "mock-engine" in registry.list_registered()

    def test_register_duplicate_overwrites(self, registry, mock_engine_class):
        """Test that registering duplicate engine overwrites."""
        registry.register(mock_engine_class)
        registry.register(mock_engine_class)

        assert registry.list_registered().count("mock-engine") == 1

    def test_unregister_engine(self, registry, mock_engine_class):
        """Test unregistering an engine."""
        registry.register(mock_engine_class)
        registry.unregister("mock-engine")

        assert "mock-engine" not in registry.list_registered()

    def test_unregister_nonexistent(self, registry):
        """Test unregistering non-existent engine does nothing."""
        registry.unregister("nonexistent")
        # Should not raise

    def test_get_engine_class(self, registry, mock_engine_class):
        """Test getting engine class by name."""
        registry.register(mock_engine_class)

        retrieved = registry.get_engine_class("mock-engine")
        assert retrieved is mock_engine_class

    def test_get_engine_class_not_found(self, registry):
        """Test getting non-existent engine class returns None."""
        result = registry.get_engine_class("nonexistent")
        assert result is None

    def test_list_registered(self, registry, mock_engine_class):
        """Test listing registered engines."""
        registry.register(mock_engine_class)

        registered = registry.list_registered()
        assert "mock-engine" in registered

    def test_list_available(self, registry, mock_engine_class):
        """Test listing available engines."""
        registry.register(mock_engine_class)

        available = registry.list_available()
        assert "mock-engine" in available

    def test_list_available_excludes_unavailable(self, registry, mock_engine_class):
        """Test that unavailable engines are excluded from available list."""
        mock_engine_class._available = False
        registry.register(mock_engine_class)

        available = registry.list_available()
        assert "mock-engine" not in available

        # Reset for other tests
        mock_engine_class._available = True

    def test_get_engine_info(self, registry, mock_engine_class):
        """Test getting engine info."""
        registry.register(mock_engine_class)

        info = registry.get_engine_info("mock-engine")
        assert info is not None
        assert info.name == "mock-engine"
        assert info.display_name == "Mock Engine"

    def test_get_engine_info_not_found(self, registry):
        """Test getting info for non-existent engine returns None."""
        info = registry.get_engine_info("nonexistent")
        assert info is None

    def test_get_all_engine_info(self, registry, mock_engine_class):
        """Test getting info for all engines."""
        registry.register(mock_engine_class)

        all_info = registry.get_all_engine_info()
        assert "mock-engine" in all_info
        assert all_info["mock-engine"].name == "mock-engine"

    def test_is_available(self, registry, mock_engine_class):
        """Test checking engine availability."""
        registry.register(mock_engine_class)

        assert registry.is_available("mock-engine") is True

    def test_is_available_not_registered(self, registry):
        """Test availability check for unregistered engine."""
        assert registry.is_available("nonexistent") is False

    def test_get_unavailable_reason(self, registry, mock_engine_class):
        """Test getting unavailability reason."""
        mock_engine_class._available = False
        registry.register(mock_engine_class)

        reason = registry.get_unavailable_reason("mock-engine")
        assert reason == "Mock unavailable reason"

        mock_engine_class._available = True

    def test_get_unavailable_reason_when_available(self, registry, mock_engine_class):
        """Test unavailability reason for available engine is None."""
        registry.register(mock_engine_class)

        reason = registry.get_unavailable_reason("mock-engine")
        assert reason is None

    def test_get_unavailable_reason_not_registered(self, registry):
        """Test unavailability reason for unregistered engine."""
        reason = registry.get_unavailable_reason("nonexistent")
        assert "not registered" in reason.lower()

    def test_create_engine(self, registry, mock_engine_class):
        """Test creating engine instance."""
        registry.register(mock_engine_class)

        engine = registry.create_engine("mock-engine")
        assert isinstance(engine, mock_engine_class)
        assert engine.name == "mock-engine"

    def test_create_engine_with_kwargs(self, registry, mock_engine_class):
        """Test creating engine with configuration kwargs."""
        registry.register(mock_engine_class)

        engine = registry.create_engine("mock-engine", device="cpu")
        assert isinstance(engine, mock_engine_class)

    def test_create_engine_not_found(self, registry):
        """Test creating non-existent engine raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            registry.create_engine("nonexistent")

    def test_create_engine_unavailable(self, registry, mock_engine_class):
        """Test creating unavailable engine raises ValueError."""
        mock_engine_class._available = False
        registry.register(mock_engine_class)

        with pytest.raises(ValueError, match="not available"):
            registry.create_engine("mock-engine")

        mock_engine_class._available = True


class TestRegisterEngineDecorator:
    """Tests for @register_engine decorator."""

    def test_decorator_returns_class(self):
        """Test that decorator returns the same class."""
        from dndsandbox_tts.engines.registry import _registry

        @register_engine
        class TestEngine(BaseTTSEngine):
            name = "test-decorator-engine"

            @classmethod
            def get_engine_info(cls) -> EngineInfo:
                return EngineInfo(
                    name="test-decorator-engine",
                    display_name="Test",
                    description="Test",
                )

            def __init__(self, **kwargs):
                pass

            def load(self) -> None:
                pass

            def unload(self) -> None:
                pass

            def synthesize(self, text: str, voice: str, speed: float = 1.0) -> AudioResult:
                import numpy as np
                return AudioResult(audio=np.zeros(100), sample_rate=24000)

            def get_voices(self) -> list[VoicePreset]:
                return []

            def is_loaded(self) -> bool:
                return False

            def get_default_voice(self) -> str:
                return "default"

        assert TestEngine.name == "test-decorator-engine"
        assert "test-decorator-engine" in _registry.list_registered()

        # Cleanup
        _registry.unregister("test-decorator-engine")


class TestDiscoverEngines:
    """Tests for engine discovery."""

    def test_discover_engines_finds_bark(self):
        """Test that discover_engines finds Bark engine."""
        from dndsandbox_tts.engines.registry import discover_engines, get_registry

        discover_engines()
        registry = get_registry()

        assert "bark" in registry.list_registered()

    def test_discover_engines_idempotent(self):
        """Test that multiple discover calls are safe."""
        from dndsandbox_tts.engines.registry import discover_engines, get_registry

        discover_engines()
        discover_engines()

        registry = get_registry()
        # Should still have bark registered once
        assert registry.list_registered().count("bark") == 1
