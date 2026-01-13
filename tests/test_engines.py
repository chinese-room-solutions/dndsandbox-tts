"""Tests for TTS engines."""

import pytest

from dndsandbox_tts.engines.bark import (
    BARK_VOICE_PRESETS,
    DEFAULT_VOICE,
    MAX_CHUNK_LENGTH,
    split_text_into_chunks,
)
from dndsandbox_tts.engines.base import AudioResult, VoicePreset


class TestSplitTextIntoChunks:
    """Tests for text chunking function."""

    def test_short_text_no_split(self):
        """Test that short text is not split."""
        text = "Hello, world!"
        chunks = split_text_into_chunks(text)
        assert chunks == [text]

    def test_text_at_max_length(self):
        """Test text exactly at max length."""
        text = "x" * MAX_CHUNK_LENGTH
        chunks = split_text_into_chunks(text)
        assert chunks == [text]

    def test_split_on_sentence_boundary(self):
        """Test splitting on sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence."
        chunks = split_text_into_chunks(text, max_length=30)
        assert len(chunks) >= 2
        # Each chunk should be valid
        for chunk in chunks:
            assert len(chunk) <= 30 or "." not in chunk  # May exceed if single sentence is long

    def test_split_long_sentence_on_comma(self):
        """Test splitting long sentences on commas."""
        text = "This is a very long sentence, with multiple clauses, separated by commas, that exceeds the limit"
        chunks = split_text_into_chunks(text, max_length=50)
        assert len(chunks) >= 2

    def test_split_long_word_sequence(self):
        """Test splitting when no punctuation available."""
        text = "word " * 100  # Long text with only spaces
        chunks = split_text_into_chunks(text.strip(), max_length=50)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 50

    def test_preserves_all_content(self):
        """Test that all content is preserved after splitting."""
        text = "The quick brown fox jumps over the lazy dog. " * 10
        chunks = split_text_into_chunks(text.strip(), max_length=100)
        # Rejoin and compare (allowing for whitespace differences)
        rejoined = " ".join(chunks)
        # All words should be present
        original_words = set(text.split())
        rejoined_words = set(rejoined.split())
        assert original_words == rejoined_words

    def test_empty_string(self):
        """Test handling of empty string."""
        chunks = split_text_into_chunks("")
        assert chunks == [""]

    def test_single_long_word(self):
        """Test handling of single very long word."""
        text = "supercalifragilisticexpialidocious"
        chunks = split_text_into_chunks(text, max_length=10)
        # Should still return the word, even if it exceeds max_length
        assert len(chunks) >= 1


class TestBarkVoicePresets:
    """Tests for Bark voice presets."""

    def test_default_voice_exists(self):
        """Test that default voice is in presets."""
        voice_ids = {v.id for v in BARK_VOICE_PRESETS}
        assert DEFAULT_VOICE in voice_ids

    def test_english_voices_present(self):
        """Test that English voice presets are present."""
        voice_ids = {v.id for v in BARK_VOICE_PRESETS}
        for i in range(10):
            assert f"v2/en_speaker_{i}" in voice_ids

    def test_multilingual_voices_present(self):
        """Test that voices for multiple languages are present."""
        languages = {v.language for v in BARK_VOICE_PRESETS}
        assert "en" in languages
        assert "de" in languages
        assert "es" in languages
        assert "fr" in languages

    def test_voice_preset_structure(self):
        """Test that voice presets have correct structure."""
        for voice in BARK_VOICE_PRESETS:
            assert isinstance(voice, VoicePreset)
            assert voice.id
            assert voice.name
            assert voice.language


class TestAudioResult:
    """Tests for AudioResult dataclass."""

    def test_audio_result_creation(self):
        """Test creating AudioResult."""
        import numpy as np

        audio = np.zeros(1000, dtype=np.float32)
        result = AudioResult(audio=audio, sample_rate=24000)

        assert result.sample_rate == 24000
        assert len(result.audio) == 1000


class TestBarkEngineUnit:
    """Unit tests for BarkEngine that don't require model loading."""

    def test_engine_name(self):
        """Test engine has correct name."""
        from dndsandbox_tts.engines.bark import BarkEngine

        engine = BarkEngine(device="cpu")
        assert engine.name == "bark"

    def test_engine_not_loaded_initially(self):
        """Test engine is not loaded on init."""
        from dndsandbox_tts.engines.bark import BarkEngine

        engine = BarkEngine(device="cpu")
        assert not engine.is_loaded()

    def test_get_voices_without_loading(self):
        """Test getting voices doesn't require model loading."""
        from dndsandbox_tts.engines.bark import BarkEngine

        engine = BarkEngine(device="cpu")
        voices = engine.get_voices()
        assert len(voices) > 0

    def test_get_default_voice(self):
        """Test getting default voice."""
        from dndsandbox_tts.engines.bark import BarkEngine

        engine = BarkEngine(device="cpu")
        default = engine.get_default_voice()
        assert default == DEFAULT_VOICE

    def test_validate_voice_valid(self):
        """Test voice validation with valid voice."""
        from dndsandbox_tts.engines.bark import BarkEngine

        engine = BarkEngine(device="cpu")
        result = engine.validate_voice("v2/en_speaker_0")
        assert result == "v2/en_speaker_0"

    def test_validate_voice_invalid_falls_back(self):
        """Test voice validation falls back to default for invalid voice."""
        from dndsandbox_tts.engines.bark import BarkEngine

        engine = BarkEngine(device="cpu")
        result = engine.validate_voice("invalid_voice")
        assert result == DEFAULT_VOICE

    def test_synthesize_raises_when_not_loaded(self):
        """Test synthesize raises error when model not loaded."""
        from dndsandbox_tts.engines.bark import BarkEngine

        engine = BarkEngine(device="cpu")
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.synthesize("Hello", "v2/en_speaker_6")


class TestModelManager:
    """Tests for ModelManager."""

    def test_register_engine(self):
        """Test registering an engine."""
        from dndsandbox_tts.config import Settings
        from dndsandbox_tts.engines.bark import BarkEngine
        from dndsandbox_tts.engines.manager import ModelManager

        settings = Settings()
        manager = ModelManager(settings)

        engine = BarkEngine(device="cpu")
        manager.register_engine(engine)

        assert "bark" in [m["id"] for m in manager.list_models()]

    def test_get_engine(self):
        """Test getting a registered engine."""
        from dndsandbox_tts.config import Settings
        from dndsandbox_tts.engines.bark import BarkEngine
        from dndsandbox_tts.engines.manager import ModelManager

        settings = Settings()
        manager = ModelManager(settings)

        engine = BarkEngine(device="cpu")
        manager.register_engine(engine)

        retrieved = manager.get_engine("bark")
        assert retrieved is engine

    def test_get_engine_not_found(self):
        """Test getting unregistered engine raises error."""
        from dndsandbox_tts.config import Settings
        from dndsandbox_tts.engines.manager import ModelManager, ModelNotFoundError

        settings = Settings()
        manager = ModelManager(settings)

        with pytest.raises(ModelNotFoundError):
            manager.get_engine("nonexistent")

    def test_list_models(self):
        """Test listing registered models."""
        from dndsandbox_tts.config import Settings
        from dndsandbox_tts.engines.bark import BarkEngine
        from dndsandbox_tts.engines.manager import ModelManager

        settings = Settings()
        manager = ModelManager(settings)

        engine = BarkEngine(device="cpu")
        manager.register_engine(engine)

        models = manager.list_models()
        assert len(models) == 1
        assert models[0]["id"] == "bark"
        assert models[0]["loaded"] is False

    def test_get_voices(self):
        """Test getting voices for a model."""
        from dndsandbox_tts.config import Settings
        from dndsandbox_tts.engines.bark import BarkEngine
        from dndsandbox_tts.engines.manager import ModelManager

        settings = Settings()
        manager = ModelManager(settings)

        engine = BarkEngine(device="cpu")
        manager.register_engine(engine)

        voices = manager.get_voices("bark")
        assert len(voices) > 0
