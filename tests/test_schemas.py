"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from dndsandbox_tts.models import (
    AudioFormat,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    ModelList,
    ServiceInfo,
    SpeechRequest,
    VoiceInfo,
    VoiceList,
)


class TestSpeechRequest:
    """Tests for SpeechRequest schema."""

    def test_valid_request_minimal(self):
        """Test valid request with only required field."""
        request = SpeechRequest(input="Hello world")
        assert request.input == "Hello world"
        assert request.model == "bark"
        assert request.voice == "v2/en_speaker_6"
        assert request.response_format == AudioFormat.MP3
        assert request.speed == 1.0

    def test_valid_request_full(self):
        """Test valid request with all fields."""
        request = SpeechRequest(
            model="bark",
            input="Hello adventurer!",
            voice="v2/en_speaker_0",
            response_format=AudioFormat.WAV,
            speed=1.5,
        )
        assert request.model == "bark"
        assert request.input == "Hello adventurer!"
        assert request.voice == "v2/en_speaker_0"
        assert request.response_format == AudioFormat.WAV
        assert request.speed == 1.5

    def test_invalid_empty_input(self):
        """Test that empty input is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SpeechRequest(input="")
        assert "String should have at least 1 character" in str(exc_info.value)

    def test_invalid_input_too_long(self):
        """Test that input exceeding max length is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SpeechRequest(input="x" * 4097)
        assert "String should have at most 4096 characters" in str(exc_info.value)

    def test_invalid_speed_too_low(self):
        """Test that speed below minimum is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SpeechRequest(input="test", speed=0.1)
        assert "greater than or equal to 0.25" in str(exc_info.value)

    def test_invalid_speed_too_high(self):
        """Test that speed above maximum is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SpeechRequest(input="test", speed=5.0)
        assert "less than or equal to 4" in str(exc_info.value)

    def test_valid_speed_boundaries(self):
        """Test speed at valid boundaries."""
        request_min = SpeechRequest(input="test", speed=0.25)
        assert request_min.speed == 0.25

        request_max = SpeechRequest(input="test", speed=4.0)
        assert request_max.speed == 4.0

    def test_audio_format_enum(self):
        """Test different audio formats."""
        for fmt in AudioFormat:
            request = SpeechRequest(input="test", response_format=fmt)
            assert request.response_format == fmt

    def test_audio_format_from_string(self):
        """Test audio format from string value."""
        request = SpeechRequest(input="test", response_format="wav")
        assert request.response_format == AudioFormat.WAV


class TestErrorResponse:
    """Tests for error response schemas."""

    def test_error_detail(self):
        """Test ErrorDetail schema."""
        error = ErrorDetail(
            message="Something went wrong",
            type="invalid_request_error",
            param="model",
            code="model_not_found",
        )
        assert error.message == "Something went wrong"
        assert error.type == "invalid_request_error"
        assert error.param == "model"
        assert error.code == "model_not_found"

    def test_error_detail_minimal(self):
        """Test ErrorDetail with only required fields."""
        error = ErrorDetail(message="Error", type="server_error")
        assert error.message == "Error"
        assert error.type == "server_error"
        assert error.param is None
        assert error.code is None

    def test_error_response(self):
        """Test ErrorResponse wrapper."""
        response = ErrorResponse(
            error=ErrorDetail(message="Test error", type="test_error")
        )
        assert response.error.message == "Test error"
        assert response.error.type == "test_error"


class TestModelSchemas:
    """Tests for model-related schemas."""

    def test_model_info(self):
        """Test ModelInfo schema."""
        model = ModelInfo(id="bark", created=1234567890, owned_by="local")
        assert model.id == "bark"
        assert model.object == "model"
        assert model.created == 1234567890
        assert model.owned_by == "local"

    def test_model_info_defaults(self):
        """Test ModelInfo default values."""
        model = ModelInfo(id="test")
        assert model.object == "model"
        assert model.created == 0
        assert model.owned_by == "local"

    def test_model_list(self):
        """Test ModelList schema."""
        models = ModelList(
            data=[
                ModelInfo(id="bark"),
                ModelInfo(id="piper"),
            ]
        )
        assert models.object == "list"
        assert len(models.data) == 2
        assert models.data[0].id == "bark"
        assert models.data[1].id == "piper"


class TestVoiceSchemas:
    """Tests for voice-related schemas."""

    def test_voice_info(self):
        """Test VoiceInfo schema."""
        voice = VoiceInfo(
            id="nova",
            name="Nova",
            language="en",
            description="Friendly voice",
        )
        assert voice.id == "nova"
        assert voice.name == "Nova"
        assert voice.language == "en"
        assert voice.description == "Friendly voice"

    def test_voice_info_defaults(self):
        """Test VoiceInfo default values."""
        voice = VoiceInfo(id="test", name="Test Voice")
        assert voice.language == "en"
        assert voice.description is None

    def test_voice_list(self):
        """Test VoiceList schema."""
        voices = VoiceList(
            voices=[
                VoiceInfo(id="nova", name="Nova"),
                VoiceInfo(id="alloy", name="Alloy"),
            ]
        )
        assert len(voices.voices) == 2


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_healthy_response(self):
        """Test healthy status response."""
        response = HealthResponse(
            status="healthy",
            model_loaded=True,
            device="cuda",
        )
        assert response.status == "healthy"
        assert response.model_loaded is True
        assert response.device == "cuda"

    def test_unhealthy_response(self):
        """Test unhealthy status response."""
        response = HealthResponse(status="unhealthy")
        assert response.status == "unhealthy"
        assert response.model_loaded is False
        assert response.device is None


class TestServiceInfo:
    """Tests for ServiceInfo schema."""

    def test_service_info(self):
        """Test ServiceInfo schema."""
        info = ServiceInfo(
            version="0.1.0",
            device="cuda",
            models_available=["bark", "piper"],
        )
        assert info.service == "dndsandbox-tts"
        assert info.version == "0.1.0"
        assert info.device == "cuda"
        assert info.models_available == ["bark", "piper"]
