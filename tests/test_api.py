"""Tests for API endpoints."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from dndsandbox_tts.engines.base import AudioResult, VoicePreset


@pytest.fixture
def mock_model_manager():
    """Create a mock model manager."""
    manager = MagicMock()
    manager.list_models.return_value = [{"id": "bark", "loaded": True}]
    manager.get_voices.return_value = [
        VoicePreset(id="v2/en_speaker_0", name="English Speaker 0", language="en"),
        VoicePreset(id="v2/en_speaker_6", name="English Speaker 6", language="en"),
    ]
    manager.synthesize.return_value = AudioResult(
        audio=np.zeros(24000, dtype=np.float32),
        sample_rate=24000,
    )
    return manager


@pytest.fixture
def client(mock_model_manager):
    """Create test client with mocked model manager."""
    with patch("dndsandbox_tts.engines.manager._model_manager", mock_model_manager):
        from dndsandbox_tts.main import app

        yield TestClient(app, raise_server_exceptions=False)


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_service_info(self, client):
        """Test that root endpoint returns service information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "dndsandbox-tts"
        assert "version" in data
        assert "device" in data
        assert "models_available" in data


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_healthy(self, client):
        """Test that health endpoint returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestModelsEndpoint:
    """Tests for models listing endpoint."""

    def test_list_models(self, client):
        """Test listing available models."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) > 0
        assert data["data"][0]["id"] == "bark"
        assert data["data"][0]["object"] == "model"


class TestVoicesEndpoint:
    """Tests for voices listing endpoint."""

    def test_list_voices(self, client):
        """Test listing available voices."""
        response = client.get("/v1/voices")

        assert response.status_code == 200
        data = response.json()
        assert "voices" in data
        assert isinstance(data["voices"], list)
        assert len(data["voices"]) > 0

        # Check Bark voice presets are present
        voice_ids = [v["id"] for v in data["voices"]]
        assert "v2/en_speaker_6" in voice_ids
        assert "v2/en_speaker_0" in voice_ids

    def test_list_voices_with_model_param(self, client):
        """Test listing voices with model parameter."""
        response = client.get("/v1/voices?model=bark")

        assert response.status_code == 200


class TestSpeechEndpoint:
    """Tests for speech synthesis endpoint."""

    @patch("dndsandbox_tts.api.routes.process_audio")
    def test_create_speech_minimal(self, mock_process, client):
        """Test speech synthesis with minimal request."""
        mock_process.return_value = b"fake audio data"

        response = client.post(
            "/v1/audio/speech",
            json={"input": "Hello, world!"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert response.content == b"fake audio data"

    @patch("dndsandbox_tts.api.routes.process_audio")
    def test_create_speech_full_params(self, mock_process, client):
        """Test speech synthesis with all parameters."""
        mock_process.return_value = b"fake audio data"

        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "bark",
                "input": "Hello, adventurer!",
                "voice": "v2/en_speaker_6",
                "response_format": "wav",
                "speed": 1.5,
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"

    def test_create_speech_empty_input(self, client):
        """Test that empty input is rejected."""
        response = client.post(
            "/v1/audio/speech",
            json={"input": ""},
        )

        assert response.status_code == 422  # Validation error

    def test_create_speech_missing_input(self, client):
        """Test that missing input is rejected."""
        response = client.post(
            "/v1/audio/speech",
            json={},
        )

        assert response.status_code == 422

    def test_create_speech_invalid_model(self, client, mock_model_manager):
        """Test that invalid model returns error."""
        from dndsandbox_tts.engines.manager import ModelNotFoundError

        mock_model_manager.synthesize.side_effect = ModelNotFoundError("invalid-model")

        response = client.post(
            "/v1/audio/speech",
            json={"model": "invalid-model", "input": "test"},
        )

        assert response.status_code == 400
        data = response.json()
        # Error wrapped in detail for FastAPI HTTPException
        error = data.get("error") or data.get("detail", {}).get("error", {})
        assert error.get("code") == "model_not_found"

    def test_create_speech_invalid_speed_low(self, client):
        """Test that speed below minimum is rejected."""
        response = client.post(
            "/v1/audio/speech",
            json={"input": "test", "speed": 0.1},
        )

        assert response.status_code == 422

    def test_create_speech_invalid_speed_high(self, client):
        """Test that speed above maximum is rejected."""
        response = client.post(
            "/v1/audio/speech",
            json={"input": "test", "speed": 5.0},
        )

        assert response.status_code == 422

    @patch("dndsandbox_tts.api.routes.process_audio")
    def test_create_speech_content_disposition(self, mock_process, client):
        """Test that response includes content disposition header."""
        mock_process.return_value = b"fake audio data"

        response = client.post(
            "/v1/audio/speech",
            json={"input": "test"},
        )

        assert "content-disposition" in response.headers
        assert "speech.mp3" in response.headers["content-disposition"]

    @patch("dndsandbox_tts.api.routes.process_audio")
    def test_create_speech_wav_format(self, mock_process, client):
        """Test speech synthesis with WAV format."""
        mock_process.return_value = b"fake wav data"

        response = client.post(
            "/v1/audio/speech",
            json={"input": "test", "response_format": "wav"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert "speech.wav" in response.headers["content-disposition"]


class TestAuthentication:
    """Tests for API authentication."""

    @pytest.fixture
    def auth_client(self, mock_model_manager):
        """Create test client with authentication required."""
        with patch("dndsandbox_tts.engines.manager._model_manager", mock_model_manager):
            from dndsandbox_tts.config import Settings, get_settings
            from dndsandbox_tts.main import app

            def get_settings_override():
                return Settings(require_auth=True, api_key="test-secret-key")

            app.dependency_overrides[get_settings] = get_settings_override
            client = TestClient(app, raise_server_exceptions=False)
            yield client
            app.dependency_overrides.clear()

    def test_missing_auth_header(self, auth_client):
        """Test that missing auth header returns 401."""
        response = auth_client.get("/v1/models")

        assert response.status_code == 401
        data = response.json()
        error = data.get("error") or data.get("detail", {}).get("error", {})
        assert error.get("code") == "missing_api_key"

    def test_invalid_auth_format(self, auth_client):
        """Test that invalid auth format returns 401."""
        response = auth_client.get(
            "/v1/models",
            headers={"Authorization": "InvalidFormat"},
        )

        assert response.status_code == 401
        data = response.json()
        error = data.get("error") or data.get("detail", {}).get("error", {})
        assert error.get("code") == "invalid_api_key"

    def test_wrong_api_key(self, auth_client):
        """Test that wrong API key returns 401."""
        response = auth_client.get(
            "/v1/models",
            headers={"Authorization": "Bearer wrong-key"},
        )

        assert response.status_code == 401
        data = response.json()
        error = data.get("error") or data.get("detail", {}).get("error", {})
        assert error.get("code") == "invalid_api_key"

    def test_valid_api_key(self, auth_client):
        """Test that valid API key allows access."""
        response = auth_client.get(
            "/v1/models",
            headers={"Authorization": "Bearer test-secret-key"},
        )

        assert response.status_code == 200

    def test_no_auth_when_not_required(self, client):
        """Test that auth is not required when disabled."""
        response = client.get("/v1/models")

        assert response.status_code == 200


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def no_cache_client(self, mock_model_manager):
        """Create test client with caching disabled."""
        with patch("dndsandbox_tts.engines.manager._model_manager", mock_model_manager):
            from dndsandbox_tts.config import Settings, get_settings
            from dndsandbox_tts.main import app

            def get_settings_override():
                return Settings(cache_enabled=False)

            app.dependency_overrides[get_settings] = get_settings_override
            client = TestClient(app, raise_server_exceptions=False)
            yield client
            app.dependency_overrides.clear()

    @patch("dndsandbox_tts.api.routes.process_audio")
    def test_audio_processing_error(self, mock_process, no_cache_client):
        """Test handling of audio processing errors."""
        from dndsandbox_tts.audio import AudioProcessingError

        mock_process.side_effect = AudioProcessingError("ffmpeg failed")

        response = no_cache_client.post(
            "/v1/audio/speech",
            json={"input": "test_error_audio"},
        )

        assert response.status_code == 500
        data = response.json()
        error = data.get("error") or data.get("detail", {}).get("error", {})
        assert error.get("code") == "audio_processing_error"

    def test_model_runtime_error(self, no_cache_client, mock_model_manager):
        """Test handling of model runtime errors."""
        mock_model_manager.synthesize.side_effect = RuntimeError("Model failed")

        response = no_cache_client.post(
            "/v1/audio/speech",
            json={"input": "test_error_model"},
        )

        assert response.status_code == 500
        data = response.json()
        error = data.get("error") or data.get("detail", {}).get("error", {})
        assert error.get("code") == "model_error"
