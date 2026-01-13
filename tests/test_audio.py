"""Tests for audio processing utilities."""

import subprocess
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dndsandbox_tts.audio import (
    AudioProcessingError,
    adjust_speed,
    get_content_type,
    numpy_to_wav_bytes,
    process_audio,
    wav_bytes_to_format,
)
from dndsandbox_tts.models import AudioFormat


class TestNumpyToWavBytes:
    """Tests for numpy_to_wav_bytes function."""

    def test_float32_audio(self):
        """Test conversion of float32 audio array."""
        # Generate 0.1 second of 440Hz sine wave
        sample_rate = 24000
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        wav_bytes = numpy_to_wav_bytes(audio, sample_rate)

        # WAV header is 44 bytes
        assert len(wav_bytes) > 44
        # Check WAV magic bytes
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"

    def test_float64_audio(self):
        """Test conversion of float64 audio array."""
        sample_rate = 24000
        audio = np.zeros(1000, dtype=np.float64)

        wav_bytes = numpy_to_wav_bytes(audio, sample_rate)

        assert wav_bytes[:4] == b"RIFF"

    def test_int16_audio(self):
        """Test conversion of int16 audio array."""
        sample_rate = 24000
        audio = np.zeros(1000, dtype=np.int16)

        wav_bytes = numpy_to_wav_bytes(audio, sample_rate)

        assert wav_bytes[:4] == b"RIFF"

    def test_clipping(self):
        """Test that audio values outside [-1, 1] are clipped."""
        sample_rate = 24000
        # Audio with values outside valid range
        audio = np.array([-2.0, -1.5, 0.0, 1.5, 2.0], dtype=np.float32)

        # Should not raise an error
        wav_bytes = numpy_to_wav_bytes(audio, sample_rate)
        assert len(wav_bytes) > 0

    def test_unsupported_dtype(self):
        """Test that unsupported dtypes raise an error."""
        sample_rate = 24000
        audio = np.zeros(1000, dtype=np.complex64)

        with pytest.raises(AudioProcessingError) as exc_info:
            numpy_to_wav_bytes(audio, sample_rate)
        assert "Unsupported audio dtype" in str(exc_info.value)


class TestWavBytesToFormat:
    """Tests for wav_bytes_to_format function."""

    @pytest.fixture
    def wav_bytes(self):
        """Create sample WAV bytes for testing."""
        sample_rate = 24000
        audio = np.zeros(1000, dtype=np.float32)
        return numpy_to_wav_bytes(audio, sample_rate)

    def test_wav_passthrough(self, wav_bytes):
        """Test that WAV format returns unchanged bytes."""
        result = wav_bytes_to_format(wav_bytes, AudioFormat.WAV)
        assert result == wav_bytes

    @patch("subprocess.run")
    def test_mp3_conversion(self, mock_run, wav_bytes):
        """Test MP3 conversion calls ffmpeg correctly."""
        mock_run.return_value = MagicMock(stdout=b"fake mp3 data")

        result = wav_bytes_to_format(wav_bytes, AudioFormat.MP3)

        assert result == b"fake mp3 data"
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert "ffmpeg" in call_args[0][0]
        assert "-f" in call_args[0][0]
        assert "mp3" in call_args[0][0]

    @patch("subprocess.run")
    def test_opus_conversion(self, mock_run, wav_bytes):
        """Test OPUS conversion calls ffmpeg correctly."""
        mock_run.return_value = MagicMock(stdout=b"fake opus data")

        result = wav_bytes_to_format(wav_bytes, AudioFormat.OPUS)

        assert result == b"fake opus data"
        call_args = mock_run.call_args
        assert "opus" in call_args[0][0]

    @patch("subprocess.run")
    def test_flac_conversion(self, mock_run, wav_bytes):
        """Test FLAC conversion calls ffmpeg correctly."""
        mock_run.return_value = MagicMock(stdout=b"fake flac data")

        result = wav_bytes_to_format(wav_bytes, AudioFormat.FLAC)

        assert result == b"fake flac data"
        call_args = mock_run.call_args
        assert "flac" in call_args[0][0]

    @patch("subprocess.run")
    def test_ffmpeg_error(self, mock_run, wav_bytes):
        """Test handling of ffmpeg errors."""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "ffmpeg", stderr=b"conversion failed"
        )

        with pytest.raises(AudioProcessingError) as exc_info:
            wav_bytes_to_format(wav_bytes, AudioFormat.MP3)
        assert "ffmpeg conversion failed" in str(exc_info.value)

    @patch("subprocess.run")
    def test_ffmpeg_not_found(self, mock_run, wav_bytes):
        """Test handling when ffmpeg is not installed."""
        mock_run.side_effect = FileNotFoundError()

        with pytest.raises(AudioProcessingError) as exc_info:
            wav_bytes_to_format(wav_bytes, AudioFormat.MP3)
        assert "ffmpeg not found" in str(exc_info.value)

    @patch("subprocess.run")
    def test_sample_rate_conversion(self, mock_run, wav_bytes):
        """Test that sample rate argument is passed to ffmpeg."""
        mock_run.return_value = MagicMock(stdout=b"data")

        wav_bytes_to_format(wav_bytes, AudioFormat.MP3, sample_rate=22050)

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "-ar" in cmd
        ar_index = cmd.index("-ar")
        assert cmd[ar_index + 1] == "22050"


class TestAdjustSpeed:
    """Tests for adjust_speed function."""

    def test_speed_1_0_unchanged(self):
        """Test that speed 1.0 returns unchanged audio."""
        audio = np.zeros(1000, dtype=np.float32)
        sample_rate = 24000

        result_audio, result_rate = adjust_speed(audio, sample_rate, 1.0)

        assert np.array_equal(result_audio, audio)
        assert result_rate == sample_rate

    def test_speed_2_0_doubles_rate(self):
        """Test that speed 2.0 doubles the sample rate."""
        audio = np.zeros(1000, dtype=np.float32)
        sample_rate = 24000

        result_audio, result_rate = adjust_speed(audio, sample_rate, 2.0)

        assert np.array_equal(result_audio, audio)
        assert result_rate == 48000

    def test_speed_0_5_halves_rate(self):
        """Test that speed 0.5 halves the sample rate."""
        audio = np.zeros(1000, dtype=np.float32)
        sample_rate = 24000

        result_audio, result_rate = adjust_speed(audio, sample_rate, 0.5)

        assert np.array_equal(result_audio, audio)
        assert result_rate == 12000


class TestProcessAudio:
    """Tests for process_audio function."""

    @patch("dndsandbox_tts.audio.processor.wav_bytes_to_format")
    def test_process_audio_default(self, mock_convert):
        """Test process_audio with default parameters."""
        mock_convert.return_value = b"processed audio"
        audio = np.zeros(1000, dtype=np.float32)
        sample_rate = 24000

        result = process_audio(audio, sample_rate)

        assert result == b"processed audio"
        mock_convert.assert_called_once()

    @patch("dndsandbox_tts.audio.processor.wav_bytes_to_format")
    def test_process_audio_with_speed(self, mock_convert):
        """Test process_audio applies speed adjustment."""
        mock_convert.return_value = b"processed audio"
        audio = np.zeros(1000, dtype=np.float32)
        sample_rate = 24000

        process_audio(audio, sample_rate, speed=2.0)

        # The WAV bytes passed to convert should have adjusted sample rate
        mock_convert.assert_called_once()

    @patch("dndsandbox_tts.audio.processor.wav_bytes_to_format")
    def test_process_audio_format(self, mock_convert):
        """Test process_audio passes correct format."""
        mock_convert.return_value = b"processed audio"
        audio = np.zeros(1000, dtype=np.float32)
        sample_rate = 24000

        process_audio(audio, sample_rate, output_format=AudioFormat.OPUS)

        call_args = mock_convert.call_args
        assert call_args[0][1] == AudioFormat.OPUS


class TestGetContentType:
    """Tests for get_content_type function."""

    def test_mp3_content_type(self):
        """Test MP3 content type."""
        assert get_content_type(AudioFormat.MP3) == "audio/mpeg"

    def test_wav_content_type(self):
        """Test WAV content type."""
        assert get_content_type(AudioFormat.WAV) == "audio/wav"

    def test_opus_content_type(self):
        """Test OPUS content type."""
        assert get_content_type(AudioFormat.OPUS) == "audio/opus"

    def test_flac_content_type(self):
        """Test FLAC content type."""
        assert get_content_type(AudioFormat.FLAC) == "audio/flac"
