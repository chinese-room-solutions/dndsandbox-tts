"""Audio processing utilities for format conversion and manipulation."""

import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from dndsandbox_tts.models import AudioFormat


class AudioProcessingError(Exception):
    """Raised when audio processing fails."""

    pass


def numpy_to_wav_bytes(audio_array: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy audio array to WAV bytes.

    Args:
        audio_array: Audio data as numpy array (float32, range -1 to 1)
        sample_rate: Audio sample rate in Hz

    Returns:
        WAV file as bytes
    """
    # Normalize to int16 range
    if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
        # Clip to valid range and convert to int16
        audio_array = np.clip(audio_array, -1.0, 1.0)
        audio_int16 = (audio_array * 32767).astype(np.int16)
    elif audio_array.dtype == np.int16:
        audio_int16 = audio_array
    else:
        raise AudioProcessingError(f"Unsupported audio dtype: {audio_array.dtype}")

    # Write to bytes buffer
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, audio_int16)
    buffer.seek(0)
    return buffer.read()


def wav_bytes_to_format(
    wav_bytes: bytes,
    output_format: AudioFormat,
    sample_rate: int | None = None,
) -> bytes:
    """Convert WAV bytes to target audio format using ffmpeg.

    Args:
        wav_bytes: Input audio as WAV bytes
        output_format: Target audio format
        sample_rate: Optional sample rate for output (resamples if different)

    Returns:
        Audio in target format as bytes

    Raises:
        AudioProcessingError: If conversion fails
    """
    if output_format == AudioFormat.WAV:
        return wav_bytes

    # Build ffmpeg command
    format_args = {
        AudioFormat.MP3: ["-f", "mp3", "-codec:a", "libmp3lame", "-q:a", "2"],
        AudioFormat.OPUS: ["-f", "opus", "-codec:a", "libopus", "-b:a", "96k"],
        AudioFormat.FLAC: ["-f", "flac", "-codec:a", "flac"],
    }

    if output_format not in format_args:
        raise AudioProcessingError(f"Unsupported output format: {output_format}")

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0"]

    if sample_rate:
        cmd.extend(["-ar", str(sample_rate)])

    cmd.extend(format_args[output_format])
    cmd.extend(["pipe:1"])

    try:
        result = subprocess.run(
            cmd,
            input=wav_bytes,
            capture_output=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise AudioProcessingError(f"ffmpeg conversion failed: {e.stderr.decode()}") from e
    except FileNotFoundError:
        raise AudioProcessingError(
            "ffmpeg not found. Please install ffmpeg for audio format conversion."
        )


def adjust_speed(audio_array: np.ndarray, sample_rate: int, speed: float) -> tuple[np.ndarray, int]:
    """Adjust audio playback speed by resampling.

    Args:
        audio_array: Input audio data
        sample_rate: Original sample rate
        speed: Speed multiplier (1.0 = normal, 2.0 = double speed)

    Returns:
        Tuple of (resampled audio, new sample rate)
    """
    if speed == 1.0:
        return audio_array, sample_rate

    # Adjust speed by changing sample rate interpretation
    # Higher speed = lower effective sample rate
    new_sample_rate = int(sample_rate * speed)

    return audio_array, new_sample_rate


def process_audio(
    audio_array: np.ndarray,
    sample_rate: int,
    output_format: AudioFormat = AudioFormat.MP3,
    speed: float = 1.0,
) -> bytes:
    """Process audio array to target format with optional speed adjustment.

    Args:
        audio_array: Audio data as numpy array
        sample_rate: Audio sample rate
        output_format: Target output format
        speed: Speed multiplier

    Returns:
        Processed audio as bytes in target format
    """
    # Adjust speed if needed
    audio_array, sample_rate = adjust_speed(audio_array, sample_rate, speed)

    # Convert to WAV bytes
    wav_bytes = numpy_to_wav_bytes(audio_array, sample_rate)

    # Convert to target format
    return wav_bytes_to_format(wav_bytes, output_format)


def get_content_type(audio_format: AudioFormat) -> str:
    """Get MIME content type for audio format.

    Args:
        audio_format: Audio format enum

    Returns:
        MIME type string
    """
    content_types = {
        AudioFormat.MP3: "audio/mpeg",
        AudioFormat.WAV: "audio/wav",
        AudioFormat.OPUS: "audio/opus",
        AudioFormat.FLAC: "audio/flac",
    }
    return content_types.get(audio_format, "application/octet-stream")
