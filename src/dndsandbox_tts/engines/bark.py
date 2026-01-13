"""Bark TTS engine implementation."""

import os
import re
from typing import Any

import numpy as np
import structlog

from collections.abc import Iterator

from dndsandbox_tts.engines.base import AudioResult, BaseTTSEngine, EngineInfo, VoicePreset
from dndsandbox_tts.engines.registry import register_engine

logger = structlog.get_logger(__name__)

# Bark voice presets organized by language
BARK_VOICE_PRESETS = [
    # English speakers
    VoicePreset(id="v2/en_speaker_0", name="English Speaker 0", language="en"),
    VoicePreset(id="v2/en_speaker_1", name="English Speaker 1", language="en"),
    VoicePreset(id="v2/en_speaker_2", name="English Speaker 2", language="en"),
    VoicePreset(id="v2/en_speaker_3", name="English Speaker 3", language="en"),
    VoicePreset(id="v2/en_speaker_4", name="English Speaker 4", language="en"),
    VoicePreset(id="v2/en_speaker_5", name="English Speaker 5", language="en"),
    VoicePreset(id="v2/en_speaker_6", name="English Speaker 6", language="en", description="Default voice"),
    VoicePreset(id="v2/en_speaker_7", name="English Speaker 7", language="en"),
    VoicePreset(id="v2/en_speaker_8", name="English Speaker 8", language="en"),
    VoicePreset(id="v2/en_speaker_9", name="English Speaker 9", language="en"),
    # German speakers
    VoicePreset(id="v2/de_speaker_0", name="German Speaker 0", language="de"),
    VoicePreset(id="v2/de_speaker_1", name="German Speaker 1", language="de"),
    VoicePreset(id="v2/de_speaker_2", name="German Speaker 2", language="de"),
    VoicePreset(id="v2/de_speaker_3", name="German Speaker 3", language="de"),
    VoicePreset(id="v2/de_speaker_4", name="German Speaker 4", language="de"),
    VoicePreset(id="v2/de_speaker_5", name="German Speaker 5", language="de"),
    VoicePreset(id="v2/de_speaker_6", name="German Speaker 6", language="de"),
    VoicePreset(id="v2/de_speaker_7", name="German Speaker 7", language="de"),
    VoicePreset(id="v2/de_speaker_8", name="German Speaker 8", language="de"),
    VoicePreset(id="v2/de_speaker_9", name="German Speaker 9", language="de"),
    # Spanish speakers
    VoicePreset(id="v2/es_speaker_0", name="Spanish Speaker 0", language="es"),
    VoicePreset(id="v2/es_speaker_1", name="Spanish Speaker 1", language="es"),
    VoicePreset(id="v2/es_speaker_2", name="Spanish Speaker 2", language="es"),
    VoicePreset(id="v2/es_speaker_3", name="Spanish Speaker 3", language="es"),
    VoicePreset(id="v2/es_speaker_4", name="Spanish Speaker 4", language="es"),
    VoicePreset(id="v2/es_speaker_5", name="Spanish Speaker 5", language="es"),
    VoicePreset(id="v2/es_speaker_6", name="Spanish Speaker 6", language="es"),
    VoicePreset(id="v2/es_speaker_7", name="Spanish Speaker 7", language="es"),
    VoicePreset(id="v2/es_speaker_8", name="Spanish Speaker 8", language="es"),
    VoicePreset(id="v2/es_speaker_9", name="Spanish Speaker 9", language="es"),
    # French speakers
    VoicePreset(id="v2/fr_speaker_0", name="French Speaker 0", language="fr"),
    VoicePreset(id="v2/fr_speaker_1", name="French Speaker 1", language="fr"),
    VoicePreset(id="v2/fr_speaker_2", name="French Speaker 2", language="fr"),
    VoicePreset(id="v2/fr_speaker_3", name="French Speaker 3", language="fr"),
    VoicePreset(id="v2/fr_speaker_4", name="French Speaker 4", language="fr"),
    VoicePreset(id="v2/fr_speaker_5", name="French Speaker 5", language="fr"),
    VoicePreset(id="v2/fr_speaker_6", name="French Speaker 6", language="fr"),
    VoicePreset(id="v2/fr_speaker_7", name="French Speaker 7", language="fr"),
    VoicePreset(id="v2/fr_speaker_8", name="French Speaker 8", language="fr"),
    VoicePreset(id="v2/fr_speaker_9", name="French Speaker 9", language="fr"),
    # Italian speakers
    VoicePreset(id="v2/it_speaker_0", name="Italian Speaker 0", language="it"),
    VoicePreset(id="v2/it_speaker_1", name="Italian Speaker 1", language="it"),
    VoicePreset(id="v2/it_speaker_2", name="Italian Speaker 2", language="it"),
    VoicePreset(id="v2/it_speaker_3", name="Italian Speaker 3", language="it"),
    VoicePreset(id="v2/it_speaker_4", name="Italian Speaker 4", language="it"),
    VoicePreset(id="v2/it_speaker_5", name="Italian Speaker 5", language="it"),
    VoicePreset(id="v2/it_speaker_6", name="Italian Speaker 6", language="it"),
    VoicePreset(id="v2/it_speaker_7", name="Italian Speaker 7", language="it"),
    VoicePreset(id="v2/it_speaker_8", name="Italian Speaker 8", language="it"),
    VoicePreset(id="v2/it_speaker_9", name="Italian Speaker 9", language="it"),
    # Japanese speakers
    VoicePreset(id="v2/ja_speaker_0", name="Japanese Speaker 0", language="ja"),
    VoicePreset(id="v2/ja_speaker_1", name="Japanese Speaker 1", language="ja"),
    VoicePreset(id="v2/ja_speaker_2", name="Japanese Speaker 2", language="ja"),
    VoicePreset(id="v2/ja_speaker_3", name="Japanese Speaker 3", language="ja"),
    VoicePreset(id="v2/ja_speaker_4", name="Japanese Speaker 4", language="ja"),
    VoicePreset(id="v2/ja_speaker_5", name="Japanese Speaker 5", language="ja"),
    VoicePreset(id="v2/ja_speaker_6", name="Japanese Speaker 6", language="ja"),
    VoicePreset(id="v2/ja_speaker_7", name="Japanese Speaker 7", language="ja"),
    VoicePreset(id="v2/ja_speaker_8", name="Japanese Speaker 8", language="ja"),
    VoicePreset(id="v2/ja_speaker_9", name="Japanese Speaker 9", language="ja"),
    # Korean speakers
    VoicePreset(id="v2/ko_speaker_0", name="Korean Speaker 0", language="ko"),
    VoicePreset(id="v2/ko_speaker_1", name="Korean Speaker 1", language="ko"),
    VoicePreset(id="v2/ko_speaker_2", name="Korean Speaker 2", language="ko"),
    VoicePreset(id="v2/ko_speaker_3", name="Korean Speaker 3", language="ko"),
    VoicePreset(id="v2/ko_speaker_4", name="Korean Speaker 4", language="ko"),
    VoicePreset(id="v2/ko_speaker_5", name="Korean Speaker 5", language="ko"),
    VoicePreset(id="v2/ko_speaker_6", name="Korean Speaker 6", language="ko"),
    VoicePreset(id="v2/ko_speaker_7", name="Korean Speaker 7", language="ko"),
    VoicePreset(id="v2/ko_speaker_8", name="Korean Speaker 8", language="ko"),
    VoicePreset(id="v2/ko_speaker_9", name="Korean Speaker 9", language="ko"),
    # Polish speakers
    VoicePreset(id="v2/pl_speaker_0", name="Polish Speaker 0", language="pl"),
    VoicePreset(id="v2/pl_speaker_1", name="Polish Speaker 1", language="pl"),
    VoicePreset(id="v2/pl_speaker_2", name="Polish Speaker 2", language="pl"),
    VoicePreset(id="v2/pl_speaker_3", name="Polish Speaker 3", language="pl"),
    VoicePreset(id="v2/pl_speaker_4", name="Polish Speaker 4", language="pl"),
    VoicePreset(id="v2/pl_speaker_5", name="Polish Speaker 5", language="pl"),
    VoicePreset(id="v2/pl_speaker_6", name="Polish Speaker 6", language="pl"),
    VoicePreset(id="v2/pl_speaker_7", name="Polish Speaker 7", language="pl"),
    VoicePreset(id="v2/pl_speaker_8", name="Polish Speaker 8", language="pl"),
    VoicePreset(id="v2/pl_speaker_9", name="Polish Speaker 9", language="pl"),
    # Portuguese speakers
    VoicePreset(id="v2/pt_speaker_0", name="Portuguese Speaker 0", language="pt"),
    VoicePreset(id="v2/pt_speaker_1", name="Portuguese Speaker 1", language="pt"),
    VoicePreset(id="v2/pt_speaker_2", name="Portuguese Speaker 2", language="pt"),
    VoicePreset(id="v2/pt_speaker_3", name="Portuguese Speaker 3", language="pt"),
    VoicePreset(id="v2/pt_speaker_4", name="Portuguese Speaker 4", language="pt"),
    VoicePreset(id="v2/pt_speaker_5", name="Portuguese Speaker 5", language="pt"),
    VoicePreset(id="v2/pt_speaker_6", name="Portuguese Speaker 6", language="pt"),
    VoicePreset(id="v2/pt_speaker_7", name="Portuguese Speaker 7", language="pt"),
    VoicePreset(id="v2/pt_speaker_8", name="Portuguese Speaker 8", language="pt"),
    VoicePreset(id="v2/pt_speaker_9", name="Portuguese Speaker 9", language="pt"),
    # Russian speakers
    VoicePreset(id="v2/ru_speaker_0", name="Russian Speaker 0", language="ru"),
    VoicePreset(id="v2/ru_speaker_1", name="Russian Speaker 1", language="ru"),
    VoicePreset(id="v2/ru_speaker_2", name="Russian Speaker 2", language="ru"),
    VoicePreset(id="v2/ru_speaker_3", name="Russian Speaker 3", language="ru"),
    VoicePreset(id="v2/ru_speaker_4", name="Russian Speaker 4", language="ru"),
    VoicePreset(id="v2/ru_speaker_5", name="Russian Speaker 5", language="ru"),
    VoicePreset(id="v2/ru_speaker_6", name="Russian Speaker 6", language="ru"),
    VoicePreset(id="v2/ru_speaker_7", name="Russian Speaker 7", language="ru"),
    VoicePreset(id="v2/ru_speaker_8", name="Russian Speaker 8", language="ru"),
    VoicePreset(id="v2/ru_speaker_9", name="Russian Speaker 9", language="ru"),
    # Turkish speakers
    VoicePreset(id="v2/tr_speaker_0", name="Turkish Speaker 0", language="tr"),
    VoicePreset(id="v2/tr_speaker_1", name="Turkish Speaker 1", language="tr"),
    VoicePreset(id="v2/tr_speaker_2", name="Turkish Speaker 2", language="tr"),
    VoicePreset(id="v2/tr_speaker_3", name="Turkish Speaker 3", language="tr"),
    VoicePreset(id="v2/tr_speaker_4", name="Turkish Speaker 4", language="tr"),
    VoicePreset(id="v2/tr_speaker_5", name="Turkish Speaker 5", language="tr"),
    VoicePreset(id="v2/tr_speaker_6", name="Turkish Speaker 6", language="tr"),
    VoicePreset(id="v2/tr_speaker_7", name="Turkish Speaker 7", language="tr"),
    VoicePreset(id="v2/tr_speaker_8", name="Turkish Speaker 8", language="tr"),
    VoicePreset(id="v2/tr_speaker_9", name="Turkish Speaker 9", language="tr"),
    # Chinese speakers
    VoicePreset(id="v2/zh_speaker_0", name="Chinese Speaker 0", language="zh"),
    VoicePreset(id="v2/zh_speaker_1", name="Chinese Speaker 1", language="zh"),
    VoicePreset(id="v2/zh_speaker_2", name="Chinese Speaker 2", language="zh"),
    VoicePreset(id="v2/zh_speaker_3", name="Chinese Speaker 3", language="zh"),
    VoicePreset(id="v2/zh_speaker_4", name="Chinese Speaker 4", language="zh"),
    VoicePreset(id="v2/zh_speaker_5", name="Chinese Speaker 5", language="zh"),
    VoicePreset(id="v2/zh_speaker_6", name="Chinese Speaker 6", language="zh"),
    VoicePreset(id="v2/zh_speaker_7", name="Chinese Speaker 7", language="zh"),
    VoicePreset(id="v2/zh_speaker_8", name="Chinese Speaker 8", language="zh"),
    VoicePreset(id="v2/zh_speaker_9", name="Chinese Speaker 9", language="zh"),
]

DEFAULT_VOICE = "v2/en_speaker_6"

# Default maximum text length for a single Bark generation (characters)
# Bark works best with shorter segments, but larger chunks are faster
DEFAULT_CHUNK_LENGTH = 250


def split_text_into_chunks(text: str, max_length: int = DEFAULT_CHUNK_LENGTH) -> list[str]:
    """Split text into chunks suitable for Bark synthesis.

    Bark works best with shorter text segments. This function splits
    text at sentence boundaries when possible.

    Args:
        text: Input text to split
        max_length: Maximum characters per chunk

    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]

    # Split on sentence boundaries
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If single sentence is too long, split on commas or words
        if len(sentence) > max_length:
            # Try splitting on commas first
            if "," in sentence:
                parts = sentence.split(",")
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    if len(current_chunk) + len(part) + 2 <= max_length:
                        current_chunk = f"{current_chunk}, {part}" if current_chunk else part
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part
            else:
                # Split on words as last resort
                words = sentence.split()
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= max_length:
                        current_chunk = f"{current_chunk} {word}" if current_chunk else word
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = word
        elif len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk = f"{current_chunk} {sentence}" if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


@register_engine
class BarkEngine(BaseTTSEngine):
    """Bark TTS engine using Suno's Bark model."""

    name = "bark"

    @classmethod
    def get_engine_info(cls) -> EngineInfo:
        """Return information about the Bark engine."""
        return EngineInfo(
            name="bark",
            display_name="Bark",
            description="High-quality neural TTS by Suno AI with multi-language support",
            version="1.0.0",
            supported_formats=["mp3", "wav", "opus", "flac"],
            supports_streaming=True,
            requires_gpu=False,  # Can run on CPU, but GPU recommended
            min_vram_gb=4.0,  # Minimum for GPU inference
            dependencies=["bark", "torch"],
        )

    @classmethod
    def is_available(cls) -> bool:
        """Check if Bark dependencies are available."""
        try:
            import bark  # noqa: F401
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def get_unavailable_reason(cls) -> str | None:
        """Return reason why Bark is unavailable."""
        if cls.is_available():
            return None

        missing = []
        try:
            import bark  # noqa: F401
        except ImportError:
            missing.append("bark")

        try:
            import torch  # noqa: F401
        except ImportError:
            missing.append("torch")

        return f"Missing dependencies: {', '.join(missing)}. Install with: pip install git+https://github.com/suno-ai/bark.git torch"

    def __init__(
        self,
        device: str = "auto",
        use_small_models: bool = False,
        use_fp16: bool = True,
        chunk_size: int = DEFAULT_CHUNK_LENGTH,
        **kwargs: Any,
    ) -> None:
        """Initialize Bark engine.

        Args:
            device: Compute device ('auto', 'cuda', 'cpu')
            use_small_models: Use smaller models for less VRAM (lower quality)
            use_fp16: Use half-precision for GPU inference (faster, less memory)
            chunk_size: Max characters per text chunk
            **kwargs: Additional configuration (ignored)
        """
        self._requested_device = device
        self.device = self._resolve_device(device)
        self.use_small_models = use_small_models
        self.use_fp16 = use_fp16 and self.device != "cpu"
        self.chunk_size = chunk_size
        self._loaded = False
        self._sample_rate: int | None = None
        self._generate_audio = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' device to actual device.

        Args:
            device: Device string ('auto', 'cuda', 'cpu', etc.)

        Returns:
            Resolved device string
        """
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    def get_device_info(self) -> dict[str, Any]:
        """Get information about the compute device.

        Returns:
            Dict with device info (name, memory, etc.)
        """
        info: dict[str, Any] = {
            "device": self.device,
            "requested_device": self._requested_device,
            "fp16_enabled": self.use_fp16,
        }

        if self.device.startswith("cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    device_idx = 0
                    if ":" in self.device:
                        device_idx = int(self.device.split(":")[1])
                    info["gpu_name"] = torch.cuda.get_device_name(device_idx)
                    info["gpu_memory_total"] = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
                    info["gpu_memory_allocated"] = torch.cuda.memory_allocated(device_idx) / (1024**3)
                    info["gpu_memory_reserved"] = torch.cuda.memory_reserved(device_idx) / (1024**3)
            except Exception:
                pass

        return info

    def load(self) -> None:
        """Load Bark models into memory."""
        if self._loaded:
            logger.info("Bark models already loaded")
            return

        logger.info(
            "Loading Bark models",
            device=self.device,
            small_models=self.use_small_models,
            fp16=self.use_fp16,
        )

        try:
            # Set environment variables before importing bark
            if self.use_small_models:
                os.environ["SUNO_USE_SMALL_MODELS"] = "1"

            # Set device for Bark
            if self.device == "cpu":
                os.environ["SUNO_OFFLOAD_CPU"] = "1"

            # Enable FP16 if requested
            if self.use_fp16 and self.device != "cpu":
                os.environ["SUNO_ENABLE_MPS"] = "0"  # Disable MPS for FP16 on CUDA

            # Patch torch.load for PyTorch 2.6+ compatibility with Bark models
            # Bark's pretrained models use numpy scalars which aren't in the safe globals
            import torch
            _original_torch_load = torch.load

            def _patched_torch_load(*args: Any, **kwargs: Any) -> Any:
                # Force weights_only=False for Bark model loading
                if "weights_only" not in kwargs:
                    kwargs["weights_only"] = False
                return _original_torch_load(*args, **kwargs)

            torch.load = _patched_torch_load

            try:
                # Import bark here to avoid loading at module import time
                from bark import SAMPLE_RATE, generate_audio, preload_models

                # Preload all models
                preload_models()
            finally:
                # Restore original torch.load
                torch.load = _original_torch_load

            self._sample_rate = SAMPLE_RATE
            self._generate_audio = generate_audio
            self._loaded = True

            device_info = self.get_device_info()
            logger.info(
                "Bark models loaded successfully",
                sample_rate=SAMPLE_RATE,
                **device_info,
            )

        except ImportError as e:
            logger.error("Failed to import Bark", error=str(e))
            raise RuntimeError(
                "Bark is not installed. Install with: pip install git+https://github.com/suno-ai/bark.git"
            ) from e
        except Exception as e:
            logger.error("Failed to load Bark models", error=str(e))
            raise

    def unload(self) -> None:
        """Unload Bark models from memory."""
        if not self._loaded:
            return

        logger.info("Unloading Bark models")

        # Clear references to allow garbage collection
        self._generate_audio = None
        self._sample_rate = None
        self._loaded = False

        # Force garbage collection to free GPU memory
        import gc
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                # Clear CUDA cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug(
                    "CUDA memory cleared",
                    allocated_gb=torch.cuda.memory_allocated() / (1024**3),
                    reserved_gb=torch.cuda.memory_reserved() / (1024**3),
                )
        except ImportError:
            pass

        logger.info("Bark models unloaded")

    def synthesize(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
    ) -> AudioResult:
        """Generate speech from text using Bark.

        Args:
            text: Text to synthesize
            voice: Bark voice preset (e.g., 'v2/en_speaker_6')
            speed: Speech speed multiplier (applied post-synthesis)

        Returns:
            AudioResult containing audio data and sample rate
        """
        if not self._loaded:
            raise RuntimeError("Bark models not loaded. Call load() first.")

        # Validate and get voice preset
        voice = self.validate_voice(voice)

        logger.info("Synthesizing text", text_length=len(text), voice=voice)

        # Split text into chunks for better quality
        chunks = split_text_into_chunks(text, max_length=self.chunk_size)
        logger.debug("Split text into chunks", chunk_count=len(chunks))

        audio_segments = []

        for i, chunk in enumerate(chunks):
            logger.debug(
                "Generating chunk",
                chunk_num=i + 1,
                total_chunks=len(chunks),
                preview=chunk[:50],
            )

            # Generate audio for this chunk
            audio_array = self._generate_audio(chunk, history_prompt=voice)
            audio_segments.append(audio_array)

        # Concatenate all audio segments
        if len(audio_segments) == 1:
            final_audio = audio_segments[0]
        else:
            # Add small silence between segments for natural pauses
            silence_samples = int(self._sample_rate * 0.1)  # 100ms silence
            silence = np.zeros(silence_samples, dtype=np.float32)

            combined = []
            for i, segment in enumerate(audio_segments):
                combined.append(segment)
                if i < len(audio_segments) - 1:
                    combined.append(silence)

            final_audio = np.concatenate(combined)

        logger.info(
            "Generated audio",
            samples=len(final_audio),
            duration_seconds=round(len(final_audio) / self._sample_rate, 2),
        )

        return AudioResult(
            audio=final_audio.astype(np.float32),
            sample_rate=self._sample_rate,
        )

    def get_voices(self) -> list[VoicePreset]:
        """Return available Bark voice presets."""
        return BARK_VOICE_PRESETS.copy()

    def is_loaded(self) -> bool:
        """Check if Bark models are loaded."""
        return self._loaded

    def get_default_voice(self) -> str:
        """Return the default voice preset."""
        return DEFAULT_VOICE

    def get_config_schema(self) -> dict[str, Any]:
        """Return configuration schema for Bark engine."""
        return {
            "type": "object",
            "properties": {
                "device": {
                    "type": "string",
                    "enum": ["auto", "cuda", "cpu"],
                    "default": "auto",
                    "description": "Compute device for inference",
                },
                "use_small_models": {
                    "type": "boolean",
                    "default": False,
                    "description": "Use smaller models (less VRAM, lower quality)",
                },
                "use_fp16": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use half-precision for faster GPU inference",
                },
            },
        }

    def synthesize_stream(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
    ) -> Iterator[AudioResult]:
        """Generate speech from text as a stream of audio chunks.

        Each chunk is synthesized and yielded immediately, allowing
        the client to start playback before the full text is processed.

        Args:
            text: Text to synthesize
            voice: Voice preset ID
            speed: Speech speed multiplier

        Yields:
            AudioResult for each chunk of audio
        """
        if not self._loaded:
            raise RuntimeError("Bark models not loaded. Call load() first.")

        voice = self.validate_voice(voice)

        logger.info(
            "Streaming synthesis started",
            text_length=len(text),
            voice=voice,
            chunk_size=self.chunk_size,
        )

        chunks = split_text_into_chunks(text, max_length=self.chunk_size)
        logger.debug("Split text into chunks for streaming", chunk_count=len(chunks))

        for i, chunk in enumerate(chunks):
            logger.debug(
                "Generating chunk",
                chunk_num=i + 1,
                total_chunks=len(chunks),
                preview=chunk[:50],
            )

            audio_array = self._generate_audio(chunk, history_prompt=voice)

            logger.debug(
                "Yielding audio chunk",
                chunk_num=i + 1,
                samples=len(audio_array),
                duration_seconds=round(len(audio_array) / self._sample_rate, 2),
            )

            yield AudioResult(
                audio=audio_array.astype(np.float32),
                sample_rate=self._sample_rate,
            )

        logger.info("Streaming synthesis complete", total_chunks=len(chunks))
