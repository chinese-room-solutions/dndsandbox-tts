"""Audio caching for repeated TTS requests."""

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CachedAudio:
    """Cached audio entry."""

    audio_bytes: bytes
    content_type: str
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1


class AudioCache:
    """LRU cache for generated audio.

    Caches audio by a hash of the synthesis parameters to avoid
    regenerating identical audio.
    """

    def __init__(
        self,
        max_size: int = 100,
        max_memory_mb: float = 500.0,
        ttl_seconds: int = 3600,
    ) -> None:
        """Initialize audio cache.

        Args:
            max_size: Maximum number of cached entries
            max_memory_mb: Maximum cache memory in MB
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.ttl_seconds = ttl_seconds

        self._cache: OrderedDict[str, CachedAudio] = OrderedDict()
        self._lock = threading.RLock()
        self._current_memory = 0

        # Statistics
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(
        text: str,
        model: str,
        voice: str,
        speed: float,
        output_format: str,
    ) -> str:
        """Generate cache key from synthesis parameters.

        Args:
            text: Input text
            model: Model name
            voice: Voice preset
            speed: Speech speed
            output_format: Output audio format

        Returns:
            SHA-256 hash of parameters
        """
        key_data = f"{model}:{voice}:{speed}:{output_format}:{text}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(
        self,
        text: str,
        model: str,
        voice: str,
        speed: float,
        output_format: str,
    ) -> tuple[bytes, str] | None:
        """Get cached audio if available.

        Args:
            text: Input text
            model: Model name
            voice: Voice preset
            speed: Speech speed
            output_format: Output audio format

        Returns:
            Tuple of (audio_bytes, content_type) or None if not cached
        """
        key = self._make_key(text, model, voice, speed, output_format)

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            # Check TTL
            if time.time() - entry.created_at > self.ttl_seconds:
                self._evict(key)
                self._misses += 1
                logger.debug("Cache entry expired", key=key[:16])
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1

            logger.debug(
                "Cache hit",
                key=key[:16],
                access_count=entry.access_count,
            )
            return entry.audio_bytes, entry.content_type

    def put(
        self,
        text: str,
        model: str,
        voice: str,
        speed: float,
        output_format: str,
        audio_bytes: bytes,
        content_type: str,
    ) -> None:
        """Store audio in cache.

        Args:
            text: Input text
            model: Model name
            voice: Voice preset
            speed: Speech speed
            output_format: Output audio format
            audio_bytes: Generated audio data
            content_type: Audio MIME type
        """
        key = self._make_key(text, model, voice, speed, output_format)
        entry_size = len(audio_bytes)

        # Don't cache if single entry exceeds memory limit
        if entry_size > self.max_memory_bytes:
            logger.debug(
                "Audio too large to cache",
                size_mb=entry_size / (1024 * 1024),
                max_mb=self.max_memory_bytes / (1024 * 1024),
            )
            return

        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._evict(key)

            # Evict entries until we have space
            while (
                len(self._cache) >= self.max_size
                or self._current_memory + entry_size > self.max_memory_bytes
            ):
                if not self._cache:
                    break
                # Remove oldest (first) entry
                oldest_key = next(iter(self._cache))
                self._evict(oldest_key)

            # Add new entry
            self._cache[key] = CachedAudio(
                audio_bytes=audio_bytes,
                content_type=content_type,
            )
            self._current_memory += entry_size

            logger.debug(
                "Cached audio",
                key=key[:16],
                size_kb=entry_size / 1024,
                cache_size=len(self._cache),
                memory_mb=self._current_memory / (1024 * 1024),
            )

    def _evict(self, key: str) -> None:
        """Remove entry from cache.

        Args:
            key: Cache key to remove
        """
        if key in self._cache:
            entry = self._cache.pop(key)
            self._current_memory -= len(entry.audio_bytes)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0
            logger.info("Cache cleared")

    def cleanup_expired(self) -> int:
        """Remove expired entries.

        Returns:
            Number of entries removed
        """
        now = time.time()
        removed = 0

        with self._lock:
            expired_keys = [
                key
                for key, entry in self._cache.items()
                if now - entry.created_at > self.ttl_seconds
            ]
            for key in expired_keys:
                self._evict(key)
                removed += 1

        if removed:
            logger.info("Cleaned up expired cache entries", removed=removed)

        return removed

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "entries": len(self._cache),
                "max_entries": self.max_size,
                "memory_mb": round(self._current_memory / (1024 * 1024), 2),
                "max_memory_mb": round(self.max_memory_bytes / (1024 * 1024), 2),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 3),
                "ttl_seconds": self.ttl_seconds,
            }


# Global cache instance
_audio_cache: AudioCache | None = None


def get_audio_cache() -> AudioCache:
    """Get the global audio cache instance.

    Returns:
        AudioCache instance
    """
    global _audio_cache
    if _audio_cache is None:
        _audio_cache = AudioCache()
    return _audio_cache


def init_audio_cache(
    max_size: int = 100,
    max_memory_mb: float = 500.0,
    ttl_seconds: int = 3600,
) -> AudioCache:
    """Initialize the global audio cache.

    Args:
        max_size: Maximum number of cached entries
        max_memory_mb: Maximum cache memory in MB
        ttl_seconds: Time-to-live for cache entries

    Returns:
        Initialized AudioCache instance
    """
    global _audio_cache
    _audio_cache = AudioCache(
        max_size=max_size,
        max_memory_mb=max_memory_mb,
        ttl_seconds=ttl_seconds,
    )
    logger.info(
        "Audio cache initialized",
        max_size=max_size,
        max_memory_mb=max_memory_mb,
        ttl_seconds=ttl_seconds,
    )
    return _audio_cache
