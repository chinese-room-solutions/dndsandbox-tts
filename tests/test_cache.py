"""Tests for audio caching."""

import time

import pytest

from dndsandbox_tts.cache import AudioCache, CachedAudio


class TestCachedAudio:
    """Tests for CachedAudio dataclass."""

    def test_cached_audio_creation(self):
        """Test creating a CachedAudio entry."""
        entry = CachedAudio(
            audio_bytes=b"test audio data",
            content_type="audio/mpeg",
        )

        assert entry.audio_bytes == b"test audio data"
        assert entry.content_type == "audio/mpeg"
        assert entry.access_count == 0
        assert entry.created_at <= time.time()

    def test_touch_updates_access(self):
        """Test that touch() updates access count and time."""
        entry = CachedAudio(
            audio_bytes=b"test",
            content_type="audio/mpeg",
        )

        initial_time = entry.last_accessed
        time.sleep(0.01)  # Small delay

        entry.touch()

        assert entry.access_count == 1
        assert entry.last_accessed > initial_time


class TestAudioCache:
    """Tests for AudioCache class."""

    @pytest.fixture
    def cache(self):
        """Create a fresh cache for testing."""
        return AudioCache(max_size=10, max_memory_mb=1.0, ttl_seconds=60)

    def test_make_key_deterministic(self):
        """Test that key generation is deterministic."""
        key1 = AudioCache._make_key("hello", "bark", "voice1", 1.0, "mp3")
        key2 = AudioCache._make_key("hello", "bark", "voice1", 1.0, "mp3")

        assert key1 == key2

    def test_make_key_different_inputs(self):
        """Test that different inputs produce different keys."""
        key1 = AudioCache._make_key("hello", "bark", "voice1", 1.0, "mp3")
        key2 = AudioCache._make_key("world", "bark", "voice1", 1.0, "mp3")
        key3 = AudioCache._make_key("hello", "bark", "voice2", 1.0, "mp3")

        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_put_and_get(self, cache):
        """Test basic put and get operations."""
        cache.put(
            text="hello",
            model="bark",
            voice="voice1",
            speed=1.0,
            output_format="mp3",
            audio_bytes=b"audio data",
            content_type="audio/mpeg",
        )

        result = cache.get(
            text="hello",
            model="bark",
            voice="voice1",
            speed=1.0,
            output_format="mp3",
        )

        assert result is not None
        audio_bytes, content_type = result
        assert audio_bytes == b"audio data"
        assert content_type == "audio/mpeg"

    def test_get_miss(self, cache):
        """Test cache miss returns None."""
        result = cache.get(
            text="nonexistent",
            model="bark",
            voice="voice1",
            speed=1.0,
            output_format="mp3",
        )

        assert result is None

    def test_lru_eviction(self, cache):
        """Test LRU eviction when max_size is reached."""
        # Fill cache with 10 entries
        for i in range(10):
            cache.put(
                text=f"text{i}",
                model="bark",
                voice="voice1",
                speed=1.0,
                output_format="mp3",
                audio_bytes=b"x" * 100,
                content_type="audio/mpeg",
            )

        # Add one more - should evict the oldest
        cache.put(
            text="text_new",
            model="bark",
            voice="voice1",
            speed=1.0,
            output_format="mp3",
            audio_bytes=b"x" * 100,
            content_type="audio/mpeg",
        )

        # First entry should be evicted
        result = cache.get(
            text="text0",
            model="bark",
            voice="voice1",
            speed=1.0,
            output_format="mp3",
        )
        assert result is None

        # Newest entry should exist
        result = cache.get(
            text="text_new",
            model="bark",
            voice="voice1",
            speed=1.0,
            output_format="mp3",
        )
        assert result is not None

    def test_memory_limit_eviction(self):
        """Test eviction when memory limit is reached."""
        # 100KB limit
        cache = AudioCache(max_size=100, max_memory_mb=0.1, ttl_seconds=60)

        # Add entries until memory limit
        for i in range(5):
            cache.put(
                text=f"text{i}",
                model="bark",
                voice="voice1",
                speed=1.0,
                output_format="mp3",
                audio_bytes=b"x" * 30000,  # 30KB each
                content_type="audio/mpeg",
            )

        # Should have evicted oldest entries
        stats = cache.get_stats()
        assert stats["memory_mb"] <= 0.1

    def test_ttl_expiration(self):
        """Test that expired entries are not returned."""
        cache = AudioCache(max_size=10, max_memory_mb=1.0, ttl_seconds=1)

        cache.put(
            text="hello",
            model="bark",
            voice="voice1",
            speed=1.0,
            output_format="mp3",
            audio_bytes=b"audio data",
            content_type="audio/mpeg",
        )

        # Immediately should work
        result = cache.get(
            text="hello",
            model="bark",
            voice="voice1",
            speed=1.0,
            output_format="mp3",
        )
        assert result is not None

        # After TTL expires
        time.sleep(1.1)
        result = cache.get(
            text="hello",
            model="bark",
            voice="voice1",
            speed=1.0,
            output_format="mp3",
        )
        assert result is None

    def test_clear(self, cache):
        """Test clearing the cache."""
        cache.put(
            text="hello",
            model="bark",
            voice="voice1",
            speed=1.0,
            output_format="mp3",
            audio_bytes=b"audio data",
            content_type="audio/mpeg",
        )

        cache.clear()

        result = cache.get(
            text="hello",
            model="bark",
            voice="voice1",
            speed=1.0,
            output_format="mp3",
        )
        assert result is None

        stats = cache.get_stats()
        assert stats["entries"] == 0
        assert stats["memory_mb"] == 0.0

    def test_get_stats(self, cache):
        """Test statistics tracking."""
        # Initial stats
        stats = cache.get_stats()
        assert stats["entries"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

        # Add an entry
        cache.put(
            text="hello",
            model="bark",
            voice="voice1",
            speed=1.0,
            output_format="mp3",
            audio_bytes=b"audio data",
            content_type="audio/mpeg",
        )

        # Cache miss
        cache.get(
            text="nonexistent",
            model="bark",
            voice="voice1",
            speed=1.0,
            output_format="mp3",
        )

        # Cache hit
        cache.get(
            text="hello",
            model="bark",
            voice="voice1",
            speed=1.0,
            output_format="mp3",
        )

        stats = cache.get_stats()
        assert stats["entries"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache = AudioCache(max_size=10, max_memory_mb=1.0, ttl_seconds=1)

        # Add entries
        for i in range(3):
            cache.put(
                text=f"text{i}",
                model="bark",
                voice="voice1",
                speed=1.0,
                output_format="mp3",
                audio_bytes=b"audio data",
                content_type="audio/mpeg",
            )

        assert cache.get_stats()["entries"] == 3

        # Wait for expiration
        time.sleep(1.1)

        # Cleanup
        removed = cache.cleanup_expired()
        assert removed == 3
        assert cache.get_stats()["entries"] == 0

    def test_oversized_entry_not_cached(self):
        """Test that entries larger than max memory are not cached."""
        cache = AudioCache(max_size=100, max_memory_mb=0.001, ttl_seconds=60)

        # Try to cache large entry (larger than 1KB limit)
        cache.put(
            text="hello",
            model="bark",
            voice="voice1",
            speed=1.0,
            output_format="mp3",
            audio_bytes=b"x" * 10000,  # 10KB
            content_type="audio/mpeg",
        )

        result = cache.get(
            text="hello",
            model="bark",
            voice="voice1",
            speed=1.0,
            output_format="mp3",
        )
        assert result is None

    def test_overwrite_existing(self, cache):
        """Test that putting same key overwrites existing entry."""
        cache.put(
            text="hello",
            model="bark",
            voice="voice1",
            speed=1.0,
            output_format="mp3",
            audio_bytes=b"old data",
            content_type="audio/mpeg",
        )

        cache.put(
            text="hello",
            model="bark",
            voice="voice1",
            speed=1.0,
            output_format="mp3",
            audio_bytes=b"new data",
            content_type="audio/mpeg",
        )

        result = cache.get(
            text="hello",
            model="bark",
            voice="voice1",
            speed=1.0,
            output_format="mp3",
        )

        assert result is not None
        audio_bytes, _ = result
        assert audio_bytes == b"new data"

    def test_thread_safety(self, cache):
        """Test concurrent access to cache."""
        import threading

        errors = []

        def writer():
            try:
                for i in range(100):
                    cache.put(
                        text=f"text{threading.current_thread().name}_{i}",
                        model="bark",
                        voice="voice1",
                        speed=1.0,
                        output_format="mp3",
                        audio_bytes=b"data",
                        content_type="audio/mpeg",
                    )
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    cache.get(
                        text="text_any",
                        model="bark",
                        voice="voice1",
                        speed=1.0,
                        output_format="mp3",
                    )
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(4):
            threads.append(threading.Thread(target=writer, name=f"writer{i}"))
            threads.append(threading.Thread(target=reader, name=f"reader{i}"))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
