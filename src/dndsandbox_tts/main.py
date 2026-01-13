"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from dndsandbox_tts import __version__
from dndsandbox_tts.api import router
from dndsandbox_tts.cache import get_audio_cache, init_audio_cache
from dndsandbox_tts.config import get_settings
from dndsandbox_tts.engines import init_model_manager
from dndsandbox_tts.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown tasks."""
    settings = get_settings()

    # Configure structlog
    configure_logging(
        log_level=settings.log_level,
        json_logs=settings.json_logs,
    )
    logger = get_logger(__name__)

    logger.info(
        "Starting dndsandbox-tts",
        version=__version__,
        device=settings.get_device(),
        preload_models=settings.preload_models,
        cache_enabled=settings.cache_enabled,
    )

    # Initialize audio cache
    if settings.cache_enabled:
        init_audio_cache(
            max_size=settings.cache_max_entries,
            max_memory_mb=settings.cache_max_memory_mb,
            ttl_seconds=settings.cache_ttl_seconds,
        )

    # Initialize model manager
    manager = init_model_manager(settings)

    # Preload models if configured
    if settings.preload_models:
        logger.info("Preloading models")
        manager.preload_all()
        logger.info("Models preloaded successfully")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down dndsandbox-tts")
    manager.unload_all()

    # Clear cache
    if settings.cache_enabled:
        cache = get_audio_cache()
        cache.clear()


app = FastAPI(
    title="dndsandbox-tts",
    description="Local TTS service for speech synthesis using Bark and other models",
    version=__version__,
    lifespan=lifespan,
)

# Include API routes
app.include_router(router)


def cli():
    """CLI entry point for running the server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "dndsandbox_tts.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    cli()
