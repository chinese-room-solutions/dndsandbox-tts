"""API routes for TTS service."""

import time
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from fastapi.responses import StreamingResponse

from dndsandbox_tts import __version__
from dndsandbox_tts.audio import AudioProcessingError, get_content_type, process_audio
from dndsandbox_tts.cache import get_audio_cache
from dndsandbox_tts.config import Settings, get_settings
from dndsandbox_tts.engines import ModelNotFoundError, get_model_manager
from dndsandbox_tts.models import (
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

logger = structlog.get_logger(__name__)

router = APIRouter()


def verify_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Verify API key if authentication is required."""
    if not settings.require_auth:
        return

    if not settings.api_key:
        # No API key configured but auth required - misconfiguration
        logger.warning("Authentication required but no API key configured")
        return None

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message="Missing Authorization header",
                    type="invalid_request_error",
                    code="missing_api_key",
                )
            ).model_dump(),
        )

    # Extract bearer token
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message="Invalid Authorization header format. Expected 'Bearer <token>'",
                    type="invalid_request_error",
                    code="invalid_api_key",
                )
            ).model_dump(),
        )

    if parts[1] != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message="Invalid API key",
                    type="invalid_request_error",
                    code="invalid_api_key",
                )
            ).model_dump(),
        )


@router.get("/", response_model=ServiceInfo)
async def root(settings: Annotated[Settings, Depends(get_settings)]) -> ServiceInfo:
    """Get service information."""
    try:
        manager = get_model_manager()
        models = [m["id"] for m in manager.list_models()]
    except RuntimeError:
        models = []

    return ServiceInfo(
        version=__version__,
        device=settings.get_device(),
        models_available=models,
    )


@router.get("/health", response_model=HealthResponse)
async def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Health check endpoint."""
    try:
        manager = get_model_manager()
        models = manager.list_models()
        any_loaded = any(m["loaded"] for m in models)
    except RuntimeError:
        any_loaded = False

    return HealthResponse(
        status="healthy",
        model_loaded=any_loaded,
        device=settings.get_device(),
    )


@router.get("/v1/models", response_model=ModelList)
async def list_models(
    settings: Annotated[Settings, Depends(get_settings)],
    _auth: Annotated[None, Depends(verify_api_key)] = None,
) -> ModelList:
    """List available TTS models."""
    try:
        manager = get_model_manager()
        models = manager.list_models()
    except RuntimeError:
        models = []

    return ModelList(
        data=[
            ModelInfo(
                id=m["id"],
                created=int(time.time()),
                owned_by="local",
            )
            for m in models
        ]
    )


@router.get("/v1/voices", response_model=VoiceList)
async def list_voices(
    settings: Annotated[Settings, Depends(get_settings)],
    model: str = "bark",
    _auth: Annotated[None, Depends(verify_api_key)] = None,
) -> VoiceList:
    """List available voices for a model."""
    try:
        manager = get_model_manager()
        voice_presets = manager.get_voices(model)
        voices = [
            VoiceInfo(
                id=v.id,
                name=v.name,
                language=v.language,
                description=v.description,
            )
            for v in voice_presets
        ]
    except RuntimeError:
        voices = []
    except ModelNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message=f"Model '{model}' not found",
                    type="invalid_request_error",
                    param="model",
                    code="model_not_found",
                )
            ).model_dump(),
        )

    return VoiceList(voices=voices)


@router.post("/v1/audio/speech")
async def create_speech(
    request: SpeechRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    _auth: Annotated[None, Depends(verify_api_key)] = None,
) -> Response:
    """Generate speech from text.

    Returns audio in the requested format (default: MP3).
    """
    logger.info(
        "Speech request received",
        model=request.model,
        voice=request.voice,
        format=request.response_format.value,
        speed=request.speed,
        text_length=len(request.input),
    )

    try:
        # Check cache first if enabled
        if settings.cache_enabled:
            cache = get_audio_cache()
            cached = cache.get(
                text=request.input,
                model=request.model,
                voice=request.voice,
                speed=request.speed,
                output_format=request.response_format.value,
            )
            if cached:
                audio_bytes, content_type = cached
                logger.info(
                    "Serving cached audio",
                    bytes=len(audio_bytes),
                    format=request.response_format.value,
                )
                return Response(
                    content=audio_bytes,
                    media_type=content_type,
                    headers={
                        "Content-Disposition": f"attachment; filename=speech.{request.response_format.value}",
                        "X-Cache": "HIT",
                    },
                )

        manager = get_model_manager()

        # Synthesize audio using the model manager
        result = manager.synthesize(
            model=request.model,
            text=request.input,
            voice=request.voice,
            speed=request.speed,
        )

        # Process audio to target format
        audio_bytes = process_audio(
            audio_array=result.audio,
            sample_rate=result.sample_rate,
            output_format=request.response_format,
            speed=1.0,  # Speed already applied in synthesis or will be applied here
        )

        content_type = get_content_type(request.response_format)

        # Cache the result if enabled
        if settings.cache_enabled:
            cache = get_audio_cache()
            cache.put(
                text=request.input,
                model=request.model,
                voice=request.voice,
                speed=request.speed,
                output_format=request.response_format.value,
                audio_bytes=audio_bytes,
                content_type=content_type,
            )

        logger.info(
            "Generated audio",
            bytes=len(audio_bytes),
            format=request.response_format.value,
        )

        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format.value}",
                "X-Cache": "MISS",
            },
        )

    except ModelNotFoundError as e:
        logger.error("Model not found", error=str(e), model=request.model)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message=str(e),
                    type="invalid_request_error",
                    param="model",
                    code="model_not_found",
                )
            ).model_dump(),
        )
    except RuntimeError as e:
        logger.error("Runtime error during synthesis", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message=str(e),
                    type="server_error",
                    code="model_error",
                )
            ).model_dump(),
        )
    except AudioProcessingError as e:
        logger.error("Audio processing error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message=f"Audio processing failed: {str(e)}",
                    type="server_error",
                    code="audio_processing_error",
                )
            ).model_dump(),
        )
    except Exception as e:
        logger.exception("Unexpected error during speech generation", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message="An unexpected error occurred during speech generation",
                    type="server_error",
                    code="internal_error",
                )
            ).model_dump(),
        )


@router.post("/v1/audio/speech/stream")
async def create_speech_stream(
    request: SpeechRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    _auth: Annotated[None, Depends(verify_api_key)] = None,
) -> StreamingResponse:
    """Generate speech from text as a stream of audio chunks.

    Returns audio chunks as they are generated, allowing playback to start
    before the full text is processed. Each chunk is a complete audio segment
    that can be played independently.

    The response is a stream of raw audio bytes. For WAV format, each chunk
    includes a valid WAV header. For other formats (MP3, etc.), chunks are
    concatenatable.

    Use this endpoint when:
    - Processing long text that would take too long to generate all at once
    - You want to start playback as soon as possible (reduce latency)
    - Integrating with streaming LLM output

    Note: Caching is disabled for streaming requests.
    """
    logger.info(
        "Streaming speech request received",
        model=request.model,
        voice=request.voice,
        format=request.response_format.value,
        speed=request.speed,
        text_length=len(request.input),
    )

    try:
        manager = get_model_manager()
        engine = manager.get_engine(request.model)

        # Check if engine supports streaming
        engine_info = engine.get_engine_info()
        if not engine_info.supports_streaming:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        message=f"Model '{request.model}' does not support streaming",
                        type="invalid_request_error",
                        param="model",
                        code="streaming_not_supported",
                    )
                ).model_dump(),
            )

        # Ensure model is loaded
        if not engine.is_loaded():
            engine.load()

        content_type = get_content_type(request.response_format)

        def audio_stream():
            """Generator that yields audio chunks."""
            chunk_num = 0
            for audio_result in engine.synthesize_stream(
                text=request.input,
                voice=request.voice,
                speed=request.speed,
            ):
                chunk_num += 1
                try:
                    # Convert each chunk to the requested format
                    audio_bytes = process_audio(
                        audio_array=audio_result.audio,
                        sample_rate=audio_result.sample_rate,
                        output_format=request.response_format,
                        speed=1.0,
                    )
                    logger.debug(
                        "Streaming audio chunk",
                        chunk_num=chunk_num,
                        bytes=len(audio_bytes),
                    )
                    yield audio_bytes
                except AudioProcessingError as e:
                    logger.error("Audio processing error in stream", error=str(e), chunk=chunk_num)
                    # Stop the stream on error
                    break

            logger.info("Streaming complete", total_chunks=chunk_num)

        return StreamingResponse(
            audio_stream(),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format.value}",
                "X-Streaming": "true",
            },
        )

    except ModelNotFoundError as e:
        logger.error("Model not found", error=str(e), model=request.model)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message=str(e),
                    type="invalid_request_error",
                    param="model",
                    code="model_not_found",
                )
            ).model_dump(),
        )
    except RuntimeError as e:
        logger.error("Runtime error during streaming synthesis", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message=str(e),
                    type="server_error",
                    code="model_error",
                )
            ).model_dump(),
        )
    except Exception as e:
        logger.exception("Unexpected error during streaming speech generation", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message="An unexpected error occurred during speech generation",
                    type="server_error",
                    code="internal_error",
                )
            ).model_dump(),
        )


@router.get("/v1/cache/stats")
async def cache_stats(
    settings: Annotated[Settings, Depends(get_settings)],
    _auth: Annotated[None, Depends(verify_api_key)] = None,
) -> dict:
    """Get cache statistics.

    Returns information about the audio cache including hit rate,
    memory usage, and entry count.
    """
    if not settings.cache_enabled:
        return {"enabled": False}

    cache = get_audio_cache()
    stats = cache.get_stats()
    stats["enabled"] = True
    return stats


@router.delete("/v1/cache")
async def clear_cache(
    settings: Annotated[Settings, Depends(get_settings)],
    _auth: Annotated[None, Depends(verify_api_key)] = None,
) -> dict:
    """Clear the audio cache.

    Removes all cached audio entries.
    """
    if not settings.cache_enabled:
        return {"cleared": False, "reason": "Cache is disabled"}

    cache = get_audio_cache()
    stats_before = cache.get_stats()
    cache.clear()

    logger.info("Cache cleared", entries_removed=stats_before["entries"])

    return {
        "cleared": True,
        "entries_removed": stats_before["entries"],
        "memory_freed_mb": stats_before["memory_mb"],
    }
