# syntax=docker/dockerfile:1

# GPU-enabled Dockerfile for dndsandbox-tts
# Supports NVIDIA CUDA for accelerated TTS inference

# =============================================================================
# Single stage build using CUDA base
# =============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

# Install Python, pip, and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install PyTorch with CUDA support first
RUN uv pip install --system --no-cache \
    torch --index-url https://download.pytorch.org/whl/cu124

# Install Bark
RUN uv pip install --system --no-cache git+https://github.com/suno-ai/bark.git

# Install application and dependencies (uvicorn, fastapi, etc.)
RUN uv pip install --system --no-cache .

# Create cache directories
RUN mkdir -p /root/.cache/suno/bark_v0 \
    && mkdir -p /root/.cache/huggingface

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    XDG_CACHE_HOME=/root/.cache \
    DEVICE=auto \
    PRELOAD_MODELS=true \
    HOST=0.0.0.0 \
    PORT=8100

EXPOSE 8100

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8100/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "dndsandbox_tts.main:app", "--host", "0.0.0.0", "--port", "8100"]
