# dndsandbox-tts

Local TTS service for speech synthesis using Bark. Provides an OpenAI-compatible API.

## Quick Start

```bash
# Install
pip install -e .

# Run
dndsandbox-tts
```

The server starts at `http://localhost:8100`.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/speech` | POST | Generate speech from text |
| `/v1/audio/speech/stream` | POST | Stream speech chunks as generated |
| `/v1/models` | GET | List available models |
| `/v1/voices` | GET | List voices for a model |
| `/v1/cache/stats` | GET | Cache statistics |
| `/v1/cache` | DELETE | Clear cache |
| `/health` | GET | Health check |

## Usage

**Standard request:**
```bash
curl -X POST http://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "voice": "v2/en_speaker_6"}' \
  --output speech.mp3
```

**Streaming request** (for long text or low latency):
```bash
curl -X POST http://localhost:8100/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{"input": "Your long text here...", "voice": "v2/en_speaker_6"}' \
  --output speech.mp3
```

The streaming endpoint returns audio chunks as they're generated, reducing time-to-first-audio for long texts.

## Configuration

Set via environment variables or `.env` file:

```bash
HOST=0.0.0.0
PORT=8100
DEVICE=auto          # auto, cuda, cpu
USE_FP16=true        # Half-precision on GPU
CACHE_ENABLED=true
CACHE_MAX_MEMORY_MB=500
LOG_LEVEL=INFO
```

See [.env.example](.env.example) for all options.

## Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## License

MIT
