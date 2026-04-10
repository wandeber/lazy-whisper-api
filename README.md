# Lazy Whisper API

OpenAI-compatible local transcription API built on top of `faster-whisper`.

It is designed for a home lab or workstation where the API should stay up, but models should wake up only on demand:

- `turbo` loads on demand on GPU and unloads after an idle timeout
- `large-v3` loads on demand on CPU and unloads immediately after each request
- `distil-multi4` loads on demand on GPU and unloads after an idle timeout
- the API itself can be managed as a persistent `systemd --user` service

## Features

- OpenAI-style endpoints:
  - `POST /v1/audio/transcriptions`
  - `POST /v1/audio/translations`
  - `GET /v1/models`
- API key authentication on `/v1/*`
- Lazy model loading and timed unloading
- Per-model device selection
- Public-health endpoint at `/healthz`
- Local `systemd --user` start/stop/status/logs workflow

## Models

Current aliases:

- `whisper-1` -> `turbo`
- `turbo` -> OpenAI Whisper Turbo
- `large-v3` -> OpenAI Whisper Large v3
- `distil` -> `distil-multi4`
- `distil-multi4` -> local distilled multilingual checkpoint for `en`, `es`, `fr`, `de`

Current default behavior:

- `turbo`: GPU, unload after 90 minutes idle
- `large-v3`: CPU, unload immediately after request
- `distil-multi4`: GPU, unload after 90 minutes idle

## Repo Layout

- [whisper_api.py](/home/wandeber/codex-playground/whisper_api.py): FastAPI application
- [whisper-api.sh](/home/wandeber/codex-playground/whisper-api.sh): manual launcher
- [whisper-service.sh](/home/wandeber/codex-playground/whisper-service.sh): persistent service controller
- [.env.example](/home/wandeber/codex-playground/.env.example): safe example config
- [whisper-large.sh](/home/wandeber/codex-playground/whisper-large.sh): direct CLI launcher for `large-v3`

## Quick Start

1. Install dependencies.

For the base Python dependencies:

```bash
make install
```

For a machine like this one with NVIDIA GPU support:

```bash
make install-gpu
```

For CPU-only:

```bash
make install-cpu
```

2. Create your local config:

```bash
cp .env.example .env
```

3. Edit `.env` and set at least:

```bash
WHISPER_API_KEY=change-me
```

4. Start the API:

```bash
./whisper-service.sh start
```

5. Check health:

```bash
curl http://127.0.0.1:43556/healthz
```

## Example Request

```bash
curl -X POST http://127.0.0.1:43556/v1/audio/transcriptions \
  -H "Authorization: Bearer your-api-key" \
  -F file=@audio.mp3 \
  -F model=whisper-1 \
  -F response_format=json
```

With the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:43556/v1",
    api_key="your-api-key",
)

with open("audio.mp3", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )

print(transcript.text)
```

## Configuration

All runtime configuration lives in `.env`.

Important variables:

- `WHISPER_API_HOST`
- `WHISPER_API_PORT`
- `WHISPER_API_KEY`
- `WHISPER_DEFAULT_MODEL`
- `WHISPER_MODEL_ALIAS_MAP`
- `WHISPER_MODEL_SOURCE_MAP`
- `WHISPER_MODEL_DEVICE_MAP`
- `WHISPER_MODEL_COMPUTE_TYPE_MAP`
- `WHISPER_MODEL_IDLE_SECONDS_MAP`
- `WHISPER_MODEL_VAD_MAP`
- `WHISPER_CPU_THREADS`
- `WHISPER_LOG_LEVEL`

After editing `.env`, apply changes with:

```bash
./whisper-service.sh restart
```

## Dependency Setup

This repo now includes a central Python project manifest:

- [pyproject.toml](/home/wandeber/codex-playground/pyproject.toml)
- [Makefile](/home/wandeber/codex-playground/Makefile)
- [requirements-common.txt](/home/wandeber/codex-playground/requirements-common.txt)
- [requirements-gpu-cu126.txt](/home/wandeber/codex-playground/requirements-gpu-cu126.txt)
- [requirements-cpu.txt](/home/wandeber/codex-playground/requirements-cpu.txt)

Closest equivalent to `package.json` here is `pyproject.toml`.

Practical commands:

```bash
make install
make install-gpu
make install-cpu
```

The `requirements-*.txt` files are still kept as simple export-friendly manifests, but the main source of truth for the repo is now [pyproject.toml](/home/wandeber/codex-playground/pyproject.toml).

## Service Control

```bash
./whisper-service.sh start
./whisper-service.sh stop
./whisper-service.sh restart
./whisper-service.sh status
./whisper-service.sh logs
```

Behavior:

- `start` enables and starts the API
- `stop` disables and stops the API
- if enabled, it stays enabled across reboots
- with `systemd --user`, it normally starts when the user session starts
- to start even before login, run `sudo loginctl enable-linger $USER`

## Distilled Multilingual Model

This repository does not commit local model weights.

The sample config expects a converted local model at:

```bash
./models/distil-multi4-ct2
```

To build it locally:

```bash
./.venv/bin/ct2-transformers-converter \
  --model bofenghuang/whisper-large-v3-distil-multi4-v0.2 \
  --output_dir models/distil-multi4-ct2 \
  --copy_files tokenizer.json preprocessor_config.json tokenizer_config.json generation_config.json vocab.json merges.txt added_tokens.json special_tokens_map.json \
  --quantization float16
```

## What Is Ignored

The repository ignores:

- `.env`
- `.venv/`
- local caches
- local converted models
- Python cache files

The committed [bin/ffmpeg](/home/wandeber/codex-playground/bin/ffmpeg) file is only a small wrapper script.
The real downloaded ffmpeg binary lives inside the virtual environment and is not committed.

That keeps secrets, large binaries, and machine-specific data out of git.

## License

MIT. See [LICENSE](/home/wandeber/codex-playground/LICENSE).
