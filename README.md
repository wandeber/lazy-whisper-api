# Lazy Whisper API

OpenAI-compatible local ASR API with lazy-loaded models, streaming transcription, realtime transcription, and multi-family backend support.

The project started as a `faster-whisper` wrapper and now works as a single public API in front of multiple model families:

- Whisper-family models run in the main API process through `faster-whisper`
- Qwen-family models run in an isolated sidecar runtime with their own `.venv`
- models wake up on demand and go back to sleep after configurable idle timeouts

## What it exposes

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`
- `WS /v1/realtime`

OpenAI-like features:

- API key auth
- `stream=true` on completed-audio transcriptions via SSE
- realtime transcription-only WebSocket sessions
- `json`, `text`, `srt`, `vtt`, and `verbose_json` response formats

## Model families

### Whisper family

- `whisper-1` -> `turbo`
- `turbo`
- `large-v3`
- `distil` -> `distil-multi4`
- `distil-multi4`

Backend:

- `faster-whisper`

### Qwen family

- `qwen-0.6b` -> `qwen3-asr-0.6b`
- `qwen-1.7b` -> `qwen3-asr-1.7b`
- `qwen3-asr-0.6b`
- `qwen3-asr-1.7b`

Backend:

- isolated `qwen-asr` worker runtime

Notes:

- Qwen is transcription-only in this project for now
- Qwen timestamps are produced with `Qwen/Qwen3-ForcedAligner-0.6B`
- on this GTX 1070 setup, Qwen is configured for `float16` on GPU and the aligner stays on CPU by default

## Resource policy

The API is intentionally conservative on this machine:

- max 1 loaded CPU model at once
- GPU scheduling uses a VRAM reservation budget instead of a fixed “N models” rule
- each model has its own max concurrent transcription limit
- if a new model needs room, the oldest idle model on the same device family is evicted first
- if the device is busy and nothing idle can be evicted, HTTP returns `503` and realtime emits an `error`

Default reservations in [.env.example](/home/wandeber/codex-playground/.env.example):

- `turbo=5200`
- `distil-multi4=4200`
- `qwen3-asr-0.6b=6500`
- `qwen3-asr-1.7b=7800`
- `large-v3=0` because it runs on CPU

## Repo layout

- [whisper_api.py](/home/wandeber/codex-playground/whisper_api.py): compatibility entrypoint for `uvicorn`
- [lazy_whisper_api/app.py](/home/wandeber/codex-playground/lazy_whisper_api/app.py): route wiring and FastAPI assembly
- [lazy_whisper_api/config.py](/home/wandeber/codex-playground/lazy_whisper_api/config.py): `ASR_*` config parsing with `WHISPER_*` legacy aliases
- [lazy_whisper_api/backends.py](/home/wandeber/codex-playground/lazy_whisper_api/backends.py): runtime abstraction plus Whisper and Qwen backend adapters
- [lazy_whisper_api/model_manager.py](/home/wandeber/codex-playground/lazy_whisper_api/model_manager.py): leases, idle unloads, and capacity enforcement
- [lazy_whisper_api/transcription.py](/home/wandeber/codex-playground/lazy_whisper_api/transcription.py): validation, uploads, decoding, and timestamp alignment helpers
- [lazy_whisper_api/streaming.py](/home/wandeber/codex-playground/lazy_whisper_api/streaming.py): SSE transcription streaming
- [lazy_whisper_api/realtime.py](/home/wandeber/codex-playground/lazy_whisper_api/realtime.py): realtime transcription-only WebSocket server
- [lazy_whisper_api/qwen_worker.py](/home/wandeber/codex-playground/lazy_whisper_api/qwen_worker.py): isolated Qwen sidecar worker
- [whisper-api.sh](/home/wandeber/codex-playground/whisper-api.sh): launcher
- [whisper-service.sh](/home/wandeber/codex-playground/whisper-service.sh): persistent `systemd --user` controller
- [setup-qwen-runtime.sh](/home/wandeber/codex-playground/setup-qwen-runtime.sh): creates and installs the isolated Qwen runtime
- [requirements-qwen-cu126.txt](/home/wandeber/codex-playground/requirements-qwen-cu126.txt): Qwen runtime dependencies for CUDA 12.6
- [docs/architecture.md](/home/wandeber/codex-playground/docs/architecture.md): deeper runtime notes
- [docs/benchmarks.md](/home/wandeber/codex-playground/docs/benchmarks.md): machine-specific benchmark notes for the classic endpoint

## Benchmark Notes

There is a small benchmark snapshot in `docs/benchmarks.md`.
It currently covers:

- model comparison on the classic non-streaming transcription endpoint
- endpoint-style comparison for the same `turbo` model across classic JSON, classic SSE, and stable realtime manual-commit flows

Those numbers are machine-specific and should be read as relative guidance, not as universal guarantees.

## Installation

Install the main API runtime:

```bash
make install-gpu
```

Install the isolated Qwen runtime:

```bash
make install-qwen-runtime
```

Create local config:

```bash
cp .env.example .env
```

Set at least:

```bash
ASR_API_KEY=change-me
```

Start the API:

```bash
./whisper-service.sh start
```

## Configuration

Primary config lives in [.env](/home/wandeber/codex-playground/.env). The project now prefers `ASR_*` variables, but still accepts the old `WHISPER_*` names as legacy aliases.

Important settings:

- `ASR_API_HOST`
- `ASR_API_PORT`
- `ASR_API_KEY`
- `ASR_DEFAULT_MODEL`
- `ASR_MODEL_ALIAS_MAP`
- `ASR_MODEL_SOURCE_MAP`
- `ASR_MODEL_FAMILY_MAP`
- `ASR_MODEL_BACKEND_MAP`
- `ASR_MODEL_DEVICE_MAP`
- `ASR_MODEL_COMPUTE_TYPE_MAP`
- `ASR_MODEL_IDLE_SECONDS_MAP`
- `ASR_MODEL_CAPABILITIES_MAP`
- `ASR_MODEL_GPU_MEMORY_RESERVATION_MB_MAP`
- `ASR_MODEL_MAX_CONCURRENT_REQUESTS_MAP`
- `ASR_MODEL_ALIGNER_SOURCE_MAP`
- `ASR_MODEL_ALIGNER_DEVICE_MAP`
- `ASR_MODEL_ALIGNER_DTYPE_MAP`
- `ASR_FAMILY_RUNTIME_PYTHON_MAP`
- `ASR_GPU_MEMORY_BUDGET_MB`

Shell note:

- map values that contain `|` are quoted in [.env.example](/home/wandeber/codex-playground/.env.example) on purpose, because the launcher sources the file from Bash

## Example requests

Health:

```bash
curl http://127.0.0.1:43556/healthz \
  -H "Authorization: Bearer your-api-key"
```

Classic transcription:

```bash
curl -X POST http://127.0.0.1:43556/v1/audio/transcriptions \
  -H "Authorization: Bearer your-api-key" \
  -F file=@audio.mp3 \
  -F model=whisper-1
```

Qwen transcription:

```bash
curl -X POST http://127.0.0.1:43556/v1/audio/transcriptions \
  -H "Authorization: Bearer your-api-key" \
  -F file=@audio.mp3 \
  -F model=qwen-0.6b
```

Completed-audio streaming:

```bash
curl --no-buffer -X POST http://127.0.0.1:43556/v1/audio/transcriptions \
  -H "Authorization: Bearer your-api-key" \
  -F file=@audio.mp3 \
  -F model=whisper-1 \
  -F stream=true \
  -F 'timestamp_granularities[]=segment'
```

Realtime transcription:

```python
import asyncio
import base64
import json
import websockets

API_KEY = "your-api-key"

async def main():
    async with websockets.connect(
        "ws://127.0.0.1:43556/v1/realtime",
        additional_headers={"Authorization": f"Bearer {API_KEY}"},
    ) as ws:
        print(json.loads(await ws.recv()))

        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "transcription": {"model": "qwen-0.6b", "language": "es"},
                        "turn_detection": None,
                    }
                },
            },
        }))
        print(json.loads(await ws.recv()))

        pcm = b"\x00\x00" * 24000
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode("ascii"),
        }))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        while True:
            event = json.loads(await ws.recv())
            print(event)
            if event["type"] == "conversation.item.input_audio_transcription.completed":
                break

asyncio.run(main())
```

## Qwen first-use behavior

The first Qwen request may take a while because it has to:

- download the model from Hugging Face if it is not cached yet
- start the isolated worker
- load the model onto GPU

After that first load, Qwen follows the same lazy policy as the rest of the API: it stays warm for its configured idle window and unloads afterwards.

## Tests

Run the automated suite:

```bash
./.venv/bin/python -m pytest -q -s
```

Current automated coverage includes:

- auth on HTTP and realtime
- classic transcription
- SSE streaming
- realtime transcription
- Qwen model exposure on `/v1/models`
- config compatibility for `ASR_*` and legacy `WHISPER_*`
- GPU-budget scheduler behavior

## License

MIT. See [LICENSE](/home/wandeber/codex-playground/LICENSE).
