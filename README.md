# Lazy Whisper API

OpenAI-compatible local ASR API with lazy-loaded models, streaming transcription, realtime transcription, and multi-family backend support.

The project started as a `faster-whisper` wrapper and now works as a single public API in front of multiple model families:

- Whisper-family models run in the main API process through `faster-whisper`
- Qwen-family models run in an isolated sidecar runtime with their own `.venv`
- models wake up on demand and go back to sleep after configurable idle timeouts

## Quick start on Apple Silicon

This is the recommended native setup for a local Apple Silicon Mac. Docker is
not required, so Qwen keeps direct access to MLX and Metal.

### 1. Prepare Hugging Face access

The pyannote diarization model is gated. Before the first installation:

1. Accept the conditions on
   [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
2. Create a Hugging Face token with read access.

The token is needed only to download or repair the local model. The API removes
it from the environment before starting any runtime process.

### 2. Create the local configuration

```bash
cp .env.apple.example .env
chmod 600 .env
```

Edit `.env` and add the Hugging Face token:

```bash
ASR_DIARIZATION_SETUP_HF_TOKEN=hf_your_read_token
```

The example is immediately usable with `ASR_API_KEY=change-me`. Changing that
local API key is recommended, especially if the service will ever listen beyond
`localhost`.

The Apple Silicon example already enables diarization and points every runtime
at its local `.venv` and model directory. Keep `.env` private; it is ignored by
Git and the setup command enforces mode `0600` on every run.

### 3. Install everything

```bash
make setup-macos
```

That single command:

- checks that the machine is an Apple Silicon Mac and that `.env` is usable
- installs Homebrew FFmpeg when its shared libraries are missing
- finds Python 3.11+ or installs Homebrew Python 3.12
- creates the main API, Qwen MLX, and pyannote virtual environments
- installs their pinned dependencies
- downloads the gated diarization model to `ASR_DIARIZATION_MODEL_PATH`
- verifies that the downloaded pyannote pipeline can load with networking disabled

The command is safe to rerun. A partial model download is resumed, while an
already-verified installation is checked locally without downloading it again.
Qwen model weights remain lazy and are downloaded on the first request for each
configured Qwen model.

### 4. Start the API

```bash
make run
```

The foreground server listens on `http://localhost:43556`. Stop it with
`Ctrl+C`. Subsequent starts only require `make run`; repeat `make setup-macos`
only after changing dependencies, deleting a runtime, or repairing a model.

Check it with the API key configured in `.env`:

```bash
curl http://localhost:43556/healthz \
  -H "Authorization: Bearer your-local-api-key"
```

Other local applications can use `http://localhost:43556/v1` as their base URL.
The `make start`, `stop`, `status`, and `logs` targets use `systemd --user` and
are intended for Linux; the native macOS foreground command is `make run`.

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
- optional local speaker diarization for `verbose_json` transcription responses

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

- isolated worker runtime selected by config:
  - `qwen-worker` for NVIDIA/CUDA through `qwen-asr`
  - `qwen-mlx-worker` for Apple Silicon through `mlx-qwen3-asr`

Notes:

- Qwen is transcription-only in this project for now
- Qwen timestamps are produced with `Qwen/Qwen3-ForcedAligner-0.6B`; aligned
  words are grouped into readable segments for `verbose_json`, `srt`, and `vtt`
- clients use the same model aliases on CUDA and Apple Silicon: `qwen-0.6b` and `qwen-1.7b`

### Speaker diarization

Speaker diarization is optional and runs in its own isolated pyannote runtime.
It is currently supported only for non-streaming `/v1/audio/transcriptions`
requests with `response_format=verbose_json`.

Default backend:

- `pyannote/speaker-diarization-community-1`

Notes:

- the model is gated on Hugging Face, so setup requires accepting the model
  conditions and providing `ASR_DIARIZATION_SETUP_HF_TOKEN` in `.env` or an
  exported `HF_TOKEN` while downloading it
- serving is strictly local: the worker loads `ASR_DIARIZATION_MODEL_PATH`,
  enables Hugging Face and Transformers offline modes, disables telemetry, and
  receives neither the setup token nor other credentials from the API process
- diarization labels distinguish local speakers such as `SPEAKER_00`; they do
  not identify real people by name
- CPU is the conservative default on Apple Silicon; a 24 GB Mac can keep the
  diarization runtime isolated and unload it after the configured idle period

## Resource policy

The API is intentionally conservative on this machine:

- max 1 loaded CPU model at once
- GPU scheduling uses a VRAM reservation budget instead of a fixed “N models” rule
- each model has its own max concurrent transcription limit
- diarization runs one job at a time and uses its own idle-unload timer
- if a new model needs room, the oldest idle model on the same device family is evicted first
- if the device is busy and nothing idle can be evicted, HTTP returns `503` and realtime emits an `error`

Default reservations in [.env.example](.env.example):

- `turbo=5200`
- `distil-multi4=4200`
- `qwen3-asr-0.6b=6500`
- `qwen3-asr-1.7b=7800`
- `large-v3=0` because it runs on CPU

## Repo layout

- [whisper_api.py](whisper_api.py): compatibility entrypoint for `uvicorn`
- [lazy_whisper_api/app.py](lazy_whisper_api/app.py): route wiring and FastAPI assembly
- [lazy_whisper_api/config.py](lazy_whisper_api/config.py): `ASR_*` config parsing with `WHISPER_*` legacy aliases
- [lazy_whisper_api/backends.py](lazy_whisper_api/backends.py): runtime abstraction plus Whisper and Qwen backend adapters
- [lazy_whisper_api/worker_protocol.py](lazy_whisper_api/worker_protocol.py): shared JSON-line subprocess transport for isolated runtimes
- [lazy_whisper_api/model_manager.py](lazy_whisper_api/model_manager.py): leases, idle unloads, and capacity enforcement
- [lazy_whisper_api/transcription.py](lazy_whisper_api/transcription.py): validation, uploads, decoding, and timestamp alignment helpers
- [lazy_whisper_api/diarization.py](lazy_whisper_api/diarization.py): diarization validation, reservation, and worker lifecycle
- [lazy_whisper_api/diarization_types.py](lazy_whisper_api/diarization_types.py): shared diarization result value objects
- [lazy_whisper_api/speaker_attribution.py](lazy_whisper_api/speaker_attribution.py): efficient timestamp reconciliation and readable speaker grouping
- [lazy_whisper_api/streaming.py](lazy_whisper_api/streaming.py): SSE transcription streaming
- [lazy_whisper_api/realtime.py](lazy_whisper_api/realtime.py): realtime transcription-only WebSocket server
- [lazy_whisper_api/qwen_worker.py](lazy_whisper_api/qwen_worker.py): isolated Qwen CUDA sidecar worker
- [lazy_whisper_api/qwen_mlx_worker.py](lazy_whisper_api/qwen_mlx_worker.py): isolated Qwen Apple Silicon sidecar worker
- [whisper-api.sh](whisper-api.sh): launcher
- [whisper-service.sh](whisper-service.sh): persistent `systemd --user` controller
- [setup-qwen-runtime.sh](setup-qwen-runtime.sh): creates and installs the isolated Qwen CUDA runtime
- [setup-qwen-mlx-runtime.sh](setup-qwen-mlx-runtime.sh): creates and installs the isolated Qwen Apple Silicon runtime
- [setup-macos.sh](setup-macos.sh): performs the complete native Apple Silicon setup and preflight
- [setup-diarization-runtime.sh](setup-diarization-runtime.sh): installs pyannote, downloads the gated model to local disk, and verifies offline loading
- [smoke-test-diarization.sh](smoke-test-diarization.sh): generates a two-voice macOS sample and verifies local multi-speaker detection
- [requirements-qwen-cu126.txt](requirements-qwen-cu126.txt): Qwen runtime dependencies for CUDA 12.6
- [requirements-qwen-mlx.txt](requirements-qwen-mlx.txt): Qwen runtime dependencies for Apple Silicon
- [requirements-diarization.txt](requirements-diarization.txt): pyannote diarization runtime dependencies
- [docs/architecture.md](docs/architecture.md): deeper runtime notes
- [docs/benchmarks.md](docs/benchmarks.md): machine-specific benchmark notes for the classic endpoint

## Benchmark Notes

There is a small benchmark snapshot in `docs/benchmarks.md`.
It currently covers:

- model comparison on the classic non-streaming transcription endpoint
- endpoint-style comparison for the same `turbo` model across classic JSON, classic SSE, and stable realtime manual-commit flows

Those numbers are machine-specific and should be read as relative guidance, not as universal guarantees.

## Manual installation and other platforms

Apple Silicon users should normally follow the one-command quick start above.
The individual targets remain available for repairs or custom installations:

```bash
make install-macos
make install-qwen-mlx-runtime
make install-diarization-runtime
```

On NVIDIA/CUDA machines:

```bash
make install-gpu
make install-qwen-cuda-runtime
cp .env.cuda.example .env
```

For a one-off diarization download without storing the token in `.env`:

```bash
read -rsp "Hugging Face token: " HF_TOKEN && export HF_TOKEN && echo
make install-diarization-runtime
unset HF_TOKEN
```

The setup-only `.env` variable and its historical aliases are removed before
starting any API or worker process. A one-off `HF_TOKEN` should be unset after
installation as shown above; it remains a supported credential for private ASR
models when deliberately supplied at runtime.

The legacy Qwen target points at the CUDA runtime:

```bash
make install-qwen-runtime
```

Verify the local model directly, without starting the HTTP API:

```bash
make smoke-diarization
```

## Configuration

Primary config lives in the ignored local `.env` file. The project now prefers
`ASR_*` variables, but still accepts the old `WHISPER_*` names as legacy aliases.

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
- `ASR_MODEL_RUNTIME_PYTHON_MAP`
- `ASR_DIARIZATION_ENABLED`
- `ASR_DIARIZATION_BACKEND`
- `ASR_DIARIZATION_MODEL_ID`
- `ASR_DIARIZATION_MODEL_PATH`
- `ASR_DIARIZATION_DEVICE`
- `ASR_DIARIZATION_IDLE_SECONDS`
- `ASR_DIARIZATION_RUNTIME_PYTHON`
- `ASR_DIARIZATION_STARTUP_TIMEOUT_SECONDS`
- `ASR_DIARIZATION_REQUEST_TIMEOUT_SECONDS`
- `ASR_DIARIZATION_SETUP_HF_TOKEN` (setup only; stripped before serving)
- `ASR_GPU_MEMORY_BUDGET_MB`

Shell note:

- map values that contain `|` are quoted in [.env.example](.env.example) on purpose, because the launcher sources the file from Bash

## Example requests

Health:

```bash
curl http://localhost:43556/healthz \
  -H "Authorization: Bearer your-api-key"
```

Classic transcription:

```bash
curl -X POST http://localhost:43556/v1/audio/transcriptions \
  -H "Authorization: Bearer your-api-key" \
  -F file=@audio.mp3 \
  -F model=whisper-1
```

Qwen transcription:

```bash
curl -X POST http://localhost:43556/v1/audio/transcriptions \
  -H "Authorization: Bearer your-api-key" \
  -F file=@audio.mp3 \
  -F model=qwen-1.7b
```

Use `qwen-1.7b` for best transcription quality and `qwen-0.6b` when startup
time, memory, or disk use matter more than the last bit of accuracy.

Speaker diarization:

```bash
curl -X POST http://localhost:43556/v1/audio/transcriptions \
  -H "Authorization: Bearer your-api-key" \
  -F file=@audio.mp3 \
  -F model=whisper-1 \
  -F response_format=verbose_json \
  -F diarize=true \
  -F min_speakers=1 \
  -F max_speakers=4
```

The response adds `speaker` to timestamped segments and words. The top-level
`diarization` object includes the detected speaker count and labels, raw turns,
processing time, and readable `speaker_segments` grouped by speaker changes.

Completed-audio streaming:

```bash
curl --no-buffer -X POST http://localhost:43556/v1/audio/transcriptions \
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
        "ws://localhost:43556/v1/realtime",
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

## Setup troubleshooting

- **The gated download returns 401 or 403:** accept the model conditions with
  the same Hugging Face account that created the token, confirm the token has
  read access, update `ASR_DIARIZATION_SETUP_HF_TOKEN`, and rerun
  `make setup-macos`.
- **A download was interrupted:** rerun `make setup-macos`. Hugging Face resumes
  partial files and the setup does not mark the model ready until an offline
  pipeline load succeeds.
- **TorchCodec reports missing FFmpeg libraries:** rerun `make setup-macos`.
  The preflight installs Homebrew FFmpeg, whose shared libraries are required
  in addition to the repository's command-line wrapper.
- **`/healthz` reports `model_missing`:** rerun `make setup-macos` and inspect
  its final offline-verification result.
- **The first Qwen request takes longer:** Qwen weights are intentionally lazy;
  this first request downloads and loads the selected Qwen model.

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
- local diarization request validation, offline isolation, lifecycle handling,
  speaker reconciliation, and response rendering
- shared worker-protocol timeout, error, and shutdown behavior
- deterministic FastAPI lifespan cleanup for all runtime managers

## License

MIT. See [LICENSE](LICENSE).
