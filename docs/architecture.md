# Architecture

This project exposes one OpenAI-like transcription API while dispatching work to different local ASR backends.

## Runtime split

### Main API process

The main process owns:

- FastAPI routes
- API key auth
- upload handling
- SSE orchestration
- realtime WebSocket state
- model scheduling and capacity decisions

Relevant files:

- [lazy_whisper_api/app.py](/home/wandeber/codex-playground/lazy_whisper_api/app.py)
- [lazy_whisper_api/config.py](/home/wandeber/codex-playground/lazy_whisper_api/config.py)
- [lazy_whisper_api/model_manager.py](/home/wandeber/codex-playground/lazy_whisper_api/model_manager.py)
- [lazy_whisper_api/streaming.py](/home/wandeber/codex-playground/lazy_whisper_api/streaming.py)
- [lazy_whisper_api/realtime.py](/home/wandeber/codex-playground/lazy_whisper_api/realtime.py)

### Whisper backend

Whisper-family models live directly in the main process through `faster-whisper`.

Relevant file:

- [lazy_whisper_api/backends.py](/home/wandeber/codex-playground/lazy_whisper_api/backends.py)

### Qwen backend

Qwen-family models live in a separate Python runtime with a separate dependency set.

Relevant files:

- [lazy_whisper_api/backends.py](/home/wandeber/codex-playground/lazy_whisper_api/backends.py)
- [lazy_whisper_api/qwen_worker.py](/home/wandeber/codex-playground/lazy_whisper_api/qwen_worker.py)
- [setup-qwen-runtime.sh](/home/wandeber/codex-playground/setup-qwen-runtime.sh)
- [requirements-qwen-cu126.txt](/home/wandeber/codex-playground/requirements-qwen-cu126.txt)

The main API talks to each Qwen worker over line-delimited JSON-RPC on `stdin/stdout`.

## Model abstraction

Each configured model has:

- `family`
- `backend`
- `source`
- `device`
- `compute_type`
- `idle_seconds`
- `capabilities`
- `gpu_memory_reservation_mb`
- `max_concurrent_requests`

That configuration is produced in [lazy_whisper_api/config.py](/home/wandeber/codex-playground/lazy_whisper_api/config.py) from `ASR_*` variables, with `WHISPER_*` preserved as legacy aliases.

## Request flow

### Batch HTTP

1. The client calls `POST /v1/audio/transcriptions` or `POST /v1/audio/translations`.
2. The upload is streamed to a temp file in chunks.
3. Request parameters are validated and model aliases are resolved.
4. [ModelManager](/home/wandeber/codex-playground/lazy_whisper_api/model_manager.py) acquires a lease.
5. The selected backend transcribes in a worker thread, not on the event loop.
6. If timestamps were requested and the backend did not return them directly, alignment runs as a second step.
7. The response is rendered as `json`, `text`, `srt`, `vtt`, or `verbose_json`.

### SSE streaming

For `stream=true` on `POST /v1/audio/transcriptions`:

- Whisper uses native segment iteration from `faster-whisper`
- Qwen uses synthetic streaming based on progressive re-transcription of decoded PCM
- the public SSE surface is the same in both cases:
  - `transcript.text.delta`
  - `transcript.text.done`
  - `error`

### Realtime

For `WS /v1/realtime`:

- the socket itself does not reserve a model
- a lease begins only when a turn actually starts transcribing
- manual and VAD-driven commits are supported
- partials are produced by periodic re-transcription of the buffered PCM
- finals may add timestamp alignment before `completed`

Supported client events:

- `session.update`
- `input_audio_buffer.append`
- `input_audio_buffer.commit`
- `input_audio_buffer.clear`

Supported server events:

- `session.created`
- `session.updated`
- `input_audio_buffer.committed`
- `conversation.item.input_audio_transcription.delta`
- `conversation.item.input_audio_transcription.completed`
- `error`

## Capacity model

### CPU

- limit is count-based
- default: 1 loaded CPU model

### GPU

- limit is VRAM-budget-based
- each model has a configured reservation
- default machine budget: `8192 MB`

Eviction rule:

- if a new model needs space, the oldest idle model on the same device family is unloaded first
- if nothing idle can be evicted, the request fails

Concurrency rule:

- each model has its own max active request count
- HTTP returns `429` on saturation
- realtime emits `error` and keeps the socket open

## First-use Qwen behavior

The first Qwen request is more expensive than a Whisper request because it may need to:

- download model files from Hugging Face
- start the isolated worker process
- load weights on GPU

Once loaded, Qwen follows the same idle-unload policy as the other models.
