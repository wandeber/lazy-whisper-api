# Architecture

This project exposes a local transcription API that looks like the OpenAI audio API, while keeping model usage lazy and resource-aware.

## Main Pieces

- [whisper_api.py](/home/wandeber/codex-playground/whisper_api.py): compatibility entrypoint used by `uvicorn whisper_api:app`
- [lazy_whisper_api/app.py](/home/wandeber/codex-playground/lazy_whisper_api/app.py): FastAPI app assembly and route wiring
- [lazy_whisper_api/config.py](/home/wandeber/codex-playground/lazy_whisper_api/config.py): environment parsing and config validation
- [lazy_whisper_api/auth.py](/home/wandeber/codex-playground/lazy_whisper_api/auth.py): fixed API key auth dependency
- [lazy_whisper_api/model_manager.py](/home/wandeber/codex-playground/lazy_whisper_api/model_manager.py): lazy loading, leases, per-device capacity limits, and idle unload timers
- [lazy_whisper_api/transcription.py](/home/wandeber/codex-playground/lazy_whisper_api/transcription.py): request validation, chunked temp-file handling, and Faster Whisper execution
- [lazy_whisper_api/streaming.py](/home/wandeber/codex-playground/lazy_whisper_api/streaming.py): `stream=true` SSE orchestration for completed audio uploads
- [lazy_whisper_api/realtime.py](/home/wandeber/codex-playground/lazy_whisper_api/realtime.py): transcription-only WebSocket sessions, PCM buffering, VAD, and realtime event emission
- [lazy_whisper_api/responses.py](/home/wandeber/codex-playground/lazy_whisper_api/responses.py): response rendering for `json`, `text`, `srt`, `vtt`, and `verbose_json`

## Request Flow

1. [whisper-api.sh](/home/wandeber/codex-playground/whisper-api.sh) loads [.env](/home/wandeber/codex-playground/.env) and starts `uvicorn`.
2. `uvicorn` imports [whisper_api.py](/home/wandeber/codex-playground/whisper_api.py), which re-exports the app created in [lazy_whisper_api/app.py](/home/wandeber/codex-playground/lazy_whisper_api/app.py).
3. The app loads validated settings from environment variables.
4. Protected `/v1/*` routes and `/healthz` enforce the configured API key.
5. Audio requests are normalized in [lazy_whisper_api/transcription.py](/home/wandeber/codex-playground/lazy_whisper_api/transcription.py).
6. The [ModelManager](/home/wandeber/codex-playground/lazy_whisper_api/model_manager.py) loads the requested model only when needed and enforces per-device and per-model limits.
7. Faster Whisper runs in a threadpool so the async request loop is not blocked by long transcriptions.
8. After transcription, the response is rendered in the requested output format.
9. Idle model timers decide whether the loaded model stays warm or unloads immediately.

## Streaming and Realtime

The app now exposes two OpenAI-like streaming surfaces:

- `POST /v1/audio/transcriptions` with `stream=true` for completed files
- `WS /v1/realtime` for audio that is still arriving

### Completed-audio streaming

- The upload is still copied to a temp file in chunks, so large files do not spike RAM.
- A model lease is acquired before the SSE response starts, so resource-limit failures still surface as normal HTTP errors.
- Faster Whisper segment iteration runs in a worker thread.
- The worker pushes SSE events back into the async response stream:
- `transcript.text.delta`
- `transcript.text.done`
- `error`
- `delta` events are segment-level and can optionally include segment timestamps.
- `POST /v1/audio/translations` deliberately rejects `stream=true` with `400` in this version.

### Realtime transcription

- Realtime is transcription-only. There are no assistant messages, tool calls, or bot responses.
- Authentication accepts either:
- `Authorization: Bearer ...`
- `?api_key=...`
- Only `audio/pcm` at `24000` Hz is accepted in v1.
- Audio arrives as base64 PCM16 mono chunks via `input_audio_buffer.append`.
- Session configuration is updated with `session.update`.
- Turns can be closed explicitly with `input_audio_buffer.commit` or automatically with `server_vad`.
- Partial transcription runs periodically in a background worker and emits:
- `conversation.item.input_audio_transcription.delta`
- Final transcription emits:
- `conversation.item.input_audio_transcription.completed`
- A realtime socket does not reserve a model by itself; the lease starts only when audio for a turn actually begins transcription.
- Because partial work and final work overlap slightly, a turn may emit one last `delta` after `input_audio_buffer.committed` and before `completed`.

## Model Lifecycle

- `large-v3` defaults to a 10-minute idle timeout and can still be set to `idle_seconds=0` if you want immediate unloads.
- GPU models such as `turbo` or `distil-multi4` can stay warm for a configurable idle window.
- By default, at most 1 CPU model and 2 GPU models can remain loaded at the same time.
- By default, each model accepts at most 2 active transcriptions at once.
- When a new model needs room, the manager evicts the oldest idle model on the same device family first.
- If CUDA is unavailable, models configured for GPU fall back to CPU automatically.
- Stopping the API process stops all model timers and unloads loaded models.
- The same capacity and concurrency limits apply to classic HTTP requests, SSE streaming requests, and realtime turns.
- If a realtime turn hits a model limit, the socket stays open and the server emits an `error` event instead of crashing the connection.

## Configuration Notes

The main config file is [.env](/home/wandeber/codex-playground/.env), with a safe template in [.env.example](/home/wandeber/codex-playground/.env.example).

Important map-based settings:

- `WHISPER_MODEL_ALIAS_MAP`
- `WHISPER_MODEL_SOURCE_MAP`
- `WHISPER_MODEL_DEVICE_MAP`
- `WHISPER_MODEL_COMPUTE_TYPE_MAP`
- `WHISPER_MODEL_IDLE_SECONDS_MAP`
- `WHISPER_MODEL_VAD_MAP`
- `WHISPER_MAX_LOADED_MODELS_CPU`
- `WHISPER_MAX_LOADED_MODELS_GPU`
- `WHISPER_MAX_CONCURRENT_REQUESTS_PER_MODEL`
- `WHISPER_UPLOAD_CHUNK_SIZE`

Relative local model paths in `WHISPER_MODEL_SOURCE_MAP` are resolved from the project root, so values like `./models/distil-multi4-ct2` are portable.
`WHISPER_MODEL_IDLE_SECONDS_MAP` is the main control for model sleep behavior.

## Service Layer

- [whisper-service.sh](/home/wandeber/codex-playground/whisper-service.sh) manages persistence with `systemd --user`.
- `start` means “enable + start now”.
- `stop` means “disable + stop now”.
- The service script reads the API key from [.env](/home/wandeber/codex-playground/.env) so it can authenticate its own readiness check against `/healthz`.
- On reboot, the API follows the service enabled/disabled state you last left it in.
