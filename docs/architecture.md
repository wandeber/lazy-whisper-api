# Architecture

This project exposes a local transcription API that looks like the OpenAI audio API, while keeping model usage lazy and resource-aware.

## Main Pieces

- [whisper_api.py](/home/wandeber/codex-playground/whisper_api.py): compatibility entrypoint used by `uvicorn whisper_api:app`
- [lazy_whisper_api/app.py](/home/wandeber/codex-playground/lazy_whisper_api/app.py): FastAPI app assembly and route wiring
- [lazy_whisper_api/config.py](/home/wandeber/codex-playground/lazy_whisper_api/config.py): environment parsing and config validation
- [lazy_whisper_api/auth.py](/home/wandeber/codex-playground/lazy_whisper_api/auth.py): fixed API key auth dependency
- [lazy_whisper_api/model_manager.py](/home/wandeber/codex-playground/lazy_whisper_api/model_manager.py): lazy loading, leases, per-device capacity limits, and idle unload timers
- [lazy_whisper_api/transcription.py](/home/wandeber/codex-playground/lazy_whisper_api/transcription.py): request validation, chunked temp-file handling, and Faster Whisper execution
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

## Model Lifecycle

- `large-v3` defaults to a 10-minute idle timeout and can still be set to `idle_seconds=0` if you want immediate unloads.
- GPU models such as `turbo` or `distil-multi4` can stay warm for a configurable idle window.
- By default, at most 1 CPU model and 2 GPU models can remain loaded at the same time.
- By default, each model accepts at most 2 active transcriptions at once.
- When a new model needs room, the manager evicts the oldest idle model on the same device family first.
- If CUDA is unavailable, models configured for GPU fall back to CPU automatically.
- Stopping the API process stops all model timers and unloads loaded models.

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
