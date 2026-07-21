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
- optional speaker-diarization scheduling

Relevant files:

- [lazy_whisper_api/app.py](../lazy_whisper_api/app.py)
- [lazy_whisper_api/audio_timeline.py](../lazy_whisper_api/audio_timeline.py)
- [lazy_whisper_api/config.py](../lazy_whisper_api/config.py)
- [lazy_whisper_api/diarization.py](../lazy_whisper_api/diarization.py)
- [lazy_whisper_api/editing.py](../lazy_whisper_api/editing.py)
- [lazy_whisper_api/editing_types.py](../lazy_whisper_api/editing_types.py)
- [lazy_whisper_api/model_manager.py](../lazy_whisper_api/model_manager.py)
- [lazy_whisper_api/silero_vad.py](../lazy_whisper_api/silero_vad.py)
- [lazy_whisper_api/streaming.py](../lazy_whisper_api/streaming.py)
- [lazy_whisper_api/realtime.py](../lazy_whisper_api/realtime.py)

FastAPI's lifespan owns both runtime managers. Graceful application shutdown
unloads diarization and ASR workers deterministically instead of relying on
interpreter-level exit callbacks.

### Whisper backend

Whisper-family models live directly in the main process through `faster-whisper`.

Relevant file:

- [lazy_whisper_api/backends.py](../lazy_whisper_api/backends.py)

### Qwen backends

Qwen-family models live in a separate Python runtime with a separate dependency set.

Relevant files:

- [lazy_whisper_api/backends.py](../lazy_whisper_api/backends.py)
- [lazy_whisper_api/qwen_worker.py](../lazy_whisper_api/qwen_worker.py)
- [lazy_whisper_api/qwen_mlx_worker.py](../lazy_whisper_api/qwen_mlx_worker.py)
- [setup-qwen-runtime.sh](../setup-qwen-runtime.sh)
- [setup-qwen-mlx-runtime.sh](../setup-qwen-mlx-runtime.sh)
- [requirements-qwen-cu126.txt](../requirements-qwen-cu126.txt)
- [requirements-qwen-mlx.txt](../requirements-qwen-mlx.txt)

The main API talks to each Qwen worker over line-delimited JSON-RPC on `stdin/stdout`.
`qwen-worker` runs the existing PyTorch/CUDA `qwen-asr` runtime. `qwen-mlx-worker`
runs `mlx-qwen3-asr` on Apple Silicon. Both expose the same public model aliases.

### Diarization backend

Speaker diarization lives in a separate pyannote runtime so heavy diarization
dependencies do not affect the main ASR environment.

Relevant files:

- [lazy_whisper_api/diarization.py](../lazy_whisper_api/diarization.py)
- [lazy_whisper_api/diarization_types.py](../lazy_whisper_api/diarization_types.py)
- [lazy_whisper_api/speaker_attribution.py](../lazy_whisper_api/speaker_attribution.py)
- [lazy_whisper_api/diarization_worker.py](../lazy_whisper_api/diarization_worker.py)
- [setup-diarization-runtime.sh](../setup-diarization-runtime.sh)
- [smoke-test-diarization.sh](../smoke-test-diarization.sh)
- [requirements-diarization.txt](../requirements-diarization.txt)

The worker uses `pyannote/speaker-diarization-community-1` by default. That
model is gated on Hugging Face, so first use requires accepted model conditions
and either `ASR_DIARIZATION_SETUP_HF_TOKEN` in the private `.env` file or an
exported `HF_TOKEN` during setup. The complete pipeline is downloaded to
`ASR_DIARIZATION_MODEL_PATH`; runtime launchers remove the setup-only variable
before starting the API, and request-time workers load only that local path.
They run with Hugging Face and Transformers offline modes enabled, telemetry
disabled, an isolated HOME, and an allowlisted environment without credentials.

On macOS, `torchcodec` also requires shared FFmpeg libraries. The repository's
static ffmpeg wrapper is enough for command-line conversion but not dynamic
library loading, so setup validates `torchcodec` explicitly and points Homebrew
users to `brew install ffmpeg` when those libraries are missing.

The high-risk native stack (`torch`, `torchaudio`, `torchcodec`, NumPy, and the
Hugging Face client) is pinned alongside pyannote. Setup also runs `pip check`
before accepting the environment, so dependency conflicts fail during setup
rather than on the first request.

### Shared worker transport

Qwen and pyannote domain proxies both use
[lazy_whisper_api/worker_protocol.py](../lazy_whisper_api/worker_protocol.py).
Worker stdout is consumed by a background reader and handed to startup/RPC
waiters through a queue. The shared transport owns protocol validation, stderr
draining, optional timeouts, signal isolation, graceful shutdown, and forced
process cleanup.
Domain-specific proxies only build arguments and normalize results; diarization
also supplies its credential-redaction function and strict offline environment.

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

That configuration is produced in [lazy_whisper_api/config.py](../lazy_whisper_api/config.py) from `ASR_*` variables, with `WHISPER_*` preserved as legacy aliases.
Worker Python paths can be configured by family with `ASR_FAMILY_RUNTIME_PYTHON_MAP`
or by canonical model with `ASR_MODEL_RUNTIME_PYTHON_MAP`.

### Model profiles

A public model ID resolves to a `ModelRoute` containing the requested ID, one
canonical scheduler/cache key, and a behavior profile. Existing IDs implicitly
use `subtitles-v1`. `qwen-1.7b-edit-max` uses `edit-max-v1` while resolving to
the same `qwen3-asr-1.7b` canonical model as `qwen-1.7b`; therefore both IDs
share one worker, aligner cache, concurrency limit, and loaded weight set.

`ASR_MODEL_PROFILE_MAP` is a replace-all public-ID-to-profile map. The edit ID
is reserved and fail-closed: if it is exposed in an explicit alias map, its
canonical target and explicit profile must be exactly
`qwen3-asr-1.7b/edit-max-v1`. An old explicit alias map that omits the ID stays
valid and simply does not expose it. This prevents a precision-editing request
from silently falling back to subtitle behavior.

## Request flow

### Batch HTTP

1. The client calls `POST /v1/audio/transcriptions` or `POST /v1/audio/translations`.
2. Request parameters are validated and model aliases are resolved.
3. A diarized request reserves the single diarization slot before upload or ASR
   work. Saturated requests therefore fail immediately with `429`.
4. The upload is streamed to a temp file in chunks.
5. [ModelManager](../lazy_whisper_api/model_manager.py) acquires an ASR lease.
6. The selected backend transcribes in a worker thread, not on the event loop.
7. If timestamps were requested and the backend did not return them directly,
   alignment runs as a second step. Diarized Whisper requests always enable word
   timestamps so speaker changes are not reduced to coarse ASR segment boundaries.
8. If `diarize=true`, pyannote runs after transcription/alignment.
9. A temporal turn index adds speaker labels to timestamped segments and words
   without rescanning and sorting the complete diarization timeline per word.
10. The response is rendered as `json`, `text`, `srt`, `vtt`, or `verbose_json`.

Diarization is intentionally limited to non-streaming transcription requests
with `response_format=verbose_json`, so speaker labels are always visible to the
client that paid the extra runtime cost.

### Edit-max batch path

The `edit-max-v1` profile is restricted to non-streaming batch transcription
with `verbose_json`. Validation rejects other surfaces before upload reads or a
model lease. Its path is deliberately separate from the legacy timestamp path:

1. PyAV decodes the first audio stream once to sequential mono PCM16 at 16 kHz
   and flushes the resampler tail.
2. One canonical temporary WAV is written from those exact samples.
3. Qwen transcribes the WAV, then its existing forced aligner aligns the exact
   transcript and returns ungrouped words. Apple Silicon uses a directly cached
   `mlx_qwen3_asr.ForcedAligner`; it does not rerun `Session.transcribe`.
4. Faster Whisper's bundled Silero v6 ONNX asset emits a probability for each
   512-sample frame. Entry/exit hysteresis produces coarse speech islands with
   no generic padding.
5. A bounded adaptive RMS search refines only each island's outside onset and
   offset. The default 10 ms windows improve transition resolution relative to
   the 32 ms VAD frames.
6. Pure sample-native fusion associates aligned words with acoustic islands,
   retains aligned quiet speech and strong transcript-free acoustic evidence,
   and never moves internal word edges. Only energy-confirmed first/last word
   edges may be adjusted within the configured 240 ms limit.
7. Readable `segments` are grouped independently from the final editing
   regions. Optional diarization uses the same canonical WAV, preserving the
   word order addressed by editing word-index ranges.

The public editing intervals are half-open integer sample ranges. Seconds are
derived from samples, and `time_origin=decoded_audio_start` states that v1 is
not a source-video/container PTS mapping. A non-empty transcript with no aligned
words fails closed; the API never markets a VAD-only degraded transcript as a
successful maximum-precision result.

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

### MLX

- Apple Silicon models use the `mlx` device family
- MLX models do not share the CPU loaded-model limit
- concurrency is still enforced per model

### Diarization

- diarization runs outside the ASR model scheduler
- default device is CPU for predictable Apple Silicon behavior
- only one diarization job runs at a time
- the diarization slot is reserved before ASR starts, so saturation fails fast
- the diarization worker unloads after its configured idle window
- the health payload distinguishes disabled, missing-runtime, missing-model,
  ready, loaded, busy, and error states without loading the model
- request-time diarization has no network or credential dependency

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
- load weights on CUDA or Apple Silicon GPU

Once loaded, Qwen follows the same idle-unload policy as the other models.
