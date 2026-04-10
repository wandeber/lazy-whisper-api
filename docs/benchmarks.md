# Benchmark Notes

This file captures a small, machine-specific benchmark snapshot for the classic
non-streaming HTTP endpoint:

- endpoint: `POST /v1/audio/transcriptions`
- `stream=true`: not used
- realtime WebSocket: not used
- all measured runs were preceded by a short warm-up request on the same basic
  endpoint so model load costs would not dominate the comparison

These results are useful as relative guidance for this specific workstation and
service configuration. They are not intended to be treated as universal model
benchmarks.

## Environment notes

- API route under test: classic request/response path in
  `lazy_whisper_api/transcription.py`
- Streaming code path in `lazy_whisper_api/streaming.py` was deliberately not
  used for these measurements
- GPU models were configured on `cuda` with `compute_type=int8`
- `large-v3` was configured on `cpu` with `compute_type=int8`
- aliases:
  - `distil` -> `distil-multi4`
  - `whisper-1` -> `turbo`

## Benchmark 1: 5 minute 29 second tutorial video

- Source: `https://www.youtube.com/watch?v=34jvGi7AiVo`
- Title: `PRESENTACIONES PROFESIONALES EN WORD`
- Duration reported by the API: `329.05s`
- Warm-up clip per model: about `5s`
- Quality scores below were based on direct transcript review plus comparison
  against YouTube auto-captions as a rough external reference

| Requested model | Canonical model | Device | Warm-up time | Measured time | Quality note |
| --- | --- | --- | ---: | ---: | --- |
| `distil` | `distil-multi4` | `cuda` | `0.90s` | `16.39s` | `7.8/10` |
| `distil-multi4` | `distil-multi4` | `cuda` | `0.81s` | `16.39s` | `7.8/10` |
| `large-v3` | `large-v3` | `cpu` | `3.74s` | `192.25s` | `8.9/10` |
| `turbo` | `turbo` | `cuda` | `1.01s` | `23.15s` | `9.1/10` |
| `whisper-1` | `turbo` | `cuda` | `0.91s` | `23.25s` | `9.1/10` |

### Reading

- `turbo` was slightly better than `large-v3` on this specific video while
  remaining much faster.
- `distil-multi4` was faster than `turbo`, but the gap in transcript quality
  was visible enough that `turbo` remained the better default trade-off.
- `distil` and `distil-multi4` produced identical transcripts.
- `turbo` and `whisper-1` produced identical transcripts.

## Benchmark 2: 20 second clean speech clip

- Source clip: `60s` to `80s` from the same video above
- Local clip duration: `20.016s`
- Warm-up clip per model: about `5s`
- This was intended to mimic short spoken prompts rather than long-form video

| Requested model | Canonical model | Device | Warm-up time | Measured time | Quality note |
| --- | --- | --- | ---: | ---: | --- |
| `distil` | `distil-multi4` | `cuda` | `0.98s` | `1.23s` | `8.6/10` |
| `distil-multi4` | `distil-multi4` | `cuda` | `0.82s` | `1.22s` | `8.6/10` |
| `large-v3` | `large-v3` | `cpu` | `7.06s` | `7.17s` | `9.1/10` |
| `turbo` | `turbo` | `cuda` | `0.92s` | `1.44s` | `9.2/10` |
| `whisper-1` | `turbo` | `cuda` | `0.92s` | `1.44s` | `9.2/10` |

### Reading

- On a short, clean clip, all models preserved the meaning.
- The main difference was transcript polish, not gross correctness.
- `turbo` remained only slightly slower than `distil-multi4`, while producing a
  cleaner transcript.
- `large-v3` stayed much slower on CPU even for a 20 second clip.

## Practical takeaway

For this machine and this deployment, the classic endpoint currently looks like
this:

- fastest raw throughput: `distil-multi4`
- best default quality/speed trade-off: `turbo`
- strongest CPU-only accuracy option: `large-v3`

That does not mean these conclusions automatically transfer to other languages,
other deployments, cold-start scenarios, or the streaming/realtime endpoints.
They are simply the local snapshot captured after the streaming work landed,
while intentionally benchmarking only the basic endpoint.
