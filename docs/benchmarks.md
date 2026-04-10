# Benchmark Notes

This file captures a small, machine-specific benchmark snapshot for the classic
non-streaming HTTP endpoint, plus one endpoint-style comparison using the same
model across multiple request surfaces:

- model comparison:
  - endpoint: `POST /v1/audio/transcriptions`
  - `stream=true`: not used
  - realtime WebSocket: not used
- endpoint comparison:
  - same requested model on classic JSON, classic SSE, and realtime manual
    commit variants
- all measured runs were preceded by a short warm-up request on the same route
  family so model load costs would not dominate the comparison

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

## Benchmark 3: Same model across endpoint styles

This section keeps the model fixed at `turbo` and compares how the same audio
behaves across the main request styles that produced stable, repeatable results
in this deployment:

- classic blocking JSON
- classic SSE with `stream=true`
- realtime manual commit with one large append
- realtime manual commit with chunked appends

The goal here is different from the model comparison above: it is about which
endpoint style is the better default for a given product shape, not which model
is better in general.

### Endpoint comparison A: 5 minute 29 second tutorial video

- Source: `https://www.youtube.com/watch?v=34jvGi7AiVo`
- Title: `PRESENTACIONES PROFESIONALES EN WORD`
- Duration reported by the API: about `329s`
- Requested model for every row: `turbo`

| Endpoint style | First text | Final transcript | Text parity vs classic JSON | Reading |
| --- | ---: | ---: | --- | --- |
| classic JSON | `23.65s` | `23.65s` | baseline | final-only response |
| classic SSE | `2.51s` | `23.11s` | exact match | same final text, much earlier visible progress |
| realtime manual commit, single append | no partial | failed on this long file | n/a | not a good fit for long completed audio uploads |
| realtime manual commit, chunked appends | `0.81s` | `23.86s` | very close, but not exact | earliest partial text, slightly slower final result, less stable intermediate text |

#### Reading

- For completed long-form audio, classic SSE was the best overall surface.
- It finished slightly faster than classic JSON while exposing useful text much
  earlier.
- Realtime manual commit with chunked appends produced the earliest visible
  text, but its final transcript was slightly less stable than the HTTP result
  on this longer sample.
- Realtime manual commit with a single huge append was not a reliable fit for
  long completed files in this deployment.

### Endpoint comparison B: 20 second clean speech clip

- Source clip: `60s` to `80s` from the same video above
- Local clip duration: about `20s`
- Requested model for every row: `turbo`

| Endpoint style | First text | Final transcript | Text parity vs classic JSON | Reading |
| --- | ---: | ---: | --- | --- |
| classic JSON | `1.47s` | `1.47s` | baseline | final-only response |
| classic SSE | `1.39s` | `1.39s` | exact match | fastest final result for completed audio |
| realtime manual commit, single append | no partial | `1.66s` | exact match | clean final result, simple WebSocket flow |
| realtime manual commit, chunked appends | `0.78s` | `2.12s` | exact match | earliest partial text, slower final result |

#### Reading

- For short completed audio, classic SSE again came out as the best default.
- If the client already speaks realtime WebSocket and only needs a final answer
  after an explicit commit, the single-append manual flow was reasonable on a
  short clip.
- If the product cares most about showing text as early as possible, chunked
  realtime manual commit won on first-text latency, but not on final latency.

## Endpoint takeaway

For this machine and this deployment:

- best endpoint for already-recorded audio: `stream=true` on the classic HTTP
  transcription route
- simplest final-only completed-audio path: classic JSON
- best short-prompt realtime/manual-commit path: realtime manual commit with a
  single append
- best earliest-partial path: realtime manual commit with chunked appends

As with the model comparison above, these are deployment-specific numbers and
should be read as relative guidance rather than as universal guarantees.
