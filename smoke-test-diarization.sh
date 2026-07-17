#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_PYTHON="${MAIN_PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
AUDIO_PATH="${1:-}"
EXPECTED_SPEAKERS="${DIARIZATION_EXPECTED_SPEAKERS:-2}"

if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
fi

# This smoke test consumes an already-downloaded local model. Do not expose the
# setup credential to its main Python process even though the sidecar also uses
# a strict environment allowlist.
unset ASR_DIARIZATION_SETUP_HF_TOKEN HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN

export ASR_DIARIZATION_ENABLED="${ASR_DIARIZATION_ENABLED:-true}"
export PATH="$SCRIPT_DIR/bin:$PATH"

if [[ ! -x "$MAIN_PYTHON" ]]; then
  echo "Main Python runtime not found at $MAIN_PYTHON" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/lazy-whisper-diarization.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

if [[ -z "$AUDIO_PATH" ]]; then
  if [[ "$(uname -s)" != "Darwin" ]] || ! command -v say >/dev/null 2>&1; then
    echo "Pass an audio file on non-macOS systems: $0 /path/to/conversation.wav" >&2
    exit 1
  fi
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg is required to build the synthetic conversation." >&2
    exit 1
  fi

  # Four alternating utterances are long enough for the embedding model to
  # distinguish the two voices while remaining quick enough for a smoke test.
  say -v Samantha -r 165 -o "$TMP_DIR/a1.aiff" \
    "Hello. I am the first speaker in this completely local diarization test."
  say -v Daniel -r 155 -o "$TMP_DIR/b1.aiff" \
    "Hello. I am the second speaker, answering with a clearly different voice."
  say -v Samantha -r 165 -o "$TMP_DIR/a2.aiff" \
    "The first speaker is back now, continuing the short test conversation."
  say -v Daniel -r 155 -o "$TMP_DIR/b2.aiff" \
    "And the second speaker closes the conversation so both labels can be checked."

  for generated_part in "$TMP_DIR"/*.aiff; do
    # macOS `say` can return success without producing frames when its speech
    # service is unavailable (for example, inside a restricted sandbox). Catch
    # that condition here so the failure points at synthesis, not TorchCodec.
    if [[ "$(wc -c < "$generated_part")" -lt 10000 ]]; then
      echo "macOS speech synthesis produced no usable audio: $generated_part" >&2
      exit 1
    fi
  done

  AUDIO_PATH="$TMP_DIR/two-speakers.wav"
  ffmpeg -hide_banner -loglevel error -y \
    -i "$TMP_DIR/a1.aiff" \
    -i "$TMP_DIR/b1.aiff" \
    -i "$TMP_DIR/a2.aiff" \
    -i "$TMP_DIR/b2.aiff" \
    -filter_complex "[0:a][1:a][2:a][3:a]concat=n=4:v=0:a=1[out]" \
    -map "[out]" -ar 16000 -ac 1 "$AUDIO_PATH"
fi

if [[ ! -f "$AUDIO_PATH" ]]; then
  echo "Audio file not found: $AUDIO_PATH" >&2
  exit 1
fi

"$MAIN_PYTHON" - "$AUDIO_PATH" "$EXPECTED_SPEAKERS" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

from lazy_whisper_api.config import load_settings
from lazy_whisper_api.diarization import DiarizationManager

audio_path = Path(sys.argv[1]).resolve()
expected_speakers = int(sys.argv[2])
settings = load_settings()
manager = DiarizationManager(settings)

try:
    result = manager.diarize(
        audio_path=audio_path,
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
    )
finally:
    manager.unload()

speakers = sorted({turn.speaker for turn in result.turns})
payload = {
    "audio": str(audio_path),
    "model": result.model,
    "device": result.device,
    "processing_seconds": result.processing_seconds,
    "speakers": speakers,
    "turns": [
        {"start": turn.start, "end": turn.end, "speaker": turn.speaker}
        for turn in result.turns
    ],
}
print(json.dumps(payload, indent=2))

if len(speakers) < expected_speakers:
    raise SystemExit(
        f"Expected at least {expected_speakers} speakers, detected {len(speakers)}."
    )
PY
