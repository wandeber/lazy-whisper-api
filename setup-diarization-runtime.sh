#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIARIZATION_VENV="${DIARIZATION_VENV:-$SCRIPT_DIR/.venv-diarization}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -f "$SCRIPT_DIR/.env" ]]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
fi

# Keep the gated-model credential as a shell-only value. In particular, pip and
# dependency build hooks do not need it and must not inherit it from `.env` or
# from the environment used to invoke this script.
HF_DOWNLOAD_TOKEN="${ASR_DIARIZATION_SETUP_HF_TOKEN:-${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}}}"
export -n HF_DOWNLOAD_TOKEN
unset \
  ASR_DIARIZATION_SETUP_HF_TOKEN \
  HF_TOKEN \
  HUGGING_FACE_HUB_TOKEN \
  HUGGINGFACE_HUB_TOKEN

MODEL_ID="${ASR_DIARIZATION_MODEL_ID:-pyannote/speaker-diarization-community-1}"
MODEL_PATH="${ASR_DIARIZATION_MODEL_PATH:-$SCRIPT_DIR/models/pyannote-speaker-diarization-community-1}"
MODEL_READY_MARKER=".lazy-whisper-offline-ready"
SKIP_MODEL_DOWNLOAD="${DIARIZATION_SKIP_MODEL_DOWNLOAD:-false}"
if [[ "$MODEL_PATH" != /* ]]; then
  MODEL_PATH="$SCRIPT_DIR/${MODEL_PATH#./}"
fi

# Keep the repository ffmpeg wrapper available for audio tooling. TorchCodec
# additionally needs FFmpeg shared libraries, which are checked explicitly after
# the isolated Python environment is installed.
export PATH="$SCRIPT_DIR/bin:$PATH"
export PIP_CACHE_DIR="$SCRIPT_DIR/.cache/pip"
mkdir -p "$PIP_CACHE_DIR"
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required by pyannote.audio but was not found." >&2
  exit 1
fi

CREATED_DIARIZATION_VENV="false"
if [[ ! -x "$DIARIZATION_VENV/bin/python" || ! -x "$DIARIZATION_VENV/bin/pip" ]]; then
  if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
  then
    "$PYTHON_BIN" -m venv "$DIARIZATION_VENV"
  else
    "$BOOTSTRAP_PYTHON" -m virtualenv "$DIARIZATION_VENV"
  fi
  CREATED_DIARIZATION_VENV="true"
fi

if [[ "$CREATED_DIARIZATION_VENV" == "true" ]]; then
  "$DIARIZATION_VENV/bin/pip" install --upgrade pip setuptools wheel
fi
"$DIARIZATION_VENV/bin/pip" install \
  --disable-pip-version-check \
  --quiet \
  -r "$SCRIPT_DIR/requirements-diarization.txt"
"$DIARIZATION_VENV/bin/pip" check

if ! "$DIARIZATION_VENV/bin/python" -c \
  'from torchcodec.decoders import AudioDecoder; print("torchcodec audio decoding available")'
then
  echo "torchcodec could not load the shared FFmpeg libraries required by pyannote.audio." >&2
  if [[ "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    echo "Install them with: brew install ffmpeg" >&2
  fi
  exit 1
fi

"$DIARIZATION_VENV/bin/python" -c \
  'from importlib.metadata import version; print("pyannote.audio", version("pyannote.audio"), "installed")'

SETUP_CACHE_ROOT="$SCRIPT_DIR/.cache/diarization-setup"
mkdir -p \
  "$SETUP_CACHE_ROOT/home" \
  "$SETUP_CACHE_ROOT/huggingface" \
  "$SETUP_CACHE_ROOT/matplotlib"

verify_offline_pipeline() {
  env \
    -u HF_TOKEN \
    -u HUGGING_FACE_HUB_TOKEN \
    -u HUGGINGFACE_HUB_TOKEN \
    HOME="$SETUP_CACHE_ROOT/home" \
    HF_HOME="$SETUP_CACHE_ROOT/huggingface" \
    HF_TOKEN_PATH="$SETUP_CACHE_ROOT/huggingface/token-disabled" \
    MPLCONFIGDIR="$SETUP_CACHE_ROOT/matplotlib" \
    PYANNOTE_METRICS_ENABLED=0 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    "$DIARIZATION_VENV/bin/python" - "$MODEL_PATH" <<'PY'
from pathlib import Path
import sys

from pyannote.audio import Pipeline

model_path = Path(sys.argv[1]).resolve()
Pipeline.from_pretrained(str(model_path))
print(f"Offline pipeline verified at {model_path}")
PY
}

marker_matches_model() {
  "$DIARIZATION_VENV/bin/python" - "$MODEL_PATH/$MODEL_READY_MARKER" "$MODEL_ID" <<'PY'
import json
from pathlib import Path
import sys

marker_path = Path(sys.argv[1])
expected_model_id = sys.argv[2]
try:
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
if not isinstance(marker, dict):
    raise SystemExit(1)
raise SystemExit(
    0
    if marker.get("format_version") == 1
    and marker.get("model_id") == expected_model_id
    else 1
)
PY
}

write_ready_marker() {
  "$DIARIZATION_VENV/bin/python" - "$MODEL_PATH" "$MODEL_READY_MARKER" "$MODEL_ID" <<'PY'
import json
from pathlib import Path
import re
import sys

model_path = Path(sys.argv[1]).resolve()
marker_path = model_path / sys.argv[2]
model_id = sys.argv[3]
revision = None
metadata_path = model_path / ".cache/huggingface/download/config.yaml.metadata"
try:
    candidate = metadata_path.read_text(encoding="utf-8").splitlines()[0].strip()
    if re.fullmatch(r"[0-9a-f]{40}", candidate):
        revision = candidate
except (OSError, IndexError):
    pass

payload = {
    "format_version": 1,
    "model_id": model_id,
    "revision": revision,
}
temporary_path = marker_path.with_name(f"{marker_path.name}.tmp")
temporary_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
temporary_path.replace(marker_path)
PY
}

if [[ "$SKIP_MODEL_DOWNLOAD" == "true" ]]; then
  echo "Skipped local model download because DIARIZATION_SKIP_MODEL_DOWNLOAD=true."
else
  MODEL_IS_READY="false"
  if marker_matches_model; then
    # Invalidate the previous result before checking current files. Health must
    # never keep reporting a stale success while revalidation or repair fails.
    rm -f "$MODEL_PATH/$MODEL_READY_MARKER"
    if [[ -f "$MODEL_PATH/config.yaml" ]] && verify_offline_pipeline >/dev/null 2>&1; then
      write_ready_marker
      MODEL_IS_READY="true"
    fi
  else
    rm -f "$MODEL_PATH/$MODEL_READY_MARKER"
  fi

  if [[ "$MODEL_IS_READY" == "true" ]]; then
    echo "Local diarization model already present and verified at $MODEL_PATH"
  else
    if [[ -z "$HF_DOWNLOAD_TOKEN" ]]; then
      echo "The gated model is missing, incomplete, or does not match MODEL_ID; no setup token is configured." >&2
      echo "Set ASR_DIARIZATION_SETUP_HF_TOKEN in .env or export HF_TOKEN once, then rerun this command." >&2
      exit 2
    fi

    mkdir -p "$MODEL_PATH"
    echo "Downloading $MODEL_ID into $MODEL_PATH"
    # Scope the token to this one process. `hf download --local-dir` is
    # idempotent and resumes any files left by an interrupted previous attempt.
    HF_TOKEN="$HF_DOWNLOAD_TOKEN" \
      "$DIARIZATION_VENV/bin/hf" download "$MODEL_ID" --local-dir "$MODEL_PATH"
    unset HF_DOWNLOAD_TOKEN

    if [[ ! -f "$MODEL_PATH/config.yaml" ]]; then
      echo "Model download completed without the expected config.yaml file." >&2
      exit 1
    fi
    echo "Verifying that the downloaded pipeline loads with networking disabled"
    verify_offline_pipeline
    write_ready_marker
  fi
fi

echo "Diarization runtime ready at $DIARIZATION_VENV"
