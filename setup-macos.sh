#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

export PIP_CACHE_DIR="$SCRIPT_DIR/.cache/pip"
mkdir -p "$PIP_CACHE_DIR"

fail() {
  echo "macOS setup failed: $*" >&2
  exit 1
}

python_is_supported() {
  local candidate="$1"
  [[ -x "$candidate" ]] || return 1
  "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
}

find_supported_python() {
  local candidate

  if [[ -n "${PYTHON_BIN:-}" ]]; then
    python_is_supported "$PYTHON_BIN" || fail "PYTHON_BIN must point to Python 3.11 or newer."
    printf '%s\n' "$PYTHON_BIN"
    return
  fi

  # Prefer Python 3.12 because every pinned native runtime is already validated
  # against it. Do not use `.venv/bin/python` as the bootstrap interpreter:
  # `make install-macos` recreates/updates that same environment.
  for candidate in \
    /opt/homebrew/bin/python3.12 \
    /opt/homebrew/bin/python3.11 \
    /opt/homebrew/bin/python3.13 \
    /usr/local/bin/python3.12 \
    /usr/local/bin/python3.11 \
    /usr/local/bin/python3.13 \
    "$(command -v python3 2>/dev/null || true)"
  do
    if python_is_supported "$candidate"; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  printf '\n'
}

[[ "$(uname -s)" == "Darwin" ]] || fail "this command is only for macOS."
[[ "$(uname -m)" == "arm64" ]] || fail "the native setup requires Apple Silicon."
command -v make >/dev/null 2>&1 || fail "make is required. Install the Xcode Command Line Tools."
command -v brew >/dev/null 2>&1 || fail "Homebrew is required: https://brew.sh"

if [[ ! -f "$ENV_FILE" ]]; then
  fail "missing .env. Run 'cp .env.apple.example .env', add the setup token, then rerun make setup-macos."
fi

# `.env` contains both the API key and the optional gated-model credential.
# Tighten its permissions before reading it, including on repeat installations.
chmod 600 "$ENV_FILE"
# shellcheck disable=SC1091
source "$ENV_FILE"

[[ -n "${ASR_API_KEY:-}" ]] || fail "ASR_API_KEY must be set in .env."
[[ "${ASR_DIARIZATION_ENABLED:-false}" == "true" ]] || fail "ASR_DIARIZATION_ENABLED=true is required for the full macOS setup."

SETUP_TOKEN="${ASR_DIARIZATION_SETUP_HF_TOKEN:-${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}}}"
export -n SETUP_TOKEN

# No dependency installer needs access to the gated-model credential. The value
# is passed only to setup-diarization-runtime.sh, which scopes it further to the
# single `hf download` subprocess when a download or repair is actually needed.
unset \
  ASR_DIARIZATION_SETUP_HF_TOKEN \
  HF_TOKEN \
  HUGGING_FACE_HUB_TOKEN \
  HUGGINGFACE_HUB_TOKEN

MODEL_PATH="${ASR_DIARIZATION_MODEL_PATH:-$SCRIPT_DIR/models/pyannote-speaker-diarization-community-1}"
if [[ "$MODEL_PATH" != /* ]]; then
  MODEL_PATH="$SCRIPT_DIR/${MODEL_PATH#./}"
fi
if [[ -z "$SETUP_TOKEN" && ! -f "$MODEL_PATH/.lazy-whisper-offline-ready" ]]; then
  fail "set ASR_DIARIZATION_SETUP_HF_TOKEN in .env after accepting the pyannote model conditions."
fi

if ! brew list --versions ffmpeg >/dev/null 2>&1; then
  echo "Installing Homebrew FFmpeg shared libraries required by TorchCodec"
  brew install ffmpeg
fi

PYTHON_EXECUTABLE="$(find_supported_python)"
if [[ -z "$PYTHON_EXECUTABLE" ]]; then
  echo "Installing Homebrew Python 3.12"
  brew install python@3.12
  PYTHON_EXECUTABLE="$(brew --prefix python@3.12)/bin/python3.12"
fi
python_is_supported "$PYTHON_EXECUTABLE" || fail "could not find a working Python 3.11 or newer."

echo "Using $($PYTHON_EXECUTABLE -c 'import sys; print(sys.executable)')"
echo "Installing the main API runtime"
make -C "$SCRIPT_DIR" PYTHON="$PYTHON_EXECUTABLE" install-macos

echo "Installing the Qwen MLX runtime"
PYTHON_BIN="$PYTHON_EXECUTABLE" \
BOOTSTRAP_PYTHON="$SCRIPT_DIR/.venv/bin/python" \
  "$SCRIPT_DIR/setup-qwen-mlx-runtime.sh"

echo "Installing and verifying the local diarization runtime"
ASR_DIARIZATION_SETUP_HF_TOKEN="$SETUP_TOKEN" \
PYTHON_BIN="$PYTHON_EXECUTABLE" \
BOOTSTRAP_PYTHON="$SCRIPT_DIR/.venv/bin/python" \
  "$SCRIPT_DIR/setup-diarization-runtime.sh"
unset SETUP_TOKEN

echo
echo "macOS setup complete. Start the local API with: make run"
echo "Then check: http://localhost:${ASR_API_PORT:-43556}/healthz"
