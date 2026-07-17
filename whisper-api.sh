#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$SCRIPT_DIR/bin:$PATH"

if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
fi

# The gated diarization token may be retained in `.env` for future model repair,
# but it is a setup-only secret and must never reach uvicorn or any ASR sidecar.
unset \
  ASR_DIARIZATION_SETUP_HF_TOKEN \
  HUGGING_FACE_HUB_TOKEN \
  HUGGINGFACE_HUB_TOKEN

shopt -s nullglob
for lib_dir in \
  "$SCRIPT_DIR"/.venv/lib/python*/site-packages/nvidia/cublas/lib \
  "$SCRIPT_DIR"/.venv/lib/python*/site-packages/nvidia/cudnn/lib
do
  export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}"
done
shopt -u nullglob

HOST="${ASR_API_HOST:-${WHISPER_API_HOST:-localhost}}"
PORT="${ASR_API_PORT:-${WHISPER_API_PORT:-43556}}"

exec "$SCRIPT_DIR/.venv/bin/uvicorn" whisper_api:app \
  --app-dir "$SCRIPT_DIR" \
  --host "$HOST" \
  --port "$PORT"
