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

shopt -s nullglob
for lib_dir in \
  "$SCRIPT_DIR"/.venv/lib/python*/site-packages/nvidia/cublas/lib \
  "$SCRIPT_DIR"/.venv/lib/python*/site-packages/nvidia/cudnn/lib
do
  export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}"
done
shopt -u nullglob

HOST="${ASR_API_HOST:-${WHISPER_API_HOST:-127.0.0.1}}"
PORT="${ASR_API_PORT:-${WHISPER_API_PORT:-43556}}"

exec "$SCRIPT_DIR/.venv/bin/uvicorn" whisper_api:app \
  --app-dir "$SCRIPT_DIR" \
  --host "$HOST" \
  --port "$PORT"
