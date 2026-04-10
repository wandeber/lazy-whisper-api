#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$SCRIPT_DIR/bin:$PATH"

exec "$SCRIPT_DIR/.venv/bin/whisper" --model large-v3 --device cpu --fp16 False "$@"
