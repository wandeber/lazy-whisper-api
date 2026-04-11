#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN_VENV="${QWEN_VENV:-$SCRIPT_DIR/.venv-qwen}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-$SCRIPT_DIR/.venv/bin/python}"

if [[ ! -x "$QWEN_VENV/bin/python" || ! -x "$QWEN_VENV/bin/pip" ]]; then
  if python3 -m venv "$QWEN_VENV" >/dev/null 2>&1; then
    :
  else
    "$BOOTSTRAP_PYTHON" -m virtualenv "$QWEN_VENV"
  fi
fi
"$QWEN_VENV/bin/pip" install --upgrade pip setuptools wheel
"$QWEN_VENV/bin/pip" install -r "$SCRIPT_DIR/requirements-qwen-cu126.txt"

echo "Qwen runtime listo en $QWEN_VENV"
