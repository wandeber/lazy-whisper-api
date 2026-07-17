#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN_MLX_VENV="${QWEN_MLX_VENV:-$SCRIPT_DIR/.venv-qwen-mlx}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CREATED_QWEN_MLX_VENV="false"

export PIP_CACHE_DIR="$SCRIPT_DIR/.cache/pip"
mkdir -p "$PIP_CACHE_DIR"

if [[ ! -x "$QWEN_MLX_VENV/bin/python" || ! -x "$QWEN_MLX_VENV/bin/pip" ]]; then
  if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
  then
    "$PYTHON_BIN" -m venv "$QWEN_MLX_VENV"
  else
    "$BOOTSTRAP_PYTHON" -m virtualenv "$QWEN_MLX_VENV"
  fi
  CREATED_QWEN_MLX_VENV="true"
fi

"$QWEN_MLX_VENV/bin/python" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("mlx-qwen3-asr requires Python >= 3.10")
PY

if [[ "$CREATED_QWEN_MLX_VENV" == "true" ]]; then
  "$QWEN_MLX_VENV/bin/pip" install --upgrade pip setuptools wheel
fi
"$QWEN_MLX_VENV/bin/pip" install \
  --disable-pip-version-check \
  --quiet \
  -r "$SCRIPT_DIR/requirements-qwen-mlx.txt"

"$QWEN_MLX_VENV/bin/python" - <<'PY'
from importlib.metadata import version

import mlx.core as mx

print(f"mlx-qwen3-asr {version('mlx-qwen3-asr')} installed; default device: {mx.default_device()}")
PY

echo "Qwen MLX runtime ready at $QWEN_MLX_VENV"
