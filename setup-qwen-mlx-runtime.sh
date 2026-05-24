#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN_MLX_VENV="${QWEN_MLX_VENV:-$SCRIPT_DIR/.venv-qwen-mlx}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

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
fi

"$QWEN_MLX_VENV/bin/python" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("mlx-qwen3-asr requires Python >= 3.10")
PY

"$QWEN_MLX_VENV/bin/pip" install --upgrade pip setuptools wheel
"$QWEN_MLX_VENV/bin/pip" install -r "$SCRIPT_DIR/requirements-qwen-mlx.txt"

echo "Qwen MLX runtime ready at $QWEN_MLX_VENV"
