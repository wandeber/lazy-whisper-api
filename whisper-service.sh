#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="whisper-api.service"
SYSTEMD_USER_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT_PATH="$SYSTEMD_USER_DIR/$SERVICE_NAME"
ENV_FILE="$SCRIPT_DIR/.env"
STATUS_HOST="127.0.0.1"
STATUS_PORT="43556"

require_systemd() {
  if ! command -v systemctl >/dev/null 2>&1; then
    echo "systemctl no esta disponible en esta maquina." >&2
    exit 1
  fi
}

ensure_env_file() {
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "Falta $ENV_FILE" >&2
    exit 1
  fi
}

load_env_file() {
  ensure_env_file
  set -a
  # shellcheck disable=SC1091
  source "$ENV_FILE"
  set +a

  STATUS_HOST="${ASR_API_HOST:-${WHISPER_API_HOST:-127.0.0.1}}"
  STATUS_PORT="${ASR_API_PORT:-${WHISPER_API_PORT:-43556}}"

  if [[ "$STATUS_HOST" == "0.0.0.0" ]]; then
    STATUS_HOST="127.0.0.1"
  fi
}

write_unit() {
  mkdir -p "$SYSTEMD_USER_DIR"
  python3 - <<'PY' "$UNIT_PATH" "$SCRIPT_DIR" "$ENV_FILE"
from pathlib import Path
import sys

unit_path = Path(sys.argv[1])
script_dir = Path(sys.argv[2])
env_file = Path(sys.argv[3])

unit_path.write_text(
    f"""[Unit]
Description=Local ASR API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={script_dir}
EnvironmentFile=-{env_file}
ExecStart={script_dir}/whisper-api.sh
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
""",
    encoding="utf-8",
)
PY
  systemctl --user daemon-reload
}

show_linger_hint() {
  if ! command -v loginctl >/dev/null 2>&1; then
    return
  fi

  local linger
  linger="$(loginctl show-user "$USER" -p Linger --value 2>/dev/null || true)"
  if [[ "$linger" != "yes" ]]; then
    echo
    echo "Nota: asi quedara persistente entre reinicios para tu usuario, pero normalmente arrancara cuando se inicie tu sesion."
    echo "Si quieres que arranque incluso antes de iniciar sesion: sudo loginctl enable-linger $USER"
  fi
}

print_status() {
  systemctl --user status "$SERVICE_NAME" --no-pager || true
}

wait_until_ready() {
  load_env_file
  local health_url="http://${STATUS_HOST}:${STATUS_PORT}/healthz"
  local attempt
  local -a curl_args=(-fsS)

  local api_key="${ASR_API_KEY:-${WHISPER_API_KEY:-}}"
  if [[ -n "$api_key" ]]; then
    curl_args+=(-H "Authorization: Bearer ${api_key}")
  fi

  for attempt in $(seq 1 60); do
    if curl "${curl_args[@]}" "$health_url" >/dev/null 2>&1; then
      echo "API lista en $health_url"
      return 0
    fi
    sleep 1
  done

  echo "La API no llego a estar lista a tiempo en $health_url" >&2
  return 1
}

start_service() {
  ensure_env_file
  write_unit
  systemctl --user enable --now "$SERVICE_NAME"
  wait_until_ready
  print_status
  show_linger_hint
}

stop_service() {
  write_unit
  systemctl --user disable --now "$SERVICE_NAME"
  print_status
  show_linger_hint
}

restart_service() {
  ensure_env_file
  write_unit
  if systemctl --user is-enabled "$SERVICE_NAME" >/dev/null 2>&1; then
    systemctl --user restart "$SERVICE_NAME"
  else
    systemctl --user start "$SERVICE_NAME"
  fi
  wait_until_ready
  print_status
}

status_service() {
  write_unit
  print_status
}

logs_service() {
  journalctl --user -u "$SERVICE_NAME" -n 100 --no-pager
}

usage() {
  echo "Uso: $0 {start|stop|restart|status|logs}"
}

require_systemd

case "${1:-}" in
  start)
    start_service
    ;;
  stop)
    stop_service
    ;;
  restart)
    restart_service
    ;;
  status)
    status_service
    ;;
  logs)
    logs_service
    ;;
  *)
    usage
    exit 1
    ;;
esac
