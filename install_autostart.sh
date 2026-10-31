#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK="${1:-digit}"
case "$TASK" in
  digit|digit_recognition) TASK=digit ;;
  pattern|pattern_recognition) TASK=pattern ;;
  *) echo "Usage: bash install_autostart.sh [digit|pattern]" >&2; exit 2 ;;
esac

RUN_USER="${SUDO_USER:-$USER}"
RUN_GROUP="$(id -gn "$RUN_USER")"
PROJECT_DIR="$SCRIPT_DIR"
SERVICE_TEMPLATE="$SCRIPT_DIR/systemd/cuadc-vision.service.in"
ENV_TEMPLATE="$SCRIPT_DIR/systemd/cuadc-vision.env"

if [[ ! -f "$SERVICE_TEMPLATE" || ! -f "$ENV_TEMPLATE" ]]; then
  echo "Missing systemd templates under $SCRIPT_DIR/systemd" >&2
  exit 1
fi
if [[ ! -x "$SCRIPT_DIR/task/bin/python" ]]; then
  echo "Missing shared environment: $SCRIPT_DIR/task/bin/python" >&2
  echo "Run the online or offline installer before enabling autostart." >&2
  exit 1
fi

chmod +x \
  "$SCRIPT_DIR/run_selected_task.sh" \
  "$SCRIPT_DIR/digit_recognition/run_localized_camera.sh" \
  "$SCRIPT_DIR/pattern_recognition/run_localized_camera.sh"

TEMP_SERVICE="$(mktemp)"
TEMP_ENV="$(mktemp)"
trap 'rm -f -- "$TEMP_SERVICE" "$TEMP_ENV"' EXIT
sed \
  -e "s|@RUN_USER@|$RUN_USER|g" \
  -e "s|@RUN_GROUP@|$RUN_GROUP|g" \
  -e "s|@PROJECT_DIR@|$PROJECT_DIR|g" \
  "$SERVICE_TEMPLATE" > "$TEMP_SERVICE"
sed "s/^VISION_TASK=.*/VISION_TASK=$TASK/" "$ENV_TEMPLATE" > "$TEMP_ENV"

sudo install -m 0644 "$TEMP_SERVICE" /etc/systemd/system/cuadc-vision.service
if [[ ! -f /etc/default/cuadc-vision ]]; then
  sudo install -m 0644 "$TEMP_ENV" /etc/default/cuadc-vision
else
  sudo sed -i "s/^VISION_TASK=.*/VISION_TASK=$TASK/" /etc/default/cuadc-vision
fi
sudo systemctl daemon-reload
sudo systemctl enable --now cuadc-vision.service

echo "Enabled cuadc-vision.service with VISION_TASK=$TASK"
echo "Configuration: /etc/default/cuadc-vision"
echo "Status: sudo systemctl status cuadc-vision.service"
