#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SHARED_ENV="${VISION_ENV_DIR:-$SCRIPT_DIR/../task}"
TASK_PYTHON="$SHARED_ENV/bin/python"
if [[ ! -x "$TASK_PYTHON" ]]; then
  TASK_PYTHON="$SCRIPT_DIR/.venv/bin/python"
fi
if [[ ! -x "$TASK_PYTHON" ]]; then
  TASK_PYTHON=python3
fi
if [[ -z "${FLIGHT_RUN_ID:-}" ]]; then
  if [[ -r /proc/sys/kernel/random/uuid ]]; then
    FLIGHT_RUN_ID="$(tr -d '-' < /proc/sys/kernel/random/uuid)"
  else
    FLIGHT_RUN_ID="run_$(date +%s)_$$"
  fi
fi
FLIGHT_LOG_DIR="${FLIGHT_LOG_DIR:-$SCRIPT_DIR/logs/$FLIGHT_RUN_ID}"
MAVLINK_ENDPOINT="${MAVLINK_ENDPOINT:-/dev/serial0}"
MAVLINK_BAUD="${MAVLINK_BAUD:-57600}"
HEIGHT_SOURCE="${HEIGHT_SOURCE:-relative-alt}"
TARGET_PLANE_ALTITUDE_ABOVE_HOME_M="${TARGET_PLANE_ALTITUDE_ABOVE_HOME_M:-0}"
CAMERA_MOUNT_ROLL="${CAMERA_MOUNT_ROLL:-0}"
CAMERA_MOUNT_PITCH="${CAMERA_MOUNT_PITCH:-0}"
CAMERA_MOUNT_YAW="${CAMERA_MOUNT_YAW:-0}"
CAMERA_OFFSET_FORWARD_M="${CAMERA_OFFSET_FORWARD_M:-0}"
CAMERA_OFFSET_RIGHT_M="${CAMERA_OFFSET_RIGHT_M:-0}"
CAMERA_OFFSET_DOWN_M="${CAMERA_OFFSET_DOWN_M:-0}"
if [[ -e "$FLIGHT_LOG_DIR/flight_observations.jsonl" \
   || -e "$FLIGHT_LOG_DIR/final_targets.json" \
   || -e "$FLIGHT_LOG_DIR/runtime.log" ]]; then
  if [[ -r /proc/sys/kernel/random/uuid ]]; then
    FLIGHT_ATTEMPT_ID="$(tr -d '-' < /proc/sys/kernel/random/uuid)"
  else
    FLIGHT_ATTEMPT_ID="attempt_$$"
  fi
  FLIGHT_LOG_DIR="${FLIGHT_LOG_DIR}_$FLIGHT_ATTEMPT_ID"
fi
mkdir -p "$FLIGHT_LOG_DIR"
echo "flight_log_dir=$FLIGHT_LOG_DIR"
cd "$SCRIPT_DIR"
exec >>"$FLIGHT_LOG_DIR/runtime.log" 2>&1
echo "run_id=$FLIGHT_RUN_ID"
echo "task_python=$TASK_PYTHON"
exec "$TASK_PYTHON" -u app_localized.py \
  --source 0 --camera-backend picamera2 \
  --width 2304 --height 1296 --sensor-width 2304 --sensor-height 1296 \
  --sensor-bit-depth 10 --fps 30 --camera-ae --camera-ev -0.75 \
  --camera-awb --camera-lens-position 0 --camera-brightness 0 \
  --camera-contrast 1.3 --camera-saturation 1 --camera-sharpness 1 \
  --infer-every 3 --target-proposal color --no-undistort --headless \
  --mavlink "$MAVLINK_ENDPOINT" --mavlink-baud "$MAVLINK_BAUD" \
  --height-source "$HEIGHT_SOURCE" \
  --target-plane-altitude-above-home-m "$TARGET_PLANE_ALTITUDE_ABOVE_HOME_M" \
  --camera-mount-roll "$CAMERA_MOUNT_ROLL" \
  --camera-mount-pitch "$CAMERA_MOUNT_PITCH" \
  --camera-mount-yaw "$CAMERA_MOUNT_YAW" \
  --camera-offset-forward-m "$CAMERA_OFFSET_FORWARD_M" \
  --camera-offset-right-m "$CAMERA_OFFSET_RIGHT_M" \
  --camera-offset-down-m "$CAMERA_OFFSET_DOWN_M" \
  --max-telemetry-age 0.3 --max-position-age 0.25 \
  --max-attitude-age 0.15 --max-distance-age 0.25 --timesync-rate-hz 2 \
  --timesync-max-rtt-ms 100 --camera-latency-ms 0 --require-telemetry \
  --run-id "$FLIGHT_RUN_ID" \
  --log-jsonl "$FLIGHT_LOG_DIR/flight_observations.jsonl" \
  --final-json "$FLIGHT_LOG_DIR/final_targets.json" "$@"
