#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${VISION_TASK:-digit}" in
  digit|digit_recognition)
    exec bash "$SCRIPT_DIR/digit_recognition/run_localized_camera.sh"
    ;;
  pattern|pattern_recognition)
    exec bash "$SCRIPT_DIR/pattern_recognition/run_localized_camera.sh"
    ;;
  *)
    echo "VISION_TASK must be digit or pattern, got: ${VISION_TASK:-}" >&2
    exit 2
    ;;
esac
