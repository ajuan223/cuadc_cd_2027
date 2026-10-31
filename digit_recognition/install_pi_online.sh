#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SHARED_ENV="${VISION_ENV_DIR:-$SCRIPT_DIR/../task}"
cd "$SCRIPT_DIR"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This deployment requires 64-bit Raspberry Pi OS (aarch64)." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  python3-venv python3-pip python3-dev python3-opencv python3-numpy python3-picamera2 \
  python3-lxml python3-serial build-essential v4l-utils ffmpeg

if [[ ! -x "$SHARED_ENV/bin/python" ]]; then
  python3 -m venv --system-site-packages "$SHARED_ENV"
fi
"$SHARED_ENV/bin/python" -m pip install -r requirements_localized.txt
"$SHARED_ENV/bin/python" check_environment.py --localized --camera 0 --camera-backend picamera2

echo "Installation complete in $SHARED_ENV. Runtime no longer needs internet access."
