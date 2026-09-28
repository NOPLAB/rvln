#!/usr/bin/env bash
set -euo pipefail

Xvfb :99 -screen 0 1280x1024x24 -ac -nolisten tcp >/tmp/isaac6-xvfb.log 2>&1 &
sleep 2
export DISPLAY=:99
exec "$@"
