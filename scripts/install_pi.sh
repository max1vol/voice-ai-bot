#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: scripts/install_pi.sh pi@192.168.1.90" >&2
  exit 2
fi

target="$1"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stage_dir="$(mktemp -d "${TMPDIR:-/tmp}/voice-ai-bot-pi-deploy.XXXXXX")"

cleanup() {
  rm -rf "$stage_dir"
}
trap cleanup EXIT

if [[ ! -f "$repo_dir/.env" ]]; then
  echo "missing .env; copy .env.example and set OPENAI_API_KEY" >&2
  exit 1
fi

python3 "$repo_dir/scripts/build_pi_deploy.py" --repo-root "$repo_dir" --output "$stage_dir" --env "$repo_dir/.env" >/dev/null

ssh "$target" 'sudo install -d -o pi -g pi /opt/voice-ai-bot /var/lib/voice-ai-bot /var/lib/voice-ai-bot/recordings /var/lib/voice-ai-bot/music /var/lib/voice-ai-bot/images'
ssh "$target" 'sudo systemctl stop voice-ai-bot-debug.service 2>/dev/null || true'
ssh "$target" 'sudo systemctl stop voice-ai-bot.service 2>/dev/null || true'

rsync -az --delete --exclude .venv "$stage_dir/" "$target:/opt/voice-ai-bot/"

ssh "$target" 'sudo find /opt/voice-ai-bot -path /opt/voice-ai-bot/.venv -prune -o -exec chown pi:pi {} + && sudo chown -R pi:pi /var/lib/voice-ai-bot && sudo chmod 600 /opt/voice-ai-bot/.env'

ssh "$target" 'sudo sh -s' <<'REMOTE'
set -eu

needs_apt=0
for command in arecord aplay ffmpeg fswebcam gpiodetect rsync v4l2-ctl; do
  if ! command -v "$command" >/dev/null 2>&1; then
    needs_apt=1
  fi
done
if ! python3 -m venv --help >/dev/null 2>&1; then
  needs_apt=1
fi
if ! python3 - <<'PY'
import importlib.util
import sys

missing = [
    module
    for module in ("gpiozero", "lgpio", "RPi.GPIO")
    if importlib.util.find_spec(module) is None
]
sys.exit(1 if missing else 0)
PY
then
  needs_apt=1
fi
if [ "$needs_apt" -eq 1 ]; then
  apt-get update
  apt-get install -y alsa-utils python3-venv python3-gpiozero python3-lgpio python3-rpi-lgpio gpiod rsync ffmpeg fswebcam v4l-utils
else
  echo "system dependencies already present; skipping apt"
fi

CONFIG=/boot/firmware/config.txt
BACKUP=/boot/firmware/config.txt.voice-ai-bot.bak
[ -f "$BACKUP" ] || cp "$CONFIG" "$BACKUP"
if ! grep -q '^dtoverlay=googlevoicehat-soundcard' "$CONFIG"; then
  cat >> "$CONFIG" <<'EOF'

# Google AIY Voice HAT
dtparam=i2c_arm=on
dtparam=i2s=on
dtoverlay=googlevoicehat-soundcard
EOF
fi

usermod -aG audio,gpio,i2c,spi,input,video pi

cd /opt/voice-ai-bot
if [ ! -x .venv/bin/python ]; then
  python3 -m venv --system-site-packages .venv
fi
.venv/bin/python - <<'PY' || .venv/bin/python -m pip install setuptools wheel
import setuptools  # noqa: F401
import wheel  # noqa: F401
PY
.venv/bin/python -m pip install --upgrade --no-build-isolation .
.venv/bin/python - <<'PY'
from pathlib import Path
from importlib import metadata

dist = metadata.distribution("voice-ai-bot")
scripts = {entry.name for entry in dist.entry_points if entry.group == "console_scripts"}
for script in ("voice-ai-bot", "voice-ai-bot-debug-web"):
    if script not in scripts:
        raise SystemExit(f"{script} console script is missing from package metadata")
    if not Path(".venv/bin", script).is_file():
        raise SystemExit(f"{script} console script was not installed into the venv")
import voice_ai_bot.config  # noqa: F401
import voice_ai_bot.debug_web  # noqa: F401
PY
python3 - <<'PY'
from pathlib import Path

env_file = Path("/opt/voice-ai-bot/.env")
if env_file.exists():
    lines = env_file.read_text().splitlines()
    cleaned = [line for line in lines if not line.startswith("PYTHONPATH=")]
    env_file.write_text("".join(f"{line}\n" for line in cleaned))
PY
rm -rf /opt/voice-ai-bot/build
rm -rf /opt/voice-ai-bot/vendor

install -m 0644 systemd/voice-ai-bot.service /etc/systemd/system/voice-ai-bot.service
install -m 0644 systemd/voice-ai-bot-debug.service /etc/systemd/system/voice-ai-bot-debug.service
systemctl daemon-reload
systemctl enable voice-ai-bot.service
systemctl enable voice-ai-bot-debug.service
systemctl restart voice-ai-bot.service
systemctl restart voice-ai-bot-debug.service
REMOTE

ssh "$target" 'systemctl --no-pager --full status voice-ai-bot.service voice-ai-bot-debug.service || true'
