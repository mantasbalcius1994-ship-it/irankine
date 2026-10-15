#!/usr/bin/env bash
# Įrankinės diegimas / atnaujinimas (Ubuntu/Debian). Kviečiamas iš setup.sh ir update.sh.
# Kartojant tik atnaujina kodą ir priklausomybes; duomenų (/var/lib/toolbox) ir /etc/toolbox.env neliečia.
set -euo pipefail

APP_DIR=/opt/toolbox
DATA_DIR=/var/lib/toolbox
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export DEBIAN_FRONTEND=noninteractive

apt-get install -y -qq python3-venv python3-pip rsync curl >/dev/null

# 2 GB swap apsaugai nuo atminties šuolių (jei swap dar nėra)
if [ "$(swapon --show | wc -l)" -eq 0 ] && [ ! -f /swapfile ]; then
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  echo "Sukurtas 2 GB swap"
fi

id toolbox &>/dev/null || useradd --system --home "$DATA_DIR" --shell /usr/sbin/nologin toolbox
mkdir -p "$APP_DIR" "$DATA_DIR"

rsync -a --delete --exclude .git --exclude data --exclude '__pycache__' --exclude '.venv' "$SRC_DIR/" "$APP_DIR/"

[ -d "$APP_DIR/.venv" ] || python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install -q --upgrade pip
"$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"

[ -f /etc/toolbox.env ] || cp "$APP_DIR/deploy/toolbox.env.example" /etc/toolbox.env
[ -f "$DATA_DIR/links.json" ] || cp "$APP_DIR/deploy/links.json.example" "$DATA_DIR/links.json"
chown -R toolbox:toolbox "$DATA_DIR"

cp "$APP_DIR/deploy/toolbox.service" /etc/systemd/system/toolbox.service
# automatinis atnaujinimas iš GitHub (tik jei kodas paimtas per git)
if [ -d "$SRC_DIR/.git" ]; then
  cp "$APP_DIR/deploy/toolbox-update.service" "$APP_DIR/deploy/toolbox-update.timer" /etc/systemd/system/
fi
systemctl daemon-reload
systemctl enable -q toolbox
systemctl restart toolbox
[ -d "$SRC_DIR/.git" ] && systemctl enable -q --now toolbox-update.timer

for i in $(seq 1 30); do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8010/api/info || true)
  if [ "$code" = "200" ] || [ "$code" = "401" ]; then
    echo "OK: Įrankinė veikia (127.0.0.1:8010)"; exit 0
  fi
  sleep 1
done
echo "KLAIDA: Įrankinė nepasileido. Žurnalas:"
journalctl -u toolbox -n 30 --no-pager
exit 1
