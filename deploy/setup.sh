#!/usr/bin/env bash
# VIENKARTINIS serverio paruošimas. Serveryje (root) įklijuoti:
#   curl -fsSL https://raw.githubusercontent.com/mantasbalcius1994-ship-it/irankine/main/deploy/setup.sh | bash
# Galima paleisti pakartotinai - nieko nesugadins.
set -euo pipefail
REPO=https://github.com/mantasbalcius1994-ship-it/irankine.git
SRC=/opt/toolbox-src
export DEBIAN_FRONTEND=noninteractive

echo "== 1/5 Paketai (1-3 min.)"
apt-get update -qq
apt-get install -y -qq git nginx python3-venv python3-pip rsync curl >/dev/null

echo "== 2/5 Kodas iš GitHub"
if [ -d "$SRC/.git" ]; then
  git -C "$SRC" fetch -q origin main && git -C "$SRC" reset -q --hard origin/main
else
  git clone -q "$REPO" "$SRC"
fi

echo "== 3/5 Įrankinės diegimas (2-5 min.)"
bash "$SRC/deploy/install.sh"

echo "== 4/5 Svetainė (nginx, prievadas 80)"
mkdir -p /root/nginx-backup
for f in /etc/nginx/sites-enabled/*; do
  [ -e "$f" ] || continue
  [ "$(basename "$f")" = toolbox ] && continue
  cp -aL "$f" /root/nginx-backup/ 2>/dev/null || true
  rm -f "$f"
  echo "   išjungta sena nginx svetainė: $(basename "$f") (kopija /root/nginx-backup)"
done
cp "$SRC/deploy/nginx-site.conf" /etc/nginx/sites-available/toolbox
ln -sf /etc/nginx/sites-available/toolbox /etc/nginx/sites-enabled/toolbox
nginx -t -q
systemctl enable -q nginx
systemctl restart nginx
if command -v ufw >/dev/null && ufw status | grep -q "Status: active"; then
  ufw allow 80/tcp >/dev/null && echo "   ugniasienė: atidarytas 80 prievadas"
fi

echo "== 5/5 Slaptažodis"
if ! grep -q '^TOOLBOX_USER=' /etc/toolbox.env; then
  U=""
  read -r -p "   Vartotojo vardas Įrankinei (Enter = be slaptažodžio): " U </dev/tty || true
  if [ -n "$U" ]; then
    read -r -s -p "   Slaptažodis: " P </dev/tty; echo
    printf 'TOOLBOX_USER=%s\nTOOLBOX_PASS=%s\n' "$U" "$P" >> /etc/toolbox.env
    systemctl restart toolbox
    echo "   slaptažodis nustatytas"
  fi
else
  echo "   jau nustatytas (keisti: nano /etc/toolbox.env, po to systemctl restart toolbox)"
fi

IP=$(hostname -I | awk '{print $1}')
echo
echo "=================================================================="
echo " Įrankinė įdiegta:  http://$IP/"
echo " Automatinis atnaujinimas iš GitHub: kas 2 min."
PID=$(ss -ltnpH 'sport = :8000' 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1 || true)
if [ -n "${PID:-}" ]; then
  UNIT=$(ps -o unit= -p "$PID" | xargs || true)
  echo
  echo " Prievadą 8000 (senas roadgrader) vis dar užima: ${UNIT:-procesas $PID}"
  if [ -n "${UNIT:-}" ] && [ "$UNIT" != "-" ]; then
    echo " Jei jo nebereikia:  systemctl disable --now $UNIT"
  fi
fi
echo "=================================================================="
