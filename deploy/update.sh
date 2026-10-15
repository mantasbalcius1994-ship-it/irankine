#!/usr/bin/env bash
# Automatinis atnaujinimas (toolbox-update.timer, kas 2 min.):
# jei GitHub'e yra nauja versija ir šiuo metu nevyksta skaičiavimas - įdiegia ją.
# Visas skriptas { } bloke: bash jį perskaito iš karto, todėl git reset gali saugiai pakeisti šį failą.
{
set -euo pipefail
SRC=/opt/toolbox-src
cd "$SRC"
git fetch -q origin main
if [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ]; then exit 0; fi
if grep -lqsE '"status": "(running|queued)"' /var/lib/toolbox/jobs/*/job.json; then
  echo "Vyksta skaičiavimas - atnaujinsiu vėliau"; exit 0
fi
git reset -q --hard origin/main
echo "Atnaujinama į $(git rev-parse --short HEAD): $(git log -1 --format=%s)"
bash deploy/install.sh
exit $?
}
