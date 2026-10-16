#!/usr/bin/env bash
# Install or restart Method on the Mac mini. Run on the mini once the
# code is on its backtester clone (git_sync pulls origin/master every 30 min):
#
#     ssh mini 'bash ~/.openclaw/workspace/backtester/method/deploy/install_mini.sh'
#
# Installs the launchd job (com.zach.method, 127.0.0.1:8506), puts it on the
# tailnet with `tailscale serve` on :8445, and checks /api/health. Cron lines
# for the refresh job and the playbook email are printed, not installed.
set -euo pipefail

REPO="$HOME/.openclaw/workspace/backtester"
LABEL=com.zach.method
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
TS=/Applications/Tailscale.app/Contents/MacOS/Tailscale
PORT=8506

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
sleep 1
# 8505 turned out to be Aspen's public connector (10/15): never share a port
if lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null; then
  echo "port $PORT is taken by another service; refusing to install:"; lsof -nP -iTCP:$PORT -sTCP:LISTEN; exit 1
fi
cp "$REPO/method/deploy/$LABEL.plist" "$PLIST"
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "launchd: $LABEL loaded"

if ! "$TS" serve status 2>/dev/null | grep -q ':8445'; then
  "$TS" serve --bg --https=8445 http://127.0.0.1:$PORT
fi
"$TS" serve status | grep -A1 ':8445' || true

ok=""
for i in 1 2 3 4 5 6 7 8 9 10; do
  # Method's health carries "mode" and "version"; anything else is the wrong server
  if curl -fsS http://127.0.0.1:$PORT/api/health 2>/dev/null | grep -q '"version"'; then ok=1; break; fi
  sleep 2
done
[ -n "$ok" ] && echo "health: Method is up on $PORT" || { echo "health: Method did not come up; see ~/Library/Logs/method.log"; exit 1; }

PY="$HOME/.venvs/cron312/bin/python3"
cat <<CRON

Cron lines (add to the live crontab and to the matching dispatcher crontab file):
# Method refresh: mode and cadence are decided inside (premarket 5 min, market/evening 15 min)
*/5 4-19 * * 1-5 cd $REPO && PYTHONPATH=$REPO $PY -m method.refresh >> $HOME/logs/cron/method_refresh.log 2>&1
# Playbook email: 2+ GO on one side -> lessons email (evening for tomorrow, 7:30 catch-up)
0 17 * * 1-5 cd $REPO && PYTHONPATH=$REPO EXITMONITOR_RISK_MODEL_PATH=$HOME/.openclaw/workspace/Intrinsic/journal/risk_model.py $PY -m scripts.playbook_email --session evening >> $HOME/logs/cron/playbook_email.log 2>&1
30 7 * * 1-5 cd $REPO && PYTHONPATH=$REPO EXITMONITOR_RISK_MODEL_PATH=$HOME/.openclaw/workspace/Intrinsic/journal/risk_model.py $PY -m scripts.playbook_email --session morning >> $HOME/logs/cron/playbook_email.log 2>&1
CRON
