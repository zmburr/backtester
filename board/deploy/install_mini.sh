#!/usr/bin/env bash
# Install or restart the Setup Board on the Mac mini. Run on the mini once the
# code is on its backtester clone (git_sync pulls origin/master every 30 min):
#
#     ssh mini 'bash ~/.openclaw/workspace/backtester/board/deploy/install_mini.sh'
#
# Installs the launchd job (com.zach.board, 127.0.0.1:8505), puts it on the
# tailnet with `tailscale serve` on :8445, and checks /api/health. Cron lines
# for the refresh job and the playbook email are printed, not installed.
set -euo pipefail

REPO="$HOME/.openclaw/workspace/backtester"
LABEL=com.zach.board
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
TS=/Applications/Tailscale.app/Contents/MacOS/Tailscale

cp "$REPO/board/deploy/$LABEL.plist" "$PLIST"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "launchd: $LABEL loaded"

if ! "$TS" serve status 2>/dev/null | grep -q ':8445'; then
  "$TS" serve --bg --https=8445 http://127.0.0.1:8505
fi
"$TS" serve status | grep -A1 ':8445' || true

for i in 1 2 3 4 5 6 7 8 9 10; do
  if curl -fsS http://127.0.0.1:8505/api/health; then echo; break; fi
  sleep 2
done

PY="$HOME/.venvs/cron312/bin/python3"
cat <<CRON

Cron lines (add to the live crontab and to the matching dispatcher crontab file):
# Setup Board refresh: mode and cadence are decided inside (premarket 5 min, market/evening 15 min)
*/5 4-19 * * 1-5 cd $REPO && PYTHONPATH=$REPO $PY -m board.refresh >> $HOME/logs/cron/board_refresh.log 2>&1
# Playbook email: 2+ GO on one side -> lessons email (evening for tomorrow, 7:30 catch-up)
0 17 * * 1-5 cd $REPO && PYTHONPATH=$REPO EXITMONITOR_RISK_MODEL_PATH=$HOME/.openclaw/workspace/Intrinsic/journal/risk_model.py $PY -m scripts.playbook_email --session evening >> $HOME/logs/cron/playbook_email.log 2>&1
30 7 * * 1-5 cd $REPO && PYTHONPATH=$REPO EXITMONITOR_RISK_MODEL_PATH=$HOME/.openclaw/workspace/Intrinsic/journal/risk_model.py $PY -m scripts.playbook_email --session morning >> $HOME/logs/cron/playbook_email.log 2>&1
CRON
