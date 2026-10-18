#!/usr/bin/env bash
# Install or restart Method's Hyperliquid watch on the Mac mini, once the code is on
# its backtester clone (git_sync pulls origin/master every 30 min):
#
#     bash ~/.openclaw/workspace/backtester/method/deploy/install_hl_watch.sh
#
# Leaves the Method server alone. Pushes go to the phones subscribed in Method.
set -euo pipefail

REPO="$HOME/.openclaw/workspace/backtester"
LABEL=com.zach.hlwatch
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/hlwatch.log"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
sleep 1
cp "$REPO/method/deploy/$LABEL.plist" "$PLIST"
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "launchd: $LABEL loaded"
sleep 5
tail -n 3 "$LOG"
