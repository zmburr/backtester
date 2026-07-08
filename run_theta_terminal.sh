#!/bin/bash
# Start Theta Terminal v3 (serves the local options-data REST API on :25503).
# Setup: iv_study/THETA_SETUP_MAC.md
#
# v3 reads credentials from a 2-line creds file (email, password). Point at it
# with THETA_CREDS, or drop it at ~/theta/creds.txt (chmod 600).
JAR="${THETA_JAR:-$HOME/theta/ThetaTerminalv3.jar}"
CREDS="${THETA_CREDS:-$HOME/theta/creds.txt}"

# Resolve java — openjdk is keg-only, so it may not be on cron/launchd PATH.
JAVA="$(command -v java)"
[ -x /opt/homebrew/opt/openjdk/bin/java ] && JAVA=/opt/homebrew/opt/openjdk/bin/java

if [ -z "$JAVA" ]; then
    echo "java not found — install with 'brew install openjdk' (see iv_study/THETA_SETUP_MAC.md)" >&2
    exit 1
fi
if [ ! -f "$JAR" ]; then
    echo "ThetaTerminalv3.jar not found at $JAR — see iv_study/THETA_SETUP_MAC.md" >&2
    exit 1
fi

if [ -f "$CREDS" ]; then
    exec "$JAVA" -jar "$JAR" --creds-file "$CREDS"
else
    echo "No creds file at $CREDS — see iv_study/THETA_SETUP_MAC.md (need email/password, 2 lines)." >&2
    exit 1
fi
