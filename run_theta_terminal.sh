#!/bin/bash
# Start Theta Terminal (serves the local options-data REST API on :25503).
# Setup: iv_study/THETA_SETUP_MAC.md
JAR="${THETA_JAR:-$HOME/theta/ThetaTerminal.jar}"
if [ ! -f "$JAR" ]; then
    echo "ThetaTerminal.jar not found at $JAR — see iv_study/THETA_SETUP_MAC.md" >&2
    exit 1
fi
exec java -jar "$JAR"
