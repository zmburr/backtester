# Theta Terminal setup on macOS

The IV top-profile block in the priority report needs a running Theta Terminal
(the Java process that serves the local REST API on `localhost:25503`). Without
it the report still sends — the IV block is skipped with one log line — so do
this whenever you want the IV data on the Mac.

## 1. Install Java (once)

```bash
brew install --cask temurin
java -version   # any recent LTS (17/21+) is fine
```

## 2. Download Theta Terminal (once)

```bash
mkdir -p ~/theta
curl -L -o ~/theta/ThetaTerminal.jar https://download-latest.thetadata.us
```

(If that URL changes, grab the latest ThetaTerminal.jar from the downloads
section at https://www.thetadata.net — same Standard-tier login as on the PC.)

## 3. Run it

```bash
./run_theta_terminal.sh          # from the backtester project root
# or directly:
java -jar ~/theta/ThetaTerminal.jar
```

First run prompts for your Theta Data email + password and stores the creds.
Leave the process running — it serves `http://localhost:25503`.

## 4. Verify

```bash
curl "http://localhost:25503/v3/option/list/expirations?symbol=AAPL&format=json" | head -c 200
```

JSON out = working. Then confirm the Python side from the project root:

```bash
venv/bin/python -m iv_study.live_profile NVDA 2024-02-12
```

That should print a profile JSON (this date is a study trade, so it also
sanity-checks the whole fetch path).

## 5. Keep it running before the morning report (optional)

Simplest: add `run_theta_terminal.sh` as a Login Item, or run it in a tmux
session. If you want launchd, a minimal `~/Library/LaunchAgents/us.thetadata.terminal.plist`
with `ProgramArguments = [java, -jar, /Users/<you>/theta/ThetaTerminal.jar]`
and `RunAtLoad + KeepAlive = true` does it; `launchctl load` the plist.

## Notes

- The client reads `THETA_BASE_URL` from the environment (defaults to
  `http://localhost:25503`) — nothing to configure if the terminal is local.
- Historical marks are pickle-cached forever under `data/options_replay_cache/`,
  so each ticker costs a handful of API calls once, then one new day per session.
- Only one machine can be logged into a Theta account at a time on some plans —
  if the Windows terminal is running, the Mac terminal may bump it (and vice
  versa). The report's graceful skip covers whichever box loses.
