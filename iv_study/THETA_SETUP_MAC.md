# Theta Terminal setup on macOS

The IV top-profile block in the priority report needs a running Theta Terminal
(the Java process that serves the local REST API on `localhost:25503`). Without
it the report still sends — the IV block is skipped with one log line — so do
this whenever you want the IV data on the Mac.

> **IMPORTANT — use the v3 terminal.** The client (`options_replay/theta_client.py`,
> `iv_study/`) targets the **v3 REST API** (`/v3` paths on port **25503**). The
> `download-latest` / `download-stable` URLs serve the **v2** terminal (v1.7.0,
> port 25510, no `/v3` handlers) — that will 404 every IV call. Get the v3 jar
> from `download-unstable` (see below).

## 1. Install Java (once)

```bash
brew install openjdk        # formula — no sudo (the temurin cask needs a password prompt)
```

openjdk is keg-only, so put it on PATH (needed by `run_theta_terminal.sh` and any
cron/login-item):

```bash
echo 'export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"' >> ~/.zshrc
exec zsh -l
java -version    # should print openjdk 21+ (Homebrew build)
```

## 2. Download the v3 Theta Terminal (once)

```bash
mkdir -p ~/theta
curl -L -o ~/theta/ThetaTerminalv3.jar https://download-unstable.thetadata.us/ThetaTerminalv3.jar
```

The v3 jar is ~42 MB (the v2 one is ~11 MB — if you got 11 MB you grabbed v2 by
mistake). Sanity check: `unzip -p ~/theta/ThetaTerminalv3.jar META-INF/MANIFEST.MF`
should show `Main-Class: net.thetadata.Main` (v2 is `net.thetadata.terminal.App`).

## 3. Credentials (once)

v3 reads a 2-line creds file — email on line 1, password on line 2 (same
Standard-tier login as on the PC):

```bash
printf 'YOUR_THETA_EMAIL\nYOUR_THETA_PASSWORD\n' > ~/theta/creds.txt
chmod 600 ~/theta/creds.txt
```

## 4. Run it

```bash
./run_theta_terminal.sh          # from the backtester project root
```

The script launches the v3 jar with `--creds-file ~/theta/creds.txt` and serves
`http://localhost:25503`. Leave the process running.

## 5. Verify

```bash
curl "http://localhost:25503/v3/option/list/expirations?symbol=AAPL&format=json" | head -c 200
```

JSON out = working. Then confirm the Python side from the project root (system
python3 — no venv on the Mac; the daily crons run the same way):

```bash
PYTHONPATH="$PWD" python3 -m iv_study.live_profile NVDA 2024-02-12
```

That should print a profile JSON (this date is a study trade, so it also
sanity-checks the whole fetch path).

## 6. Keep it running before the morning report (optional)

Simplest: add `run_theta_terminal.sh` as a Login Item, or run it in a tmux
session. If you want launchd, a minimal `~/Library/LaunchAgents/us.thetadata.terminal.plist`
running `/opt/homebrew/opt/openjdk/bin/java -jar /Users/<you>/theta/ThetaTerminalv3.jar
--creds-file /Users/<you>/theta/creds.txt` with `RunAtLoad + KeepAlive = true`
does it; `launchctl load` the plist.

## Notes

- The client reads `THETA_BASE_URL` from the environment (defaults to
  `http://localhost:25503`) — nothing to configure if the terminal is local.
- v3 is REST-only (no streaming). If you ever need the v2 streaming terminal too,
  it listens on 25510/25520 and can run alongside — but they share the one
  account login (see below).
- Historical marks are pickle-cached forever under `data/options_replay_cache/`,
  so each ticker costs a handful of API calls once, then one new day per session.
- Only one machine can be logged into a Theta account at a time on some plans —
  if the Windows terminal is running, the Mac terminal may bump it (and vice
  versa). The report's graceful skip covers whichever box loses.
