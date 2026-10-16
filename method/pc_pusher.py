"""
Method's real-time feed: Trillium 1-min bars from this PC to Method on the mini.

The mini can't reach Trillium (SHEL only resolves on the PC's network). This
loop posts each name's bar summary (method/live.summarize_bars) to /api/live:
the names that matter now every 15 s, the whole board every 2 min. Method
re-grades those rows as they arrive and falls back to Polygon when posts stop.

Runs 04:00-20:00 ET on session days. Task Scheduler starts it at 4 AM (waking
the PC); while it runs it keeps Windows from sleeping, and it exits at 8 PM.
With --update it first moves its own checkout to origin/master, so a push goes
live the next morning. It only updates a detached, clean checkout: never a
working folder on a branch.

Usage (from a checkout of the backtester repo):
    python -m method.pc_pusher               # the day's loop
    python -m method.pc_pusher --update      # git update, then the loop (Task Scheduler)
    python -m method.pc_pusher --once --dry  # one pass over the board: print, don't post

Needs SHEL_API_PWD (from the backtester .env, METHOD_ENV_FILE) and the token
Method made in the mini's ~/.method/live_token, copied to ~/.method/live_token here.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from method.live import summarize_bars

TZ = "US/Eastern"
REPO = Path(__file__).resolve().parents[1]
METHOD_URL = os.environ.get("METHOD_URL", "https://zacharys-mac-mini.tailca90a1.ts.net:8445")
ENV_FILE = os.environ.get("METHOD_ENV_FILE", r"C:\Users\zmbur\PycharmProjects\backtester\.env")
TOKEN_FILE = Path.home() / ".method" / "live_token"
LOG_DIR = Path.home() / "logs" / "method_pusher"
ACTIVE_EVERY_S = 15
FULL_EVERY_S = 120
START, STOP = datetime.time(4, 0), datetime.time(20, 0)

logger = logging.getLogger("method.pc_pusher")


def now_et() -> datetime.datetime:
    return pd.Timestamp.now(tz=TZ).to_pydatetime()


def is_session_day(day: datetime.date) -> bool:
    import pandas_market_calendars as mcal
    return not mcal.get_calendar("NYSE").schedule(start_date=day, end_date=day).empty


def keep_awake(on: bool) -> None:
    """Hold off Windows sleep while the feed runs (the display may still turn off)."""
    if os.name != "nt":
        return
    import ctypes
    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system_required if on else 0))


# ---------------------------------------------------------------- names

def pick_names(snap: dict) -> tuple[list[str], list[str]]:
    """(active, board). Active = what can change a decision now: names the board or
    the priority report rates GO/CAUTION, and every reversal row. The board also carries
    the market-breadth names near their flag price until the open's count locks, so
    the count moves in real time (method/breadth.py)."""
    bounce = (snap.get("bounce") or {}).get("rows", [])
    reversal = (snap.get("reversal") or {}).get("rows", [])
    active = [r["ticker"] for r in bounce
              if r.get("rec") in ("GO", "CAUTION") or r.get("report_rec") in ("GO", "CAUTION")]
    active += [r["ticker"] for r in reversal]
    board = [r["ticker"] for r in bounce] + [r["ticker"] for r in reversal]
    mb = snap.get("market_breadth") or {}
    if mb.get("status") == "ok" and not mb.get("locked"):
        board += list(mb.get("names") or []) + [x["ticker"] for x in mb.get("near") or []]
    return list(dict.fromkeys(active)), list(dict.fromkeys(board))


def fetch_snapshot() -> dict:
    r = requests.get(f"{METHOD_URL}/api/snapshot", timeout=20)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------- Trillium

def bars_frame(aggs: list) -> pd.DataFrame | None:
    """SHEL bar-1min messages -> OHLC indexed by bar START (ET), like Polygon's.
    SHEL stamps a bar with its close time, so each stamp moves back one minute."""
    rows = [a if isinstance(a, dict) else vars(a) for a in aggs]
    df = pd.DataFrame(rows)
    if df.empty or "close-time" not in df.columns:
        return None
    start = pd.to_datetime(df["close-time"], unit="ns", utc=True).dt.tz_convert(TZ) - pd.Timedelta(minutes=1)
    out = df[["open", "high", "low", "close"]].astype(float)
    out.index = pd.DatetimeIndex(start)
    return out.sort_index()


def trillium_summaries(tickers: list[str], day: str) -> dict:
    """{ticker: bar summary} over one SHEL session; a name that fails is left out."""
    import sheldatagateway
    from sheldatagateway import environments
    user, pwd = os.getenv("SHEL_USER", "zburr"), os.getenv("SHEL_API_PWD")
    d = datetime.date.fromisoformat(day)
    out = {}
    with sheldatagateway.Session(environments.env_defs.Prod, user, pwd) as session:
        for t in tickers:
            aggs: list = []
            try:
                handle = session.request_data(callback=aggs.append, symbol=t, start_date=d,
                                              end_date=d, subscriptions=["bar-1min"])
                handle.wait()
                handle.raise_on_error()
                summ = summarize_bars(bars_frame(aggs), day)
            except Exception as e:
                logger.debug(f"{t}: {e}")
                continue
            if summ:
                out[t] = summ
    return out


# ---------------------------------------------------------------- posting

def read_token() -> str:
    tok = os.environ.get("METHOD_LIVE_TOKEN") or (TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else "")
    if not tok:
        raise SystemExit(f"no token: copy the mini's ~/.method/live_token to {TOKEN_FILE}")
    return tok


def post(day: str, names: dict, token: str) -> dict:
    r = requests.post(f"{METHOD_URL}/api/live", json={"day": day, "names": names},
                      headers={"X-Method": "1", "X-Method-Token": token}, timeout=20)
    r.raise_for_status()
    return r.json()


def run_loop(dry: bool = False, once: bool = False) -> int:
    now = now_et()
    day = now.date()
    if not once and not is_session_day(day):
        logger.info(f"{day} is not a session day; nothing to stream")
        return 0
    token = "" if dry else read_token()
    stop = datetime.datetime.combine(day, STOP, tzinfo=now.tzinfo)
    active, board, last_full, fails = [], [], 0.0, 0
    keep_awake(True)
    try:
        while once or now_et() < stop:
            t0 = time.time()
            full = once or t0 - last_full >= FULL_EVERY_S or not board
            try:
                if full:
                    active, board = pick_names(fetch_snapshot())
                tickers = board if full else active
                summ = trillium_summaries(tickers, day.isoformat()) if tickers else {}
                if dry:
                    for t, s in list(summ.items())[:12]:
                        print(f"{t:6} {s['price']:>10.2f} @ {s['bar_time']}  pm_low {s.get('pm_low')}  open {s.get('open')}")
                    print(f"{len(summ)}/{len(tickers)} names from Trillium in {time.time() - t0:.1f} s")
                elif summ:
                    res = post(day.isoformat(), summ, token)
                    if full or res.get("alerts"):
                        logger.info(f"{'board' if full else 'active'}: {len(summ)}/{len(tickers)} names, "
                                    f"re-graded {res.get('regraded')}, alerts {res.get('alerts') or []} "
                                    f"({time.time() - t0:.1f} s)")
                elif tickers:
                    logger.warning(f"Trillium returned nothing for {len(tickers)} names")
                if full:
                    last_full = t0
                fails = 0
            except Exception as e:
                fails += 1
                logger.warning(f"pass failed ({fails} in a row): {e}")
            if once:
                break
            time.sleep(max(1.0, ACTIVE_EVERY_S * (1 if fails < 4 else 4) - (time.time() - t0)))
    finally:
        keep_awake(False)
    logger.info("feed stopped")
    return 0


# ---------------------------------------------------------------- self-update

def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                          timeout=120).stdout.strip()


def self_update() -> bool:
    """Move this checkout to origin/master if it's a detached, clean pusher checkout."""
    if _git("symbolic-ref", "-q", "HEAD"):
        logger.warning("checkout is on a branch: not updating it (use a detached pusher worktree)")
        return False
    if _git("status", "--porcelain", "--untracked-files=no"):
        logger.warning("checkout has local edits: not updating it")
        return False
    _git("fetch", "-q", "origin", "master")
    target = _git("rev-parse", "FETCH_HEAD")
    if target and target != _git("rev-parse", "HEAD"):
        _git("checkout", "-q", "--detach", target)
        logger.info(f"updated to {target[:8]}")
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Trillium -> Method feed (runs on the PC)")
    ap.add_argument("--update", action="store_true", help="git-update this checkout first, then run")
    ap.add_argument("--once", action="store_true", help="one pass over the whole board")
    ap.add_argument("--dry", action="store_true", help="print what would be posted")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(),
                logging.FileHandler(LOG_DIR / f"{datetime.date.today().isoformat()}.log", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=handlers)
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE)

    if args.update:
        self_update()
        # the updated code runs in a fresh process
        rest = [a for a in sys.argv[1:] if a != "--update"]
        sys.exit(subprocess.run([sys.executable, "-m", "method.pc_pusher", *rest], cwd=str(REPO)).returncode)
    sys.exit(run_loop(dry=args.dry, once=args.once))


if __name__ == "__main__":
    main()
