"""
Method's real-time feed: Trillium 1-min bars from this PC to Method on the mini.

The mini can't reach Trillium (SHEL only resolves on the PC's network). This
loop posts each name's bar summary (method/live.summarize_bars) to /api/live:
the names that matter now every 15 s, the whole board every 2 min. Method
re-grades those rows as they arrive and falls back to Polygon when posts stop.

On a breadth day each flagged name's summary also carries the 2-min entry rule's
state (method/watch.rule_state, from the same bars), and the server pushes new
entries to the phone. Passes start 3 s after each 15-s mark, so one lands right
after every 2-min bar closes.

Runs 04:00-20:00 ET on session days. Task Scheduler starts it at 4 AM (waking
the PC); while it runs it keeps Windows from sleeping, and it exits at 8 PM.
With --update it first moves its own checkout to origin/master, so a push goes
live the next morning. It only updates a detached, clean checkout: never a
working folder on a branch.

Usage (from a checkout of the backtester repo):
    python -m method.pc_pusher               # the day's loop
    python -m method.pc_pusher --update      # git update, then the loop (Task Scheduler)
    python -m method.pc_pusher --once --dry  # one pass over the board: print, don't post
    python -m method.pc_pusher --playback 2026-10-15   # rehearse a past breadth morning's entry pushes

Needs SHEL_API_PWD (from the backtester .env, METHOD_ENV_FILE) and the token
Method made in the mini's ~/.method/live_token, copied to ~/.method/live_token here.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import math
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
PASS_OFFSET_S = 3                # past each 15-s mark: a 2-min bar's last 1-min bar is in by then
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
    active += list(rule_names(snap))          # a breadth day's names are watched for entries
    return list(dict.fromkeys(active)), list(dict.fromkeys(board + active))


def rule_names(snap: dict) -> dict:
    """{ticker: ATR} for the names the entry rule runs on: in market hours on a breadth day,
    every flagged name in the counted universe (method/watch.watch_names), with its row's ATR."""
    mb = snap.get("market_breadth") or {}
    if snap.get("mode") != "market" or mb.get("status") != "ok" or mb.get("state") != "breadth":
        return {}
    detail = mb.get("detail") or {}
    atr = {r["ticker"]: r.get("atr") for r in (snap.get("bounce") or {}).get("rows", [])}
    names = list(mb.get("names") or []) + list(mb.get("relaxed") or [])
    return {t: atr.get(t) for t in dict.fromkeys(names) if (detail.get(t) or {}).get("counted")}


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


def trillium_bars(tickers: list[str], day: str):
    """Yield (ticker, 1-min bars) for each name over one SHEL session; a name that fails is skipped."""
    import sheldatagateway
    from sheldatagateway import environments
    user, pwd = os.getenv("SHEL_USER", "zburr"), os.getenv("SHEL_API_PWD")
    d = datetime.date.fromisoformat(day)
    with sheldatagateway.Session(environments.env_defs.Prod, user, pwd) as session:
        for t in tickers:
            aggs: list = []
            try:
                handle = session.request_data(callback=aggs.append, symbol=t, start_date=d,
                                              end_date=d, subscriptions=["bar-1min"])
                handle.wait()
                handle.raise_on_error()
                bars = bars_frame(aggs)
            except Exception as e:
                logger.debug(f"{t}: {e}")
                continue
            if bars is not None and not bars.empty:
                yield t, bars


def trillium_summaries(tickers: list[str], day: str, rules_for: dict | None = None) -> dict:
    """{ticker: bar summary}; names in `rules_for` ({ticker: ATR}) also carry the entry rule's state."""
    rule_state, failed = None, []
    if rules_for:
        try:
            from method.watch import rule_state
        except Exception as e:                # prices still go out; the alerts are what's lost
            logger.warning(f"entry rule unavailable, posting prices only: {e}")
    out = {}
    for t, bars in trillium_bars(tickers, day):
        try:
            summ = summarize_bars(bars, day)
        except Exception as e:
            logger.debug(f"{t}: {e}")
            continue
        if not summ:
            continue
        if rule_state and t in rules_for:
            try:
                rules = rule_state(bars, rules_for[t])
                if rules:
                    summ["rules"] = rules
            except Exception as e:
                failed.append(f"{t}: {e}")
        out[t] = summ
    if failed:
        logger.warning(f"entry rule failed on {len(failed)} names, first {failed[0]}")
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
    active, board, rules_for, last_full, fails = [], [], {}, 0.0, 0
    keep_awake(True)
    try:
        while once or now_et() < stop:
            t0 = time.time()
            full = once or t0 - last_full >= FULL_EVERY_S or not board
            try:
                if full:
                    snap = fetch_snapshot()
                    active, board = pick_names(snap)
                    rules_for = rule_names(snap)
                tickers = board if full else active
                summ = trillium_summaries(tickers, day.isoformat(), rules_for) if tickers else {}
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
            time.sleep(wait_s(time.time(), ACTIVE_EVERY_S * (1 if fails < 4 else 4)))
    finally:
        keep_awake(False)
    logger.info("feed stopped")
    return 0


def wait_s(now_s: float, every: float, offset: float = PASS_OFFSET_S) -> float:
    """Seconds to the next pass: `offset` s past the next `every`-s mark of the clock (at
    least 1 s away), so 15-s passes start just after each 2-min bar closes."""
    nxt = math.floor((now_s - offset) / every + 1) * every + offset
    if nxt - now_s < 1.0:
        nxt += every
    return nxt - now_s


# ---------------------------------------------------------------- rehearsal

def polygon_bars(tickers: list[str], day: str):
    """Yield (ticker, 1-min bars, ET) from Polygon's cache: the replay's own bars, for a
    rehearsal when Trillium is closed (nights and weekends)."""
    from bounce_entry_study.fetch_bars import fetch_day
    for t in tickers:
        bars = fetch_day(t, day)
        if bars is None or bars.empty:
            continue
        idx = pd.DatetimeIndex(bars.index)
        bars = bars.set_axis(idx.tz_localize(TZ) if idx.tz is None else idx.tz_convert(TZ))
        yield t, bars[["open", "high", "low", "close"]]


def playback(day: str, until: str = "11:00", source: str = "trillium") -> int:
    """Rehearse a past breadth morning: the day's flagged names (its record in
    data/breadth_signals) on Trillium bars (or Polygon's), stepped like the live loop (3 s past
    every 15-s mark), through the server's own alert code with a push that prints. Fetch time
    isn't modelled."""
    import json
    import tempfile
    from method import breadth, watch
    rec = json.loads(breadth.signal_path(day).read_text())
    sigs = {s["ticker"]: s for s in rec["signals"]}
    mb = {"status": "ok", "session": day, "locked": True, "state": rec["state"], "n": rec["n"],
          "band": breadth.band_for(rec["n"]),
          "names": [t for t, s in sigs.items() if s.get("recommendation") == "STRICT"],
          "relaxed": [t for t, s in sigs.items() if s.get("recommendation") == "RELAXED"],
          "detail": {t: {"counted": bool(s.get("counted"))} for t, s in sigs.items()}}
    names = watch.watch_names(mb)
    if not names:
        print(f"{day}: {rec['state']} day ({rec['n']} names), no entry alerts")
        return 0
    rows, atr = [], {}
    for t in names:
        m = sigs[t].get("metrics") or {}
        atr[t] = m["atr_pct"] * m["open"] if m.get("atr_pct") and m.get("open") else None
        rows.append({"ticker": t, "score_open": sigs[t].get("score"), "atr": atr[t]})
    t0 = time.time()
    bars = dict((polygon_bars if source == "polygon" else trillium_bars)(names, day))
    print(f"{day}: {mb['band']} names ({rec['n']}), watching {len(names)}; {source} bars for "
          f"{len(bars)} in {time.time() - t0:.1f} s")
    path = Path(tempfile.mkdtemp()) / f"entry_alerts_{day}.json"
    pushes, delays = [], []
    clock = {"now": None}

    def send(title, body, url="/", tag=""):
        pushes.append(clock["now"])
        print(f"\n{clock['now']:%H:%M:%S}  {title}\n{body}")
        return 1

    now = pd.Timestamp(f"{day} 09:30:{PASS_OFFSET_S:02d}", tz=TZ)
    stop = pd.Timestamp(f"{day} {until}", tz=TZ)
    while now <= stop:
        clock["now"] = now
        feed = {t: {"rules": watch.rule_state(b, atr.get(t), now=now)} for t, b in bars.items()}
        for key in watch.entry_alerts(day, mb, rows, feed, now.to_pydatetime(), send, path):
            t, n = key.split("#")
            e = feed[t]["rules"]["entries"][int(n) - 1]
            delays.append((now - pd.Timestamp(f"{day} {e['t']}", tz=TZ)).total_seconds())
        now += pd.Timedelta(seconds=ACTIVE_EVERY_S)
    late = (json.loads(path.read_text()).get("late") if path.exists() else None) or []
    print(f"\n{len(pushes)} pushes, {len(delays)} entries pushed (each {max(delays, default=0):.0f} s or "
          f"less after its bar closed), {len(late)} late")
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
    ap.add_argument("--playback", metavar="YYYY-MM-DD", help="rehearse a past breadth morning's entry pushes")
    ap.add_argument("--until", default="11:00", help="--playback: stop at this ET time")
    ap.add_argument("--polygon", action="store_true", help="--playback on Polygon's bars (Trillium closed)")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(),
                logging.FileHandler(LOG_DIR / f"{datetime.date.today().isoformat()}.log", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=handlers)
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE)

    if args.playback:
        sys.exit(playback(args.playback, args.until, "polygon" if args.polygon else "trillium"))
    if args.update:
        self_update()
        # the updated code runs in a fresh process
        rest = [a for a in sys.argv[1:] if a != "--update"]
        sys.exit(subprocess.run([sys.executable, "-m", "method.pc_pusher", *rest], cwd=str(REPO)).returncode)
    sys.exit(run_loop(dry=args.dry, once=args.once))


if __name__ == "__main__":
    main()
