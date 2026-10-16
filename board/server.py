"""
Setup Board server: one page for bounce and reversal setups, always current.

It serves what board/refresh.py writes (data/board/snapshot.json) plus the
lessons notes, past setup days, chart bars and the day's shot counter. It
does no scoring itself. Tailnet only: launchd runs it on 127.0.0.1:8505 and
`tailscale serve` puts it on https://zacharys-mac-mini.<tailnet>.ts.net:8445.

Run:  python -m board.server [--port 8505]
With BOARD_AUTORESTART=1 (launchd) it exits when the repo's HEAD moves, so a
git pull goes live on the next KeepAlive restart.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

import pandas as pd
from flask import Flask, abort, jsonify, request, send_from_directory

from board import core
from board.refresh import BOARD_DIR, SIGNALS_DIR, SNAPSHOT, DATA_DIR, read_breadth
from data_queries import polygon_queries as pq
from scripts import playbook_email as pe
from scripts.evening_signal_log import next_trading_day

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("board.server")

STATIC = Path(__file__).resolve().parent / "static"
REPO = Path(__file__).resolve().parents[1]
OUTCOMES = DATA_DIR / "signal_outcomes.csv"
SHOTS = BOARD_DIR / "shots.json"
BARS_TTL_S = 300
TZ = "US/Eastern"

_bars_cache: dict[str, tuple[float, dict]] = {}
_history_cache: dict = {"key": None, "data": None}
_shots_lock = threading.Lock()


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _et_epoch(ts: pd.Timestamp) -> int:
    """Shift so the chart's UTC axis reads ET wall-clock time."""
    return int(ts.tz_localize(None).timestamp()) if ts.tzinfo is None else \
        int(ts.tz_convert(TZ).tz_localize(None).timestamp())


def chart_bars(ticker: str) -> dict:
    """~6 months of daily bars and the latest session's 1-min bars (ext hours)."""
    today = pd.Timestamp.now(tz=TZ).date()
    daily = pq.get_levels_data(ticker, today.isoformat(), 190, 1, "day")
    day_rows = [] if daily is None else [
        {"time": ts.strftime("%Y-%m-%d"), "open": float(r.open), "high": float(r.high),
         "low": float(r.low), "close": float(r.close), "value": float(r.volume)}
        for ts, r in daily.iterrows()]
    intra, day = None, today
    for back in range(6):
        day = today - datetime.timedelta(days=back)
        intra = pq.get_intraday(ticker, day.isoformat(), 1, "minute")
        if intra is not None and not intra.empty:
            break
    min_rows = [] if intra is None or intra.empty else [
        {"time": _et_epoch(ts), "open": float(r.open), "high": float(r.high),
         "low": float(r.low), "close": float(r.close), "value": float(r.volume)}
        for ts, r in intra.iterrows()]
    return {"ticker": ticker, "daily": day_rows, "intraday": min_rows,
            "intraday_date": day.isoformat()}


def history() -> list[dict]:
    files = sorted(SIGNALS_DIR.glob("*_*.json"))
    key = (len(files), max((f.stat().st_mtime for f in files), default=0),
           OUTCOMES.stat().st_mtime if OUTCOMES.exists() else 0)
    if _history_cache["key"] == key:
        return _history_cache["data"]
    reports = {}
    for f in files:
        date_str, _, session = f.stem.partition("_")
        if session in ("morning", "evening"):
            rep = core.read_json(f)
            if rep:
                reports[(date_str, session)] = rep
    outcomes = []
    if OUTCOMES.exists():
        with OUTCOMES.open(newline="", encoding="utf-8") as fh:
            outcomes = [r for r in csv.DictReader(fh) if r.get("cohort") != "control"]
    data = core.build_episodes(reports, outcomes, next_trading_day)
    _history_cache.update(key=key, data=data)
    return data


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    version = git_head()

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/static/<path:name>")
    def static_file(name):
        return send_from_directory(STATIC, name)

    @app.get("/manifest.webmanifest")
    def manifest():
        return send_from_directory(STATIC, "manifest.webmanifest")

    @app.get("/api/health")
    def health():
        snap = core.read_json(SNAPSHOT) or {}
        age = None
        if snap.get("generated_at"):
            age = (datetime.datetime.now(datetime.timezone.utc)
                   - datetime.datetime.fromisoformat(snap["generated_at"])).total_seconds()
        return jsonify(ok=True, version=version, mode=snap.get("mode"), snapshot_age_s=age)

    @app.get("/api/snapshot")
    def snapshot():
        snap = core.read_json(SNAPSHOT)
        if not snap:
            return jsonify(empty=True, version=version)
        snap["breadth"] = read_breadth(snap["session"])
        snap["version"] = version
        return jsonify(core.clean(snap))

    @app.get("/api/lessons")
    def lessons():
        side = request.args.get("side", "bounce")
        if side not in pe.SIDES:
            abort(400)
        note = pe.find_note(side)
        sections = pe.parse_lessons(note.read_text(encoding="utf-8")) if note else None
        return jsonify(side=side, found=note is not None, tiers=pe.tiers_line(),
                       note=f"Consolidated Playbook › {pe.NOTE_NAMES[side][:-3]}",
                       sections=[{"heading": h, "items": [pe.md_inline(b) for b in items]}
                                 for h, items in (sections or [])])

    @app.get("/api/history")
    def api_history():
        return jsonify(episodes=history())

    @app.get("/api/bars")
    def bars():
        ticker = (request.args.get("ticker") or "").upper()
        if not ticker.isascii() or not ticker.replace(".", "").isalnum() or len(ticker) > 10:
            abort(400)
        hit = _bars_cache.get(ticker)
        if hit and time.time() - hit[0] < BARS_TTL_S:
            return jsonify(hit[1])
        try:
            data = core.clean(chart_bars(ticker))
        except Exception as e:
            logger.warning(f"bars {ticker}: {e}")
            return jsonify(ticker=ticker, daily=[], intraday=[], error=str(e))
        _bars_cache[ticker] = (time.time(), data)
        return jsonify(data)

    @app.route("/api/shots", methods=["GET", "POST"])
    def shots():
        day = request.args.get("date") or pd.Timestamp.now(tz=TZ).date().isoformat()
        with _shots_lock:
            book = core.read_json(SHOTS, {}) or {}
            if request.method == "POST":
                body = request.get_json(silent=True) or {}
                day = body.get("date") or day
                entries = book.setdefault(day, [])
                kind = body.get("kind")
                if kind == "undo":
                    if entries:
                        entries.pop()
                elif kind in ("shot", "stop") and body.get("ticker") and body.get("side") in pe.SIDES:
                    entries.append({"side": body["side"], "ticker": str(body["ticker"]).upper()[:10],
                                    "kind": kind, "t": pd.Timestamp.now(tz=TZ).strftime("%H:%M")})
                else:
                    abort(400)
                core.write_json(SHOTS, book)
            return jsonify(date=day, entries=book.get(day, []))

    return app


def _watch_head(start: str):
    """launchd KeepAlive restarts us; exiting on a new HEAD makes a pull go live."""
    while True:
        time.sleep(120)
        now = git_head()
        if now and start and now != start:
            logger.info(f"HEAD moved {start} -> {now}; exiting for a restart")
            os._exit(0)


def main():
    ap = argparse.ArgumentParser(description="Setup Board server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("BOARD_PORT", "8505")))
    args = ap.parse_args()
    if os.environ.get("BOARD_AUTORESTART") == "1":
        threading.Thread(target=_watch_head, args=(git_head(),), daemon=True).start()
    create_app().run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
