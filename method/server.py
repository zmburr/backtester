"""
Method server: one page for bounce and reversal setups, always current.

It serves what method/refresh.py writes (data/method/snapshot.json) plus the
lessons notes, past setup days, chart bars and the day's shot counter. It
does no scoring itself. Tailnet only: launchd runs it on 127.0.0.1:8506 and
`tailscale serve` puts it on https://zacharys-mac-mini.<tailnet>.ts.net:8445.

Run:  python -m method.server [--port 8506]
With METHOD_AUTORESTART=1 (launchd) it exits when the repo's HEAD moves, so a
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

from method import core
from method import review as rv
from method.refresh import METHOD_DIR, SIGNALS_DIR, SNAPSHOT, DATA_DIR, read_breadth
from data_queries import polygon_queries as pq
from scripts import playbook_email as pe
from scripts.evening_signal_log import next_trading_day

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("method.server")

STATIC = Path(__file__).resolve().parent / "static"
REPO = Path(__file__).resolve().parents[1]
OUTCOMES = DATA_DIR / "signal_outcomes.csv"
SHOTS = METHOD_DIR / "shots.json"
REVIEW_DIR = METHOD_DIR / "review"
REVIEW_CACHE_AFTER_DAYS = 5      # past days without a scorecard row are cached once they're this old
BARS_TTL_S = 300
TZ = "US/Eastern"

_bars_cache: dict[str, tuple[float, dict]] = {}
_history_cache: dict = {"key": None, "data": None}
_outcomes_cache: dict = {"mtime": None, "rows": [], "index": {}}
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


def outcomes() -> tuple[list[dict], dict]:
    """The scorecard's rows (control cohort dropped) and their (target, side, ticker) index."""
    mtime = OUTCOMES.stat().st_mtime if OUTCOMES.exists() else None
    if _outcomes_cache["mtime"] != mtime:
        rows = []
        if OUTCOMES.exists():
            with OUTCOMES.open(newline="", encoding="utf-8") as fh:
                rows = [r for r in csv.DictReader(fh) if r.get("cohort") != "control"]
        _outcomes_cache.update(mtime=mtime, rows=rows, index=core.index_outcomes(rows))
    return _outcomes_cache["rows"], _outcomes_cache["index"]


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
    rows, _ = outcomes()
    data = core.build_episodes(reports, rows, next_trading_day)
    _history_cache.update(key=key, data=data)
    return data


def _valid_ticker(t: str) -> bool:
    return bool(t) and t.isascii() and t.replace(".", "").isalnum() and len(t) <= 10


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
        return send_from_directory(STATIC, "manifest.webmanifest", mimetype="application/manifest+json")

    @app.get("/apple-touch-icon.png")
    @app.get("/apple-touch-icon-precomposed.png")
    def touch_icon():
        return send_from_directory(STATIC / "icons", "apple-touch-icon.png")

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

    @app.get("/api/review")
    def api_review():
        """One day replayed: ladder as of the night before, the open's grade, the rules'
        trades. Past days are cached on disk; today is replayed live each call."""
        side = request.args.get("side", "bounce")
        ticker = (request.args.get("ticker") or "").upper()
        date = request.args.get("date") or ""
        try:
            day = datetime.date.fromisoformat(date)
        except ValueError:
            abort(400)
        if side not in ("bounce", "reversal") or not _valid_ticker(ticker):
            abort(400)
        today = pd.Timestamp.now(tz=TZ).date()
        _, index = outcomes()
        row = index.get((date, side, ticker))
        outcome = None if row is None else {k: core.to_num(row.get(k)) for k in (
            "d0_pct", "d1_pct", "mfe_atr_d0", "exit_r", "entry_r", "atr_pct", "entry_open")}
        path = REVIEW_DIR / f"{side}_{ticker}_{date}.json"
        data = core.read_json(path) if day < today else None
        if data is None:
            from scanners.premarket_bounce_scanner import CAP_CACHE_FILE, _load_json, get_cap
            cap = get_cap(ticker, _load_json(CAP_CACHE_FILE))
            try:
                data = core.clean(rv.review(ticker, date, side, cap, outcome=outcome, live=day >= today))
            except Exception as e:
                logger.warning(f"review {side} {ticker} {date}: {e}")
                return jsonify(ticker=ticker, date=date, side=side, error=str(e), candles=[], trades=[])
            old_enough = (today - day).days > REVIEW_CACHE_AFTER_DAYS
            if day < today and data.get("candles") and (row is not None or old_enough):
                core.write_json(path, data)
        data["outcome"] = outcome
        return jsonify(data)

    @app.get("/api/bars")
    def bars():
        ticker = (request.args.get("ticker") or "").upper()
        if not _valid_ticker(ticker):
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
    ap = argparse.ArgumentParser(description="Method server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("METHOD_PORT", "8506")))
    args = ap.parse_args()
    if os.environ.get("METHOD_AUTORESTART") == "1":
        threading.Thread(target=_watch_head, args=(git_head(),), daemon=True).start()
    create_app().run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
