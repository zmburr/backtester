"""
Method server: one page for bounce and reversal setups, always current.

It serves what method/refresh.py writes (data/method/snapshot.json) plus the
lessons notes, past setup days, chart bars and the day's shot counter. Its only
scoring is the PC's Trillium posts (/api/live): those rows are re-graded on the
spot from the session's cached statics (refresh.apply_live). Tailnet only: launchd runs it on 127.0.0.1:8506 and
`tailscale serve` puts it on https://zacharys-mac-mini.<tailnet>.ts.net:8445.

Only the owner's tailnet login gets in (as Intrinsic): tailscale serve adds a
Tailscale-User-Login header; METHOD_ALLOWED_LOGINS lists who may pass (default the
owner) and METHOD_DEV_LOGIN stands in for the header on local runs. Writes must be
JSON with X-Method: 1. Price posts also need the token in ~/.method/live_token
(the PC keeps a copy; see method/pc_pusher.py).

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
from flask import Flask, abort, jsonify, make_response, request, send_from_directory

from method import core
from method import library as lib
from method import live
from method import push
from method import review as rv
from method.refresh import METHOD_DIR, SIGNALS_DIR, SNAPSHOT, DATA_DIR, apply_live, read_breadth
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
BARS_TTL_S = 300
TZ = "US/Eastern"

_bars_cache: dict[str, tuple[float, dict]] = {}
_history_cache: dict = {"key": None, "data": None}
_outcomes_cache: dict = {"mtime": None, "rows": [], "index": {}}
_library_cache: dict = {"mtime": None, "data": None}
CAPS = ("ETF", "Large", "Medium", "Small", "Micro")
_shots_lock = threading.Lock()
_live_lock = threading.Lock()


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
    # Polygon's delayed plan refuses a range ending on a day that hasn't traded yet
    # (after midnight until the 4 AM bars arrive), so fall back a day.
    daily = None
    for end in (today, today - datetime.timedelta(days=1)):
        try:
            daily = pq.get_levels_data(ticker, end.isoformat(), 190, 1, "day")
            break
        except Exception as e:
            logger.info(f"bars {ticker}: daily to {end} refused ({e}); trying a day earlier")
    day_rows = [] if daily is None else [
        {"time": ts.strftime("%Y-%m-%d"), "open": float(r.open), "high": float(r.high),
         "low": float(r.low), "close": float(r.close), "value": float(r.volume)}
        for ts, r in daily.iterrows()]
    intra, day = None, today
    for back in range(6):
        day = today - datetime.timedelta(days=back)
        try:
            intra = pq.get_intraday(ticker, day.isoformat(), 1, "minute")
        except Exception:
            continue
        if intra is not None and not intra.empty:
            break
    min_rows = [] if intra is None or intra.empty else [
        {"time": _et_epoch(ts), "open": float(r.open), "high": float(r.high),
         "low": float(r.low), "close": float(r.close), "value": float(r.volume)}
        for ts, r in intra.iterrows()]
    return {"ticker": ticker, "daily": day_rows, "intraday": min_rows,
            "intraday_date": day.isoformat()}


def outcomes() -> tuple[list[dict], dict]:
    """The scorecard's surfaced rows (control and breadth cohorts dropped) and their
    (target, side, ticker) index."""
    mtime = OUTCOMES.stat().st_mtime if OUTCOMES.exists() else None
    if _outcomes_cache["mtime"] != mtime:
        rows = []
        if OUTCOMES.exists():
            with OUTCOMES.open(newline="", encoding="utf-8") as fh:
                rows = [r for r in csv.DictReader(fh) if core.surfaced(r)]
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


def allowed_logins() -> set[str]:
    raw = os.environ.get("METHOD_ALLOWED_LOGINS", "zmburr@gmail.com")
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def _valid_ticker(t: str) -> bool:
    return bool(t) and t.isascii() and t.replace(".", "").isalnum() and len(t) <= 10


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    version = git_head()
    logins = allowed_logins()

    @app.before_request
    def gate():
        if request.path == "/api/health":
            return None
        login = (request.headers.get("Tailscale-User-Login") or os.environ.get("METHOD_DEV_LOGIN") or "").lower()
        if login not in logins:
            # fail closed: not via tailscale serve, or someone else on the tailnet
            return ("Not available", 403)
        if request.method == "POST" and (not request.is_json or request.headers.get("X-Method") != "1"):
            return jsonify(error="Expected a Method JSON request"), 400
        return None

    @app.get("/sw.js")
    def service_worker():
        resp = make_response(send_from_directory(STATIC, "sw.js", mimetype="application/javascript"))
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["Service-Worker-Allowed"] = "/"
        return resp

    @app.get("/api/push")
    def push_info():
        return jsonify(public_key=push.ensure_keys(), subscriptions=push.count())

    @app.post("/api/push/subscribe")
    def push_subscribe():
        body = request.get_json(silent=True) or {}
        try:
            n = push.subscribe(body.get("endpoint", ""), body.get("keys") or {})
        except ValueError as e:
            return jsonify(error=str(e)), 400
        return jsonify(ok=True, subscriptions=n)

    @app.post("/api/push/test")
    def push_test():
        sent = push.send("Method", "Notifications are on. Setups forming and grade upgrades arrive here.",
                         url="/#/bounce", tag="method-test")
        return jsonify(sent=sent)

    @app.post("/api/live")
    def live_post():
        if not live.token_ok(push.home() / "live_token", request.headers.get("X-Method-Token")):
            return jsonify(error="bad token"), 403
        try:
            day, names = live.clean_post(request.get_json(silent=True) or {})
        except (TypeError, ValueError) as e:
            return jsonify(error=str(e)), 400
        with _live_lock:
            out = apply_live(day, names)
        return jsonify(ok=True, **out)

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
        cap = request.args.get("cap")
        cap = cap if cap in CAPS else None
        _, index = outcomes()
        row = index.get((date, side, ticker))
        try:
            data = lib.cached_review(side, ticker, date, REVIEW_DIR, cap=cap, row=row)
        except Exception as e:
            logger.warning(f"review {side} {ticker} {date}: {e}")
            return jsonify(ticker=ticker, date=date, side=side, error=str(e), candles=[], trades=[])
        data = dict(data)
        outcome = lib.outcome_fields(row)
        data["outcome"] = outcome
        return jsonify(data)

    @app.get("/api/library")
    def api_library():
        """Your graded trades with what the systems said (method/library.py builds it)."""
        path = lib.LIBRARY
        mtime = path.stat().st_mtime if path.exists() else None
        if _library_cache["mtime"] != mtime:
            data = core.read_json(path) or {"rows": []}
            try:
                lib.tag_days(data.get("rows") or [])           # the day type of each trade's session
            except Exception as e:
                logger.warning(f"library day tags: {e}")
            _library_cache.update(mtime=mtime, data=data)
        return jsonify(_library_cache["data"])

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
    live.ensure_token(push.home() / "live_token")
    create_app().run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
