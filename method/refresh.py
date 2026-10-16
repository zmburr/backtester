"""
Method refresh: one scoring pass into data/method/snapshot.json.

Cron runs it every 5 min, 04:00-20:00 ET on weekdays. It picks the mode from
the clock and skips passes the mode doesn't need:
  premarket 04:00-09:30, every 5 min:  live score, premarket low, open scenario
  market    09:30-16:00, every 15 min: scenario locked at the open + price now
  evening   16:30-20:00, every 15 min: scored at the after-hours price, for
                                       tomorrow's session (GO level = the open
                                       that makes it 5/6)
A fresh process per pass keeps stock_screener's import-time date and the
report module's heavy imports out of the long-running server.

Data is Polygon (15-min delayed); the mini can't reach Trillium. Every
snapshot carries the time of the newest bar it used.

Bounce scoring is the premarket scanner's own: completed_daily +
static_from_hist (cached once per session, setup type fixed at the prior
close) + metrics_from_price + BouncePretrade. Reversal scoring is the
priority report's own: get_pretrade_metrics + score_pretrade_setup, run on
the report's reversal names plus watchlist names closing 15%+ above the 9EMA.

Usage:
    python -m method.refresh              # from cron
    python -m method.refresh --force      # run now whatever the clock says
    python -m method.refresh --force --mode premarket --dry
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd

from analyzers.bounce_scorer import BouncePretrade
from method import core
from data_queries import polygon_queries as pq
from scanners.premarket_bounce_scanner import (
    CAP_CACHE_FILE, _load_json, completed_daily, get_cap, load_watchlist, metrics_from_price,
    static_from_hist,
)
from scripts.evening_signal_log import go_gap_trigger, next_trading_day
from scripts.playbook_email import PRINCIPALS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("method.refresh")

TZ = "US/Eastern"
DATA_DIR = CAP_CACHE_FILE.parent
SIGNALS_DIR = DATA_DIR / "priority_signals"
METHOD_DIR = Path(os.environ.get("METHOD_DIR") or DATA_DIR / "method")
SNAPSHOT = METHOD_DIR / "snapshot.json"
REVERSAL_9EMA_MIN = 0.15        # watchlist names this far above the 9EMA get reversal-scored
STATIC_DRIFT = 0.01             # evening: rebuild a static whose close disagrees with the tape


def now_et() -> datetime.datetime:
    return pd.Timestamp.now(tz=TZ).to_pydatetime()


def is_session_day(day: datetime.date) -> bool:
    import pandas_market_calendars as mcal
    return not mcal.get_calendar("NYSE").schedule(start_date=day, end_date=day).empty


# ---------------------------------------------------------------------------
# Statics (daily context), cached once per planned session
# ---------------------------------------------------------------------------

def _statics_path(session: str):
    return METHOD_DIR / f"statics_{session}.json"


def board_static(ticker: str, session: str, cap: str, checker) -> dict | None:
    """The premarket scanner's statics as-of the session (setup type fixed at
    the prior close), plus the GO level and the 9EMA stretch of the last
    completed close (for picking reversal candidates). One daily-bar fetch."""
    hist = completed_daily(ticker, session)
    if hist is None:
        return None
    s = static_from_hist(hist)
    s["go_gap"] = go_gap_trigger(ticker, s, cap, checker)
    ema9 = hist["close"].ewm(span=9, adjust=False).mean().iloc[-1]
    s["ema9"] = float(ema9)
    s["pct_from_9ema"] = float(hist["close"].iloc[-1] / ema9 - 1)
    return s


def load_statics(session: str, tickers: list[str], caps: dict, checker) -> dict:
    path = _statics_path(session)
    cache = core.read_json(path, {}) or {}
    missing = [t for t in tickers if t not in cache]
    for t in missing:
        try:
            s = board_static(t, session, caps[t], checker)
        except Exception as e:
            logger.warning(f"{t}: statics failed: {e}")
            s = None
        if s:
            cache[t] = s
        time.sleep(0.05)
    if missing:
        core.write_json(path, cache)
        logger.info(f"statics for {session}: {len(cache)} names ({len(missing)} built)")
    return cache


# ---------------------------------------------------------------------------
# Bounce
# ---------------------------------------------------------------------------

def _failing(res) -> list[dict]:
    """The checklist's own wording already carries the bar ('Gap down >= 4%')."""
    return [{"name": i.description or i.name, "actual": i.actual_display, "threshold": None}
            for i in res.items if not i.passed]


def bounce_pass(tickers, statics, caps, checker, mode: str, today: str,
                now: datetime.datetime, session: str) -> tuple[list[dict], str | None]:
    """Score every watchlist name off its latest Polygon minute bar. Returns
    rows and the time of the newest bar used."""
    open_ts = pd.Timestamp(f"{today} 09:30", tz=TZ)
    close_ts = pd.Timestamp(f"{today} 16:00", tz=TZ)
    asof = pd.Timestamp(now)
    rows, newest = [], None
    rebuilt = False
    for t in tickers:
        s = statics.get(t)
        if not s:
            continue
        try:
            bars = pq.get_intraday(t, today, 1, "minute")
            if bars is None or bars.empty:
                continue
            bars = bars[bars.index <= asof]
            if bars.empty:
                continue
            pm = bars[bars.index < open_ts]
            rth = bars[(bars.index >= open_ts) & (bars.index < close_ts)]
            ah = bars[bars.index >= close_ts]

            if mode == "evening" and not rth.empty:
                tape_close = float(rth["close"].iloc[-1])
                if abs(s["prior_close"] / tape_close - 1) > STATIC_DRIFT:
                    # daily bar wasn't final when the statics were built
                    fresh = board_static(t, session, caps[t], checker)
                    if fresh:
                        statics[t] = s = fresh
                        rebuilt = True

            price = float(bars["close"].iloc[-1])
            last_bar = bars.index[-1]
            newest = max(newest, last_bar) if newest is not None else last_bar
            m = metrics_from_price(s, price)
            # weak/strong fixed at the prior close, as the scanner does (SKHY 10/15)
            res = checker.validate(t, m, cap=caps[t], force_setup=s.get("setup_type"))
            go_gap = s.get("go_gap")
            prior = s["prior_close"]
            row = {
                "ticker": t, "cap": caps[t], "setup": res.setup_type,
                "score": res.score, "max_score": res.max_score, "rec": res.recommendation,
                "price": price, "prior_close": prior, "gap": m.get("gap_pct"),
                "go_gap": go_gap, "go_price": prior * (1 + go_gap) if go_gap is not None else None,
                "down_days": s.get("consecutive_down_days"), "off_30d": m.get("pct_off_30d_high"),
                "selloff": m.get("selloff_total_pct"), "atr": s.get("atr"),
                "atr_pct": s["atr"] / price if price else None,
                "mid_bb": s.get("mid_bb"),
                "to_mid_bb": (s["mid_bb"] / price - 1) if s.get("mid_bb") and price else None,
                "failing": _failing(res), "warnings": list(res.warnings or []),
                "bonuses": list(res.bonuses or []), "principal": PRINCIPALS.get(t),
                "bar_time": last_bar.strftime("%H:%M"),
            }
            if not pm.empty and mode != "evening":      # today's PM low means nothing for tomorrow
                row["pm_low"] = float(pm["low"].min())
                row["pm_low_time"] = pm["low"].idxmin().strftime("%H:%M")
                row["pm_high"] = float(pm["high"].max())
            if mode == "market" and not rth.empty:
                open_px = float(rth["open"].iloc[0])
                open_gap = open_px / prior - 1
                row.update(open=open_px, open_gap=open_gap, day_low=float(rth["low"].min()),
                           scenario=core.bounce_scenario(open_gap, go_gap),
                           scenario_now=core.bounce_scenario(row["gap"], go_gap))
            else:
                row["scenario"] = core.bounce_scenario(row["gap"], go_gap)
            if mode == "evening" and not ah.empty:
                row["ah_low"] = float(ah["low"].min())
            rows.append(row)
        except Exception as e:
            logger.warning(f"{t}: bounce pass failed: {e}")
        time.sleep(0.05)
    if rebuilt:
        core.write_json(_statics_path(session), statics)
    order = {"GO": 0, "CAUTION": 1}
    rows.sort(key=lambda r: (order.get(r["rec"], 2), -r["score"], r["gap"] or 0))
    return rows, newest.strftime("%H:%M") if newest is not None else None


# ---------------------------------------------------------------------------
# Reversal
# ---------------------------------------------------------------------------

def _fmt_crit(name, value) -> str:
    """The report's criteria carry raw numbers: ratios for range/volume, else fractions."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "–" if value is None else str(value)
    label = (name or "").upper()
    if "ATR" in label or "RVOL" in label or "VOL" in label:
        return f"{v:.2f}x"
    return f"{v * 100:+.0f}%"


def reversal_candidates(reports: list[dict], statics: dict) -> list[str]:
    names = []
    for rep in reports:
        names += [s["ticker"] for s in rep.get("signals", []) if s.get("bucket") == "reversal"]
    names += [t for t, s in statics.items()
              if (s.get("pct_from_9ema") or 0) >= REVERSAL_9EMA_MIN]
    return list(dict.fromkeys(names))


def reversal_pass(candidates: list[str], caps: dict, today: str,
                  report_setups: dict) -> list[dict]:
    if not candidates:
        return []
    # The priority report's scorer. Heavy import, so only when there's a name.
    from scripts.generate_report import get_pretrade_metrics, score_pretrade_setup
    rows = []
    for t in candidates:
        try:
            cap = caps.get(t) or get_cap(t, _load_json(CAP_CACHE_FILE))
            m = get_pretrade_metrics(t, today)
            if not m or m.get("prior_close") is None:
                continue
            res = score_pretrade_setup(t, m, cap)
            prior = float(m["prior_close"])
            gap = m.get("gap_pct")
            failing = [{"name": c.get("name"), "actual": _fmt_crit(c.get("name"), c.get("actual")),
                        "threshold": _fmt_crit(c.get("name"), c.get("threshold"))}
                       for c in res.get("criteria", []) if not c.get("passed")]
            gates = []
            if res.get("range_gate_passed") is False:
                gates.append("range gate")
            if res.get("archetype_passed") is False:
                gates.append("archetype gate")
            if res.get("momentum_pctile_passed") is False:
                gates.append("momentum gate")
            rows.append({
                "ticker": t, "cap": cap, "setup": report_setups.get(t, ""),
                "score": res.get("score"), "max_score": res.get("max_score", 5),
                "rec": res.get("recommendation"),
                "price": prior * (1 + gap) if gap is not None else prior,
                "prior_close": prior, "gap": gap,
                "gap_level": prior * (1 + core.REVERSAL_GAP_GATE),
                "pct_from_9ema": m.get("pct_from_9ema"), "ema9": m.get("ema_9"),
                "chg_3d": m.get("pct_change_3"), "atr_pct": m.get("atr_pct"),
                "rvol": m.get("prior_day_rvol"), "pm_rvol": m.get("premarket_rvol"),
                "scenario": core.reversal_scenario(gap),
                "failing": failing, "gates_failed": gates, "principal": PRINCIPALS.get(t),
            })
        except Exception as e:
            logger.warning(f"{t}: reversal pass failed: {e}")
    order = {"GO": 0, "CAUTION": 1}
    rows.sort(key=lambda r: (order.get(r["rec"], 2), -(r["score"] or 0)))
    return rows


# ---------------------------------------------------------------------------
# Reports + snapshot
# ---------------------------------------------------------------------------

def recent_reports(today: datetime.date, mode: str) -> list[dict]:
    """Newest first: the reports that speak to the session being planned."""
    found = []
    for f in SIGNALS_DIR.glob("*_*.json"):
        date_str, _, session = f.stem.partition("_")
        if session not in ("morning", "evening") or date_str > today.isoformat():
            continue
        if mode != "evening" and date_str == today.isoformat() and session == "evening":
            continue          # today's 3:15 run doesn't plan today's open
        found.append(((date_str, session == "evening"), f))
    out = []
    for _, f in sorted(found, key=lambda x: x[0], reverse=True):
        rep = core.read_json(f)
        if rep:
            out.append(rep)
        if len(out) >= 3:
            break
    return out


def _report_view(rep: dict) -> dict:
    stamp = (rep.get("generated_at") or "")[11:16]
    view = {"date": rep.get("date"), "session": rep.get("session"), "time": stamp,
            "cluster": rep.get("cluster_signal")}
    for side in ("bounce", "reversal"):
        sigs = [s for s in rep.get("signals", []) if s.get("bucket") == side]
        view[side] = {
            "GO": [s["ticker"] for s in sigs if s.get("recommendation") == "GO"],
            "CAUTION": [s["ticker"] for s in sigs if s.get("recommendation") == "CAUTION"],
            "odds": {s["ticker"]: s.get("odds") for s in sigs if s.get("odds")},
            "analogs": {s["ticker"]: s.get("analogs") for s in sigs if s.get("analogs")},
        }
    return view


def _counts(rows):
    return {"GO": sum(r["rec"] == "GO" for r in rows),
            "CAUTION": sum(r["rec"] == "CAUTION" for r in rows)}


def append_breadth(session: str, entry: dict) -> list[dict]:
    path = METHOD_DIR / f"breadth_{session}.jsonl"
    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(core.clean(entry)) + "\n")
    return read_breadth(session)


def read_breadth(session: str) -> list[dict]:
    path = METHOD_DIR / f"breadth_{session}.jsonl"
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, ValueError):
        return []


def run(now: datetime.datetime | None = None, force: bool = False, mode: str | None = None,
        dry: bool = False) -> dict | None:
    now = now or now_et()
    today = now.date()
    mode = mode or core.mode_for(now, is_session_day(today))
    if mode == "closed":
        logger.info("closed; nothing to refresh")
        return None
    prev = core.read_json(SNAPSHOT)
    if not force and prev and prev.get("mode") == mode:
        last = datetime.datetime.fromisoformat(prev["generated_at"])
        if not core.due(mode, last, now):
            return None

    session = core.session_for(mode, today, next_trading_day).isoformat()
    tickers = load_watchlist()
    caps_cache = _load_json(CAP_CACHE_FILE)
    caps = {t: get_cap(t, caps_cache) for t in tickers}
    checker = BouncePretrade()
    statics = load_statics(session, tickers, caps, checker)

    bounce, data_time = bounce_pass(tickers, statics, caps, checker, mode,
                                    today.isoformat(), now, session)
    reports = recent_reports(today, mode)
    setups = {s["ticker"]: s.get("setup_type") or "" for rep in reports
              for s in rep.get("signals", []) if s.get("bucket") == "reversal"}
    reversal = reversal_pass(reversal_candidates(reports, statics), caps,
                             today.isoformat(), setups)

    report_views = [_report_view(r) for r in reports]
    latest = report_views[0] if report_views else None
    if latest:
        for side, rows in (("bounce", bounce), ("reversal", reversal)):
            for r in rows:
                v = latest[side]
                r["report_rec"] = ("GO" if r["ticker"] in v["GO"]
                                   else "CAUTION" if r["ticker"] in v["CAUTION"] else None)
                if r["ticker"] in v["odds"]:
                    r["odds"] = v["odds"][r["ticker"]]
                if r["ticker"] in v["analogs"]:
                    r["analogs"] = v["analogs"][r["ticker"]]

    bc, rc = _counts(bounce), _counts(reversal)
    entry = {"t": now.strftime("%H:%M"), "date": today.isoformat(), "mode": mode,
             "bounce_go": bc["GO"], "bounce_caution": bc["CAUTION"],
             "reversal_go": rc["GO"], "reversal_caution": rc["CAUTION"]}
    snapshot = {
        "generated_at": now.isoformat(timespec="seconds"), "mode": mode,
        "session": session, "data_time": data_time, "source": "Polygon, 15-min delayed",
        "bounce": {"rows": bounce, "counts": bc},
        "reversal": {"rows": reversal, "counts": rc},
        "reports": report_views,
        "scenario_text": core.SCENARIO_TEXT,
    }
    if dry:
        logger.info(f"--dry {mode} for {session}: bounce {bc}, reversal {rc}, data {data_time}")
        for r in bounce[:8] + reversal[:5]:
            logger.info(f"  {r['ticker']:6} {r['score']}/{r['max_score']} {r['rec']:8} "
                        f"gap {(r['gap'] or 0) * 100:+.1f}% scenario {r.get('scenario')}")
        return snapshot
    snapshot["breadth"] = append_breadth(session, entry)
    core.write_json(SNAPSHOT, snapshot)
    logger.info(f"{mode} for {session}: bounce {bc}, reversal {rc}, data {data_time}")
    return snapshot


def main():
    ap = argparse.ArgumentParser(description="Method refresh pass")
    ap.add_argument("--force", action="store_true", help="run whatever the clock and cadence say")
    ap.add_argument("--mode", choices=["premarket", "market", "evening"])
    ap.add_argument("--dry", action="store_true", help="score and log, write nothing but caches")
    args = ap.parse_args()
    run(force=args.force or bool(args.mode), mode=args.mode, dry=args.dry)


if __name__ == "__main__":
    main()
