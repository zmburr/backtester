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

Data is Polygon (15-min delayed) unless the PC is posting Trillium bars
(method/live.py, method/pc_pusher.py): a name posted in the last 2 minutes is
scored off the PC's bars instead, and the server re-grades it on each post.
Overnight, Yahoo's Blue Ocean print. Every snapshot carries the time of the
newest bar it used and which feed it came from.

Bounce scoring is the bounce game plan's (analyzers/bounce_ladder.py), which
is the premarket scanner's checklist plus intensity: completed_daily +
static_from_hist (cached once per session, setup type fixed at the prior
close), the grade ladder built once per session, score_at for the live price. Reversal scoring is the
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

from analyzers.bounce_ladder import GRADE_RANK, historical_grade_stats, lookup_bounce_odds
from analyzers.bounce_scorer import BouncePretrade
from method import breadth
from method import core
from method import live
from method import push
from method.ladder import grade_at, ladder_payload, next_better
from data_queries import polygon_queries as pq
from scanners.premarket_bounce_scanner import (
    CAP_CACHE_FILE, _load_json, completed_daily, get_cap, load_watchlist, static_from_hist,
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
LIVE = METHOD_DIR / "live.json"
MARKET_BREADTH = METHOD_DIR / "market_breadth.json"   # the snapshot's block + per-name Polygon prices
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
    the prior close), plus the GO level, the game plan's grade ladder and the
    9EMA stretch of the last completed close (for picking reversal
    candidates). One daily-bar fetch; the ladder costs ~0.6 s."""
    hist = completed_daily(ticker, session)
    if hist is None:
        return None
    s = static_from_hist(hist)
    s["go_gap"] = go_gap_trigger(ticker, s, cap, checker)
    s["ladder"] = ladder_payload(s, cap, ticker)
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


def score_row(t: str, s: dict, cap: str, price: float, bar_time: str) -> dict:
    """One bounce row at `price`: the game plan's scoring (weak/strong fixed at the
    prior close, SKHY 10/15), its grade, the next better grade and the context."""
    g = grade_at(s, cap, price, t)
    res, m = g["result"], g["metrics"]
    go_gap = s.get("go_gap")
    prior = s["prior_close"]
    return {
        "ticker": t, "cap": cap, "setup": res.setup_type,
        "score": res.score, "max_score": res.max_score, "rec": res.recommendation,
        "grade": g["grade"], "intensity": g["intensity"],
        "next": next_better(s["ladder"]["levels"], price, g["grade"]), "ladder": s["ladder"],
        "odds_pop": lookup_bounce_odds(res.score, cap),
        "price": price, "prior_close": prior, "gap": m.get("gap_pct"),
        "go_gap": go_gap, "go_price": prior * (1 + go_gap) if go_gap is not None else None,
        "down_days": s.get("consecutive_down_days"), "off_30d": m.get("pct_off_30d_high"),
        "selloff": m.get("selloff_total_pct"), "atr": s.get("atr"),
        "atr_pct": s["atr"] / price if price else None,
        "mid_bb": s.get("mid_bb"),
        "to_mid_bb": (s["mid_bb"] / price - 1) if s.get("mid_bb") and price else None,
        "failing": _failing(res), "warnings": list(res.warnings or []),
        "bonuses": list(res.bonuses or []), "principal": PRINCIPALS.get(t),
        "bar_time": bar_time,
    }


def _sort_rows(rows: list[dict]) -> list[dict]:
    order = {"GO": 0, "CAUTION": 1}
    rows.sort(key=lambda r: (order.get(r["rec"], 2), -(r["score"] or 0), r["gap"] or 0))
    return rows


def row_from_summary(t: str, s: dict, cap: str, b: dict, mode: str, src: str) -> dict:
    """A bounce row from one name's bar summary (method/live.summarize_bars), whichever
    feed it came from: the grade at the last price, plus the premarket low before the
    open and the open's own grade once the session is running."""
    row = score_row(t, s, cap, b["price"], b["bar_time"])
    row.update(src=src, src_label=live.SRC_LABEL.get(src, src))
    if b.get("pm_low") is not None and mode != "evening":    # today's PM low means nothing for tomorrow
        row.update(pm_low=b["pm_low"], pm_low_time=b.get("pm_low_time"), pm_high=b.get("pm_high"))
    if mode == "market" and b.get("open") is not None:
        at_open = grade_at(s, cap, b["open"], t)
        row.update(open=b["open"], open_gap=b["open"] / s["prior_close"] - 1, day_low=b.get("day_low"),
                   day_low_time=b.get("day_low_time"), grade_open=at_open["grade"],
                   score_open=at_open["score"], intensity_open=at_open["intensity"])
    if mode == "evening" and b.get("ah_low") is not None:
        row["ah_low"] = b["ah_low"]
    return row


def bounce_pass(tickers, statics, caps, checker, mode: str, today: str,
                now: datetime.datetime, session: str,
                feed: dict | None = None) -> tuple[list[dict], str | None]:
    """Score every watchlist name off its latest minute bar: the PC's Trillium bars
    when `feed` has the name fresh, else Polygon's. Returns rows and the time of the
    newest bar used."""
    asof = pd.Timestamp(now)
    feed = feed or {}
    rows, newest = [], None
    rebuilt = False
    for t in tickers:
        s = statics.get(t)
        if not s:
            continue
        try:
            if "ladder" not in s:                  # statics cached before the ladder existed
                s["ladder"] = ladder_payload(s, caps[t], t)
                rebuilt = True
            src = "T" if t in feed else "P"
            if src == "T":
                b = feed[t]
            else:
                b = live.summarize_bars(pq.get_intraday(t, today, 1, "minute"), today, asof)
                time.sleep(0.05)
            if not b:
                continue

            if mode == "evening" and b.get("rth_close"):
                if abs(s["prior_close"] / b["rth_close"] - 1) > STATIC_DRIFT:
                    # daily bar wasn't final when the statics were built
                    fresh = board_static(t, session, caps[t], checker)
                    if fresh:
                        statics[t] = s = fresh
                        rebuilt = True

            last_bar = pd.Timestamp(b["bar_ts"])
            newest = max(newest, last_bar) if newest is not None else last_bar
            rows.append(row_from_summary(t, s, caps[t], b, mode, src))
        except Exception as e:
            logger.warning(f"{t}: bounce pass failed: {e}")
    if rebuilt:
        core.write_json(_statics_path(session), statics)
    return _sort_rows(rows), newest.strftime("%H:%M") if newest is not None else None


def live_reversal(rows: list[dict], feed: dict) -> list[dict]:
    """Premarket: a reversal row's gap and open scenario follow the PC's live price."""
    for r in rows:
        b = feed.get(r["ticker"])
        if b and r.get("prior_close"):
            r["price"] = b["price"]
            r["gap"] = b["price"] / r["prior_close"] - 1
            r["scenario"] = core.reversal_scenario(r["gap"])
            r.update(src="T", src_label=live.SRC_LABEL["T"], bar_time=b["bar_time"])
    return rows


def set_source(snap: dict) -> dict:
    """The page's headline feed: Blue Ocean overnight, Trillium when the PC posted at
    least half the board, else Polygon. Rows on another feed say so themselves."""
    rows = snap["bounce"]["rows"]
    if snap["mode"] == "overnight":
        snap["src"], snap["source"] = "Y", "Blue Ocean overnight via Yahoo"
    elif rows and sum(r.get("src") == "T" for r in rows) * 2 >= len(rows):
        snap["src"], snap["source"] = "T", "Trillium, live from the PC"
    else:
        snap["src"], snap["source"] = "P", "Polygon, 15-min delayed"
    return snap


# ---------------------------------------------------------------------------
# Overnight (8 PM-4 AM): Polygon has no trades, Yahoo has the Blue Ocean print
# ---------------------------------------------------------------------------

SRC_LABEL = {"Y": "Blue Ocean", "M": "your quote", "P": "after hours"}


def overnight_prices(tickers: list[str], prev: dict | None) -> dict:
    """{ticker: (price, asof, src)}: the overnight tracker's sources, newest wins
    (scanners/overnight_bounce_tracker.choose_price): your data/overnight_prices.json
    quotes, Yahoo's Blue Ocean print, else the evening snapshot's after-hours price."""
    from scanners.overnight_bounce_tracker import choose_price, load_prices_file, yahoo_overnight_prices
    yahoo = yahoo_overnight_prices(tickers)
    manual = load_prices_file()
    last = {}
    if prev and prev.get("generated_at"):
        asof = pd.Timestamp(prev["generated_at"])
        for side in ("bounce", "reversal"):
            for r in (prev.get(side) or {}).get("rows", []):
                if r.get("price") is not None:
                    last[r["ticker"]] = (float(r["price"]), asof)
    out = {}
    for t in tickers:
        chosen = choose_price(manual.get(t), last.get(t), yahoo.get(t))
        if chosen:
            out[t] = chosen
    return out


def overnight_pass(tickers, statics, caps, prices: dict) -> tuple[list[dict], str | None]:
    rows, newest = [], None
    for t in tickers:
        s = statics.get(t)
        if not s or t not in prices:
            continue
        price, asof, src = prices[t]
        try:
            if "ladder" not in s:
                s["ladder"] = ladder_payload(s, caps[t], t)
            row = score_row(t, s, caps[t], float(price), asof.strftime("%H:%M"))
            row.update(src=src, src_label=SRC_LABEL.get(src, src))
            rows.append(row)
            if src != "P":
                newest = max(newest, asof) if newest is not None else asof
        except Exception as e:
            logger.warning(f"{t}: overnight pass failed: {e}")
    return _sort_rows(rows), newest.strftime("%H:%M") if newest is not None else None


def overnight_reversal(prev: dict | None, prices: dict) -> list[dict]:
    """Reversal rows keep their evening score; the overnight print moves price, gap, scenario."""
    rows = []
    for r in ((prev or {}).get("reversal") or {}).get("rows", []):
        r = dict(r)
        hit = prices.get(r["ticker"])
        if hit and hit[2] != "P" and r.get("prior_close"):
            r["price"] = float(hit[0])
            r["gap"] = r["price"] / r["prior_close"] - 1
            r["scenario"] = core.reversal_scenario(r["gap"])
            r.update(src=hit[2], src_label=SRC_LABEL.get(hit[2]), bar_time=hit[1].strftime("%H:%M"))
        rows.append(r)
    return rows


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
        if mode not in ("evening", "overnight") and date_str == today.isoformat() and session == "evening":
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


def grade_alerts(rows: list[dict], session: str, mode: str) -> list[str]:
    """Push when a name the priority report rated GO or CAUTION reaches A or A+ at the
    live price, overnight or premarket (user, 2026-10-15). Once per name per grade per
    session; an A that becomes an A+ pushes again. Nothing is recorded unless a phone got it."""
    if mode not in ("premarket", "overnight"):
        return []
    path = METHOD_DIR / f"grade_alerts_{session}.json"
    state = core.read_json(path, {}) or {}
    sent = []
    for r in rows:
        g = r.get("grade")
        if r.get("report_rec") not in ("GO", "CAUTION") or g not in ("A", "A+"):
            continue
        if GRADE_RANK[g] <= GRADE_RANK.get(state.get(r["ticker"], "-"), 0):
            continue
        nxt = r.get("next")
        body = (f"{r['ticker']} at {r['price']:,.2f} ({(r['gap'] or 0) * 100:+.1f}% vs the close) is an {g}. "
                + (f"{nxt['grade']} at {nxt['price']:,.2f}." if nxt else "Best grade on the ladder."))
        try:
            n = push.send(f"{r['ticker']} is an {g}", body, url=f"/#/bounce/{r['ticker']}",
                          tag=f"grade-{r['ticker']}-{session}")
        except Exception as e:
            logger.warning(f"grade alert push failed: {e}")
            n = 0
        if n:
            state[r["ticker"]] = g
            sent.append(r["ticker"])
    if sent:
        core.write_json(path, state)
        logger.info(f"grade alerts pushed: {', '.join(sent)}")
    return sent


def _counts(rows):
    """Watchlist GO/CAUTION counts (the market-breadth rows have their own count)."""
    rows = [r for r in rows if not r.get("market")]
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


# ---------------------------------------------------------------------------
# Market breadth (method/breadth.py): the day's count, market rows, the edge
# ---------------------------------------------------------------------------

def market_breadth(session: str, mode: str, today: str, now: datetime.datetime, feed: dict,
                   snapshot_prices=None) -> dict | None:
    """This pass's breadth block, or None outside the modes it runs in. One all-tickers
    Polygon call, skipped once the open's count is locked."""
    if mode not in ("premarket", "market", "evening"):
        return None
    if not breadth.load_candidates(session):
        return {"session": session, "status": "not built"}
    prev = core.read_json(MARKET_BREADTH)
    prices = None
    if not (prev and prev.get("session") == session and prev.get("locked")):
        try:
            prices = (snapshot_prices or breadth.snapshot_prices)(today)
        except Exception as e:
            logger.warning(f"breadth snapshot failed: {e}")
    mb = breadth.evaluate(session, mode, now, feed, prev=prev, snap_prices=prices)
    core.write_json(MARKET_BREADTH, mb)
    if mb.get("status") == "ok" and mb.get("locked"):
        # what the card said at the open, kept for the day's record (the evening pass moves
        # market_breadth.json on to the next session before the record is written)
        core.write_json(breadth.live_path(session), {k: mb.get(k) for k in (
            "n", "n_etf", "state", "band", "basis", "asof", "names", "relaxed", "trillium", "pushed")})
        cutoff = (datetime.date.fromisoformat(session) - datetime.timedelta(days=14)).isoformat()
        for p in breadth.METHOD_DIR.glob("breadth_live_*.json"):
            if p.stem.rsplit("_", 1)[-1] < cutoff:
                p.unlink()
    return mb


def tag_rows(rows: list[dict], mb: dict | None) -> list[dict]:
    """Market rows say so; every row carries the replayed edge for its score on a day like today."""
    table = breadth.edge_table()
    names = set((mb or {}).get("names") or [])
    state = (mb or {}).get("state")
    relaxed = set((mb or {}).get("relaxed") or []) if state == "breadth" else set()   # breadth days only
    detail = (mb or {}).get("detail") or {}
    for r in rows:
        t = r["ticker"]
        r.pop("edge", None)
        if t in names or t in relaxed:
            r["breadth_flag"] = "strict" if t in names else "relaxed"
            r["counted"] = bool((detail.get(t) or {}).get("counted"))
        else:
            r.pop("breadth_flag", None)
        score = r.get("score_open") if r.get("score_open") is not None else r.get("score")
        e = breadth.edge_for(table, state, score, relaxed=r.get("breadth_flag") == "relaxed")
        if e:
            r["edge"] = {k: e.get(k) for k in ("mean_r", "win", "n", "shrunk", "entry_mean_r", "entry_win")}
            w = (table.get("watchlist") or {}).get(state) or {}
            if not r.get("market") and w.get("n"):
                # your priority report's GO/CAUTION bounces on days like this (the scorecard)
                r["edge"]["watch"] = {k: w.get(k) for k in ("n", "mean_r", "win")}
    return rows


def breadth_alert(mb: dict | None) -> bool:
    if not mb:
        return False
    if breadth.breadth_push(mb, push.send):
        core.write_json(MARKET_BREADTH, mb)
        logger.info(f"breadth push: {mb['n']} names ({mb.get('basis')})")
        return True
    return False


def apply_live(day: str, names: dict, now: datetime.datetime | None = None, alerts: bool = True) -> dict:
    """The server's half of the PC feed: store the post, re-grade those rows in the
    snapshot on the spot (statics come from the session's cache: no Polygon calls,
    ~4 ms a name) and push any grade upgrade. generated_at is left alone so the cron's
    cadence still runs the full pass."""
    now = now or now_et()
    feed = live.merge_feed(core.read_json(LIVE), day, names, now)
    core.write_json(LIVE, feed)
    snap = core.read_json(SNAPSHOT)
    if (not snap or snap.get("mode") not in ("premarket", "market", "evening")
            or (snap.get("generated_at") or "")[:10] != day):
        return {"stored": len(names), "regraded": 0}
    mode, session = snap["mode"], snap["session"]
    statics = core.read_json(_statics_path(session), {}) or {}
    fresh = live.fresh_names(feed, day, now)
    rows, n = [], 0
    for r in snap["bounce"]["rows"]:
        b, s = fresh.get(r["ticker"]), statics.get(r["ticker"])
        if b and s and "ladder" in s:
            try:
                r = {**r, **row_from_summary(r["ticker"], s, r["cap"], b, mode, "T")}
                n += 1
            except Exception as e:
                logger.warning(f"{r['ticker']}: live re-grade failed: {e}")
        rows.append(r)
    if mode == "premarket":
        live_reversal(snap["reversal"]["rows"], fresh)
        snap["reversal"]["counts"] = _counts(snap["reversal"]["rows"])
    mb = None
    if mode in ("premarket", "market"):
        try:
            mb = breadth.evaluate(session, mode, now, fresh, prev=core.read_json(MARKET_BREADTH))
            if mb.get("status") == "ok":
                core.write_json(MARKET_BREADTH, mb)
                snap["market_breadth"] = breadth.public(mb)
                tag_rows(rows, mb)
        except Exception as e:
            logger.warning(f"live breadth re-count failed: {e}")
    snap["bounce"] = {"rows": _sort_rows(rows), "counts": _counts(rows)}
    times = [snap.get("data_time") or ""] + [b["bar_time"] for b in fresh.values()]
    snap["data_time"] = max(times) or None
    snap["live_at"] = now.isoformat(timespec="seconds")
    set_source(snap)
    core.write_json(SNAPSHOT, snap)
    sent = grade_alerts(rows, session, mode) if alerts else []
    if alerts and mb and breadth_alert(mb):
        snap["market_breadth"] = breadth.public(mb)
        core.write_json(SNAPSHOT, snap)
    return {"stored": len(names), "regraded": n, "alerts": sent}


def run(now: datetime.datetime | None = None, force: bool = False, mode: str | None = None,
        dry: bool = False) -> dict | None:
    now = now or now_et()
    today = now.date()
    if mode is None:
        mode = core.mode_for(now, is_session_day(today))
        if mode == "closed" and core.is_overnight(now):
            mode = "overnight"
    if mode == "closed":
        logger.info("closed; nothing to refresh")
        return None
    prev = core.read_json(SNAPSHOT)
    if not force and prev and prev.get("mode") == mode:
        last = datetime.datetime.fromisoformat(prev["generated_at"])
        if not core.due(mode, last, now):
            return None

    if mode == "overnight":       # before midnight: tomorrow's session; after: today's, if it trades
        session = (next_trading_day(today) if now.time() >= core.OVERNIGHT[0] or not is_session_day(today)
                   else today).isoformat()
    else:
        session = core.session_for(mode, today, next_trading_day).isoformat()
    watchlist = load_watchlist()
    feed = live.fresh_names(core.read_json(LIVE), today.isoformat(), now) if mode != "overnight" else {}
    mb = None
    if not dry:
        try:
            mb = market_breadth(session, mode, today.isoformat(), now, feed)
        except Exception as e:
            logger.warning(f"market breadth failed: {e}")
    market = [t for t in breadth.board_names(mb) if t not in watchlist]
    tickers = watchlist + market
    caps_cache = _load_json(CAP_CACHE_FILE)
    caps = {t: get_cap(t, caps_cache) for t in tickers}
    checker = BouncePretrade()
    statics = load_statics(session, tickers, caps, checker)

    reports = recent_reports(today, mode)
    if mode == "overnight":
        held = [r["ticker"] for r in ((prev or {}).get("reversal") or {}).get("rows", [])]
        prices = overnight_prices(list(dict.fromkeys(tickers + held)), prev)
        bounce, data_time = overnight_pass(tickers, statics, caps, prices)
        reversal = overnight_reversal(prev, prices)
    else:
        bounce, data_time = bounce_pass(tickers, statics, caps, checker, mode,
                                        today.isoformat(), now, session, feed)
        for r in bounce:
            if r["ticker"] in market:
                r["market"] = True
        setups = {s["ticker"]: s.get("setup_type") or "" for rep in reports
                  for s in rep.get("signals", []) if s.get("bucket") == "reversal"}
        reversal = reversal_pass(reversal_candidates(reports, statics), caps,
                                 today.isoformat(), setups)
        if mode == "premarket":
            live_reversal(reversal, feed)

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

    if mode == "overnight" and prev and (prev.get("market_breadth") or {}).get("session") == session:
        mb = prev["market_breadth"]          # the evening's read stands until the premarket
    tag_rows(bounce, mb)
    if not dry:
        grade_alerts(bounce, session, mode)
        if mode in ("premarket", "market"):
            breadth_alert(mb)
    bc, rc = _counts(bounce), _counts(reversal)
    entry = {"t": now.strftime("%H:%M"), "date": today.isoformat(), "mode": mode,
             "bounce_go": bc["GO"], "bounce_caution": bc["CAUTION"],
             "reversal_go": rc["GO"], "reversal_caution": rc["CAUTION"]}
    snapshot = {
        "generated_at": now.isoformat(timespec="seconds"), "mode": mode,
        "session": session, "data_time": data_time,
        "bounce": {"rows": bounce, "counts": bc},
        "reversal": {"rows": reversal, "counts": rc},
        "reports": report_views,
        "scenario_text": core.SCENARIO_TEXT,
        "grade_stats": historical_grade_stats(),
        "market_breadth": breadth.public(mb),
    }
    set_source(snapshot)
    if dry:
        logger.info(f"--dry {mode} for {session}: bounce {bc}, reversal {rc}, data {data_time}")
        for r in bounce[:8] + reversal[:5]:
            logger.info(f"  {r['ticker']:6} {r['score']}/{r['max_score']} {r['rec']:8} "
                        f"gap {(r['gap'] or 0) * 100:+.1f}% grade {r.get('grade') or r.get('scenario')}")
        return snapshot
    snapshot["breadth"] = append_breadth(session, entry)
    core.write_json(SNAPSHOT, snapshot)
    logger.info(f"{mode} for {session}: bounce {bc}, reversal {rc}, data {data_time}")
    return snapshot


def main():
    ap = argparse.ArgumentParser(description="Method refresh pass")
    ap.add_argument("--force", action="store_true", help="run whatever the clock and cadence say")
    ap.add_argument("--mode", choices=["premarket", "market", "evening", "overnight"])
    ap.add_argument("--dry", action="store_true", help="score and log, write nothing but caches")
    args = ap.parse_args()
    run(force=args.force or bool(args.mode), mode=args.mode, dry=args.dry)


if __name__ == "__main__":
    main()
