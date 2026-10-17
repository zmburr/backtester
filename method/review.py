"""
Replay one setup day for Method's charts: the night-before grade ladder, the
open and its grade, and the trades the rules would have taken on the 2-min
frame they fire on (entries, stops, partials, exits, R).

Bounce: the morning watcher's live rules as the scorecard replays them
(scripts/backfill_entry_r.rule_trades): the first 2-min close above the prior
bar's high after a fresh low, stop at the low of day, exit on the LOD stop or,
once +1.5 ATR, the prior 2-min low; max 2 attempts. With the scorecard's ATR
(atr_pct x entry_open) the R matches its exit_r.

Reversal: the reversal entry study's best rule (reversal_entry_study, top of
configs_ranked.csv): the first 2-min close below the prior bar's low between
9:32 and 10:30, stop at the running high of day, risk floored at 0.25 ATR; then
the cover matrix (reversal_trader._check_matrix_cover_triggers): half into a
4-ATR flush from the open, after 12:30 a 25% retrace of the day range is the
tripwire, else hold to 15:55. A name stops after two stop-outs (the trader's
per-name cap), three attempts at most.

Candles are 1-min (the page draws 1, 2 or 5 min from them), with each bar's
VWAP; past days also carry the next session and the daily bars from the trade
day on, so a chart can show the overnight hold and what came after.

Past days are cached by the server; today is always fetched fresh, and never
through the study's permanent bar cache. VERSION bumps when the shape changes,
so older cached replays are rebuilt.
"""

from __future__ import annotations

import datetime
from datetime import time as dt_time

import pandas as pd

from bounce_entry_study.detectors import build_2min
from bounce_entry_study.fetch_bars import fetch_day
from data_queries import polygon_queries as pq
from method.ladder import grade_at, ladder_payload
from reversal_entry_study.run_study import _two_min
from scanners.premarket_bounce_scanner import completed_daily, static_from_hist
from scripts.backfill_entry_r import rule_trades
from scripts.evening_signal_log import next_trading_day

VERSION = 2
TZ = "US/Eastern"
BAR = pd.Timedelta(minutes=2)
REV_FIRST_END = dt_time(9, 32)       # one completed 2-min bar
REV_GATE_END = dt_time(10, 30)
REV_RISK_FLOOR_ATR = 0.25
REV_MAX_ATTEMPTS = 3
REV_MAX_STOPS = 2
FLUSH_ATR = 4.0
TRIPWIRE_AFTER = dt_time(12, 30)
TRIPWIRE_RETRACE = 0.25
EOD_EXIT = dt_time(15, 55)


def et_epoch(ts) -> int:
    """Epoch seconds shifted so a chart's UTC axis reads ET wall-clock time."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(TZ).tz_localize(None)
    return int(ts.timestamp())


LIVE_FRAME = None    # set by the server: (ticker, date) -> today's Trillium bars, or None (method/livebars.py)


def day_bars(ticker: str, date: str, live: bool) -> pd.DataFrame | None:
    if live:
        lb = LIVE_FRAME(ticker, date) if LIVE_FRAME else None
        if lb is not None and not lb.empty:
            return lb
        return pq.get_intraday(ticker, date, 1, "minute")
    return fetch_day(ticker, date)


def _candles(bars: pd.DataFrame) -> list[dict]:
    """Full-session 1-min candles (premarket and after hours included), each with its
    bar VWAP (`vw`) when Polygon has one, for the page's day VWAP."""
    vw = "vwap" in bars.columns
    out = []
    for ts, r in bars.iterrows():
        c = {"time": et_epoch(ts), "open": float(r.open), "high": float(r.high), "low": float(r.low),
             "close": float(r.close), "value": float(r.volume)}
        if vw and pd.notna(r.vwap):
            c["vw"] = float(r.vwap)
        out.append(c)
    return out


def _daily_rows(df: pd.DataFrame) -> list[dict]:
    return [{"time": ts.strftime("%Y-%m-%d"), "open": float(r.open), "high": float(r.high),
             "low": float(r.low), "close": float(r.close), "value": float(r.volume)}
            for ts, r in df.iterrows()]


def after_days(ticker: str, date: str, n: int = 6) -> list[dict]:
    """Daily bars from the trade day on (the trade day plus n-1 sessions), past days only."""
    day = datetime.date.fromisoformat(date)
    end = min(day + datetime.timedelta(days=12), pd.Timestamp.now(tz=TZ).date() - datetime.timedelta(days=1))
    if end < day:
        return []
    df = pq.get_levels_data(ticker, end.isoformat(), 14, 1, "day")
    if df is None or df.empty:
        return []
    return _daily_rows(df[df.index.date >= day].iloc[:n])


def _session(bars: pd.DataFrame) -> dict:
    out = {}
    pm = bars.between_time("04:00", "09:29")
    rth = bars.between_time("09:30", "15:59")
    if not pm.empty:
        out.update(pm_low=float(pm["low"].min()), pm_low_time=pm["low"].idxmin().strftime("%H:%M"),
                   pm_high=float(pm["high"].max()))
    if not rth.empty:
        out.update(open=float(rth["open"].iloc[0]), high=float(rth["high"].max()),
                   high_time=rth["high"].idxmax().strftime("%H:%M"),
                   low=float(rth["low"].min()), low_time=rth["low"].idxmin().strftime("%H:%M"),
                   close=float(rth["close"].iloc[-1]), last_time=rth.index[-1].strftime("%H:%M"))
    return out


# ---------------------------------------------------------------------------
# Bounce
# ---------------------------------------------------------------------------

def bounce_trades(bars: pd.DataFrame, atr: float | None, live: bool) -> list[dict]:
    two = build_2min(bars)
    if two.empty:
        return []
    last_end = two.index[-1] + BAR
    out = []
    for a in rule_trades(two, atr):
        if a.stopped:
            kind = "stop"
        elif a.exit_time == last_end:
            kind = "open" if live else "close"
        else:
            kind = "trail"
        out.append({"entry_t": et_epoch(a.entry_time - BAR), "entry": float(a.entry),
                    "stop": float(a.stop), "exit_t": et_epoch(a.exit_time - BAR),
                    "exit": float(a.exit), "kind": kind, "r": float(a.r), "half": None})
    return out


# ---------------------------------------------------------------------------
# Reversal
# ---------------------------------------------------------------------------

def _tripwire_index(b2: pd.DataFrame, open_: float, atr: float) -> int | None:
    """First 2-min bar after 12:30 whose close retraced 25% of the day range (running
    low), with the matrix noise floor: as reversal_entry_study.run_study._tripwire_time."""
    lod = b2["low"].cummin()
    min_range = max(atr, 0.02 * open_)
    for i, ts in enumerate(b2.index):
        if ts.time() < TRIPWIRE_AFTER:
            continue
        rng = open_ - float(lod.iloc[i])
        if rng < min_range:
            continue
        if (float(b2["close"].iloc[i]) - float(lod.iloc[i])) / rng >= TRIPWIRE_RETRACE:
            return i
    return None


def reversal_trades(bars: pd.DataFrame, atr: float, live: bool) -> list[dict]:
    rth = bars.between_time("09:30", "15:59")
    if rth.empty or not atr:
        return []
    b2 = _two_min(rth)
    if len(b2) < 2:
        return []
    open_ = float(rth["open"].iloc[0])
    trip = _tripwire_index(b2, open_, atr)
    flush_px = open_ - FLUSH_ATR * atr
    out, pos, stops = [], None, 0

    def close_out(i, price, kind):
        whole = (pos["entry"] - price) / pos["risk"]
        r = whole if pos["half"] is None else 0.5 * (pos["entry"] - pos["half"]["price"]) / pos["risk"] + 0.5 * whole
        out.append({"entry_t": pos["entry_t"], "entry": pos["entry"], "stop": pos["stop"],
                    "exit_t": et_epoch(b2.index[i]), "exit": float(price), "kind": kind,
                    "r": float(r), "half": pos["half"]})

    for i in range(len(b2)):
        ts = b2.index[i]
        hi, lo, close = (float(b2[c].iloc[i]) for c in ("high", "low", "close"))
        if pos is None:
            if stops >= REV_MAX_STOPS or len(out) >= REV_MAX_ATTEMPTS:
                break
            if trip is not None and i >= trip:
                break                                   # tripwire = the day's trend is over
            end = (ts + BAR).time()
            if i >= 1 and REV_FIRST_END <= end <= REV_GATE_END and close < float(b2["low"].iloc[i - 1]):
                stop = float(b2["hod"].iloc[i])
                pos = {"entry_t": et_epoch(ts), "entry": close, "stop": stop,
                       "risk": max(stop - close, REV_RISK_FLOOR_ATR * atr), "half": None}
            continue
        if hi >= pos["stop"]:
            close_out(i, pos["stop"], "stop")
            stops += 1
            pos = None
            continue
        if pos["half"] is None and lo <= flush_px:
            pos["half"] = {"t": et_epoch(ts), "price": float(flush_px)}
        if trip is not None and i >= trip:
            close_out(i, close, "tripwire")
            pos = None
            continue
        if ts.time() >= EOD_EXIT:
            close_out(i, close, "close")
            pos = None
            break
    if pos is not None:
        close_out(len(b2) - 1, float(b2["close"].iloc[-1]), "open" if live else "close")
    return out


# ---------------------------------------------------------------------------
# One day
# ---------------------------------------------------------------------------

def review(ticker: str, date: str, side: str, cap: str, outcome: dict | None = None,
           live: bool = False) -> dict:
    """Everything a review chart needs for (ticker, date). `outcome` is the
    scorecard row for that day, when there is one: its ATR is the one the
    scorecard's exit_r used, and its results are shown beside the replay."""
    out = {"v": VERSION, "ticker": ticker, "date": date, "side": side, "cap": cap, "live": live,
           "candles": [], "next_candles": [], "trades": [], "session": {}}
    hist = completed_daily(ticker, date)
    if hist is None:
        out["error"] = "no daily history"
        return out
    static = static_from_hist(hist)
    out.update(prior_close=float(static["prior_close"]), mid_bb=static.get("mid_bb"),
               atr=float(static["atr"]))
    out["daily"] = _daily_rows(hist.iloc[-180:])        # 180 so the 50-day average covers the view
    if side == "bounce":
        out["ladder"] = ladder_payload(static, cap, ticker)
    else:
        ema9 = hist["close"].ewm(span=9, adjust=False).mean().iloc[-1]
        out.update(ema9=float(ema9), gap_level=float(static["prior_close"]) * 1.04)

    bars = day_bars(ticker, date, live)
    if bars is None or bars.empty:
        return out
    bars = bars[bars.index.strftime("%Y-%m-%d") == date]
    if bars.empty:
        return out
    out["candles"] = _candles(bars)
    out["session"] = sess = _session(bars)
    open_ = sess.get("open")
    if open_ is None:
        return out

    atr = out["atr"]
    if outcome and outcome.get("atr_pct") and outcome.get("entry_open"):
        atr = float(outcome["atr_pct"]) * float(outcome["entry_open"])
    if side == "bounce":
        g = grade_at(static, cap, open_, ticker)
        out["at_open"] = {"price": open_, "gap": open_ / out["prior_close"] - 1, "grade": g["grade"],
                          "score": g["score"], "rec": g["rec"], "intensity": g["intensity"]}
        out["trades"] = bounce_trades(bars, atr, live)
    else:
        out["at_open"] = {"price": open_, "gap": open_ / out["prior_close"] - 1}
        out["trades"] = reversal_trades(bars, atr, live)
    out["rules_r"] = sum(t["r"] for t in out["trades"]) if out["trades"] else None
    if not live:
        day = datetime.date.fromisoformat(date)
        today = pd.Timestamp.now(tz=TZ).date()
        after = after_days(ticker, date)
        out["daily"] += after
        nxt = next_trading_day(day)
        if nxt < today:                  # never through fetch_day's permanent cache while it's partial
            nb = day_bars(ticker, nxt.isoformat(), live=False)
            if nb is not None and not nb.empty:
                nb = nb[nb.index.strftime("%Y-%m-%d") == nxt.isoformat()]
                out["next_candles"] = _candles(nb)
                out["next_date"] = nxt.isoformat()
                out["next_session"] = _session(nb)
        # safe to cache for good: the next day is over and the days after are in
        out["complete"] = nxt < today and (len(after) >= 6 or (today - day).days > 12)
    return out
