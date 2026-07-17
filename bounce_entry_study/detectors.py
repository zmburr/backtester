"""Pure bottom-structure detectors on a 2-min RTH bar frame.

Every detector returns a sorted list of FIRE TIMES (pd.Timestamp = the 2-min
bar's END, i.e. the moment the information exists). No lookahead: pivot-based
signals confirm k bars after the pivot and fire at the confirming bar's end.

These functions are the port target for a live morning-watcher evaluator —
keep them dependency-free (pandas only) and stateless.

Frame contract (build_2min): index = bar START time (ET), columns
open/high/low/close/volume, strictly RTH 9:30–16:00, plus a parallel
``vwap_at_end`` series for D4 (session VWAP from 4:00 AM, sampled at bar end).
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

BAR_MIN = 2  # detector timeframe in minutes


def build_2min(bars_1min: pd.DataFrame) -> pd.DataFrame:
    """1-min full-session bars -> 2-min RTH frame with vwap_at_end."""
    day = bars_1min.index[0].date()
    # Session VWAP from the 4 AM premarket, matching the live watcher.
    pv = (bars_1min["vwap"].fillna(bars_1min["close"]) * bars_1min["volume"]).cumsum()
    vv = bars_1min["volume"].cumsum()
    session_vwap = (pv / vv.replace(0, pd.NA)).ffill()

    rth = bars_1min.between_time("09:30", "15:59")
    if rth.empty:
        return pd.DataFrame()
    two = rth.resample(f"{BAR_MIN}min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "high", "low", "close"])
    # bar END time index for "when do I know this" bookkeeping
    ends = two.index + pd.Timedelta(minutes=BAR_MIN)
    # session VWAP as of each bar end (last 1-min cum value at/before end)
    two["vwap_at_end"] = session_vwap.reindex(
        session_vwap.index.union(ends)).ffill().reindex(ends).to_numpy()
    two.attrs["day"] = day
    return two


def _ends(two: pd.DataFrame) -> pd.DatetimeIndex:
    return two.index + pd.Timedelta(minutes=BAR_MIN)


# ---------------------------------------------------------------------------
# D1 — 2-min prior-bar break to the upside
# ---------------------------------------------------------------------------

def pbb_up(two: pd.DataFrame, held: int = 0) -> List[pd.Timestamp]:
    """Close above the prior 2-min bar's high.

    held=0: fires at the break bar's end. held=h: fires h bars later, only
    if none of those h bars took out the break bar's low (a HELD break).
    """
    ends = _ends(two)
    h_prev = two["high"].shift(1)
    is_break = two["close"] > h_prev
    fires: List[pd.Timestamp] = []
    n = len(two)
    for i in range(1, n):
        if not is_break.iloc[i]:
            continue
        if held == 0:
            fires.append(ends[i])
            continue
        j = i + held
        if j >= n:
            continue
        if two["low"].iloc[i + 1:j + 1].min() > two["low"].iloc[i]:
            fires.append(ends[j])
    return fires


# ---------------------------------------------------------------------------
# pivot helpers (shared by D2 / D3)
# ---------------------------------------------------------------------------

def _pivot_lows(two: pd.DataFrame, k: int) -> List[int]:
    """Indices i where low[i] is the minimum of the (2k+1)-bar neighborhood."""
    lows = two["low"]
    out = []
    for i in range(k, len(two) - k):
        win = lows.iloc[i - k:i + k + 1]
        if lows.iloc[i] <= win.min():
            out.append(i)
    return out


def _pivot_highs(two: pd.DataFrame, k: int) -> List[int]:
    highs = two["high"]
    out = []
    for i in range(k, len(two) - k):
        win = highs.iloc[i - k:i + k + 1]
        if highs.iloc[i] >= win.max():
            out.append(i)
    return out


# ---------------------------------------------------------------------------
# D2 — higher low
# ---------------------------------------------------------------------------

def higher_low(two: pd.DataFrame, k: int = 2, floor_frac: float = 0.0,
               atr: Optional[float] = None) -> List[pd.Timestamp]:
    """Confirmed swing low sitting above the prior session low.

    A swing low at bar i (pivot width k) confirms at bar i+k's end. It fires
    if its low is above the session low BEFORE it (cummin through i-1) by
    at least floor_frac*ATR (floor 0 when ATR unknown).
    """
    ends = _ends(two)
    floor = (floor_frac * atr) if (atr and atr > 0) else 0.0
    cummin_prev = two["low"].cummin().shift(1)
    fires: List[pd.Timestamp] = []
    for i in _pivot_lows(two, k):
        prior_low = cummin_prev.iloc[i]
        if pd.isna(prior_low):
            continue
        if two["low"].iloc[i] > prior_low + floor:
            j = i + k
            if j < len(two):
                fires.append(ends[j])
    return fires


# ---------------------------------------------------------------------------
# D3 — structure break (break of the most recent confirmed swing high)
# ---------------------------------------------------------------------------

def structure_break(two: pd.DataFrame, k: int = 2) -> List[pd.Timestamp]:
    """Close above the most recent CONFIRMED swing high; re-arms per swing.

    The objective downtrend-line break: on a down day the latest swing high
    is a lower high, and closing through it is the change of character.
    """
    ends = _ends(two)
    # swing high i becomes *known* at end of bar i+k
    known = sorted((i + k, i) for i in _pivot_highs(two, k) if i + k < len(two))
    fires: List[pd.Timestamp] = []
    ptr = 0
    armed_level: Optional[float] = None
    armed_at: Optional[int] = None
    for i in range(len(two)):
        # confirm any swings whose confirmation bar has completed (< i: the
        # level must be known BEFORE the bar that breaks it)
        while ptr < len(known) and known[ptr][0] < i:
            conf_bar, swing_i = known[ptr]
            armed_level = two["high"].iloc[swing_i]
            armed_at = conf_bar
            ptr += 1
        if armed_level is None:
            continue
        if two["close"].iloc[i] > armed_level:
            fires.append(ends[i])
            armed_level = None   # disarm until the next confirmed swing high
    return fires


# ---------------------------------------------------------------------------
# D4 — VWAP reclaim
# ---------------------------------------------------------------------------

def vwap_reclaim(two: pd.DataFrame, min_minutes_below: int = 30) -> List[pd.Timestamp]:
    """2-min close crosses above session VWAP after >= m minutes below it."""
    ends = _ends(two)
    fires: List[pd.Timestamp] = []
    below_since: Optional[pd.Timestamp] = None
    for i in range(len(two)):
        vw = two["vwap_at_end"].iloc[i]
        if pd.isna(vw):
            continue
        c = two["close"].iloc[i]
        if c < vw:
            if below_since is None:
                below_since = ends[i]
        elif c > vw:
            if (below_since is not None
                    and (ends[i] - below_since) >= pd.Timedelta(minutes=min_minutes_below)):
                fires.append(ends[i])
            below_since = None
    return fires


DETECTORS = {
    "pbb": lambda two, atr, p: pbb_up(two, held=p.get("held", 0)),
    "hl": lambda two, atr, p: higher_low(two, k=p.get("k", 2),
                                         floor_frac=p.get("floor", 0.0), atr=atr),
    "sb": lambda two, atr, p: structure_break(two, k=p.get("k", 2)),
    "vwap": lambda two, atr, p: vwap_reclaim(two, min_minutes_below=p.get("m", 30)),
}
