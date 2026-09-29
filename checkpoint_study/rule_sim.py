"""Deterministic simulator of orderPipe's live exit rule on historical 5s bars.

Mirrors trader/trade_manager.py + trade_watcher.run_stop_watcher exactly:

  * clock      time_elapsed = 5s bar CLOSE-time - anchor (headline_time)
  * < 2 min    ref stop: ref -/+ get_stop_offset(ref); if that is on the wrong
               side of the first close, first close -/+ offset (TradeManager.set_stop)
  * >= 2 min   2_min_quick: level = low (long) / high (short) of the latest
               COMPLETED 2-min bar — label strictly < t (the 2-min bar ending T
               arrives just after the 5s bar ending T) — bad-print filtered.
               Not a ratchet. Breach = 5s close strictly through the level.
               2_min_close (sensitivity): latest completed 2-min CLOSE through
               the bar before it.
  * checkpoint first 5s bar with elapsed >= k min. check_time_alerts runs BEFORE
               the stop check in the same loop pass, so a survivor at k has no
               breach strictly before the checkpoint bar.

Policies from checkpoint k (forward R from the checkpoint close px_k):
  hold      keep the rule; exit at its first breach at/after t_k
  trail_1m  from t_k the level is the TIGHTER of the latest completed 1-min
            bar's low/high and the 2-min level
  exit_now  flatten at t_k
  trim_half 0.5 * hold + 0.5 * exit_now  (identity — reported, not tested)
Fills: close of the first 5s bar at/after trigger + latency (manual close).
No breach by the horizon (anchor + horizon, capped at 16:00): exit at the last
bar, flagged `capped`. The trader's actual exit is never an input.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

NS = 1_000_000_000


def stop_offset(price: float) -> float:
    """Mirror of orderPipe calculators/stop_calculator.get_stop_offset."""
    return 0.01 if price < 10 else 0.05 if price < 100 else 0.20


@dataclass(frozen=True)
class SimParams:
    trail: str = "quick"            # 'quick' | 'close'
    min_hold_s: int = 120           # TradeManager.should_watch_stops
    latency_s: int = 10             # alert -> manual close
    horizon_min: int = 120


@dataclass
class DayBars:
    """One ticker-day: full-session 5s bars plus derived (filtered) 1/2-min bars."""
    fives: pd.DataFrame
    ones: pd.DataFrame
    twos: pd.DataFrame
    _cache: dict = field(default_factory=dict)

    @classmethod
    def build(cls, fives: pd.DataFrame, minute_bars: Callable, extremes_fn=None) -> "DayBars":
        return cls(fives=fives, ones=minute_bars(fives, 1, extremes_fn),
                   twos=minute_bars(fives, 2, extremes_fn))


def _latest_before(labels: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Index of the latest bar whose label (close-time) is strictly before t; -1 if none."""
    return np.searchsorted(labels, t, side="left") - 1


def _fill(t: np.ndarray, c: np.ndarray, i: int, latency_ns: int) -> tuple:
    """(index, price) of the first bar at/after t[i] + latency; last bar if none."""
    j = int(np.searchsorted(t, t[i] + latency_ns, side="left"))
    j = min(j, len(t) - 1)
    return j, float(c[j])


def _first_true(mask: np.ndarray, start: int) -> Optional[int]:
    hits = np.flatnonzero(mask[start:])
    return int(start + hits[0]) if len(hits) else None


def simulate(day: DayBars, *, anchor: pd.Timestamp, side: int, ref: float, r_ps: float,
             p: SimParams = SimParams(), checkpoints=(6, 17, 30),
             features_fn: Optional[Callable] = None, entry: Optional[float] = None) -> dict:
    """Run the rule for one episode. Returns {'path': {...}, 'k': {k: row or None}}.

    features_fn(fives_upto_tk, side=, entry=, r_ps=, trail_level=) -> dict, if given,
    is evaluated at each surviving checkpoint on bars (anchor, t_k] only.
    """
    close_16 = anchor.normalize() + pd.Timedelta(hours=16)
    horizon_end = min(anchor + pd.Timedelta(minutes=p.horizon_min), close_16)
    f = day.fives[(day.fives.index > anchor) & (day.fives.index <= horizon_end)]
    if len(f) < 2:
        return {"path": None, "k": {k: None for k in checkpoints}}
    t = f.index.asi8
    c = f["close"].to_numpy(dtype=float)
    a_ns = anchor.value
    elapsed_s = (t - a_ns) / NS
    lat = p.latency_s * NS

    # Ref-phase stop (TradeManager.set_stop against the first close).
    off = stop_offset(ref)
    stop = round(ref - side * off, 4)
    if not side * (c[0] - stop) > 0:
        stop = round(c[0] - side * off, 4)

    L2 = day.twos.index.asi8
    lo2, hi2, cl2 = (day.twos[x].to_numpy(dtype=float) for x in ("low", "high", "close"))
    i2 = _latest_before(L2, t)
    ok2 = i2 >= 0
    lvl2 = np.full(len(t), np.nan)
    lvl2[ok2] = (lo2 if side > 0 else hi2)[i2[ok2]]
    ref_phase = (elapsed_s < p.min_hold_s) | np.isnan(lvl2)

    if p.trail == "quick":
        trail_breach = side * (c - lvl2) < 0
    else:  # 'close': latest completed bar's close through the bar before it
        prev = np.clip(i2 - 1, 0, None)
        prior = (lo2 if side > 0 else hi2)[prev]
        latest_close = cl2[np.clip(i2, 0, None)]
        trail_breach = (i2 >= 1) & (side * (latest_close - prior) < 0)
    breach = np.where(ref_phase, side * (c - stop) < 0, trail_breach)

    L1 = day.ones.index.asi8
    lo1, hi1 = day.ones["low"].to_numpy(dtype=float), day.ones["high"].to_numpy(dtype=float)
    i1 = _latest_before(L1, t)
    lvl1 = np.full(len(t), np.nan)
    ok1 = i1 >= 0
    lvl1[ok1] = (lo1 if side > 0 else hi1)[i1[ok1]]
    tight = np.fmax(lvl1, lvl2) if side > 0 else np.fmin(lvl1, lvl2)
    breach_1m = np.where(ref_phase, breach, side * (c - tight) < 0)

    first_breach = _first_true(breach, 0)
    path = {
        "stop": stop,
        "first_breach_ts": pd.Timestamp(t[first_breach], tz="UTC").tz_convert(anchor.tz)
        if first_breach is not None else None,
        "first_breach_phase": (("ref" if ref_phase[first_breach] else "trail")
                               if first_breach is not None else None),
        "horizon_end": horizon_end,
        "horizon_capped_16": horizon_end == close_16,
    }

    out = {}
    for k in checkpoints:
        ik = int(np.searchsorted(elapsed_s, k * 60, side="left"))
        if ik >= len(t) or (first_breach is not None and first_breach < ik):
            out[k] = None
            continue
        px = float(c[ik])

        def policy(mask):
            j = _first_true(mask, ik)
            if j is None:
                return float(c[-1]), len(t) - 1, True
            jf, price = _fill(t, c, j, lat)
            return price, jf, False

        hold_px, hold_j, hold_capped = policy(breach)
        t1m_px, t1m_j, t1m_capped = policy(breach_1m)
        now_j, now_px = _fill(t, c, ik, lat)
        R = lambda x: side * (x - px) / r_ps
        row = {
            "t_k": pd.Timestamp(t[ik], tz="UTC").tz_convert(anchor.tz),
            "px_k": px,
            "trail_level_k": float(lvl2[ik]) if not np.isnan(lvl2[ik]) else None,
            "fwd_hold": R(hold_px), "hold_exit_min": (t[hold_j] - t[ik]) / NS / 60,
            "hold_capped": hold_capped,
            "fwd_trail_1m": R(t1m_px), "trail_1m_exit_min": (t[t1m_j] - t[ik]) / NS / 60,
            "trail_1m_capped": t1m_capped,
            "fwd_exit_now": R(now_px),
        }
        row["fwd_trim_half"] = 0.5 * row["fwd_hold"] + 0.5 * row["fwd_exit_now"]
        if features_fn is not None:
            row.update(features_fn(f.iloc[:ik + 1], side=side, entry=entry, r_ps=r_ps,
                                   trail_level=row["trail_level_k"]))
        out[k] = row
    return {"path": path, "k": out}
