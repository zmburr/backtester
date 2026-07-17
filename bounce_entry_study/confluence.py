"""K-of-N confluence over detector fire streams + the day simulator.

Simulation contract (user decisions, 2026-07-17):
  * enter at the confluence fire bar's CLOSE, at fire time exactly;
  * stop = raw session low as of the fire bar (low of day, no buffer);
  * a later bar trading BELOW the stop exits at the stop for -1R;
  * after a stop-out the next fresh confluence fire re-enters (fires reset
    on every new session low, so re-entry needs post-flush evidence);
  * open position exits at the last bar's close;
  * per-attempt risk = entry - stop, floored at 0.25% of entry so a fire
    bar sitting on the day low can't mint absurd R.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .detectors import BAR_MIN

RISK_FLOOR_FRAC = 0.0025


# ---------------------------------------------------------------------------
# confluence fires
# ---------------------------------------------------------------------------

def confluence_fires(two: pd.DataFrame, det_fires: Dict[str, List[pd.Timestamp]],
                     k: int, window_min: int, gate: str = "09:30",
                     reset_on_new_low: bool = True) -> List[pd.Timestamp]:
    """Bar-end times where >= k distinct detectors fired within the window.

    reset_on_new_low: fires older than the latest new-session-low bar don't
    count — a break that preceded a flush is stale evidence.
    """
    ends = two.index + pd.Timedelta(minutes=BAR_MIN)
    day = ends[0].date()
    gate_ts = pd.Timestamp(f"{day} {gate}", tz=ends[0].tz)
    window = pd.Timedelta(minutes=window_min)

    # bar-end time of every new RTH session low
    cummin = two["low"].cummin()
    is_new_low = two["low"] <= cummin  # low[i] == running min including itself
    new_low_ends = [ends[i] for i in range(len(two))
                    if is_new_low.iloc[i] and (i == 0 or two["low"].iloc[i] < cummin.iloc[i - 1])]

    out: List[pd.Timestamp] = []
    for i in range(len(two)):
        t = ends[i]
        if t < gate_ts:
            continue
        start = t - window
        if reset_on_new_low:
            lows_before = [nl for nl in new_low_ends if nl < t]
            if lows_before:
                start = max(start, lows_before[-1])
        n_live = sum(
            1 for fires in det_fires.values()
            if any(start < f <= t for f in fires)
        )
        if n_live >= k:
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# day simulation
# ---------------------------------------------------------------------------

@dataclass
class Attempt:
    entry_time: pd.Timestamp
    entry: float
    stop: float
    risk: float
    exit_time: Optional[pd.Timestamp] = None
    exit: Optional[float] = None
    stopped: bool = False
    r: float = 0.0


@dataclass
class DayResult:
    attempts: List[Attempt] = field(default_factory=list)
    total_r: float = 0.0
    n_stopped: int = 0
    survived: bool = False           # an attempt made it to the close
    first_entry: Optional[pd.Timestamp] = None
    surviving_entry: Optional[pd.Timestamp] = None
    lag_min: Optional[float] = None  # surviving entry vs true RTH low (minutes)


def simulate_day(two: pd.DataFrame, fires: List[pd.Timestamp]) -> DayResult:
    res = DayResult()
    if two.empty:
        return res
    ends = two.index + pd.Timedelta(minutes=BAR_MIN)
    end_to_i = {t: i for i, t in enumerate(ends)}
    cummin = two["low"].cummin()
    last_close = two["close"].iloc[-1]
    true_low_end = ends[int(two["low"].values.argmin())]

    pos: Optional[Attempt] = None
    fire_iter = [f for f in fires if f in end_to_i]

    for i in range(len(two)):
        t = ends[i]
        # manage the open position first (this bar can stop us out)
        if pos is not None and t > pos.entry_time:
            if two["low"].iloc[i] < pos.stop:
                pos.exit_time, pos.exit, pos.stopped = t, pos.stop, True
                pos.r = -1.0
                res.attempts.append(pos)
                pos = None
        # then consider an entry at this bar's close
        if pos is None and t in fire_iter:
            after_last_exit = (not res.attempts
                               or t > res.attempts[-1].exit_time)
            if after_last_exit:
                entry = float(two["close"].iloc[i])
                stop = float(cummin.iloc[i])
                risk = max(entry - stop, RISK_FLOOR_FRAC * entry)
                pos = Attempt(entry_time=t, entry=entry, stop=stop, risk=risk)

    if pos is not None:
        pos.exit_time, pos.exit = ends[-1], float(last_close)
        pos.r = (pos.exit - pos.entry) / pos.risk
        res.attempts.append(pos)
        res.survived = True
        res.surviving_entry = pos.entry_time
        res.lag_min = (pos.entry_time - true_low_end).total_seconds() / 60.0

    res.total_r = sum(a.r for a in res.attempts)
    res.n_stopped = sum(1 for a in res.attempts if a.stopped)
    res.first_entry = res.attempts[0].entry_time if res.attempts else None
    return res
