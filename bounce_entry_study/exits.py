"""Exit strategies layered on the entry simulator.

All strategies keep the raw-LOD disaster stop (checked first, fills at the
stop, -1R). On top of that:

  close            hold to the last bar's close (baseline)
  trail(arm_atr)   trailing prior-2-min-bar-low: once ARMED (immediately if
                   arm_atr=0, else once the trade has moved entry+arm*ATR in
                   favor), the first bar trading below the PRIOR completed
                   2-min bar's low exits at that level
  vwap             profit-take at session VWAP: first bar whose high reaches
                   VWAP (only when VWAP > entry) exits at VWAP — the user's
                   real-life habit, here to be measured
  atr_target(T)    exit at entry + T*ATR when a bar's high reaches it

Re-entry model: a NON-PROFITABLE exit (LOD stop, or a trail-out at/below
entry) re-arms the campaign — the next fresh confluence fire re-enters.
A profitable exit ends the day's campaign (you took your money).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .confluence import RISK_FLOOR_FRAC, Attempt, DayResult
from .detectors import BAR_MIN


@dataclass(frozen=True)
class ExitSpec:
    kind: str                 # close | trail | vwap | atr_target
    arm_atr: float = 0.0      # trail: favorable move (in ATRs) required to arm
    target_atr: float = 0.0   # atr_target: target distance in ATRs

    @property
    def name(self) -> str:
        if self.kind == "trail":
            return f"trail(arm {self.arm_atr:g}ATR)"
        if self.kind == "atr_target":
            return f"target {self.target_atr:g}ATR"
        return self.kind


def simulate_day_exits(two: pd.DataFrame, fires: List[pd.Timestamp],
                       spec: ExitSpec, atr: Optional[float],
                       risk_floor_atr: float = 0.0) -> DayResult:
    """Entry rules identical to confluence.simulate_day; exits per spec.

    risk_floor_atr > 0 additionally floors per-attempt risk at that many
    ATRs (robustness view — kills tiny-stop R inflation).
    """
    res = DayResult()
    if two.empty:
        return res
    ends = two.index + pd.Timedelta(minutes=BAR_MIN)
    end_to_i = {t: i for i, t in enumerate(ends)}
    cummin = two["low"].cummin()
    last_close = float(two["close"].iloc[-1])
    true_low_end = ends[int(two["low"].values.argmin())]
    fire_set = set(f for f in fires if f in end_to_i)

    pos: Optional[Attempt] = None
    entry_i: Optional[int] = None
    peak: float = 0.0
    campaign_over = False

    def _finish(a: Attempt, t, fill: float, stopped: bool) -> bool:
        """Close out an attempt; returns True if the campaign continues."""
        a.exit_time, a.exit, a.stopped = t, fill, stopped
        a.r = -1.0 if stopped else (fill - a.entry) / a.risk
        res.attempts.append(a)
        return stopped or fill <= a.entry   # non-profitable exit -> re-arm

    for i in range(len(two)):
        t = ends[i]
        lo = float(two["low"].iloc[i])
        hi = float(two["high"].iloc[i])

        if pos is not None and t > pos.entry_time:
            exited = False
            # 1) disaster stop at LOD — always first, worst fill
            if lo < pos.stop:
                if not _finish(pos, t, pos.stop, stopped=True):
                    campaign_over = True
                pos, exited = None, True
            # 2) strategy exit
            elif spec.kind == "trail":
                armed = (spec.arm_atr <= 0 or
                         (atr and peak >= pos.entry + spec.arm_atr * atr))
                trail_level = float(two["low"].iloc[i - 1]) if i > 0 else None
                if armed and trail_level is not None and lo < trail_level:
                    if not _finish(pos, t, trail_level, stopped=False):
                        campaign_over = True
                    pos, exited = None, True
            elif spec.kind == "vwap":
                vw = two["vwap_at_end"].iloc[i]
                if not pd.isna(vw) and vw > pos.entry and hi >= vw:
                    if not _finish(pos, t, float(vw), stopped=False):
                        campaign_over = True
                    pos, exited = None, True
            elif spec.kind == "atr_target" and atr:
                tgt = pos.entry + spec.target_atr * atr
                if hi >= tgt:
                    if not _finish(pos, t, tgt, stopped=False):
                        campaign_over = True
                    pos, exited = None, True
            if not exited and pos is not None:
                peak = max(peak, hi)

        if pos is None and not campaign_over and t in fire_set:
            after_last = (not res.attempts or t > res.attempts[-1].exit_time)
            if after_last:
                entry = float(two["close"].iloc[i])
                stop = float(cummin.iloc[i])
                risk = max(entry - stop, RISK_FLOOR_FRAC * entry)
                if risk_floor_atr > 0 and atr:
                    risk = max(risk, risk_floor_atr * atr)
                pos = Attempt(entry_time=t, entry=entry, stop=stop, risk=risk)
                entry_i = i
                peak = entry

    if pos is not None:
        pos.exit_time, pos.exit = ends[-1], last_close
        pos.r = (last_close - pos.entry) / pos.risk
        res.attempts.append(pos)

    survivors = [a for a in res.attempts if not a.stopped and a.r > 0]
    res.survived = bool(survivors)
    if survivors:
        res.surviving_entry = survivors[0].entry_time
        res.lag_min = (survivors[0].entry_time - true_low_end).total_seconds() / 60.0
    res.total_r = sum(a.r for a in res.attempts)
    res.n_stopped = sum(1 for a in res.attempts if a.stopped)
    res.first_entry = res.attempts[0].entry_time if res.attempts else None
    return res
