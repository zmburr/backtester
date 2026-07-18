"""Study #2: when does ADDING into a working bounce pay?

Base system unchanged: first 2-min break entry (unit 1), stop at LOD,
re-entry after a full stop-out, trail armed at +1.5 ATR (whole position).

Add rule tested: while in a position, each FRESH confluence fire adds one
more 1R unit — but only once price has moved >= add_arm ATRs above the
initial entry, up to max_units total. Each unit's risk is its own entry
minus the LOD at that moment (floored 0.25%); a LOD stop takes every unit
out at -1R each; the trail/close exit closes all units at one price.

Accounting is per-unit R (your 1R-per-decision framework): a day's total
R = sum of unit Rs. "r_per_unit" is the efficiency read — whether the
marginal add carried its weight.

    python -m bounce_entry_study.run_add_study
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .confluence import RISK_FLOOR_FRAC, confluence_fires
from .detectors import BAR_MIN
from .run_study import REPORTS, prepare_days

log = logging.getLogger(__name__)

ENTRY = {"dets": ("pbb:h0",), "k": 1, "window": 16, "gate": "09:30"}
TRAIL_ARM_ATR = 1.5


@dataclass
class Unit:
    entry_time: pd.Timestamp
    entry: float
    risk: float
    r: float = 0.0


def simulate_with_adds(two: pd.DataFrame, fires: List[pd.Timestamp],
                       atr: Optional[float], add_arm: float,
                       max_units: int) -> dict:
    ends = two.index + pd.Timedelta(minutes=BAR_MIN)
    cummin = two["low"].cummin()
    fire_set = set(fires)
    last_close = float(two["close"].iloc[-1])

    units: List[Unit] = []          # open units
    done_r: List[float] = []        # realized unit Rs
    n_adds = n_stops = 0
    campaign_over = False
    stop_level: Optional[float] = None
    entry0: Optional[float] = None
    peak = 0.0
    armed = False
    trail: Optional[float] = None
    last_exit_t: Optional[pd.Timestamp] = None

    def flat_all(fill: float, stopped: bool):
        nonlocal units, n_stops
        for u in units:
            u.r = -1.0 if stopped else (fill - u.entry) / u.risk
            done_r.append(u.r)
        if stopped:
            n_stops += len(units)
        units = []

    for i in range(len(two)):
        t = ends[i]
        lo = float(two["low"].iloc[i])
        hi = float(two["high"].iloc[i])
        c = float(two["close"].iloc[i])

        # roll 2-min trail bookkeeping
        if units and t > units[0].entry_time:
            peak = max(peak, hi)
            if not armed and atr and peak >= entry0 + TRAIL_ARM_ATR * atr:
                armed = True
                trail = None
            # LOD stop first
            if lo < stop_level:
                flat_all(stop_level, stopped=True)
                last_exit_t = t
                armed = False
            elif armed and trail is not None and lo < trail:
                fill = trail
                flat_all(fill, stopped=False)
                last_exit_t = t
                if fill > entry0:
                    campaign_over = True
                armed = False
        # two IS the 2-min frame — the prior row's low is the trail level
        if armed and i > 0:
            trail = float(two["low"].iloc[i - 1])

        if t in fire_set and not campaign_over:
            if not units:
                if last_exit_t is None or t > last_exit_t:
                    entry0 = c
                    stop_level = float(cummin.iloc[i])
                    risk = max(c - stop_level, RISK_FLOOR_FRAC * c)
                    units.append(Unit(t, c, risk))
                    peak, armed, trail = c, False, None
            elif (len(units) < max_units and atr
                  and c >= entry0 + add_arm * atr):
                stop_now = float(cummin.iloc[i])
                risk = max(c - stop_now, RISK_FLOOR_FRAC * c)
                units.append(Unit(t, c, risk))
                n_adds += 1

    if units:
        for u in units:
            u.r = (last_close - u.entry) / u.risk
            done_r.append(u.r)
        units = []

    total = sum(done_r)
    return {"total_r": total, "n_units": len(done_r), "n_adds": n_adds,
            "n_stopped_units": n_stops,
            "r_per_unit": total / len(done_r) if done_r else 0.0}


def run() -> None:
    hist, _ = prepare_days()
    days = [d for d in hist if d.two.attrs.get("atr")]

    fires_by_day = {}
    for d in days:
        fires_by_day[id(d)] = confluence_fires(
            d.two, {v: d.det_fires[v] for v in ENTRY["dets"]},
            k=ENTRY["k"], window_min=ENTRY["window"], gate=ENTRY["gate"])

    # score buckets from study #1 for the interaction cut
    try:
        sc = pd.read_csv(REPORTS / "score_outcome.csv")
        score_by_day = {(r["ticker"], r["date"]): r["score_low"] for _, r in sc.iterrows()}
    except Exception:
        score_by_day = {}

    configs = [("baseline (no adds)", None, 1)]
    for arm in (0.25, 0.5, 1.0):
        for mx in (2, 3):
            configs.append((f"add>= {arm:g}ATR, max {mx}", arm, mx))

    rows = []
    detail_best = None
    for name, arm, mx in configs:
        per_day = []
        for d in days:
            r = simulate_with_adds(d.two, fires_by_day[id(d)],
                                   d.two.attrs.get("atr"),
                                   add_arm=arm if arm is not None else 99.0,
                                   max_units=mx)
            r["ticker"], r["date"] = d.ticker, d.date_iso
            r["score_low"] = score_by_day.get((d.ticker, d.date_iso))
            per_day.append(r)
        df = pd.DataFrame(per_day)
        rows.append({
            "config": name,
            "mean_day_r": round(df["total_r"].mean(), 3),
            "median_day_r": round(df["total_r"].median(), 3),
            "pct_days_pos": round(100 * (df["total_r"] > 0).mean(), 1),
            "worst_day_r": round(df["total_r"].min(), 2),
            "mean_units": round(df["n_units"].mean(), 2),
            "r_per_unit": round((df["total_r"].sum() / df["n_units"].sum()), 3),
            "adds_per_day": round(df["n_adds"].mean(), 2),
        })
        if name.startswith("add>= 0.5ATR, max 3"):
            detail_best = df

    table = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS / "add_study.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== ADD STUDY (per-unit R accounting) ===")
    print(table.to_string(index=False))

    if detail_best is not None and detail_best["score_low"].notna().any():
        detail_best["bucket"] = pd.cut(detail_best["score_low"],
                                       bins=[0, 40, 55, 70, 100],
                                       labels=["<40", "40-55", "55-70", ">=70"])
        cut = detail_best.groupby("bucket", observed=True).agg(
            n=("total_r", "size"), mean_r=("total_r", "mean"),
            adds=("n_adds", "mean")).round(2)
        print("\nadd>=0.5ATR max3 — by score-at-low bucket:")
        print(cut.to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
