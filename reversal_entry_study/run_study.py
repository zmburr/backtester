"""Reversal entry study — detector x config sweep on curated + control days.

    python -m reversal_entry_study.run_study

See PLAN.md. Outputs reports/reversal_entry_study/configs_ranked.csv and
top_config_days.csv, plus a console summary. Short-side conventions:
entry at fire-bar close, stop = running HOD (+buffer), R per attempt =
(entry - exit) / max(risk, 0.25 ATR).
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from datetime import time as dt_time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .fetch_bars import cache_path, load_universe

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "reversal_entry_study"

RTH_OPEN = dt_time(9, 30)
FIRST_FIRE = dt_time(9, 32)          # need one completed 2-min bar
EOD_EXIT = dt_time(15, 55)
RISK_FLOOR_ATR = 0.25                # guards tight-stop R inflation
CONF_WINDOW_MIN = 10                 # conf2of3: 2 fires within this window

DETECTORS = ["pbb_down", "vwap_loss", "open_fail", "pm_low_break", "conf2of3"]
GATE_ENDS = [dt_time(10, 30), dt_time(11, 0), dt_time(15, 30)]
STOP_BUFFERS = [0.0, 0.15]           # x ATR above running HOD
MAX_ATTEMPTS = [1, 2, 3]

GAP_FADE_SETUPS = {"3DGapFade", "2DGapFade", "GapDownTrendBreak"}


# ---------------------------------------------------------------------------
# Per-day preparation: 2-min bars + detector fire times
# ---------------------------------------------------------------------------

@dataclass
class Day:
    ticker: str
    date_iso: str
    cohort: str
    setup: str
    open_: float
    atr: float
    pm_low: Optional[float]
    bars2: pd.DataFrame            # 2-min RTH bars: open/high/low/close/vwap/hod
    fires: Dict[str, List[pd.Timestamp]]
    day_low: float
    day_close: float


def _two_min(rth: pd.DataFrame) -> pd.DataFrame:
    g = rth.resample("2min", label="left", closed="left")
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(),
        "volume": g["volume"].sum(),
    }).dropna(subset=["close"])
    pv = (rth["close"] * rth["volume"]).resample("2min", label="left", closed="left").sum()
    out["cum_pv"] = pv.cumsum()
    out["cum_vol"] = out["volume"].cumsum()
    out["vwap"] = out["cum_pv"] / out["cum_vol"].replace(0, np.nan)
    out["hod"] = out["high"].cummax()
    return out


def prep_day(row: pd.Series) -> Optional[Day]:
    path = cache_path(row["ticker"], row["date_iso"])
    if not path.exists():
        return None
    bars = pd.read_pickle(path)
    if bars is None or bars.empty or "close" not in bars:
        return None
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("US/Eastern")
    else:
        bars.index = bars.index.tz_convert("US/Eastern")
    day_mask = bars.index.strftime("%Y-%m-%d") == row["date_iso"]
    bars = bars[day_mask]
    pm = bars[bars.index.time < RTH_OPEN]
    rth = bars[(bars.index.time >= RTH_OPEN) & (bars.index.time <= dt_time(16, 0))]
    if len(rth) < 60:
        return None
    open_ = float(rth["open"].iloc[0])
    atr_pct = pd.to_numeric(row.get("atr_pct"), errors="coerce")
    if pd.isna(atr_pct) or atr_pct <= 0 or open_ <= 0:
        return None
    atr = float(atr_pct) * open_
    pm_low = float(pm["low"].min()) if len(pm) else None

    b2 = _two_min(rth)
    if len(b2) < 10:
        return None

    fires: Dict[str, List[pd.Timestamp]] = {d: [] for d in DETECTORS}
    was_above_vwap = False
    was_above_open = False
    vwap_lost = False
    for i in range(1, len(b2)):
        ts = b2.index[i]
        close = b2["close"].iloc[i]
        prior_low = b2["low"].iloc[i - 1]
        vwap = b2["vwap"].iloc[i]
        if close > open_:
            was_above_open = True
        if pd.notna(vwap) and close > vwap:
            was_above_vwap = True
            vwap_lost = False
        bar_end = (pd.Timestamp(ts) + pd.Timedelta(minutes=2)).time()
        if bar_end < FIRST_FIRE:
            continue
        if close < prior_low:
            fires["pbb_down"].append(ts)
        if pd.notna(vwap) and close < vwap and was_above_vwap and not vwap_lost:
            fires["vwap_loss"].append(ts)
            vwap_lost = True
        if close < open_ and was_above_open:
            fires["open_fail"].append(ts)
            was_above_open = False   # re-arm only after a close back above
        if pm_low is not None and close < pm_low:
            fires["pm_low_break"].append(ts)

    # conf2of3: bar where >= 2 of {pbb, vwap_loss, open_fail} fired within 10 min
    core = ["pbb_down", "vwap_loss", "open_fail"]
    all_fires = sorted((ts, d) for d in core for ts in fires[d])
    for ts, _ in all_fires:
        recent = {d for t2, d in all_fires
                  if ts - pd.Timedelta(minutes=CONF_WINDOW_MIN) <= t2 <= ts}
        if len(recent) >= 2:
            fires["conf2of3"].append(ts)
    fires["conf2of3"] = sorted(set(fires["conf2of3"]))

    return Day(
        ticker=row["ticker"], date_iso=row["date_iso"], cohort=row["cohort"],
        setup=str(row.get("setup") or ""), open_=open_, atr=atr, pm_low=pm_low,
        bars2=b2, fires=fires,
        day_low=float(rth["low"].min()), day_close=float(rth["close"].iloc[-1]),
    )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def sim_day(day: Day, detector: str, gate_end: dt_time, stop_buffer: float,
            max_attempts: int, tripwire_exit: bool = False) -> Optional[Dict]:
    """Replay one day through one config. Returns per-day results or None
    if the detector never fired inside the window."""
    fire_list = [ts for ts in day.fires[detector]
                 if FIRST_FIRE <= (ts + pd.Timedelta(minutes=2)).time() <= gate_end]
    if not fire_list:
        return None

    b2 = day.bars2
    attempts = []
    in_pos = False
    entry = stop = risk = None
    fire_iter = iter(fire_list)
    next_fire = next(fire_iter, None)
    n_stopped = 0

    tripwire_ts = _tripwire_time(day) if tripwire_exit else None

    for i in range(len(b2)):
        ts = b2.index[i]
        row = b2.iloc[i]
        if not in_pos:
            if next_fire is None or len(attempts) >= max_attempts:
                break
            if ts == next_fire:
                entry = float(row["close"])
                hod = float(row["hod"])
                stop = hod + stop_buffer * day.atr
                risk = max(stop - entry, RISK_FLOOR_ATR * day.atr)
                in_pos = True
                next_fire = next(fire_iter, None)
            elif ts > next_fire:
                next_fire = next(fire_iter, None)
            continue
        # in position (short)
        if float(row["high"]) >= stop:
            attempts.append({"entry": entry, "exit": stop, "r": (entry - stop) / risk,
                             "stopped": True, "entry_ts": None})
            n_stopped += 1
            in_pos = False
            # skip fires that occurred while in the trade
            while next_fire is not None and next_fire <= ts:
                next_fire = next(fire_iter, None)
            continue
        if tripwire_ts is not None and ts >= tripwire_ts:
            attempts.append({"entry": entry, "exit": float(row["close"]),
                             "r": (entry - float(row["close"])) / risk,
                             "stopped": False, "entry_ts": None})
            in_pos = False
            next_fire = None   # tripwire = day trend over, no re-entry
            continue
        if ts.time() >= EOD_EXIT:
            attempts.append({"entry": entry, "exit": float(row["close"]),
                             "r": (entry - float(row["close"])) / risk,
                             "stopped": False, "entry_ts": None})
            in_pos = False
            break

    if in_pos:   # session ended mid-position (short session) — exit at last close
        last = float(b2["close"].iloc[-1])
        attempts.append({"entry": entry, "exit": last, "r": (entry - last) / risk,
                         "stopped": False, "entry_ts": None})

    if not attempts:
        return None
    first_entry = attempts[0]["entry"]
    rng = day.open_ - day.day_low
    entry_frac = (day.open_ - first_entry) / rng if rng > 0 else np.nan
    return {
        "ticker": day.ticker, "date": day.date_iso, "cohort": day.cohort,
        "setup": day.setup,
        "day_r": sum(a["r"] for a in attempts),
        "attempts": len(attempts), "stopped": n_stopped,
        "entry_frac": entry_frac,
        "first_entry": first_entry,
    }


def _tripwire_time(day: Day) -> Optional[pd.Timestamp]:
    """First 2-min bar after 12:30 whose close retraced >= 25% of the day
    range (running LOD), with the matrix noise floor."""
    b2 = day.bars2
    lod = b2["low"].cummin()
    min_range = max(day.atr, 0.02 * day.open_)
    for i in range(len(b2)):
        ts = b2.index[i]
        if ts.time() < dt_time(12, 30):
            continue
        day_range = day.open_ - float(lod.iloc[i])
        if day_range < min_range:
            continue
        retrace = (float(b2["close"].iloc[i]) - float(lod.iloc[i])) / day_range
        if retrace >= 0.25:
            return ts
    return None


# ---------------------------------------------------------------------------
# Sweep + report
# ---------------------------------------------------------------------------

def _agg(rows: List[Dict]) -> Dict:
    if not rows:
        return {"n": 0}
    r = np.array([x["day_r"] for x in rows], dtype=float)
    stops = sum(x["stopped"] for x in rows)
    atts = sum(x["attempts"] for x in rows)
    ef = np.array([x["entry_frac"] for x in rows], dtype=float)
    return {
        "n": len(rows),
        "mean_r": round(float(r.mean()), 2),
        "med_r": round(float(np.median(r)), 2),
        "pct_pos": round(float((r > 0).mean()), 3),
        "stop_rate": round(stops / atts, 3) if atts else np.nan,
        "att_per_day": round(atts / len(rows), 2),
        "med_entry_frac": round(float(np.nanmedian(ef)), 3),
    }


def main() -> None:
    uni = load_universe()
    days: List[Day] = []
    skipped = 0
    for _, row in uni.iterrows():
        d = prep_day(row)
        if d is None:
            skipped += 1
        else:
            days.append(d)
    curated = [d for d in days if d.cohort == "curated"]
    gapfade = [d for d in curated if d.setup in GAP_FADE_SETUPS]
    controls = [d for d in days if d.cohort == "control"]
    log.info(f"prepared {len(days)} days ({len(curated)} curated / "
             f"{len(gapfade)} gap-fade / {len(controls)} controls), {skipped} skipped")

    results = []
    for det, gate, buf, k in itertools.product(DETECTORS, GATE_ENDS, STOP_BUFFERS, MAX_ATTEMPTS):
        cfg = {"detector": det, "gate_end": gate.strftime("%H:%M"),
               "stop_buffer": buf, "max_attempts": k}
        cur_rows = [r for d in gapfade if (r := sim_day(d, det, gate, buf, k))]
        ctl_rows = [r for d in controls if (r := sim_day(d, det, gate, buf, k))]
        entry = {**cfg}
        entry.update({f"cur_{k2}": v for k2, v in _agg(cur_rows).items()})
        entry.update({f"ctl_{k2}": v for k2, v in _agg(ctl_rows).items()})
        entry["cur_fire_rate"] = round(len(cur_rows) / len(gapfade), 3) if gapfade else np.nan
        entry["ctl_fire_rate"] = round(len(ctl_rows) / len(controls), 3) if controls else np.nan
        results.append(entry)

    df = pd.DataFrame(results)
    # Rank: control mean R is the honest test; curated median breaks ties.
    df = df.sort_values(["ctl_mean_r", "cur_med_r"], ascending=False).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "configs_ranked.csv", index=False)

    top = df.iloc[0]
    print("\n=== TOP 12 CONFIGS (ranked by CONTROL mean R, the honest test) ===")
    cols = ["detector", "gate_end", "stop_buffer", "max_attempts",
            "cur_n", "cur_mean_r", "cur_med_r", "cur_pct_pos", "cur_stop_rate",
            "ctl_n", "ctl_mean_r", "ctl_med_r", "ctl_pct_pos", "ctl_stop_rate",
            "cur_fire_rate", "ctl_fire_rate"]
    print(df[cols].head(12).to_string(index=False))

    # Winner detail + tripwire overlay
    det, gate = top["detector"], dt_time(*map(int, top["gate_end"].split(":")))
    buf, k = float(top["stop_buffer"]), int(top["max_attempts"])
    win_rows = [r for d in gapfade if (r := sim_day(d, det, gate, buf, k))]
    pd.DataFrame(win_rows).sort_values("day_r").to_csv(OUT / "top_config_days.csv", index=False)

    trip_cur = [r for d in gapfade if (r := sim_day(d, det, gate, buf, k, tripwire_exit=True))]
    trip_ctl = [r for d in controls if (r := sim_day(d, det, gate, buf, k, tripwire_exit=True))]
    print("\n=== WINNER with 12:30 tripwire exit overlay (vs hold-to-close) ===")
    print(f"  curated : {_agg(win_rows)}  ->  tripwire {_agg(trip_cur)}")
    base_ctl = [r for d in controls if (r := sim_day(d, det, gate, buf, k))]
    print(f"  control : {_agg(base_ctl)}  ->  tripwire {_agg(trip_ctl)}")

    # All-curated (incl. non-gap-fade setups) slice for the winner
    all_cur = [r for d in curated if (r := sim_day(d, det, gate, buf, k))]
    print(f"\n  winner on ALL curated setups: {_agg(all_cur)}")
    print(f"\nwrote {OUT / 'configs_ranked.csv'} and top_config_days.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
