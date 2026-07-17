"""Grid study: detector configs x curated bounce days -> ranked results.

    python -m bounce_entry_study.run_study
    python -m bounce_entry_study.run_study --top 20

Outputs (reports/bounce_entry_study/):
  * configs_ranked.csv — one row per config, ranked by mean total day R
  * top_config_days.csv — per-day detail for the best config
  * console: top-N table + today's partial-day fires for the best config
"""
from __future__ import annotations

import argparse
import itertools
import logging
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .confluence import DayResult, confluence_fires, simulate_day
from .detectors import DETECTORS, build_2min
from .fetch_bars import cache_path, load_universe

log = logging.getLogger(__name__)

REPORTS = Path(__file__).resolve().parent.parent / "reports" / "bounce_entry_study"

# Detector variants — computed once per day, shared across configs.
VARIANTS: Dict[str, Dict] = {
    "pbb:h0": {"det": "pbb", "held": 0},
    "pbb:h2": {"det": "pbb", "held": 2},
    "hl:k1f0": {"det": "hl", "k": 1, "floor": 0.0},
    "hl:k2f0": {"det": "hl", "k": 2, "floor": 0.0},
    "hl:k2f.1": {"det": "hl", "k": 2, "floor": 0.10},
    "sb:k1": {"det": "sb", "k": 1},
    "sb:k2": {"det": "sb", "k": 2},
    "vwap:m30": {"det": "vwap", "m": 30},
}

# (family variant options per subset slot)
_PBB = ["pbb:h0", "pbb:h2"]
_HL = ["hl:k1f0", "hl:k2f0", "hl:k2f.1"]
_SB = ["sb:k1", "sb:k2"]
_VW = ["vwap:m30"]

WINDOWS = [16, 30]
GATES = ["09:30", "09:45"]


@dataclass(frozen=True)
class Config:
    dets: Tuple[str, ...]
    k: int
    window: int
    gate: str

    @property
    def name(self) -> str:
        return f"{'+'.join(self.dets)} k{self.k} w{self.window} g{self.gate}"


def build_grid() -> List[Config]:
    combos: List[Tuple[Tuple[str, ...], int]] = []
    for v in _PBB:
        combos.append(((v,), 1))
    for v in _HL:
        combos.append(((v,), 1))
    for v in _SB:
        combos.append(((v,), 1))
    for a, b in itertools.product(_PBB, _HL):
        combos.append(((a, b), 2))
    for a, b in itertools.product(_PBB, _SB):
        combos.append(((a, b), 2))
    for a, b in itertools.product(_HL, _SB):
        combos.append(((a, b), 2))
    for a, b, c in itertools.product(_PBB, _HL, _SB):
        combos.append(((a, b, c), 2))
        combos.append(((a, b, c), 3))
    for a, b, c in itertools.product(_PBB, _HL, _SB):
        combos.append(((a, b, c, _VW[0]), 3))
    return [Config(dets, k, w, g)
            for (dets, k) in combos for w in WINDOWS for g in GATES]


# ---------------------------------------------------------------------------
# per-day preparation
# ---------------------------------------------------------------------------

@dataclass
class DayData:
    ticker: str
    date_iso: str
    setup: str
    cap: str
    grade: str
    two: pd.DataFrame
    det_fires: Dict[str, List[pd.Timestamp]]  # variant name -> fire times


def prepare_days() -> Tuple[List[DayData], List[DayData]]:
    """Returns (historical_days, today_days)."""
    uni = load_universe()
    today_iso = _date.today().isoformat()
    hist: List[DayData] = []
    today: List[DayData] = []
    for _, row in uni.iterrows():
        path = cache_path(row["ticker"], row["date_iso"])
        if not path.exists():
            continue
        bars = pd.read_pickle(path)
        two = build_2min(bars)
        if two.empty or len(two) < 10:
            log.warning(f"thin frame, skipping {row['ticker']} {row['date_iso']}")
            continue
        atr = None
        try:
            atr_pct = float(row["atr_pct"])
            if atr_pct > 0:
                atr = atr_pct * float(two["open"].iloc[0])
        except (TypeError, ValueError):
            pass
        fires = {name: DETECTORS[spec["det"]](two, atr, spec)
                 for name, spec in VARIANTS.items()}
        dd = DayData(row["ticker"], row["date_iso"], str(row.get("Setup", "")),
                     str(row.get("cap", "")), str(row.get("trade_grade", "")),
                     two, fires)
        (today if row["date_iso"] == today_iso else hist).append(dd)
    return hist, today


# ---------------------------------------------------------------------------
# study
# ---------------------------------------------------------------------------

def run_config(cfg: Config, days: List[DayData]) -> Tuple[dict, List[Tuple[DayData, DayResult]]]:
    per_day: List[Tuple[DayData, DayResult]] = []
    for d in days:
        fires = confluence_fires(d.two, {v: d.det_fires[v] for v in cfg.dets},
                                 k=cfg.k, window_min=cfg.window, gate=cfg.gate)
        per_day.append((d, simulate_day(d.two, fires)))

    n = len(per_day)
    traded = [r for _, r in per_day if r.attempts]
    survived = [r for _, r in per_day if r.survived]
    total_rs = [r.total_r for _, r in per_day]
    attempts = sum(len(r.attempts) for _, r in per_day)
    stopped = sum(r.n_stopped for _, r in per_day)
    lags = [r.lag_min for _, r in per_day if r.lag_min is not None]
    rs = pd.Series(total_rs)
    row = {
        "config": cfg.name,
        "dets": "+".join(cfg.dets), "k": cfg.k, "window": cfg.window, "gate": cfg.gate,
        "n_days": n,
        "miss_pct": round(100 * (n - len(traded)) / n, 1) if n else None,
        "mean_day_r": round(rs.mean(), 3),
        "median_day_r": round(rs.median(), 3),
        "pct_days_pos": round(100 * (rs > 0).mean(), 1),
        "mean_attempts": round(attempts / max(len(traded), 1), 2),
        "stop_rate_pct": round(100 * stopped / attempts, 1) if attempts else None,
        "survived_pct": round(100 * len(survived) / n, 1) if n else None,
        "median_lag_min": round(pd.Series(lags).median(), 1) if lags else None,
    }
    return row, per_day


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    hist, today = prepare_days()
    log.info(f"prepared {len(hist)} historical days, {len(today)} today rows")

    grid = build_grid()
    log.info(f"grid: {len(grid)} configs")

    rows = []
    best: Optional[Tuple[Config, List]] = None
    for cfg in grid:
        row, per_day = run_config(cfg, hist)
        rows.append(row)
        if best is None or row["mean_day_r"] > best[2]["mean_day_r"]:
            best = (cfg, per_day, row)

    results = pd.DataFrame(rows).sort_values("mean_day_r", ascending=False)
    REPORTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS / "configs_ranked.csv", index=False)

    pd.set_option("display.width", 250)
    print("\n=== TOP CONFIGS (ranked by mean total day R) ===")
    print(results.head(args.top).to_string(index=False))

    # ---- per-day detail for the best config ----
    cfg, per_day, _ = best
    detail = []
    for d, r in per_day:
        detail.append({
            "date": d.date_iso, "ticker": d.ticker, "setup": d.setup,
            "grade": d.grade, "cap": d.cap,
            "attempts": len(r.attempts), "stopped": r.n_stopped,
            "total_r": round(r.total_r, 2), "survived": r.survived,
            "first_entry": r.first_entry, "surviving_entry": r.surviving_entry,
            "lag_min": r.lag_min,
        })
    dd = pd.DataFrame(detail).sort_values("date")
    dd.to_csv(REPORTS / "top_config_days.csv", index=False)
    print(f"\nbest config: {cfg.name}")
    print(f"per-day detail -> {REPORTS / 'top_config_days.csv'}")

    # ---- today (partial day) demo with the best config ----
    if today:
        print(f"\n=== TODAY (partial day) — {cfg.name} ===")
        for d in today:
            fires = confluence_fires(d.two, {v: d.det_fires[v] for v in cfg.dets},
                                     k=cfg.k, window_min=cfg.window, gate=cfg.gate)
            r = simulate_day(d.two, fires)
            f_str = ", ".join(t.strftime("%H:%M") for t in fires) or "none yet"
            a_str = "; ".join(
                f"{a.entry_time.strftime('%H:%M')} @ {a.entry:.2f} stop {a.stop:.2f} "
                f"{'STOPPED' if a.stopped else f'open/last {a.exit:.2f}'} ({a.r:+.1f}R)"
                for a in r.attempts) or "no entry"
            print(f"  {d.ticker}: confluence fires [{f_str}] -> {a_str}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
