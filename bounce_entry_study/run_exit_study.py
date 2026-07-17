"""Exit-strategy comparison + entry robustness checks.

Entry fixed at the v1 winner (first 2-min close over prior high, 9:30 gate,
re-entry after non-profitable exits). Exits compared on the same days.

    python -m bounce_entry_study.run_exit_study
"""
from __future__ import annotations

import logging

import pandas as pd

from .confluence import confluence_fires
from .exits import ExitSpec, simulate_day_exits
from .run_study import REPORTS, prepare_days

log = logging.getLogger(__name__)

ENTRY = {"dets": ("pbb:h0",), "k": 1, "window": 16, "gate": "09:30"}

EXITS = [
    ExitSpec("close"),
    ExitSpec("trail", arm_atr=0.0),
    ExitSpec("trail", arm_atr=0.5),
    ExitSpec("trail", arm_atr=1.0),
    ExitSpec("trail", arm_atr=1.5),
    ExitSpec("vwap"),
    ExitSpec("atr_target", target_atr=0.5),
    ExitSpec("atr_target", target_atr=1.0),
    ExitSpec("atr_target", target_atr=1.5),
    ExitSpec("atr_target", target_atr=2.0),
    ExitSpec("atr_target", target_atr=3.0),
]


def _day_fires(d):
    return confluence_fires(d.two, {v: d.det_fires[v] for v in ENTRY["dets"]},
                            k=ENTRY["k"], window_min=ENTRY["window"],
                            gate=ENTRY["gate"])


def _atr(d):
    # DayData carries no atr; recompute the same way prepare_days did — from
    # the universe row. Simplest: stash on the frame attrs in prepare_days…
    return d.two.attrs.get("atr")


def run() -> None:
    hist, today = prepare_days()
    # keep only days with a usable ATR so every strategy sees the same days
    days = [d for d in hist if _atr(d)]
    log.info(f"{len(days)}/{len(hist)} historical days with ATR")

    fires_by_day = {id(d): _day_fires(d) for d in days}

    # ---------------- exit grid ----------------
    rows = []
    per_exit_days = {}
    for spec in EXITS:
        results = [(d, simulate_day_exits(d.two, fires_by_day[id(d)], spec, _atr(d)))
                   for d in days]
        rs = pd.Series([r.total_r for _, r in results])
        attempts = sum(len(r.attempts) for _, r in results)
        stopped = sum(r.n_stopped for _, r in results)
        rows.append({
            "exit": spec.name,
            "mean_day_r": round(rs.mean(), 3),
            "median_day_r": round(rs.median(), 3),
            "pct_days_pos": round(100 * (rs > 0).mean(), 1),
            "worst_day_r": round(rs.min(), 2),
            "best_day_r": round(rs.max(), 2),
            "mean_attempts": round(attempts / len(days), 2),
            "lod_stop_rate_pct": round(100 * stopped / attempts, 1) if attempts else None,
        })
        per_exit_days[spec.name] = results
    table = pd.DataFrame(rows).sort_values("mean_day_r", ascending=False)
    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS / "exit_strategies.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== EXIT STRATEGIES (entry: first 2-min break, re-entry on non-profitable exits) ===")
    print(table.to_string(index=False))

    # ---------------- robustness of the entry finding ----------------
    print("\n=== ROBUSTNESS (hold-to-close baseline) ===")
    base = per_exit_days["close"]

    # (a) 0.5-ATR risk floor — kills tiny-stop R inflation
    floored = [(d, simulate_day_exits(d.two, fires_by_day[id(d)], ExitSpec("close"),
                                      _atr(d), risk_floor_atr=0.5)) for d in days]
    rs_f = pd.Series([r.total_r for _, r in floored])
    rs_b = pd.Series([r.total_r for _, r in base])
    print(f"raw risk:            mean {rs_b.mean():+.2f}R  median {rs_b.median():+.2f}R")
    print(f"risk >= 0.5 ATR:     mean {rs_f.mean():+.2f}R  median {rs_f.median():+.2f}R  "
          f"(tiny-stop inflation removed)")

    # (b) excluding IntradayCapitch
    ex = [(d, r) for d, r in base if "IntradayCapitch" not in d.setup]
    rs_e = pd.Series([r.total_r for _, r in ex])
    print(f"excl IntradayCapitch: mean {rs_e.mean():+.2f}R  median {rs_e.median():+.2f}R  "
          f"({len(ex)} days)")

    # (c) by year — stability check
    yr = pd.DataFrame([{"year": d.date_iso[:4], "r": r.total_r} for d, r in base])
    by_year = yr.groupby("year")["r"].agg(["count", "mean", "median"]).round(2)
    print("\nby year (hold-to-close, raw risk):")
    print(by_year.to_string())

    # (d) by setup type
    st = pd.DataFrame([{"setup": d.setup or "?", "r": r.total_r} for d, r in base])
    by_setup = (st.groupby("setup")["r"].agg(["count", "mean", "median"])
                .round(2).sort_values("mean", ascending=False))
    print("\nby setup (hold-to-close, raw risk):")
    print(by_setup.to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
