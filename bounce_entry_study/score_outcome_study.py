"""Study #1: does the bounce-intensity score at the low predict the day's R?

For every curated bounce day, rebuild the LIVE scorer's inputs as of that
morning (daily history strictly BEFORE the day), compute the score at the
open and at the session low (leave-one-out vs the reference book), and
regress against what the validated entry system actually made that day.

Also writes the score-at-low distribution to data/bounce_score_dist.json —
the live watcher uses it to render "74/100 · p82" context.

    python -m bounce_entry_study.score_outcome_study
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from data_queries.polygon_queries import get_levels_data

from .confluence import confluence_fires
from .exits import ExitSpec, simulate_day_exits
from .run_study import REPORTS, prepare_days

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DAILY_CACHE = ROOT / "data" / "bounce_entry_study" / "daily"
DIST_OUT = ROOT / "data" / "bounce_score_dist.json"

ENTRY = {"dets": ("pbb:h0",), "k": 1, "window": 16, "gate": "09:30"}

SPEC = [
    ("pct_change_3",      0.30),
    ("pct_change_15",     0.20),
    ("selloff_total_pct", 0.15),
    ("gap_pct",           0.15),
    ("pct_off_30d_high",  0.15),
    ("pct_off_52wk_high", 0.05),
]


def _pctrank(arr: np.ndarray, s: float) -> float:
    return 100.0 * (np.sum(arr < s) + 0.5 * np.sum(arr == s)) / arr.size


def _daily_before(ticker: str, date_iso: str) -> Optional[pd.DataFrame]:
    """Daily bars strictly BEFORE date_iso, cached."""
    DAILY_CACHE.mkdir(parents=True, exist_ok=True)
    path = DAILY_CACHE / f"{ticker}_{date_iso}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    try:
        df = get_levels_data(ticker, date_iso, 550, 1, "day")
    except Exception as e:
        log.warning(f"daily fetch failed {ticker} {date_iso}: {e}")
        return None
    if df is None or df.empty:
        return None
    df = df[df.index.date < pd.Timestamp(date_iso).date()]
    df.to_pickle(path)
    return df


def _refs(daily: pd.DataFrame) -> Optional[Dict[str, float]]:
    if len(daily) < 20:
        return None
    closes = daily["close"].astype(float)
    highs = daily["high"].astype(float)
    run = 0
    for i in range(len(daily) - 1, 0, -1):
        if closes.iloc[i] < closes.iloc[i - 1]:
            run += 1
        else:
            break
    anchor_i = max(0, len(daily) - 1 - run)
    try:
        return {
            "c3": float(closes.iloc[-4]),
            "c15": float(closes.iloc[-16]) if len(closes) >= 16 else float(closes.iloc[0]),
            "h30": float(highs.tail(30).max()),
            "h52": float(highs.tail(252).max()),
            "anchor": float(highs.iloc[anchor_i]),
            "prior_close": float(closes.iloc[-1]),
        }
    except (IndexError, ValueError):
        return None


def _metrics(price: float, today_open: float, r: Dict[str, float]) -> Dict[str, float]:
    return {
        "pct_change_3": price / r["c3"] - 1,
        "pct_change_15": price / r["c15"] - 1,
        "selloff_total_pct": price / r["anchor"] - 1,
        "gap_pct": today_open / r["prior_close"] - 1,
        "pct_off_30d_high": price / r["h30"] - 1,
        "pct_off_52wk_high": price / r["h52"] - 1,
    }


def _score(metrics: Dict[str, float], ref_arrays: Dict[str, np.ndarray]) -> Optional[float]:
    w = t = 0.0
    for col, weight in SPEC:
        arr = ref_arrays.get(col)
        v = metrics.get(col)
        if arr is None or arr.size == 0 or v is None or v != v:
            continue
        w += (100.0 - _pctrank(arr, v)) * weight
        t += weight
    return round(w / t, 1) if t > 0 else None


def run() -> None:
    # Reference book (excl IntradayCapitch), keyed for leave-one-out drops.
    book = pd.read_csv(ROOT / "data" / "bounce_data.csv")
    book = book[~book["Setup"].str.contains("IntradayCapitch", case=False, na=False)]
    book["date_iso"] = pd.to_datetime(book["date"], format="%m/%d/%Y").dt.strftime("%Y-%m-%d")

    hist, _today = prepare_days()
    rows = []
    for d in hist:
        daily = _daily_before(d.ticker, d.date_iso)
        if daily is None:
            continue
        refs = _refs(daily)
        if refs is None:
            continue
        two = d.two
        today_open = float(two["open"].iloc[0])
        day_low = float(two["low"].min())
        day_close = float(two["close"].iloc[-1])

        # leave-one-out reference arrays
        loo = book[~((book["ticker"].str.upper() == d.ticker.upper())
                     & (book["date_iso"] == d.date_iso))]
        ref_arrays = {col: loo[col].dropna().to_numpy(dtype=float)
                      for col, _ in SPEC if col in loo.columns}

        s_open = _score(_metrics(today_open, today_open, refs), ref_arrays)
        s_low = _score(_metrics(day_low, today_open, refs), ref_arrays)

        fires = confluence_fires(two, {v: d.det_fires[v] for v in ENTRY["dets"]},
                                 k=ENTRY["k"], window_min=ENTRY["window"],
                                 gate=ENTRY["gate"])
        res = simulate_day_exits(two, fires, ExitSpec("close"), two.attrs.get("atr"))

        rows.append({
            "date": d.date_iso, "ticker": d.ticker, "setup": d.setup,
            "grade": d.grade, "cap": d.cap,
            "score_open": s_open, "score_low": s_low,
            "total_r": round(res.total_r, 2),
            "low_to_close_pct": round((day_close / day_low - 1) * 100, 2),
        })

    df = pd.DataFrame(rows).dropna(subset=["score_low"])
    REPORTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS / "score_outcome.csv", index=False)
    print(f"\n{len(df)} days scored (see {REPORTS / 'score_outcome.csv'})")

    rho = df["score_low"].corr(df["total_r"], method="spearman")
    rho_o = df["score_open"].corr(df["total_r"], method="spearman")
    print(f"\nSpearman rho — score_at_low vs day R:  {rho:+.3f}")
    print(f"Spearman rho — score_at_open vs day R: {rho_o:+.3f}")

    df["bucket"] = pd.cut(df["score_low"], bins=[0, 40, 55, 70, 100],
                          labels=["<40", "40-55", "55-70", ">=70"])
    tbl = df.groupby("bucket", observed=True).agg(
        n=("total_r", "size"),
        mean_r=("total_r", "mean"),
        median_r=("total_r", "median"),
        pct_ge_2r=("total_r", lambda s: 100 * (s >= 2).mean()),
        mean_low_to_close=("low_to_close_pct", "mean"),
    ).round(2)
    print("\nscore-at-low buckets:")
    print(tbl.to_string())

    weak = df[df["setup"].str.contains("weakstock", case=False, na=False)]
    strong = df[~df["setup"].str.contains("weakstock", case=False, na=False)]
    for name, sub in (("weakstock", weak), ("strongstock/other", strong)):
        if len(sub) > 5:
            print(f"  {name}: n={len(sub)}  rho={sub['score_low'].corr(sub['total_r'], method='spearman'):+.3f}")

    # distribution artifact for the live watcher's percentile context
    dist = sorted(float(x) for x in df["score_low"])
    DIST_OUT.write_text(json.dumps({
        "built": "2026-07-17", "n": len(dist), "kind": "score_at_low",
        "scores": dist,
    }))
    print(f"\nscore-at-low distribution -> {DIST_OUT}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
