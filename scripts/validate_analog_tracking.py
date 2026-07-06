"""Validate the analog-tracking premise behind the live analog chart.

Question: do bounce days actually share intraday SHAPE with their kNN
analogs (matched on setup-day features), or does the overlay just look
compelling on a good day?

Method — leave-one-out over the curated bounces in data/bounce_data.csv:
  for each bounce day D:
    * drop D from the reference set, find its top-5 analogs
      (scripts.priority_report.find_historical_comps — the same kNN the
      live pipeline uses);
    * fetch 1-min close paths (9:30-16:00, % from 9:30 open) for D and its
      analogs; average the analogs and take the min-max envelope ("band");
    * measure: path correlation (full day + 9:30-11:00), whether D sat
      inside the band at 11:00, and what happened after band exits.

Outputs a printed report + data/analog_tracking_stats.json consumed by the
priority report (chart caption) IF the splits are real. All minute paths
cache to data/analog_path_cache.pkl so reruns are free.

Usage:
    python -m scripts.validate_analog_tracking
"""
from __future__ import annotations

import datetime
import json
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from data_queries.polygon_queries import get_intraday
from scripts.priority_report import BOUNCE_COMP_COLUMNS, find_historical_comps

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_PATH = _DATA_DIR / "analog_path_cache.pkl"
STATS_PATH = _DATA_DIR / "analog_tracking_stats.json"

_MINUTES = 391          # 9:30..16:00 inclusive
_M_1000 = 30            # 10:00
_M_1100 = 90            # 11:00
_EXIT_CONSEC = 3        # consecutive 1-min closes below band_lo = band exit


# ------------------------------------------------------------------ paths

def _norm_date(raw: str) -> Optional[str]:
    s = str(raw).strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"):
        try:
            return datetime.datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _load_cache() -> Dict:
    if CACHE_PATH.exists():
        try:
            return pickle.loads(CACHE_PATH.read_bytes())
        except Exception:
            return {}
    return {}


def _fetch_path(ticker: str, date: str, cache: Dict) -> Optional[np.ndarray]:
    """1-min close path, % from 9:30 open, on a fixed 391-minute grid."""
    key = (ticker, date)
    if key in cache:
        return cache[key]
    try:
        df = get_intraday(ticker, date, multiplier=1, timespan="minute")
        df = df.between_time("09:30", "16:00")
        if df is None or df.empty:
            cache[key] = None
            return None
        o = float(df["open"].iloc[0])
        s = (df["close"] / o - 1) * 100
        idx = [(t.hour * 60 + t.minute) - (9 * 60 + 30) for t in s.index]
        ser = pd.Series(s.values, index=idx)
        ser = ser[~ser.index.duplicated()]
        arr = ser.reindex(range(_MINUTES)).interpolate(limit_direction="both").to_numpy()
        cache[key] = arr
        return arr
    except Exception:
        cache[key] = None
        return None


# ------------------------------------------------------------------ study

def main() -> None:
    df = pd.read_csv(_DATA_DIR / "bounce_data.csv").dropna(subset=["ticker", "date"])
    cache = _load_cache()

    rows = []
    skipped = 0
    for idx, row in df.iterrows():
        date = _norm_date(row["date"])
        ticker = str(row["ticker"]).strip().upper()
        if date is None:
            skipped += 1
            continue

        day_path = _fetch_path(ticker, date, cache)
        if day_path is None or np.isnan(day_path).all():
            skipped += 1
            continue

        metrics = {c: row.get(c) for c in BOUNCE_COMP_COLUMNS}
        ref = df.drop(index=idx)
        comps = find_historical_comps(metrics, ref, BOUNCE_COMP_COLUMNS,
                                      str(row.get("cap", "")))
        if comps is None or comps.empty:
            skipped += 1
            continue

        analog_paths = []
        for _, c in comps.iterrows():
            cd = _norm_date(c["date"])
            if cd is None:
                continue
            p = _fetch_path(str(c["ticker"]).strip().upper(), cd, cache)
            if p is not None and not np.isnan(p).all():
                analog_paths.append(p)
        if len(analog_paths) < 3:
            skipped += 1
            continue

        mat = np.vstack(analog_paths)
        avg = np.nanmean(mat, axis=0)
        band_lo = np.nanmin(mat, axis=0)

        # correlations
        def _corr(a, b):
            m = ~(np.isnan(a) | np.isnan(b))
            if m.sum() < 30 or np.nanstd(a[m]) == 0 or np.nanstd(b[m]) == 0:
                return np.nan
            return float(np.corrcoef(a[m], b[m])[0, 1])

        corr_full = _corr(day_path, avg)
        corr_am = _corr(day_path[:_M_1100], avg[:_M_1100])

        # in-band at 11:00 (min-max envelope)
        band_hi = np.nanmax(mat, axis=0)
        v11 = day_path[_M_1100]
        in_band_11 = bool(band_lo[_M_1100] <= v11 <= band_hi[_M_1100])

        # band exit after 10:00: three consecutive closes below band_lo
        exit_m = None
        run = 0
        for m in range(_M_1000, _MINUTES):
            if np.isnan(day_path[m]) or np.isnan(band_lo[m]):
                run = 0
                continue
            if day_path[m] < band_lo[m]:
                run += 1
                if run >= _EXIT_CONSEC:
                    exit_m = m - _EXIT_CONSEC + 1
                    break
            else:
                run = 0

        close_pct = day_path[~np.isnan(day_path)][-1]
        rows.append({
            "ticker": ticker, "date": date,
            "corr_full": corr_full, "corr_am": corr_am,
            "in_band_11": in_band_11,
            "close_pct": float(close_pct),
            "green": bool(close_pct > 0),
            "exited_band": exit_m is not None,
            "drift_after_exit": (float(close_pct - day_path[exit_m])
                                 if exit_m is not None else None),
        })

    CACHE_PATH.write_bytes(pickle.dumps(cache))
    res = pd.DataFrame(rows)
    n = len(res)
    print(f"\n=== Analog tracking validation ===")
    print(f"days evaluated: {n} (skipped {skipped} — no data / too few analog paths)")

    print(f"\n--- shape correlation (day vs analog average) ---")
    print(f"full-day corr: median {res['corr_full'].median():.2f} "
          f"(q25 {res['corr_full'].quantile(.25):.2f}, q75 {res['corr_full'].quantile(.75):.2f})")
    print(f"9:30-11:00 corr: median {res['corr_am'].median():.2f}")
    print(f"pct of days with full-day corr > 0.5: {(res['corr_full'] > 0.5).mean():.0%}")

    print(f"\n--- in-band at 11:00 (min-max envelope of 5 analogs) ---")
    for flag, name in ((True, "IN band"), (False, "OUT of band")):
        sub = res[res["in_band_11"] == flag]
        if len(sub):
            print(f"{name:12s}: n={len(sub):3d}  p_green={sub['green'].mean():.0%}  "
                  f"med close {sub['close_pct'].median():+.1f}%")

    print(f"\n--- band exit (3 consecutive 1-min closes below envelope low, after 10:00) ---")
    ex = res[res["exited_band"]]
    print(f"days that exited: {len(ex)}/{n} ({len(ex)/n:.0%})")
    if len(ex):
        print(f"median drift AFTER exit (exit point -> close): "
              f"{ex['drift_after_exit'].median():+.1f}%")
        print(f"p_green given exit: {ex['green'].mean():.0%} "
              f"vs {res[~res['exited_band']]['green'].mean():.0%} for non-exits")

    stats = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "n": n,
        "med_corr_full": round(float(res["corr_full"].median()), 3),
        "med_corr_am": round(float(res["corr_am"].median()), 3),
        "in_band_11_n": int(res["in_band_11"].sum()),
        "in_band_11_p_green": round(float(res[res["in_band_11"]]["green"].mean()), 3)
            if res["in_band_11"].any() else None,
        "in_band_11_med_close": round(float(res[res["in_band_11"]]["close_pct"].median()), 2)
            if res["in_band_11"].any() else None,
        "out_band_11_p_green": round(float(res[~res["in_band_11"]]["green"].mean()), 3)
            if (~res["in_band_11"]).any() else None,
        "out_band_11_med_close": round(float(res[~res["in_band_11"]]["close_pct"].median()), 2)
            if (~res["in_band_11"]).any() else None,
        "exit_rate": round(float(res["exited_band"].mean()), 3),
        "below_band_med_drift": round(float(ex["drift_after_exit"].median()), 2) if len(ex) else None,
        "exit_p_green": round(float(ex["green"].mean()), 3) if len(ex) else None,
        "nonexit_p_green": round(float(res[~res["exited_band"]]["green"].mean()), 3)
            if (~res["exited_band"]).any() else None,
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2))
    print(f"\nWrote {STATS_PATH}")


if __name__ == "__main__":
    main()
