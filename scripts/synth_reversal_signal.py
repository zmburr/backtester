"""Synthesize a priority-signal JSON for a curated reversal day — playback fuel.

The morning watcher's reversal context tooling (odds card, analog chart,
matrix covering, checkpoints) consumes fields the priority report only ships
going forward. This builds a signal file for a HISTORICAL reversal day from
curated reversal_data.csv so `--playback` can exercise the full pipeline:

    python -m scripts.synth_reversal_signal EWY 2026-02-26

Writes data/priority_signals/<date>_morning.json (refuses to overwrite a
real report file unless it was itself synthetic). The target day's own row
is EXCLUDED from analogs/intensity — no leakage into the replay.

Field provenance:
  metrics       — from the curated CSV row (pct_from_9ema proxied by
                  pct_from_10mav; the CSV doesn't store the 9-EMA distance)
  score/rec     — scored against reversal_stats.json per-cap thresholds
                  (>=4 GO, 3 CAUTION, else NO-GO)
  odds          — reversal_odds.json bucket for (score band, cap)
  analogs       — z-scored kNN over the curated book (excluding the day)
  setup_type    — the CSV's own setup label; setup_stats from reversal_stats
  intensity     — mean percentile of gap/3d-run/ATR/range vs the curated book
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SIGNAL_DIR = DATA / "priority_signals"

KNN_COLS = ["gap_pct", "pct_change_3", "one_day_before_range_pct", "atr_pct"]
INTENSITY_COLS = ["gap_pct", "pct_change_3", "atr_pct", "one_day_before_range_pct"]


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _score(metrics: dict, cap: str, stats: dict) -> tuple:
    """Score the 5 pre-trade criteria against reversal_stats per-cap thresholds."""
    th = (stats or {}).get("thresholds", {}).get(cap)
    if not th:
        return 3, "CAUTION", []  # no thresholds — neutral fixture default
    checks = [
        ("9EMA distance", metrics.get("pct_from_9ema"), th.get("pct_from_9ema")),
        ("range/ATR", metrics.get("prior_day_range_atr"), th.get("prior_day_range_atr")),
        ("RVOL", metrics.get("prior_day_rvol"), th.get("rvol_score")),
        ("3d run", metrics.get("pct_change_3"), th.get("pct_change_3")),
        ("gap", metrics.get("gap_pct"), th.get("gap_pct")),
    ]
    results = [(name, v is not None and t is not None and v >= t) for name, v, t in checks]
    score = sum(1 for _, ok in results if ok)
    rec = "GO" if score >= 4 else ("CAUTION" if score == 3 else "NO-GO")
    return score, rec, results


def _lookup_odds(odds: dict, score: int, cap: str, setup_type: str = None):
    if not odds:
        return None
    # Prefer setup-conditioned buckets — the typed gate discriminates far
    # better than the generic score (same order as priority_report).
    if setup_type:
        setup_only = None
        for b in odds.get("setup_buckets", []):
            if b.get("setup") != setup_type:
                continue
            if b.get("cap") == cap:
                out = {k: b.get(k) for k in ("n", "p_fade", "p_fade5", "p_close_low", "med_return")}
                out["base_p_fade5"] = (odds.get("base") or {}).get("p_fade5")
                out["bucket"] = f"{setup_type} · {cap}"
                return out
            if b.get("cap") is None:
                setup_only = b
        if setup_only is not None:
            out = {k: setup_only.get(k) for k in ("n", "p_fade", "p_fade5", "p_close_low", "med_return")}
            out["base_p_fade5"] = (odds.get("base") or {}).get("p_fade5")
            out["bucket"] = f"{setup_type} · all caps"
            return out
    best = None
    for b in odds.get("buckets", []):
        if not (b.get("score_min", 0) <= score <= b.get("score_max", 99)):
            continue
        if b.get("cap") == cap:
            best = b
            break
        if b.get("cap") is None and best is None:
            best = b
    src = best or odds.get("base")
    if not src:
        return None
    out = {k: src.get(k) for k in ("n", "p_fade", "p_fade5", "p_close_low", "med_return")}
    base = odds.get("base") or {}
    if base.get("p_fade5") is not None:
        out["base_p_fade5"] = base["p_fade5"]
    out["bucket"] = (f"score {best.get('score_min')}-{best.get('score_max')} · {best.get('cap') or 'all caps'}"
                     if best else "all screened days")
    return out


def _knn_analogs(df: pd.DataFrame, row: pd.Series, k: int = 5):
    pool = df.drop(row.name)
    feats = pool[KNN_COLS].apply(pd.to_numeric, errors="coerce")
    target = pd.to_numeric(row[KNN_COLS], errors="coerce")
    mu, sd = feats.mean(), feats.std().replace(0, 1.0)
    z = ((feats - mu) / sd).fillna(0.0)
    zt = ((target - mu) / sd).fillna(0.0)
    dist = np.sqrt(((z - zt) ** 2).sum(axis=1))
    top = pool.assign(_distance=dist).nsmallest(k, "_distance")
    comps = []
    for _, r in top.iterrows():
        d = pd.to_datetime(r["date"]).strftime("%Y-%m-%d")
        comps.append({
            "ticker": r["ticker"], "date": d,
            "open_low_pct": _num(r.get("reversal_open_low_pct")),
            "open_close_pct": _num(r.get("reversal_open_close_pct")),
            "distance": round(float(r["_distance"]), 4),
        })
    return {
        "comps": comps, "n": len(comps),
        "med_open_low": _num(pd.to_numeric(top["reversal_open_low_pct"], errors="coerce").median()),
        "med_open_close": _num(pd.to_numeric(top["reversal_open_close_pct"], errors="coerce").median()),
    }


def _intensity(df: pd.DataFrame, row: pd.Series) -> float:
    pool = df.drop(row.name)
    pcts = []
    for col in INTENSITY_COLS:
        v = pd.to_numeric(row.get(col), errors="coerce")
        ref = pd.to_numeric(pool[col], errors="coerce").dropna()
        if pd.isna(v) or ref.empty:
            continue
        pcts.append((ref < v).mean())
    return round(100.0 * float(np.mean(pcts)), 1) if pcts else 50.0


def _num(v):
    try:
        f = float(v)
        return None if f != f else round(f, 6)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ticker")
    ap.add_argument("date", help="YYYY-MM-DD (must exist in reversal_data.csv)")
    args = ap.parse_args()
    ticker = args.ticker.upper()
    day = dt.datetime.strptime(args.date, "%Y-%m-%d").date()

    df = pd.read_csv(DATA / "reversal_data.csv")
    dates = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    mask = (df["ticker"].str.upper() == ticker) & (dates.dt.date == day)
    if not mask.any():
        print(f"ERROR: {ticker} {day} not found in reversal_data.csv")
        return 1
    row = df[mask].iloc[0]

    cap = str(row.get("cap") or "Medium")
    setup_type = str(row.get("setup") or "") or None
    metrics = {
        "gap_pct": _num(row.get("gap_pct")),
        "atr_pct": _num(row.get("atr_pct")),
        "pct_change_3": _num(row.get("pct_change_3")),
        # CSV has no 9-EMA distance; 10-day MAV distance is the closest proxy.
        "pct_from_9ema": _num(row.get("pct_from_10mav")),
        "prior_day_range_atr": _num(row.get("one_day_before_range_pct")),
        "prior_day_rvol": (_num(row.get("vol_one_day_before")) or 0)
                          / max(_num(row.get("avg_daily_vol")) or 1, 1),
    }
    metrics["prior_day_rvol"] = round(metrics["prior_day_rvol"], 4)

    stats_file = _load_json(DATA / "reversal_stats.json")
    score, rec, checks = _score(metrics, cap, stats_file)
    odds = _lookup_odds(_load_json(DATA / "reversal_odds.json") or {}, score, cap,
                        setup_type=setup_type)
    analogs = _knn_analogs(df, row)
    intensity = _intensity(df, row)

    setup_stats = None
    if setup_type and stats_file:
        block = stats_file.get(setup_type.lower())
        if isinstance(block, dict):
            ci = block.get("ab_win_rate_ci") or {}
            if ci.get("point") is not None:
                setup_stats = {
                    "setup": setup_type,
                    "n": ci.get("n") or block.get("ab_count"),
                    "win_rate": ci["point"] / 100.0,
                    "ci_lo": (ci.get("ci_lower") or 0) / 100.0,
                    "ci_hi": (ci.get("ci_upper") or 0) / 100.0,
                    "avg_pnl": (block.get("ab_avg_pnl") or 0) / 100.0,
                }

    signal = {
        "ticker": ticker, "bucket": "reversal", "cap": cap,
        "recommendation": rec, "score": f"{score}/5",
        "metrics": metrics,
        "intensity": intensity,
        "archetype_passed": True,
        "archetype_detail": f"curated {row.get('trade_grade')}-grade {setup_type or 'reversal'}",
    }
    if odds:
        signal["odds"] = odds
    if analogs:
        signal["analogs"] = analogs
    if setup_type:
        signal["setup_type"] = setup_type
        signal["setup_match"] = f"{setup_type} (curated book)"
    if setup_stats:
        signal["setup_stats"] = setup_stats

    payload = {
        "date": day.strftime("%Y-%m-%d"), "session": "morning",
        "generated_at": dt.datetime.now().isoformat(),
        "_synthetic": True,
        "go_count": 1 if rec == "GO" else 0,
        "caution_count": 1 if rec == "CAUTION" else 0,
        "signals": [signal],
    }

    out = SIGNAL_DIR / f"{day.strftime('%Y-%m-%d')}_morning.json"
    if out.exists() and not (_load_json(out) or {}).get("_synthetic"):
        print(f"ERROR: {out} exists and is a REAL report file — refusing to overwrite")
        return 1
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(f"wrote {out}")
    print(f"  {ticker} {rec} {score}/5 ({', '.join(n for n, ok in checks if ok)} passed)")
    print(f"  intensity {intensity} · setup {setup_type} · odds p_fade5 "
          f"{(odds or {}).get('p_fade5')} vs base {(odds or {}).get('base_p_fade5')}")
    print(f"  analogs: {[(c['ticker'], c['date']) for c in analogs['comps']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
