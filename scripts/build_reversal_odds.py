"""Build the empirical reversal-odds table from the UNCONDITIONED reversal universe.

The curated reversal_data.csv is a winners-only set (selected by hand, ~100%
"reversed" by construction) — useless as a denominator for "what are the odds a
gap-up extended stock actually fades today?". This script instead reads the
reversal *universe* CSVs produced by scan_reversal_universe.py, whose loose
pre-filters (pct_from_9ema>0.04, gap_pct>0, price>$5, ADV>500K) deliberately
keep the non-reversers in the population:

    python -m scripts.scan_reversal_universe --start 2020-01-01 --end 2025-12-31 \
        -o data/reversal_universe_2020-01-01_2025-12-31.csv

and writes data/reversal_odds.json: P(fade / short worked), P(fade >= 5%), and
P(closed near lows) by score band x cap, with small buckets collapsed into the
score band. The priority report attaches the matching bucket to each reversal
signal so the morning watcher can show "days like this faded N% of the time
(n=...)".

This mirrors build_bounce_odds.py (same structure, bands, collapse rule and
JSON envelope) but for the short side. The success direction is inverted:
fade_day_return < 0 means the short worked.

Usage:
    python -m scripts.build_reversal_odds
    python -m scripts.build_reversal_odds --population data/one.csv data/two.csv
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# Default population = every reversal_universe_*.csv in data/ (2020-2025 + 2026 YTD).
DEFAULT_POPULATION_GLOB = str(_DATA_DIR / "reversal_universe_*.csv")
OUTPUT_PATH = _DATA_DIR / "reversal_odds.json"

# Score bands (inclusive). The reversal pre-trade screen scores 0-6.
SCORE_BANDS = [(0, 2), (3, 3), (4, 4), (5, 6)]

# A (band, cap) bucket below this n collapses into the band-only bucket.
MIN_BUCKET_N = 50

# Success labels (short side — negative fade_day_return = the short worked).
FADE_THRESHOLD = -0.05        # fade_day_return <= -5% (a real fade)
CLOSE_LOW_THRESHOLD = 0.25    # fade_day_close_position <= 0.25 (closed near lows)

# Columns an outcome row must have to be counted.
_REQUIRED = ["score", "fade_day_return", "fade_day_close_position"]


def _bucket_stats(df: pd.DataFrame) -> dict:
    return {
        "n": int(len(df)),
        "p_fade": round(float((df["fade_day_return"] < 0).mean()), 4),
        "p_fade5": round(float((df["fade_day_return"] <= FADE_THRESHOLD).mean()), 4),
        "p_close_low": round(float((df["fade_day_close_position"] <= CLOSE_LOW_THRESHOLD).mean()), 4),
        "med_return": round(float(df["fade_day_return"].median()), 4),
    }


def load_population(paths: list[Path]) -> pd.DataFrame:
    """Read + concat one or more universe CSVs, dedup on (date, ticker) keep first."""
    frames = []
    for p in paths:
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    before = len(df)
    df = df.drop_duplicates(subset=["date", "ticker"], keep="first").reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        print(f"  deduped {removed} overlapping (date,ticker) rows across {len(paths)} file(s)")
    return df


def build(paths: list[Path]) -> dict:
    df = load_population(paths)
    # Coerce score to numeric first so dirty values become NaN and drop with the
    # dropna below, instead of crashing the astype(int).
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=_REQUIRED)
    df["score"] = df["score"].astype(int)

    buckets = []
    for lo, hi in SCORE_BANDS:
        band = df[(df["score"] >= lo) & (df["score"] <= hi)]
        if band.empty:
            continue
        band_bucket = {"score_min": lo, "score_max": hi, "cap": None, **_bucket_stats(band)}
        collapsed = []
        for cap, cap_df in band.groupby("cap"):
            if len(cap_df) >= MIN_BUCKET_N:
                buckets.append({"score_min": lo, "score_max": hi, "cap": str(cap),
                                **_bucket_stats(cap_df)})
            else:
                collapsed.append(f"{cap}(n={len(cap_df)})")
        if collapsed:
            print(f"  band {lo}-{hi}: collapsed thin cap buckets into band: {', '.join(collapsed)}")
        buckets.append(band_bucket)

    return {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "population_files": [p.name for p in paths],
        "n_total": int(len(df)),
        "fade_threshold": FADE_THRESHOLD,
        "close_low_threshold": CLOSE_LOW_THRESHOLD,
        "base": _bucket_stats(df),
        "buckets": buckets,
    }


def sanity_table(paths: list[Path]) -> None:
    """GO/CAUTION/NO-GO separation on the UNCONDITIONED universe — the real test
    of whether the pre-trade score discriminates fade outcomes. Also reports the
    score-band monotonicity of p_fade and the curated-vs-population selection gap."""
    df = load_population(paths).dropna(subset=["fade_day_return", "fade_day_close_position"])

    def _agg(g):
        return g.agg(
            n=("fade_day_return", "size"),
            p_fade=("fade_day_return", lambda s: (s < 0).mean()),
            p_fade5=("fade_day_return", lambda s: (s <= FADE_THRESHOLD).mean()),
            p_close_low=("fade_day_close_position", lambda s: (s <= CLOSE_LOW_THRESHOLD).mean()),
            med_return=("fade_day_return", "median"),
        ).round(3)

    print("\n=== Sanity: recommendation vs fade outcome (unconditioned universe) ===")
    print(_agg(df.groupby("recommendation")).to_string())

    print("\n=== By score band ===")
    banded = df.dropna(subset=["score"]).copy()
    banded["score"] = banded["score"].astype(int)

    def _band(s):
        for lo, hi in SCORE_BANDS:
            if lo <= s <= hi:
                return f"{lo}-{hi}"
        return "other"

    banded["band"] = banded["score"].map(_band)
    band_tbl = _agg(banded.groupby("band"))
    print(band_tbl.to_string())

    # Monotonicity: does p_fade rise as the score band rises?
    print("\n=== Monotonicity (p_fade across score bands, low -> high) ===")
    ordered = [f"{lo}-{hi}" for lo, hi in SCORE_BANDS if f"{lo}-{hi}" in band_tbl.index]
    p_fades = [(b, float(band_tbl.loc[b, "p_fade"])) for b in ordered]
    seq = " <= ".join(f"{b}:{v:.3f}" for b, v in p_fades)
    strictly_increasing = all(p_fades[i][1] <= p_fades[i + 1][1] for i in range(len(p_fades) - 1))
    print(f"  {seq}")
    print(f"  monotonic non-decreasing: {strictly_increasing}")

    # Curated selection gap: in_reversal_csv==True (hand-picked winners) vs full pop.
    print("\n=== Curated (in_reversal_csv==True) vs full population ===")
    if "in_reversal_csv" in df.columns:
        flag = df["in_reversal_csv"].astype(str).str.lower().isin({"true", "1"})
        curated = df[flag]
        gap = pd.DataFrame({
            "full_population": pd.Series({
                "n": len(df),
                "p_fade": round((df["fade_day_return"] < 0).mean(), 3),
                "p_fade5": round((df["fade_day_return"] <= FADE_THRESHOLD).mean(), 3),
                "p_close_low": round((df["fade_day_close_position"] <= CLOSE_LOW_THRESHOLD).mean(), 3),
                "med_return": round(df["fade_day_return"].median(), 3),
            }),
            "curated_only": pd.Series({
                "n": len(curated),
                "p_fade": round((curated["fade_day_return"] < 0).mean(), 3) if len(curated) else float("nan"),
                "p_fade5": round((curated["fade_day_return"] <= FADE_THRESHOLD).mean(), 3) if len(curated) else float("nan"),
                "p_close_low": round((curated["fade_day_close_position"] <= CLOSE_LOW_THRESHOLD).mean(), 3) if len(curated) else float("nan"),
                "med_return": round(curated["fade_day_return"].median(), 3) if len(curated) else float("nan"),
            }),
        })
        print(gap.to_string())
        if len(curated):
            lift = (curated["fade_day_return"] < 0).mean() - (df["fade_day_return"] < 0).mean()
            print(f"  curation lift in p_fade: {lift:+.3f} "
                  f"(hand-picked set fades {lift * 100:+.1f}pp more often than the raw universe)")
    else:
        print("  (in_reversal_csv column not present)")


def _resolve_populations(args_population) -> list[Path]:
    if args_population:
        return [Path(p) for p in args_population]
    matches = sorted(glob.glob(DEFAULT_POPULATION_GLOB))
    return [Path(m) for m in matches]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reversal_odds.json from unconditioned universe")
    parser.add_argument("--population", nargs="+", default=None,
                        help="One or more universe CSVs (default: data/reversal_universe_*.csv)")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    paths = _resolve_populations(args.population)
    if not paths:
        raise SystemExit(f"no population files found matching {DEFAULT_POPULATION_GLOB} — "
                         "run scripts.scan_reversal_universe first (see module docstring)")
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"population file(s) not found: {', '.join(str(m) for m in missing)}")

    print(f"Building reversal odds from {len(paths)} file(s): {', '.join(p.name for p in paths)} ...")
    payload = build(paths)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.output} (n_total={payload['n_total']}, "
          f"base p_fade={payload['base']['p_fade']:.1%})")

    sanity_table(paths)


if __name__ == "__main__":
    main()
