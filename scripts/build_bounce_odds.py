"""Build the empirical bounce-odds table from the UNCONDITIONED backscanner population.

The archived bounce_backscanner_*.csv files were generated with --min-bounce 2,
i.e. every row already bounced >=2% low-to-close — useless as a denominator.
This script expects a population produced with:

    python -m scanners.bounce_backscanner --start 2022-01-01 --end 2026-06-30 \
        --min-bounce 0 --min-score 0 --cap Large,ETF,Medium,Small \
        -o data/bounce_population_2022_2026.csv

and writes data/bounce_odds.json: P(green close from open) and
P(low-to-close >= 5%) by score band x cap, with small buckets collapsed
into the score band. The priority report attaches the matching bucket to
each bounce signal so the morning watcher can show "days like this were
green N% of the time (n=...)".

Usage:
    python -m scripts.build_bounce_odds
    python -m scripts.build_bounce_odds --population data/other.csv
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_POPULATION = _DATA_DIR / "bounce_population_2022_2026.csv"
OUTPUT_PATH = _DATA_DIR / "bounce_odds.json"

# Score bands (inclusive). The pre-trade screen scores 0-6.
SCORE_BANDS = [(0, 2), (3, 3), (4, 4), (5, 6)]

# A (band, cap) bucket below this n collapses into the band-only bucket.
MIN_BUCKET_N = 50

# Success labels
BOUNCE_LTC_THRESHOLD = 0.05  # low-to-close >= 5%


def _bucket_stats(df: pd.DataFrame) -> dict:
    return {
        "n": int(len(df)),
        "p_green": round(float((df["bounce_day_return"] > 0).mean()), 4),
        "p_bounce5": round(float((df["bounce_low_to_close"] >= BOUNCE_LTC_THRESHOLD).mean()), 4),
        "med_return": round(float(df["bounce_day_return"].median()), 4),
    }


def build(population_path: Path) -> dict:
    df = pd.read_csv(population_path)
    df = df.dropna(subset=["score", "bounce_day_return", "bounce_low_to_close"])
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
        "population_file": population_path.name,
        "n_total": int(len(df)),
        "bounce_ltc_threshold": BOUNCE_LTC_THRESHOLD,
        "base": _bucket_stats(df),
        "buckets": buckets,
    }


def sanity_table(population_path: Path) -> None:
    """GO/CAUTION/NO-GO separation on the UNCONDITIONED population — the real
    test of whether the pre-trade score discriminates outcomes."""
    df = pd.read_csv(population_path).dropna(subset=["bounce_day_return"])
    print("\n=== Sanity: recommendation vs outcome (unconditioned population) ===")
    g = df.groupby("recommendation").agg(
        n=("bounce_day_return", "size"),
        p_green=("bounce_day_return", lambda s: (s > 0).mean()),
        med_return=("bounce_day_return", "median"),
        p_bounce5=("bounce_low_to_close", lambda s: (s >= BOUNCE_LTC_THRESHOLD).mean()),
    ).round(3)
    print(g.to_string())
    print("\n=== By score ===")
    g2 = df.dropna(subset=["score"]).astype({"score": int}).groupby("score").agg(
        n=("bounce_day_return", "size"),
        p_green=("bounce_day_return", lambda s: (s > 0).mean()),
        med_return=("bounce_day_return", "median"),
        p_bounce5=("bounce_low_to_close", lambda s: (s >= BOUNCE_LTC_THRESHOLD).mean()),
    ).round(3)
    print(g2.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bounce_odds.json from unconditioned population")
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    if not args.population.exists():
        raise SystemExit(f"population file not found: {args.population} — run the backscanner first "
                         "(see module docstring)")

    print(f"Building odds table from {args.population.name} ...")
    payload = build(args.population)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.output} (n_total={payload['n_total']}, "
          f"base p_green={payload['base']['p_green']:.1%})")

    sanity_table(args.population)


if __name__ == "__main__":
    main()
