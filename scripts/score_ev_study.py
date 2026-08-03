"""Expected value per score tier, measured through the trader's own rules.

The score's job is not pass/fail — it is "how hard do I attack this day, and what
is my EV". Answering that needs realised R by score, not a hit-rate on an MFE
gate measured from the open.

Method: take the universe-wide screen (data/bounce_population_2022_2026.csv,
218k ticker-days, already on disk — no rescan), replay the live 2-min
prior-bar-break entry and the +1.5-ATR-armed trailing exit over each day's
intraday bars, and tabulate realised R by score tier.

The replay is the one validated in scripts/backfill_entry_r.py: on the curated
bounce book it returns +2.98R/day mean / 85% win against the study's published
+3.2R/day raw.

Scored cohorts:
  * every GO/CAUTION day in the screen (1,321) — what the scanner would have you attack
  * the curated book's trades that the screen rated NO-GO — the detection-failure set
  * a NO-GO control sample — the baseline the score is supposed to beat

Usage:
    python -m scripts.score_ev_study --dry-run
    python -m scripts.score_ev_study --nogo-sample 600
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bounce_entry_study.detectors import build_2min  # noqa: E402
from bounce_entry_study.fetch_bars import fetch_day  # noqa: E402
from scripts.backfill_entry_r import replay_entry_day  # noqa: E402

POP = PROJECT_ROOT / "data" / "bounce_population_2022_2026.csv"
BOOK = PROJECT_ROOT / "data" / "bounce_data.csv"
OUT = PROJECT_ROOT / "data" / "score_ev_study.csv"

CARRY = ["date", "ticker", "cap", "setup_type", "score", "recommendation",
         "selloff_total_pct", "pct_off_30d_high", "gap_pct", "prior_day_range_atr",
         "prior_day_rvol", "pct_change_3", "pct_off_52wk_high", "atr_pct",
         "bounce_low_to_close"]


def load_book_keys() -> set:
    book = pd.read_csv(BOOK)
    iso = pd.to_datetime(book["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return {(d, str(t).strip()) for d, t in zip(iso, book["ticker"]) if isinstance(d, str)}


def select_rows(nogo_sample: int, seed: int = 7) -> list[dict]:
    """GO/CAUTION days + curated-but-NO-GO days + a random NO-GO control."""
    book_keys = load_book_keys()
    picked, nogo_pool = [], []
    with open(POP, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["date"], row["ticker"])
            rec = row["recommendation"]
            row["in_book"] = key in book_keys
            if rec in ("GO", "CAUTION"):
                row["cohort"] = "surfaced"
                picked.append(row)
            elif row["in_book"]:
                row["cohort"] = "book_missed"     # the detection-failure set
                picked.append(row)
            else:
                nogo_pool.append(row)

    rng = random.Random(seed)
    for row in rng.sample(nogo_pool, min(nogo_sample, len(nogo_pool))):
        row["cohort"] = "nogo_control"
        picked.append(row)
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nogo-sample", type=int, default=600,
                    help="how many random NO-GO days to score as a baseline")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = select_rows(args.nogo_sample)
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} ticker-days selected: {dict(Counter(r['cohort'] for r in rows))}")
    print(f"  of which in the curated book: {sum(1 for r in rows if r['in_book'])}")
    if args.dry_run:
        print("\n--dry-run: nothing fetched")
        return 0

    out_rows, no_bars, no_signal = [], 0, 0
    for i, r in enumerate(rows, 1):
        try:
            bars = fetch_day(r["ticker"], r["date"])
            if bars is None or bars.empty:
                no_bars += 1
                continue
            two = build_2min(bars)
            atr_pct = float(r["atr_pct"]) if r.get("atr_pct") else None
            atr_abs = atr_pct * float(two["open"].iloc[0]) if atr_pct else None
            res = replay_entry_day(two, atr_abs)
        except Exception:  # noqa: BLE001 - one bad day must not stop the sweep
            no_bars += 1
            continue
        if not res["entry_signals"]:
            no_signal += 1
        rec = {c: r.get(c, "") for c in CARRY}
        rec.update({"cohort": r["cohort"], "in_book": r["in_book"], **res})
        out_rows.append(rec)
        if i % 200 == 0:
            print(f"  ...{i}/{len(rows)}  (scored {len(out_rows)}, no bars {no_bars})")

    if out_rows:
        with open(OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
    print(f"\nscored {len(out_rows)}  |  no bars {no_bars}  |  no valid entry {no_signal}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
