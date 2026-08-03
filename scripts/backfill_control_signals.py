"""Rebuild the below-bar (control) signal cohort from the unified ledger.

Why this exists
---------------
``priority_report`` only ever wrote GO/CAUTION signals to ``priority_signals/``,
so ``signal_outcomes.csv`` — and therefore the whole Signal Analysis feedback
loop — contains no signal that failed the entry bar. That makes the bar itself
untestable: the analysis can compare GO vs CAUTION, but never "does score >= 4
beat score <= 3?".

``priority_report`` now emits a control lane forward (``control_signals/``).
This script backfills the history that lane never had, by replaying NO-GO rows
out of ``signal_ledger.csv`` — which HAS logged every scored signal, including
NO-GO, since 2026-07-09. Output files match the live schema exactly, so
``signal_scorecard.collect_signals`` needs no special case; running the scorecard
afterwards fills real D0..D+3 outcomes.

Known gap: LEDGER_COLUMNS carries atr_pct / gap_pct / pct_change_3 /
prior_day_range_atr / pct_from_9ema / prior_day_rvol / premarket_rvol, but NOT
the bounce depth features (selloff_total_pct, pct_off_30d_high,
pct_off_52wk_high). Backfilled rows therefore support score-level and
gap/pct_change_3 analysis but not the depth-criterion question. Rows logged
forward by priority_report carry the full metric set.

Usage:
    python -m scripts.backfill_control_signals --dry-run
    python -m scripts.backfill_control_signals
    python -m scripts.backfill_control_signals --since 2026-07-15
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from support.signal_ledger import LEDGER_PATH  # noqa: E402
from scripts.signal_scorecard import trading_days  # noqa: E402

CONTROL_DIR = PROJECT_ROOT / "data" / "control_signals"

# Sources whose ledger ``date`` is already the TARGET trading day rather than the
# run day. evening_signal_log calls log_signals(date_str=target_str), while every
# other source logs the run day. collect_signals() reads an evening file's date
# as the run day and advances it, so these rows must be shifted back one trading
# day or their outcome window lands a day late.
_TARGET_DATED_SOURCES = {"evening_board"}

# Buckets the scorecard scores. breakout is logged to the ledger but has no
# outcome semantics in signal_outcomes.csv, so it is excluded here too.
BUCKETS = ("bounce", "reversal")

# Ledger metric columns worth carrying into the signal JSON. These are the
# intersection of LEDGER_COLUMNS and signal_scorecard.METRIC_COLUMNS, plus
# atr_pct which the scorecard reads separately to size its tradeable gate.
METRIC_KEYS = (
    "atr_pct",
    "gap_pct",
    "prior_day_rvol",
    "premarket_rvol",
    "pct_change_3",
    "prior_day_range_atr",
    "pct_from_9ema",
)

# Row identity is (file_date, session, ticker, bucket): the ledger holds several
# `source` rows per ticker-session (priority_report, watchlist_report,
# evening_board) describing the same setup, so the first one wins. Keying on the
# normalised file_date rather than the raw date is what lets a target-dated
# evening_board row dedupe against the run-dated priority_report row beside it.


def _to_float(v):
    try:
        if v is None or str(v).strip() == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _prev_trading_day(date: str) -> str | None:
    """Trading day immediately before `date` on the NYSE calendar."""
    prior = [d for d in trading_days() if d < date]
    return prior[-1] if prior else None


def _file_date(row: dict) -> str | None:
    """The date under which this row's signal file should be written.

    Normalises every source to run-day semantics, which is what
    signal_scorecard.collect_signals assumes when it derives the target day.
    """
    date = (row.get("date") or "").strip()
    if not date:
        return None
    if (row.get("source") or "").strip() in _TARGET_DATED_SOURCES:
        return _prev_trading_day(date)
    return date


def load_control_rows(since: str | None) -> tuple[dict, Counter]:
    """Return {(date, session): [signal_entry, ...]} plus a skip tally."""
    if not LEDGER_PATH.exists():
        raise SystemExit(f"ledger not found: {LEDGER_PATH}")

    with open(LEDGER_PATH, newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))

    by_session: dict[tuple, list] = defaultdict(list)
    seen: set = set()
    skips: Counter = Counter()

    for r in raw:
        date = _file_date(r)
        if not date or (since and date < since):
            skips["out_of_range"] += 1
            continue
        if (r.get("recommendation") or "").strip() != "NO-GO":
            skips["not_below_bar"] += 1
            continue
        if (r.get("bucket") or "").strip() not in BUCKETS:
            skips["other_bucket"] += 1
            continue

        # atr_pct sizes the tradeable gate (signal_scorecard.gate_atr); without a
        # positive value the row can never produce a verdict, so drop it here
        # rather than storing an unscoreable row.
        atr = _to_float(r.get("atr_pct"))
        if atr is None or atr <= 0:
            skips["no_atr"] += 1
            continue

        session = (r.get("session") or "morning").strip()
        key = (date, session, (r.get("ticker") or "").strip(), (r.get("bucket") or "").strip())
        if key in seen:
            skips["duplicate_source"] += 1
            continue
        seen.add(key)

        metrics = {}
        for k in METRIC_KEYS:
            v = _to_float(r.get(k))
            if v is not None:
                metrics[k] = round(v, 6)

        by_session[(date, session)].append({
            "ticker": (r.get("ticker") or "").strip(),
            "bucket": (r.get("bucket") or "").strip(),
            "cap": (r.get("cap") or "").strip(),
            "recommendation": "NO-GO",
            "score": (r.get("score_str") or "").strip(),
            "metrics": metrics,
        })

    return by_session, skips


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--since", help="only backfill sessions on/after this date (YYYY-MM-DD)")
    args = ap.parse_args()

    by_session, skips = load_control_rows(args.since)

    if not by_session:
        print("No control rows found — nothing to backfill.")
        return 0

    total = sum(len(v) for v in by_session.values())
    print(f"{total} control signals across {len(by_session)} sessions "
          f"({min(k[0] for k in by_session)} .. {max(k[0] for k in by_session)})")
    print(f"skipped: {dict(skips)}\n")

    for (date, session), entries in sorted(by_session.items()):
        buckets = Counter(e["bucket"] for e in entries)
        print(f"  {date}_{session:<7} {len(entries):>4} signals  {dict(buckets)}")

    if args.dry_run:
        print("\n--dry-run: no files written")
        return 0

    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for (date, session), entries in sorted(by_session.items()):
        payload = {
            "date": date,
            "session": session,
            "generated_at": f"{date}T00:00:00",  # synthetic: replayed from ledger
            "go_count": 0,
            "caution_count": 0,
            "backfilled_from": LEDGER_PATH.name,
            "signals": entries,
        }
        (CONTROL_DIR / f"{date}_{session}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        written += 1

    print(f"\nWrote {written} files to {CONTROL_DIR}")
    print("Next: python -m scripts.signal_scorecard   (fills D0..D+3 outcomes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
