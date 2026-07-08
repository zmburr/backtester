"""Fill forward outcomes on the unified signal ledger.

Each ledger row's ``date`` is the setup day. Once that session's OHLCV is
available (i.e. the day after, or later), this script fetches same-day daily bars
from Polygon and computes the excursion columns, mirroring the dispatcher's
``validate_signal`` math EXACTLY:

    outcome_pct        = (close - open) / open
    max_favorable_pct  = (open - low)  / open   for reversals  (short: down move)
                       = (high - open) / open   for bounces / breakouts (long: up move)
    atr_move           = abs(outcome_pct) / atr_pct        (row's stored atr_pct)
    mfe_atr            = max_favorable_pct / atr_pct

Breakout note: the dispatcher's validate_signal has no breakout branch — its
``else`` treats everything non-reversal as a long (close > open, high-side MFE).
We follow that convention here: breakout uses the bounce/long MFE definition.

Usage:
    python -m scripts.fill_signal_outcomes                 # fill unfilled rows from the last 7 calendar days
    python -m scripts.fill_signal_outcomes --all           # backfill ALL unfilled past rows
    python -m scripts.fill_signal_outcomes --date 2026-07-07   # fill only that date's unfilled rows

The no-arg default is bounded to the last 7 calendar days so the morning run
stays fast (one get_daily per row, rate-limited); use --all for a full backfill.
"""

from __future__ import annotations

import argparse
import datetime
import math
import time
from pathlib import Path

import pandas as pd

from support.signal_ledger import LEDGER_PATH, LEDGER_COLUMNS
from support.csv_utils import save_csv_atomic
from data_queries.polygon_queries import get_daily

# Gentle Polygon rate-limiting between distinct API calls.
_SLEEP_BETWEEN_CALLS = 0.15

# Buckets whose max-favorable excursion is measured to the DOWNSIDE (short).
# Everything else (bounce, breakout, anything unknown) is treated long-side,
# matching the dispatcher's validate_signal else-branch.
_SHORT_BUCKETS = {"reversal"}


def _to_float(v):
    try:
        if v is None or v == "" or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _is_unfilled(row) -> bool:
    """A row is unfilled when it has no outcome_filled_date yet."""
    v = row.get("outcome_filled_date")
    return v is None or v == "" or (isinstance(v, float) and pd.isna(v))


def _compute_outcome(row: dict, bar) -> dict | None:
    """Return the outcome-column dict for a row given its same-day daily bar.

    Mirrors dispatcher.signal_scorecard.validate_signal exactly.
    """
    o = _to_float(getattr(bar, "open", None))
    h = _to_float(getattr(bar, "high", None))
    l = _to_float(getattr(bar, "low", None))
    c = _to_float(getattr(bar, "close", None))
    vol = getattr(bar, "volume", None)
    # A valid daily bar has all OHLC present, finite, and strictly positive. A
    # zero/negative/non-finite price yields a fake 0.0 outcome, so return None to
    # leave the row unfilled for a later retry instead of stamping bad data.
    if not all(v is not None and math.isfinite(v) and v > 0 for v in (o, h, l, c)):
        ticker = str(row.get("ticker", "")).strip()
        date = str(row.get("date", "")).strip()
        print(f"  {ticker} {date}: invalid daily bar (OHLC={o},{h},{l},{c}) — left unfilled")
        return None

    outcome_pct = (c - o) / o

    bucket = (row.get("bucket") or "").strip().lower()
    if bucket in _SHORT_BUCKETS:
        max_favorable_pct = (o - l) / o
    else:  # bounce, breakout, unknown -> long-side, like dispatcher's else branch
        max_favorable_pct = (h - o) / o

    atr_pct = _to_float(row.get("atr_pct"))
    if atr_pct and atr_pct > 0:
        atr_move = abs(outcome_pct) / atr_pct
        mfe_atr = max_favorable_pct / atr_pct
    else:
        atr_move = None
        mfe_atr = None

    return {
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": vol if vol is not None else "",
        "outcome_pct": round(outcome_pct, 6),
        "max_favorable_pct": round(max_favorable_pct, 6),
        "atr_move": round(atr_move, 2) if atr_move is not None else "",
        "mfe_atr": round(mfe_atr, 2) if mfe_atr is not None else "",
        "outcome_filled_date": datetime.date.today().strftime("%Y-%m-%d"),
    }


def fill_outcomes(target_date: str | None, backfill_all: bool = False) -> tuple[int, int]:
    """Fill unfilled rows. Returns (n_filled, n_failed).

    Scope: --date fills exactly that date; otherwise only settled (past) days,
    bounded to the last 7 calendar days unless backfill_all is True.
    """
    if not LEDGER_PATH.exists():
        print(f"No ledger at {LEDGER_PATH} — nothing to fill.")
        return 0, 0

    df = pd.read_csv(LEDGER_PATH, dtype=str, keep_default_na=False)
    if df.empty:
        print("Ledger is empty — nothing to fill.")
        return 0, 0

    # Ensure every expected column exists (older ledgers may predate a column).
    for col in LEDGER_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    cutoff_str = (datetime.date.today() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")

    def _row_in_scope(row) -> bool:
        if not _is_unfilled(row):
            return False
        d = row.get("date", "")
        if target_date is not None:
            return d == target_date
        if not (bool(d) and d < today_str):  # only settled (past) days
            return False
        if backfill_all:
            return True
        return d >= cutoff_str  # no-arg: last 7 calendar days only

    scope_idx = [i for i, row in df.iterrows() if _row_in_scope(row)]
    if not scope_idx:
        print("No unfilled rows in scope.")
        return 0, 0

    print(f"{len(scope_idx)} unfilled row(s) in scope.")

    # Cache bars per (ticker, date) so duplicate signals don't double-fetch.
    bar_cache: dict[tuple[str, str], object] = {}
    n_filled = 0
    n_failed = 0

    for i in scope_idx:
        row = df.loc[i].to_dict()
        ticker = str(row.get("ticker", "")).strip()
        date = str(row.get("date", "")).strip()
        if not ticker or not date:
            n_failed += 1
            continue

        key = (ticker, date)
        if key not in bar_cache:
            bar_cache[key] = get_daily(ticker, date)
            time.sleep(_SLEEP_BETWEEN_CALLS)
        bar = bar_cache[key]

        if bar is None:
            print(f"  {ticker} {date}: no Polygon data — skipped")
            n_failed += 1
            continue

        outcome = _compute_outcome(row, bar)
        if outcome is None:
            print(f"  {ticker} {date}: incomplete bar — skipped")
            n_failed += 1
            continue

        for col, val in outcome.items():
            df.at[i, col] = val
        n_filled += 1
        print(
            f"  {ticker} {date} ({row.get('bucket')}/{row.get('recommendation')}): "
            f"{outcome['outcome_pct'] * 100:+.2f}%  MFE={outcome['mfe_atr']}"
        )

    if n_filled:
        save_csv_atomic(df, LEDGER_PATH)

    return n_filled, n_failed


def main():
    parser = argparse.ArgumentParser(description="Fill forward outcomes on the signal ledger.")
    parser.add_argument("--date", help="Fill only rows with this date (YYYY-MM-DD).")
    parser.add_argument("--all", action="store_true",
                        help="Backfill ALL unfilled past rows (default: last 7 days only).")
    args = parser.parse_args()

    n_filled, n_failed = fill_outcomes(args.date, backfill_all=args.all)
    if args.date:
        scope_desc = "date " + args.date
    else:
        scope_desc = "all settled days" if args.all else "last 7 days"
    print("\n" + "=" * 44)
    print(f"  Signal ledger outcome fill — {scope_desc}")
    print(f"  filled: {n_filled}    failed/skipped: {n_failed}")
    print("=" * 44)


if __name__ == "__main__":
    main()
