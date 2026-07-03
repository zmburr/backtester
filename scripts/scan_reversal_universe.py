"""
Reversal Universe Scanner

Scans historical data for ALL stocks that triggered parabolic reversal criteria
on any given day — including the ones that did NOT reverse. This produces the
"universe" of potential setups with real outcomes, giving us the false positive
population that reversal_data.csv lacks (72-0 training, 100% winners by selection).

Uses the same infrastructure as HistoricalBackscanner (bulk Polygon data, pickle
caching, per-ticker index) but with deliberately loose pre-filters:
  - pct_from_9ema > 0.04 (4% above 9EMA)
  - gap_pct > 0 (any gap up)
  - price > $5, ADV > 500K

Scoring pipeline per hit:
  1. classify_reversal_setup(metrics) → typed setup or None
  2. If typed → ReversalPretrade.validate() → score, grade, recommendation
  3. If no typed match → ReversalScorer.score_setup() → pretrade score/grade/rec
  4. Apply readiness gate (pct_change_3 vs cap threshold)
  5. Record outcome: fade_day_return = (close - open) / open

Output: data/reversal_universe_{start}_{end}.csv (separate from reversal_data.csv)

Usage:
    python scripts/scan_reversal_universe.py --start 2024-01-01 --end 2024-12-31
    python scripts/scan_reversal_universe.py --start 2024-02-01 --end 2024-02-28
    python scripts/scan_reversal_universe.py --start 2020-01-01 --end 2025-12-31 --cap Medium,Large
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scanners.historical_backscanner import HistoricalBackscanner, MIN_PRICE, MIN_AVG_VOLUME
from analyzers.reversal_pretrade import (
    ReversalPretrade,
    classify_reversal_setup,
)
from analyzers.reversal_scorer import (
    ReversalScorer,
    EUPHORIC_SETUPS,
    READINESS_THRESHOLDS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
REVERSAL_CSV = os.path.join(DATA_DIR, 'reversal_data.csv')


def load_reversal_csv_lookup() -> set:
    """Load reversal_data.csv and build a (ticker, date_YYYY-MM-DD) set for cross-referencing."""
    lookup = set()
    if not os.path.exists(REVERSAL_CSV):
        logger.warning(f"reversal_data.csv not found at {REVERSAL_CSV}")
        return lookup

    df = pd.read_csv(REVERSAL_CSV)
    for _, row in df.iterrows():
        ticker = row.get('ticker', '')
        date_raw = str(row.get('date', ''))
        # Convert M/D/YYYY to YYYY-MM-DD
        try:
            dt = pd.to_datetime(date_raw)
            date_str = dt.strftime('%Y-%m-%d')
        except Exception:
            date_str = date_raw
        lookup.add((ticker, date_str))

    logger.info(f"Loaded {len(lookup)} trades from reversal_data.csv for cross-reference")
    return lookup


def scan_universe_date(scanner: HistoricalBackscanner, date: str,
                       pretrade: ReversalPretrade, scorer: ReversalScorer,
                       reversal_lookup: set,
                       min_ema_pct: float = 0.03,
                       cap_filter: Optional[List[str]] = None) -> List[Dict]:
    """
    Scan all tickers on a single date with loose pre-filters.

    Pre-filter: pct_from_9ema > min_ema_pct AND gap_pct >= -1%
    (must stay looser than the loosest per-cap gate: ETF ema9_min=0.04,
    Medium gap_min=-0.01). Then classify → score → record outcome.
    """
    scanner._ensure_ticker_index()

    day_df = scanner.daily_data.get(date)
    if day_df is None or day_df.empty:
        return []

    # Basic price filter + no special chars
    candidates = day_df[
        (day_df['close'] > MIN_PRICE) &
        (~day_df['ticker'].str.contains(r'[.\-/]', regex=True, na=False))
    ]

    matches = []

    for ticker in candidates['ticker'].values:
        ticker_df = scanner._ticker_index.get(ticker)
        if ticker_df is None or len(ticker_df) < 15:
            continue

        # Slice history up to scan date
        history = ticker_df.loc[ticker_df.index <= date]
        if len(history) < 15:
            continue
        if len(history) > 220:
            history = history.iloc[-220:]

        # Need today's row for gap and outcome
        if history.index[-1] != date:
            continue

        hist_only = history.iloc[:-1]
        if len(hist_only) < 10:
            continue

        # Quick volume filter
        adv_window = min(20, len(hist_only))
        avg_vol = hist_only['volume'].iloc[-adv_window:].mean()
        if avg_vol < MIN_AVG_VOLUME:
            continue

        # Quick pre-filters (loose — must stay looser than the loosest
        # per-cap gate in GATE_THRESHOLDS: Medium allows gap >= -1%)
        # 1. Gap check (allow small gap-down / flat opens)
        today_open = history.iloc[-1]['open']
        prior_close = hist_only.iloc[-1]['close']
        if prior_close <= 0 or today_open < prior_close * 0.99:
            continue

        # 2. 9EMA extension check (>= min_ema_pct)
        if len(hist_only) >= 9:
            ema_9 = hist_only['close'].ewm(span=9, adjust=False).mean().iloc[-1]
            if ema_9 > 0 and (prior_close - ema_9) / ema_9 < min_ema_pct:
                continue
        else:
            continue

        # Passed pre-filters — compute full metrics
        metrics = scanner.compute_metrics_for_ticker(ticker, date, history)
        if metrics is None:
            continue

        # Estimate cap
        cap = HistoricalBackscanner.estimate_cap(
            metrics.get('current_price', 0),
            metrics.get('avg_daily_vol', 0),
        )

        # Cap filter
        if cap_filter and cap not in cap_filter:
            continue

        # --- Scoring pipeline ---
        setup_type = classify_reversal_setup(metrics, cap=cap)
        score = None
        grade = None
        recommendation = None
        readiness_passed = None

        if setup_type is not None:
            # Typed setup → ReversalPretrade
            result = pretrade.validate(
                ticker=ticker, metrics=metrics,
                setup_type=setup_type, cap=cap,
            )
            score = result.score
            grade = result.classification_details.get('grade', 'F')
            recommendation = result.recommendation
            # Readiness is already applied inside validate() for euphoric setups
            # Check if readiness was the reason for NO-GO
            readiness_passed = True
            if setup_type in EUPHORIC_SETUPS:
                thresh = READINESS_THRESHOLDS.get(cap, 0.03)
                pc3 = metrics.get('pct_change_3')
                if pc3 is None or (isinstance(pc3, float) and pd.isna(pc3)) or pc3 < thresh:
                    readiness_passed = False
        else:
            # No typed match → ReversalScorer (generic 6-criteria, use pretrade score)
            setup_type = 'generic'
            scored = scorer.score_setup(
                ticker=ticker, date=date, cap=cap,
                metrics=metrics, setup=None,
            )
            score = scored['pretrade_score']
            grade = scored['pretrade_grade']
            recommendation = scored['pretrade_recommendation']
            readiness_passed = scored.get('readiness_passed', True)

        # --- Compute pct_change_3_with_gap (includes gap, matches CSV definition) ---
        # CSV: pct_change_3 = (trade_day_open - close_3d_ago) / close_3d_ago
        pct_change_3_with_gap = None
        if len(hist_only) >= 3:
            close_3d_ago = hist_only.iloc[-3]['close']
            if close_3d_ago > 0:
                pct_change_3_with_gap = (today_open - close_3d_ago) / close_3d_ago

        # Cross-reference with reversal_data.csv
        in_reversal_csv = (ticker, date) in reversal_lookup

        matches.append({
            'date': date,
            'ticker': ticker,
            'cap': cap,
            'setup_type': setup_type,
            'score': score,
            'grade': grade,
            'recommendation': recommendation,
            'readiness_passed': readiness_passed,
            'pct_from_9ema': metrics.get('pct_from_9ema'),
            'prior_day_range_atr': metrics.get('prior_day_range_atr'),
            'rvol_score': metrics.get('rvol_score'),
            'consecutive_up_days': metrics.get('consecutive_up_days'),
            'gap_pct': metrics.get('gap_pct'),
            'pct_change_3': metrics.get('pct_change_3'),
            'pct_change_3_with_gap': pct_change_3_with_gap,
            'pct_change_15': metrics.get('pct_change_15'),
            'pct_change_30': metrics.get('pct_change_30'),
            'pct_from_50mav': metrics.get('pct_from_50mav'),
            'current_price': metrics.get('current_price'),
            'atr_pct': metrics.get('atr_pct'),
            'fade_day_return': metrics.get('fade_day_return'),
            'fade_day_close_position': metrics.get('fade_day_close_position'),
            'in_reversal_csv': in_reversal_csv,
        })

    return matches


def print_summary(results_df: pd.DataFrame, reversal_lookup: set,
                  start: str, end: str, scan_days: int):
    """Print summary statistics after scan."""
    print(f"\n{'=' * 80}")
    print(f"REVERSAL UNIVERSE SCAN RESULTS")
    print(f"Period: {start} to {end} ({scan_days} trading days)")
    print(f"{'=' * 80}")

    if results_df.empty:
        print("  No setups found.")
        return

    total = len(results_df)

    # By recommendation
    print(f"\nTotal setups found: {total:,}")
    for rec in ['GO', 'CAUTION', 'NO-GO']:
        subset = results_df[results_df['recommendation'] == rec]
        n = len(subset)
        if n == 0:
            continue
        pct = n / total * 100
        reversed_mask = subset['fade_day_return'] < 0
        reversed_n = reversed_mask.sum()
        wr = reversed_n / n * 100 if n > 0 else 0
        avg_ret = subset['fade_day_return'].mean() * 100
        print(f"  {rec:8s}: {n:5,} ({pct:4.1f}%) — reversed: {reversed_n:,} ({wr:.0f}% WR) — avg fade: {avg_ret:+.1f}%")

    # By setup type
    print(f"\nBy setup type:")
    for stype, group in results_df.groupby('setup_type'):
        n = len(group)
        reversed_n = (group['fade_day_return'] < 0).sum()
        wr = reversed_n / n * 100 if n > 0 else 0
        print(f"  {stype:20s}: {n:5,} — reversed: {reversed_n:,} ({wr:.0f}% WR)")

    # By cap
    print(f"\nBy cap:")
    for cap in ['ETF', 'Large', 'Medium', 'Small', 'Micro']:
        subset = results_df[results_df['cap'] == cap]
        if len(subset) == 0:
            continue
        n = len(subset)
        reversed_n = (subset['fade_day_return'] < 0).sum()
        wr = reversed_n / n * 100 if n > 0 else 0
        print(f"  {cap:8s}: {n:5,} — reversed: {reversed_n:,} ({wr:.0f}% WR)")

    # Readiness gate analysis
    print(f"\nReadiness gate:")
    for passed_val in [True, False]:
        subset = results_df[results_df['readiness_passed'] == passed_val]
        n = len(subset)
        if n == 0:
            continue
        reversed_n = (subset['fade_day_return'] < 0).sum()
        wr = reversed_n / n * 100 if n > 0 else 0
        label = "PASSED" if passed_val else "FAILED"
        print(f"  {label:8s}: {n:5,} — reversed: {reversed_n:,} ({wr:.0f}% WR)")

    # Cross-reference with reversal_data.csv
    in_csv = results_df[results_df['in_reversal_csv'] == True]
    not_in_csv = results_df[results_df['in_reversal_csv'] == False]
    total_csv_trades = len(reversal_lookup)

    print(f"\nCross-reference with reversal_data.csv:")
    print(f"  In CSV:     {len(in_csv):,}/{total_csv_trades} found ({len(in_csv) / total_csv_trades * 100:.0f}% match rate)" if total_csv_trades > 0 else "  In CSV: 0")
    print(f"  Not in CSV: {len(not_in_csv):,} setups you didn't trade")

    # GO setups not traded (the interesting ones)
    go_not_traded = not_in_csv[not_in_csv['recommendation'] == 'GO']
    if len(go_not_traded) > 0:
        reversed_n = (go_not_traded['fade_day_return'] < 0).sum()
        wr = reversed_n / len(go_not_traded) * 100
        avg_ret = go_not_traded['fade_day_return'].mean() * 100
        print(f"  GO + not traded: {len(go_not_traded):,} — reversed: {reversed_n:,} ({wr:.0f}% WR) — avg fade: {avg_ret:+.1f}%")

    print()


def main():
    parser = argparse.ArgumentParser(
        description='Scan historical data for ALL reversal universe candidates (including false positives)'
    )
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output CSV path (default: data/reversal_universe_{start}_{end}.csv)')
    parser.add_argument('--cap', default=None,
                        help='Filter to specific cap(s), comma-separated (e.g. "Medium,Large")')
    parser.add_argument('--lookback', type=int, default=310,
                        help='Calendar days of lookback for data fetch (default: 310)')
    parser.add_argument('--min-ema-pct', type=float, default=0.04,
                        help='Minimum pct_from_9ema pre-filter (default: 0.04 = 4%%)')
    parser.add_argument('--summary', action='store_true', default=True,
                        help='Print summary statistics (default: True)')

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        args.output = os.path.join(DATA_DIR, f"reversal_universe_{args.start}_{args.end}.csv")

    cap_filter = [c.strip() for c in args.cap.split(',')] if args.cap else None

    # Initialize components
    scanner = HistoricalBackscanner(setup_type=None, cap_filter=None)  # No setup type filter
    pretrade = ReversalPretrade()
    scorer = ReversalScorer()

    # Load reversal_data.csv for cross-referencing
    reversal_lookup = load_reversal_csv_lookup()

    # Compute fetch range (scan dates need lookback history)
    fetch_start_dt = datetime.strptime(args.start, '%Y-%m-%d') - timedelta(days=args.lookback)
    fetch_start = fetch_start_dt.strftime('%Y-%m-%d')

    # Phase 1: Fetch market data
    print(f"\n{'=' * 80}")
    print(f"REVERSAL UNIVERSE SCANNER")
    print(f"{'=' * 80}")
    print(f"  Fetch range:    {fetch_start} to {args.end}")
    print(f"  Scan range:     {args.start} to {args.end}")
    print(f"  Pre-filter:     9EMA > {args.min_ema_pct * 100:.0f}%, gap > 0%")
    if cap_filter:
        print(f"  Cap filter:     {', '.join(cap_filter)}")
    print(f"  Output:         {args.output}")
    print()

    t0 = time.time()
    scanner.fetch_market_data(fetch_start, args.end)
    fetch_time = time.time() - t0
    print(f"Data fetch complete: {len(scanner.daily_data)} days in {fetch_time:.1f}s\n")

    # Phase 2: Scan all dates
    scan_dates = scanner.get_trading_dates(args.start, args.end)
    logger.info(f"Scanning {len(scan_dates)} trading days for reversal universe")

    all_matches = []

    t1 = time.time()
    for i, date in enumerate(scan_dates):
        matches = scan_universe_date(
            scanner, date, pretrade, scorer, reversal_lookup,
            min_ema_pct=args.min_ema_pct,
            cap_filter=cap_filter,
        )
        all_matches.extend(matches)

        if (i + 1) % 10 == 0 or (i + 1) == len(scan_dates):
            go_count = sum(1 for m in matches if m['recommendation'] == 'GO')
            logger.info(
                f"  [date {i + 1}/{len(scan_dates)}] {date} — "
                f"{len(matches)} hits ({go_count} GO) — "
                f"running total: {len(all_matches):,}"
            )

    scan_time = time.time() - t1

    # Build results DataFrame
    results_df = pd.DataFrame(all_matches)

    # Dedupe on ticker+date (keep highest-scored occurrence)
    if not results_df.empty:
        before = len(results_df)
        results_df = results_df.sort_values('score', ascending=False)
        results_df = results_df.drop_duplicates(subset=['ticker', 'date'], keep='first')
        results_df = results_df.sort_values('date').reset_index(drop=True)
        dupes_removed = before - len(results_df)
        if dupes_removed > 0:
            logger.info(f"Removed {dupes_removed} duplicate ticker+date rows (kept highest score)")

    # Save CSV
    if not results_df.empty:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        results_df.to_csv(args.output, index=False)
        logger.info(f"Results saved to {args.output}")

    # Print summary
    if args.summary:
        print_summary(results_df, reversal_lookup, args.start, args.end, len(scan_dates))

    print(f"Scan complete: {len(results_df):,} setups in {scan_time:.1f}s")
    print(f"Total time: {fetch_time + scan_time:.1f}s")


if __name__ == '__main__':
    main()
