"""
Historical Backscanner

Scans historical market data across ALL tickers for reversal setup types
(starting with 3DGapFade). Uses Polygon's get_grouped_daily_aggs() endpoint
for massive efficiency: 1 API call per date returns OHLCV for all ~12,000 tickers.

Data pipeline:
  1. fetch_market_data(start, end) — bulk fetch grouped daily bars per date
  2. Build per-ticker DataFrames from bulk data
  3. compute_metrics(ticker, date) — compute 9EMA, ATR, consecutive up days,
     gap %, RVOL etc. locally from cached data (zero API calls)
  4. scan_date(date) — apply classification + scoring to all tickers
  5. Filter: price > $5, avg volume > 500K

Efficiency: ~560 API calls for 1 year + 310-day lookback vs 250,000 per-ticker calls.

Usage:
    python -m scanners.historical_backscanner --start 2024-01-01 --end 2024-12-31 --setup 3DGapFade
    python -m scanners.historical_backscanner --start 2020-01-01 --end 2024-12-31 --setup 3DGapFade
"""

import os
import sys
import argparse
import logging
import time
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from polygon.rest import RESTClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzers.reversal_pretrade import (
    ReversalPretrade,
    classify_reversal_setup,
    REVERSAL_SETUP_PROFILES,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

POLYGON_API_KEY = "pcwUY7TnSF66nYAPIBCApPMyVrXTckJY"

# Minimum filters to reduce ticker universe
MIN_PRICE = 5.0
MIN_AVG_VOLUME = 500_000


class HistoricalBackscanner:
    """
    Scans historical market data across all tickers for reversal setup types.

    Uses Polygon grouped daily endpoint (1 call per date) for bulk data,
    then computes all metrics locally.
    """

    def __init__(self, setup_type: str = '3DGapFade', cache_dir: str = None):
        self.setup_type = setup_type
        self.pretrade = ReversalPretrade()
        self.client = RESTClient(api_key=POLYGON_API_KEY)
        self.nyse = mcal.get_calendar('NYSE')

        # Cache directory for fetched data
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data', 'backscanner_cache'
            )
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # In-memory data: date_str -> DataFrame of all tickers
        self.daily_data: Dict[str, pd.DataFrame] = {}

        # Pre-built ticker index: ticker -> DataFrame (date-indexed OHLCV)
        # Built lazily by _ensure_ticker_index()
        self._ticker_index: Optional[Dict[str, pd.DataFrame]] = None
        self._sorted_dates: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Trading calendar
    # ------------------------------------------------------------------

    def get_trading_dates(self, start: str, end: str) -> List[str]:
        """Get NYSE trading dates in range [start, end]."""
        schedule = self.nyse.valid_days(start_date=start, end_date=end)
        return [d.strftime('%Y-%m-%d') for d in schedule]

    # ------------------------------------------------------------------
    # Bulk data fetching
    # ------------------------------------------------------------------

    def fetch_market_data(self, start: str, end: str) -> None:
        """
        Fetch grouped daily bars for each trading day in range.
        Caches to pickle file for resume capability.

        Each API call returns OHLCV for ALL tickers on that date.
        """
        cache_file = os.path.join(
            self.cache_dir,
            f"grouped_daily_{start}_{end}.pkl"
        )

        # Resume from cache if available
        if os.path.exists(cache_file):
            logger.info(f"Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.daily_data = pickle.load(f)
            self._ticker_index = None  # Force rebuild
            logger.info(f"Loaded {len(self.daily_data)} days from cache")
            return

        dates = self.get_trading_dates(start, end)
        logger.info(f"Fetching grouped daily data for {len(dates)} trading days ({start} to {end})")

        for i, date in enumerate(dates):
            if date in self.daily_data:
                continue

            try:
                aggs = self.client.get_grouped_daily_aggs(date)
                if aggs:
                    rows = []
                    for a in aggs:
                        rows.append({
                            'ticker': a.ticker,
                            'open': a.open,
                            'high': a.high,
                            'low': a.low,
                            'close': a.close,
                            'volume': a.volume,
                            'vwap': getattr(a, 'vwap', None),
                        })
                    self.daily_data[date] = pd.DataFrame(rows)
                else:
                    self.daily_data[date] = pd.DataFrame()

                if (i + 1) % 50 == 0 or (i + 1) == len(dates):
                    logger.info(f"  [{i + 1}/{len(dates)}] Fetched {date}")

            except Exception as e:
                logger.error(f"  Error fetching {date}: {e}")
                self.daily_data[date] = pd.DataFrame()
                time.sleep(1)

            # Rate limiting: Polygon free tier = 5 calls/min
            # Paid tier is much higher, but add a small delay to be safe
            time.sleep(0.15)

        # Save cache and invalidate ticker index
        self._ticker_index = None
        logger.info(f"Saving cache to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(self.daily_data, f)

    def _ensure_ticker_index(self) -> None:
        """
        Pre-build a ticker-indexed lookup from the date-grouped daily_data.
        Converts {date -> all_tickers_df} into {ticker -> date-indexed OHLCV df}.
        This is O(total_rows) once, then O(1) per ticker lookup.
        """
        if self._ticker_index is not None:
            return

        logger.info("Building ticker index from grouped daily data...")
        self._sorted_dates = sorted(self.daily_data.keys())

        # Concatenate all daily DataFrames into one big frame
        frames = []
        for date_str in self._sorted_dates:
            df = self.daily_data.get(date_str)
            if df is not None and not df.empty:
                df_copy = df.copy()
                df_copy['date'] = date_str
                frames.append(df_copy)

        if not frames:
            self._ticker_index = {}
            return

        all_data = pd.concat(frames, ignore_index=True)

        # Group by ticker and build per-ticker DataFrames
        self._ticker_index = {}
        for ticker, group in all_data.groupby('ticker'):
            group_sorted = group.sort_values('date').set_index('date')
            self._ticker_index[ticker] = group_sorted

        logger.info(f"Ticker index built: {len(self._ticker_index)} tickers")

    def _build_ticker_history(self, ticker: str, as_of_date: str,
                              lookback_days: int = 220) -> Optional[pd.DataFrame]:
        """
        Build a DataFrame of daily bars for a ticker from the pre-built index.

        Args:
            ticker: Stock ticker
            as_of_date: Reference date (inclusive)
            lookback_days: Number of trading days to look back

        Returns:
            DataFrame with OHLCV indexed by date string, or None if insufficient data
        """
        self._ensure_ticker_index()

        ticker_df = self._ticker_index.get(ticker)
        if ticker_df is None or len(ticker_df) < 15:
            return None

        # Slice up to and including as_of_date
        mask = ticker_df.index <= as_of_date
        history = ticker_df.loc[mask]

        if len(history) < 15:
            return None

        # Limit to lookback window
        if len(history) > lookback_days:
            history = history.iloc[-lookback_days:]

        return history

    # ------------------------------------------------------------------
    # Metric computation (all local, zero API calls)
    # ------------------------------------------------------------------

    def compute_metrics_for_ticker(self, ticker: str, date: str,
                                   history: pd.DataFrame) -> Optional[Dict]:
        """
        Compute screening metrics from local historical data.

        Same math as SetupScreener.compute_metrics() but from cached data.

        Args:
            ticker: Stock ticker
            date: The scan date
            history: DataFrame with OHLCV up to and including `date`

        Returns:
            Dict of metrics or None if insufficient data
        """
        if history is None or len(history) < 15:
            return None

        metrics = {'ticker': ticker, 'date': date}

        # The last row is `date` (the potential trade day)
        # hist = completed bars before trade day
        has_today = (history.index[-1] == date)
        if has_today and len(history) > 1:
            hist = history.iloc[:-1]
            today_row = history.iloc[-1]
        elif has_today:
            return None
        else:
            hist = history
            today_row = None

        if len(hist) < 10:
            return None

        closes = hist['close']
        current_close = hist.iloc[-1]['close']
        metrics['current_price'] = current_close

        # --- Moving averages ---
        if len(closes) >= 9:
            ema_9 = closes.ewm(span=9, adjust=False).mean().iloc[-1]
            metrics['pct_from_9ema'] = (current_close - ema_9) / ema_9 if ema_9 != 0 else 0

        if len(closes) >= 50:
            sma_50 = closes.rolling(50).mean().iloc[-1]
            if not pd.isna(sma_50) and sma_50 != 0:
                metrics['pct_from_50mav'] = (current_close - sma_50) / sma_50

        if len(closes) >= 200:
            sma_200 = closes.rolling(200).mean().iloc[-1]
            if not pd.isna(sma_200) and sma_200 != 0:
                metrics['pct_from_200mav'] = (current_close - sma_200) / sma_200

        # --- ATR (14-day) ---
        if len(hist) >= 2:
            hl = hist['high'] - hist['low']
            hpc = abs(hist['high'] - hist['close'].shift(1))
            lpc = abs(hist['low'] - hist['close'].shift(1))
            tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)

            atr_window = min(14, len(tr))
            atr = tr.rolling(window=atr_window, min_periods=1).mean().iloc[-1]
            metrics['atr'] = atr
            metrics['atr_pct'] = atr / current_close if current_close > 0 else 0

            # Prior day range as multiple of ATR
            prior_range = hist.iloc[-1]['high'] - hist.iloc[-1]['low']
            metrics['prior_day_range_atr'] = prior_range / atr if atr > 0 else 0

        # --- Consecutive up days (close > prior_close) ---
        consecutive_up = 0
        for i in range(len(hist) - 1, 0, -1):
            if hist.iloc[i]['close'] > hist.iloc[i - 1]['close']:
                consecutive_up += 1
            else:
                break
        metrics['consecutive_up_days'] = consecutive_up

        # --- RVOL ---
        adv_window = min(20, len(hist))
        adv = hist['volume'].rolling(window=adv_window, min_periods=1).mean().iloc[-1]
        prior_day_vol = hist.iloc[-1]['volume']
        metrics['avg_daily_vol'] = adv
        metrics['rvol_score'] = prior_day_vol / adv if adv > 0 else 0

        # --- Gap % (trade day open vs prior close) ---
        if today_row is not None and current_close > 0:
            metrics['gap_pct'] = (today_row['open'] - current_close) / current_close
        else:
            metrics['gap_pct'] = 0.0

        # --- Percent changes ---
        for days, key in [(3, 'pct_change_3'), (15, 'pct_change_15'),
                          (30, 'pct_change_30')]:
            if len(hist) >= days:
                old_close = hist.iloc[-days]['close']
                if old_close > 0:
                    metrics[key] = (current_close - old_close) / old_close

        return metrics

    # ------------------------------------------------------------------
    # Market cap estimation from price + volume
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_cap(price: float, avg_vol: float) -> str:
        """
        Rough market cap estimation from price and average volume.
        Not perfect, but sufficient for threshold selection.
        """
        daily_dollar_vol = price * avg_vol
        if daily_dollar_vol > 5_000_000_000:
            return 'Large'
        elif daily_dollar_vol > 500_000_000:
            return 'Medium' if price > 20 else 'Large'
        elif daily_dollar_vol > 50_000_000:
            return 'Medium' if price > 10 else 'Small'
        elif daily_dollar_vol > 5_000_000:
            return 'Small'
        else:
            return 'Micro'

    # ------------------------------------------------------------------
    # Scan a single date
    # ------------------------------------------------------------------

    def scan_date(self, date: str) -> List[Dict]:
        """
        Screen all tickers on a single date for the target setup type.

        Uses the pre-built ticker index for O(1) per-ticker lookups.

        Args:
            date: Trading date (YYYY-MM-DD)

        Returns:
            List of dicts for matched setups
        """
        self._ensure_ticker_index()

        day_df = self.daily_data.get(date)
        if day_df is None or day_df.empty:
            return []

        # Quick pre-filter: price > MIN_PRICE, no special chars
        candidates = day_df[
            (day_df['close'] > MIN_PRICE) &
            (~day_df['ticker'].str.contains(r'[.\-/]', regex=True, na=False))
        ]

        matches = []

        for ticker in candidates['ticker'].values:
            # Fast lookup from pre-built index
            ticker_df = self._ticker_index.get(ticker)
            if ticker_df is None or len(ticker_df) < 15:
                continue

            # Slice history up to scan date
            history = ticker_df.loc[ticker_df.index <= date]
            if len(history) < 15:
                continue

            # Limit lookback
            if len(history) > 220:
                history = history.iloc[-220:]

            # Quick volume filter from last rows before building full metrics
            hist_only = history.iloc[:-1] if history.index[-1] == date else history
            if len(hist_only) < 10:
                continue
            adv_window = min(20, len(hist_only))
            avg_vol = hist_only['volume'].iloc[-adv_window:].mean()
            if avg_vol < MIN_AVG_VOLUME:
                continue

            # Quick 3DGapFade pre-filter: check consecutive up days and gap
            # before computing full metrics (saves ~80% of compute time)
            if self.setup_type == '3DGapFade':
                # Check consecutive up closes at end of hist_only
                up_count = 0
                for i in range(len(hist_only) - 1, 0, -1):
                    if hist_only.iloc[i]['close'] > hist_only.iloc[i - 1]['close']:
                        up_count += 1
                    else:
                        break
                if up_count < 2:
                    continue

                # Check gap up on trade day
                if history.index[-1] == date:
                    today_open = history.iloc[-1]['open']
                    prior_close = hist_only.iloc[-1]['close']
                    if prior_close > 0 and today_open <= prior_close:
                        continue

            # Full metric computation
            metrics = self.compute_metrics_for_ticker(ticker, date, history)
            if metrics is None:
                continue

            # Classify
            setup_type = classify_reversal_setup(metrics)
            if setup_type is None:
                continue
            if self.setup_type and setup_type != self.setup_type:
                continue

            # Estimate cap
            cap = self.estimate_cap(
                metrics.get('current_price', 0),
                metrics.get('avg_daily_vol', 0),
            )

            # Score
            result = self.pretrade.validate(
                ticker=ticker, metrics=metrics,
                setup_type=setup_type, cap=cap,
            )

            grade = result.classification_details.get('grade', 'F')

            matches.append({
                'date': date,
                'ticker': ticker,
                'cap': cap,
                'setup_type': setup_type,
                'score': result.score,
                'grade': grade,
                'recommendation': result.recommendation,
                'pct_from_9ema': metrics.get('pct_from_9ema'),
                'prior_day_range_atr': metrics.get('prior_day_range_atr'),
                'rvol_score': metrics.get('rvol_score'),
                'consecutive_up_days': metrics.get('consecutive_up_days'),
                'gap_pct': metrics.get('gap_pct'),
                'pct_change_3': metrics.get('pct_change_3'),
                'pct_change_15': metrics.get('pct_change_15'),
                'pct_change_30': metrics.get('pct_change_30'),
                'pct_from_50mav': metrics.get('pct_from_50mav'),
                'current_price': metrics.get('current_price'),
                'atr_pct': metrics.get('atr_pct'),
            })

        return matches

    # ------------------------------------------------------------------
    # Scan a date range
    # ------------------------------------------------------------------

    def scan_range(self, start: str, end: str,
                   output_csv: Optional[str] = None) -> pd.DataFrame:
        """
        Scan all dates in range for matching setups.

        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            output_csv: Optional path to save results CSV

        Returns:
            DataFrame of all matches
        """
        scan_dates = self.get_trading_dates(start, end)
        logger.info(f"Scanning {len(scan_dates)} trading days for {self.setup_type} setups")

        all_matches = []

        for i, date in enumerate(scan_dates):
            matches = self.scan_date(date)
            all_matches.extend(matches)

            if (i + 1) % 10 == 0 or (i + 1) == len(scan_dates):
                go_count = sum(1 for m in matches if m['recommendation'] == 'GO')
                logger.info(
                    f"  [date {i + 1}/{len(scan_dates)}] {date} — "
                    f"{len(matches)} candidates ({go_count} GO)"
                )

        results_df = pd.DataFrame(all_matches)

        if output_csv and not results_df.empty:
            results_df.to_csv(output_csv, index=False)
            logger.info(f"Results saved to {output_csv}")

        # Print summary
        print("\n" + "=" * 80)
        print(f"HISTORICAL BACKSCANNER RESULTS: {self.setup_type}")
        print(f"Period: {start} to {end} ({len(scan_dates)} trading days)")
        print("=" * 80)

        if results_df.empty:
            print("  No matches found.")
            return results_df

        print(f"\nTotal matches: {len(results_df)}")
        print(f"\nBy recommendation:")
        print(results_df['recommendation'].value_counts().to_string())
        print(f"\nBy grade:")
        print(results_df['grade'].value_counts().to_string())

        # Top GO candidates
        go_results = results_df[results_df['recommendation'] == 'GO']
        if not go_results.empty:
            print(f"\nGO candidates ({len(go_results)}):")
            print("-" * 80)
            display_cols = ['date', 'ticker', 'cap', 'score', 'grade',
                            'pct_from_9ema', 'rvol_score', 'consecutive_up_days',
                            'gap_pct', 'current_price']
            available_cols = [c for c in display_cols if c in go_results.columns]
            print(go_results[available_cols].to_string(index=False))

        print()
        return results_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Historical backscanner for reversal setup types'
    )
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--setup', default='3DGapFade',
                        help='Setup type to scan for (default: 3DGapFade)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output CSV path (default: data/backscanner_results_{setup}_{start}_{end}.csv)')
    parser.add_argument('--lookback', type=int, default=310,
                        help='Calendar days of lookback for data fetch (default: 310)')

    args = parser.parse_args()

    # Compute fetch range: scan dates need lookback history
    fetch_start_dt = datetime.strptime(args.start, '%Y-%m-%d') - timedelta(days=args.lookback)
    fetch_start = fetch_start_dt.strftime('%Y-%m-%d')

    # Default output path
    if args.output is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'
        )
        args.output = os.path.join(
            data_dir,
            f"backscanner_results_{args.setup}_{args.start}_{args.end}.csv"
        )

    scanner = HistoricalBackscanner(setup_type=args.setup)

    # Phase 1: Fetch all market data
    print(f"\n{'=' * 80}")
    print(f"PHASE 1: Fetching market data")
    print(f"  Fetch range: {fetch_start} to {args.end}")
    print(f"  Scan range:  {args.start} to {args.end}")
    print(f"  Setup type:  {args.setup}")
    print(f"{'=' * 80}\n")

    t0 = time.time()
    scanner.fetch_market_data(fetch_start, args.end)
    fetch_time = time.time() - t0
    print(f"\nData fetch complete: {len(scanner.daily_data)} days in {fetch_time:.1f}s")

    # Phase 2: Scan
    print(f"\n{'=' * 80}")
    print(f"PHASE 2: Scanning for {args.setup} setups")
    print(f"{'=' * 80}\n")

    t1 = time.time()
    results = scanner.scan_range(args.start, args.end, output_csv=args.output)
    scan_time = time.time() - t1
    print(f"Scan complete: {len(results)} matches in {scan_time:.1f}s")
    print(f"Total time: {fetch_time + scan_time:.1f}s")


if __name__ == '__main__':
    main()
