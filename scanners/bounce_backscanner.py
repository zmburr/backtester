"""
Historical Bounce Backscanner

Scans historical market data across ALL tickers for capitulation bounce setups.
Uses Polygon's get_grouped_daily_aggs() endpoint for massive efficiency:
1 API call per date returns OHLCV for all ~12,000 tickers.

Data pipeline:
  1. fetch_market_data(start, end) — bulk fetch grouped daily bars per date
  2. Build per-ticker DataFrames from bulk data
  3. Pre-filter: consecutive down days, gap down, discount from high
  4. compute_bounce_metrics(ticker, date) — compute selloff_total_pct,
     consecutive_down_days, pct_off_30d_high, gap_pct, etc. locally (zero API calls)
  5. classify_stock() + BouncePretrade.validate() from bounce_scorer.py
  6. Filter by score/grade/recommendation
  7. Output CSV + optional verification against bounce_data.csv

Efficiency: ~560 API calls for 1 year + 310-day lookback vs 250,000 per-ticker calls.

Usage:
    python -m scanners.bounce_backscanner --start 2024-01-01 --end 2024-12-31
    python -m scanners.bounce_backscanner --start 2024-08-05 --end 2024-08-05
    python -m scanners.bounce_backscanner --start 2024-01-01 --end 2024-12-31 --verify
    python -m scanners.bounce_backscanner --start 2024-01-01 --end 2024-12-31 --cap Large,ETF,Medium
"""

import os
import sys
import argparse
import logging
import time
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from polygon.rest import RESTClient
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzers.bounce_scorer import BouncePretrade, classify_stock
from scanners.bounce_trader import KNOWN_ETFS

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# Minimum filters to reduce ticker universe
MIN_PRICE = 5.0
MIN_AVG_VOLUME = 500_000


class BounceBackscanner:
    """
    Scans historical market data across all tickers for capitulation bounce setups.

    Uses Polygon grouped daily endpoint (1 call per date) for bulk data,
    then computes all metrics locally and scores via BouncePretrade.validate().
    """

    def __init__(self, cache_dir: str = None,
                 cap_filter: Optional[List[str]] = None,
                 min_score: int = 3,
                 min_bounce: float = 0.0):
        self.cap_filter = cap_filter
        self.min_score = min_score
        self.min_bounce = min_bounce
        self.pretrade = BouncePretrade()
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
    # Bulk data fetching (identical to historical_backscanner.py)
    # ------------------------------------------------------------------

    def fetch_market_data(self, start: str, end: str) -> None:
        """
        Fetch grouped daily bars for each trading day in range.
        Caches to pickle file for resume capability.
        """
        cache_file = os.path.join(
            self.cache_dir,
            f"grouped_daily_{start}_{end}.pkl"
        )

        if os.path.exists(cache_file):
            logger.info(f"Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.daily_data = pickle.load(f)
            self._ticker_index = None
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

            time.sleep(0.15)

        self._ticker_index = None
        logger.info(f"Saving cache to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(self.daily_data, f)

    def _ensure_ticker_index(self) -> None:
        """
        Pre-build a ticker-indexed lookup from the date-grouped daily_data.
        Converts {date -> all_tickers_df} into {ticker -> date-indexed OHLCV df}.
        """
        if self._ticker_index is not None:
            return

        logger.info("Building ticker index from grouped daily data...")
        self._sorted_dates = sorted(self.daily_data.keys())

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

        self._ticker_index = {}
        for ticker, group in all_data.groupby('ticker'):
            group_sorted = group.sort_values('date').set_index('date')
            self._ticker_index[ticker] = group_sorted

        logger.info(f"Ticker index built: {len(self._ticker_index)} tickers")

    # ------------------------------------------------------------------
    # Market cap estimation with ETF detection
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_cap(ticker: str, price: float, avg_vol: float) -> str:
        """
        Estimate market cap from ticker, price, and average volume.
        Checks KNOWN_ETFS set first, then uses dollar-volume heuristic.
        """
        if ticker.upper() in KNOWN_ETFS:
            return 'ETF'

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
    # Bounce metric computation (all local, zero API calls)
    # ------------------------------------------------------------------

    def compute_bounce_metrics(self, ticker: str, date: str,
                               history: pd.DataFrame) -> Optional[Dict]:
        """
        Compute bounce screening metrics from local historical data.

        Args:
            ticker: Stock ticker
            date: The scan date (potential bounce day)
            history: DataFrame with OHLCV up to and including `date`

        Returns:
            Dict of metrics compatible with BouncePretrade.validate(), or None
        """
        if history is None or len(history) < 15:
            return None

        metrics = {'ticker': ticker, 'date': date}

        # Separate today (bounce day) from completed history
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

        # --- Moving averages (for classification) ---
        if len(closes) >= 50:
            sma_50 = closes.rolling(50).mean().iloc[-1]
            if not pd.isna(sma_50) and sma_50 != 0:
                metrics['pct_from_50mav'] = (current_close - sma_50) / sma_50

        if len(closes) >= 200:
            sma_200 = closes.rolling(200).mean().iloc[-1]
            if not pd.isna(sma_200) and sma_200 != 0:
                metrics['pct_from_200mav'] = (current_close - sma_200) / sma_200

        # --- ATR (14-day) ---
        if len(hist) < 2:
            return None

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

        # --- Consecutive down days (close < prior_close) ---
        consecutive_down = 0
        for i in range(len(hist) - 1, 0, -1):
            if hist.iloc[i]['close'] < hist.iloc[i - 1]['close']:
                consecutive_down += 1
            else:
                break
        metrics['consecutive_down_days'] = consecutive_down

        # --- Selloff total pct ---
        if consecutive_down > 0:
            selloff_start_idx = len(hist) - consecutive_down
            if selloff_start_idx >= 0 and selloff_start_idx < len(hist):
                first_open = hist.iloc[selloff_start_idx]['open']
                metrics['selloff_total_pct'] = (current_close - first_open) / first_open if first_open != 0 else 0
            else:
                metrics['selloff_total_pct'] = 0.0
        else:
            metrics['selloff_total_pct'] = 0.0

        # --- RVOL (prior day volume vs 20d avg) ---
        adv_window = min(20, len(hist))
        adv = hist['volume'].rolling(window=adv_window, min_periods=1).mean().iloc[-1]
        prior_day_vol = hist.iloc[-1]['volume']
        metrics['avg_daily_vol'] = adv
        metrics['prior_day_rvol'] = prior_day_vol / adv if adv > 0 else 0

        # Premarket RVOL not available from daily aggs
        metrics['premarket_rvol'] = None

        # --- Pct off 30d high ---
        window_30 = hist.iloc[-30:] if len(hist) >= 30 else hist
        high_30d = window_30['high'].max()
        if high_30d and high_30d != 0:
            metrics['pct_off_30d_high'] = (current_close - high_30d) / high_30d
        else:
            metrics['pct_off_30d_high'] = None

        # --- Pct off 52wk high ---
        high_52wk = hist['high'].max()
        if high_52wk and high_52wk != 0:
            metrics['pct_off_52wk_high'] = (current_close - high_52wk) / high_52wk
        else:
            metrics['pct_off_52wk_high'] = 0

        # --- Gap pct (trade day open vs prior close) ---
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

        # --- Bounce day metrics (trade day OHLC) ---
        if today_row is not None:
            day_open = today_row['open']
            day_high = today_row['high']
            day_low = today_row['low']
            day_close = today_row['close']

            metrics['bounce_day_return'] = (day_close - day_open) / day_open if day_open > 0 else 0
            metrics['bounce_low_to_close'] = (day_close - day_low) / day_low if day_low > 0 else 0
            day_range = day_high - day_low
            metrics['bounce_day_close_position'] = (day_close - day_low) / day_range if day_range > 0 else 0.5

        return metrics

    # ------------------------------------------------------------------
    # Pre-filters (fast disqualification before full metrics)
    # ------------------------------------------------------------------

    def _passes_prefilter(self, ticker: str, date: str,
                          ticker_df: pd.DataFrame) -> bool:
        """
        Quick bounce pre-filters using raw OHLCV. Eliminates ~95% of tickers.

        Checks:
        1. Consecutive down days >= 2
        2. Gap down on scan day (today_open < prior_close)
        3. Minimum 5% off 30-day high (loosest threshold across all caps)
        """
        # Slice history up to scan date
        history = ticker_df.loc[ticker_df.index <= date]
        if len(history) < 15:
            return False

        has_today = (history.index[-1] == date)
        if not has_today:
            return False

        hist = history.iloc[:-1]
        today_row = history.iloc[-1]

        if len(hist) < 10:
            return False

        # 1. Consecutive down days >= 2
        down_count = 0
        for i in range(len(hist) - 1, 0, -1):
            if hist.iloc[i]['close'] < hist.iloc[i - 1]['close']:
                down_count += 1
            else:
                break
        if down_count < 2:
            return False

        # 2. Gap down on scan day
        prior_close = hist.iloc[-1]['close']
        if prior_close <= 0 or today_row['open'] >= prior_close:
            return False

        # 3. At least 5% off 30-day high
        window_30 = hist.iloc[-30:] if len(hist) >= 30 else hist
        high_30d = window_30['high'].max()
        if high_30d > 0:
            pct_off = (prior_close - high_30d) / high_30d
            if pct_off > -0.05:
                return False

        return True

    # ------------------------------------------------------------------
    # Scan a single date
    # ------------------------------------------------------------------

    def scan_date(self, date: str) -> List[Dict]:
        """
        Screen all tickers on a single date for bounce setups.

        Args:
            date: Trading date (YYYY-MM-DD)

        Returns:
            List of dicts for matched setups
        """
        self._ensure_ticker_index()

        day_df = self.daily_data.get(date)
        if day_df is None or day_df.empty:
            return []

        # Quick pre-filter: price > MIN_PRICE, no special chars in ticker
        candidates = day_df[
            (day_df['close'] > MIN_PRICE) &
            (~day_df['ticker'].str.contains(r'[.\-/]', regex=True, na=False))
        ]

        matches = []

        for ticker in candidates['ticker'].values:
            ticker_df = self._ticker_index.get(ticker)
            if ticker_df is None or len(ticker_df) < 15:
                continue

            # Quick volume filter
            hist_slice = ticker_df.loc[ticker_df.index < date]
            if len(hist_slice) < 10:
                continue
            adv_window = min(20, len(hist_slice))
            avg_vol = hist_slice['volume'].iloc[-adv_window:].mean()
            if avg_vol < MIN_AVG_VOLUME:
                continue

            # Bounce pre-filters (fast disqualification)
            if not self._passes_prefilter(ticker, date, ticker_df):
                continue

            # Build full history for metric computation
            history = ticker_df.loc[ticker_df.index <= date]
            if len(history) > 310:
                history = history.iloc[-310:]

            # Full metric computation
            metrics = self.compute_bounce_metrics(ticker, date, history)
            if metrics is None:
                continue

            # Estimate cap
            cap = self.estimate_cap(
                ticker,
                metrics.get('current_price', 0),
                metrics.get('avg_daily_vol', 0),
            )

            # Cap filter
            if self.cap_filter and cap not in self.cap_filter:
                continue

            # Score via BouncePretrade
            result = self.pretrade.validate(
                ticker=ticker,
                metrics=metrics,
                cap=cap,
            )

            # Min score filter
            if result.score < self.min_score:
                continue

            # Confirmed bounce filter: low-to-close must exceed threshold
            bounce_ltc = metrics.get('bounce_low_to_close')
            if self.min_bounce > 0 and (bounce_ltc is None or bounce_ltc < self.min_bounce):
                continue

            grade = result.classification_details.get('grade', '')
            if not grade:
                # Derive grade from score (7 pre-trade criteria in V2)
                if result.score >= 7:
                    grade = 'A+'
                elif result.score >= 6:
                    grade = 'A'
                elif result.score == 5:
                    grade = 'B'
                elif result.score == 4:
                    grade = 'C'
                else:
                    grade = 'F'

            matches.append({
                'date': date,
                'ticker': ticker,
                'cap': cap,
                'setup_type': result.setup_type,
                'score': result.score,
                'grade': grade,
                'recommendation': result.recommendation,
                'selloff_total_pct': metrics.get('selloff_total_pct'),
                'consecutive_down_days': metrics.get('consecutive_down_days'),
                'pct_off_30d_high': metrics.get('pct_off_30d_high'),
                'gap_pct': metrics.get('gap_pct'),
                'prior_day_range_atr': metrics.get('prior_day_range_atr'),
                'prior_day_rvol': metrics.get('prior_day_rvol'),
                'pct_from_200mav': metrics.get('pct_from_200mav'),
                'pct_from_50mav': metrics.get('pct_from_50mav'),
                'pct_change_3': metrics.get('pct_change_3'),
                'pct_change_15': metrics.get('pct_change_15'),
                'pct_change_30': metrics.get('pct_change_30'),
                'pct_off_52wk_high': metrics.get('pct_off_52wk_high'),
                'current_price': metrics.get('current_price'),
                'atr_pct': metrics.get('atr_pct'),
                'bounce_day_return': metrics.get('bounce_day_return'),
                'bounce_low_to_close': metrics.get('bounce_low_to_close'),
                'bounce_day_close_position': metrics.get('bounce_day_close_position'),
            })

        return matches

    # ------------------------------------------------------------------
    # Scan a date range
    # ------------------------------------------------------------------

    def scan_range(self, start: str, end: str,
                   output_csv: Optional[str] = None) -> pd.DataFrame:
        """
        Scan all dates in range for bounce setups.

        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            output_csv: Optional path to save results CSV

        Returns:
            DataFrame of all matches
        """
        scan_dates = self.get_trading_dates(start, end)
        logger.info(f"Scanning {len(scan_dates)} trading days for bounce setups")

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

        # Dedupe on ticker+date (keep first/highest-scored occurrence)
        if not results_df.empty:
            before = len(results_df)
            results_df = results_df.sort_values('score', ascending=False)
            results_df = results_df.drop_duplicates(subset=['ticker', 'date'], keep='first')
            results_df = results_df.sort_values('date').reset_index(drop=True)
            dupes_removed = before - len(results_df)
            if dupes_removed > 0:
                logger.info(f"Removed {dupes_removed} duplicate ticker+date rows (kept highest score)")

        if output_csv and not results_df.empty:
            results_df.to_csv(output_csv, index=False)
            logger.info(f"Results saved to {output_csv}")

        # Print summary
        print("\n" + "=" * 80)
        print(f"BOUNCE BACKSCANNER RESULTS")
        print(f"Period: {start} to {end} ({len(scan_dates)} trading days)")
        print(f"Min score: {self.min_score}")
        if self.min_bounce > 0:
            print(f"Min bounce (low-to-close): {self.min_bounce*100:.0f}%")
        if self.cap_filter:
            print(f"Cap filter: {', '.join(self.cap_filter)}")
        print("=" * 80)

        if results_df.empty:
            print("  No matches found.")
            return results_df

        print(f"\nTotal matches: {len(results_df)}")
        print(f"\nBy recommendation:")
        print(results_df['recommendation'].value_counts().to_string())
        print(f"\nBy grade:")
        print(results_df['grade'].value_counts().to_string())
        print(f"\nBy setup type:")
        print(results_df['setup_type'].value_counts().to_string())
        if 'cap' in results_df.columns:
            print(f"\nBy cap:")
            print(results_df['cap'].value_counts().to_string())

        # Top GO candidates
        go_results = results_df[results_df['recommendation'] == 'GO']
        if not go_results.empty:
            print(f"\nGO candidates ({len(go_results)}):")
            print("-" * 80)
            display_cols = ['date', 'ticker', 'cap', 'setup_type', 'score', 'grade',
                            'selloff_total_pct', 'consecutive_down_days',
                            'gap_pct', 'pct_off_30d_high', 'current_price']
            available_cols = [c for c in display_cols if c in go_results.columns]
            with pd.option_context('display.max_rows', None, 'display.width', 120):
                print(go_results[available_cols].to_string(index=False))

        # Dates with most candidates (crash days)
        date_counts = results_df.groupby('date').size().sort_values(ascending=False)
        if len(date_counts) > 0:
            print(f"\nTop dates by candidate count:")
            print("-" * 40)
            for date_str, count in date_counts.head(10).items():
                go_on_date = len(results_df[(results_df['date'] == date_str) &
                                            (results_df['recommendation'] == 'GO')])
                print(f"  {date_str}: {count} candidates ({go_on_date} GO)")

        print()
        return results_df

    # ------------------------------------------------------------------
    # Verification against bounce_data.csv
    # ------------------------------------------------------------------

    def verify_against_known_trades(self, results_df: pd.DataFrame,
                                    verify_csv: str, start: str, end: str):
        """
        Check how many known (date, ticker) pairs from bounce_data.csv
        the scanner catches. Reports recall by grade and lists missed trades.
        """
        if results_df.empty:
            print("\nNo scanner results to verify against.")
            return

        try:
            known_df = pd.read_csv(verify_csv)
        except Exception as e:
            print(f"\nCould not load verification CSV: {e}")
            return

        # Convert dates from M/D/YYYY to YYYY-MM-DD
        known_df['date_normalized'] = pd.to_datetime(
            known_df['date'], format='mixed'
        ).dt.strftime('%Y-%m-%d')

        # Filter to scan range
        known_in_range = known_df[
            (known_df['date_normalized'] >= start) &
            (known_df['date_normalized'] <= end)
        ].copy()

        if known_in_range.empty:
            print(f"\nNo known bounce trades in {start} to {end}.")
            return

        print("\n" + "=" * 80)
        print(f"VERIFICATION: Scanner vs bounce_data.csv")
        print(f"Known trades in range: {len(known_in_range)}")
        print("=" * 80)

        # Build set of (date, ticker) from scanner results
        scanner_pairs = set(
            zip(results_df['date'], results_df['ticker'].str.upper())
        )

        # Check each known trade
        caught = []
        missed = []

        for _, row in known_in_range.iterrows():
            date_str = row['date_normalized']
            ticker = str(row['ticker']).upper()
            grade = row.get('trade_grade', '?')
            cap = row.get('cap', '?')
            setup = row.get('Setup', '?')

            if (date_str, ticker) in scanner_pairs:
                # Find the scanner result for this pair
                match = results_df[
                    (results_df['date'] == date_str) &
                    (results_df['ticker'].str.upper() == ticker)
                ]
                scanner_score = match.iloc[0]['score'] if not match.empty else '?'
                scanner_rec = match.iloc[0]['recommendation'] if not match.empty else '?'
                caught.append({
                    'date': date_str, 'ticker': ticker, 'grade': grade,
                    'cap': cap, 'setup': setup,
                    'scanner_score': scanner_score, 'scanner_rec': scanner_rec,
                })
            else:
                missed.append({
                    'date': date_str, 'ticker': ticker, 'grade': grade,
                    'cap': cap, 'setup': setup,
                })

        total = len(known_in_range)
        recall = len(caught) / total * 100 if total > 0 else 0

        print(f"\nOverall recall: {len(caught)}/{total} ({recall:.0f}%)")

        # Recall by grade
        print(f"\nRecall by trade grade:")
        print("-" * 50)
        for grade_val in ['A', 'B', 'C']:
            grade_total = len(known_in_range[known_in_range['trade_grade'] == grade_val])
            grade_caught = len([c for c in caught if c['grade'] == grade_val])
            grade_recall = grade_caught / grade_total * 100 if grade_total > 0 else 0
            print(f"  Grade {grade_val}: {grade_caught}/{grade_total} ({grade_recall:.0f}%)")

        # Caught trades
        if caught:
            print(f"\nCAUGHT ({len(caught)}):")
            print("-" * 70)
            print(f"  {'Date':<12} {'Ticker':<8} {'Grade':<6} {'Cap':<8} {'Scanner Score':<14} {'Rec':<10}")
            for c in caught:
                print(f"  {c['date']:<12} {c['ticker']:<8} {c['grade']:<6} {c['cap']:<8} "
                      f"{c['scanner_score']:<14} {c['scanner_rec']:<10}")

        # Missed trades
        if missed:
            print(f"\nMISSED ({len(missed)}):")
            print("-" * 70)
            print(f"  {'Date':<12} {'Ticker':<8} {'Grade':<6} {'Cap':<8} {'Setup':<20}")
            for m in missed:
                print(f"  {m['date']:<12} {m['ticker']:<8} {m['grade']:<6} {m['cap']:<8} {m['setup']:<20}")

            # Diagnose why trades were missed
            print(f"\nMISS DIAGNOSIS:")
            print("-" * 70)
            self._ensure_ticker_index()
            for m in missed:
                reasons = self._diagnose_miss(m['date'], m['ticker'], m['cap'])
                print(f"  {m['date']} {m['ticker']}: {reasons}")

        print()

    def _diagnose_miss(self, date: str, ticker: str, known_cap: str) -> str:
        """Diagnose why a known trade was missed by the scanner."""
        reasons = []

        ticker_df = self._ticker_index.get(ticker) if self._ticker_index else None
        if ticker_df is None:
            return "Not in ticker index (may not have been in Polygon data)"

        history = ticker_df.loc[ticker_df.index <= date]
        if len(history) < 15:
            return f"Insufficient history ({len(history)} bars)"

        has_today = (history.index[-1] == date)
        if not has_today:
            return "No data on scan date"

        hist = history.iloc[:-1]
        today_row = history.iloc[-1]

        if len(hist) < 10:
            return f"Insufficient prior history ({len(hist)} bars)"

        # Check price
        if hist.iloc[-1]['close'] <= MIN_PRICE:
            reasons.append(f"price ${hist.iloc[-1]['close']:.2f} < ${MIN_PRICE}")

        # Check volume
        adv_window = min(20, len(hist))
        avg_vol = hist['volume'].iloc[-adv_window:].mean()
        if avg_vol < MIN_AVG_VOLUME:
            reasons.append(f"avg vol {avg_vol:,.0f} < {MIN_AVG_VOLUME:,}")

        # Check consecutive down days
        down_count = 0
        for i in range(len(hist) - 1, 0, -1):
            if hist.iloc[i]['close'] < hist.iloc[i - 1]['close']:
                down_count += 1
            else:
                break
        if down_count < 2:
            reasons.append(f"only {down_count} down days (need >= 2)")

        # Check gap
        prior_close = hist.iloc[-1]['close']
        if prior_close > 0 and today_row['open'] >= prior_close:
            gap = (today_row['open'] - prior_close) / prior_close
            reasons.append(f"no gap down (gap={gap*100:+.1f}%)")

        # Check 30d high discount
        window_30 = hist.iloc[-30:] if len(hist) >= 30 else hist
        high_30d = window_30['high'].max()
        if high_30d > 0:
            pct_off = (prior_close - high_30d) / high_30d
            if pct_off > -0.05:
                reasons.append(f"only {pct_off*100:.1f}% off 30d high (need <= -5%)")

        # Check cap filter
        cap = self.estimate_cap(ticker, hist.iloc[-1]['close'], avg_vol)
        if self.cap_filter and cap not in self.cap_filter:
            reasons.append(f"cap={cap} not in filter {self.cap_filter}")

        # Check score
        if not reasons:
            metrics = self.compute_bounce_metrics(ticker, date, history.iloc[-310:] if len(history) > 310 else history)
            if metrics:
                result = self.pretrade.validate(ticker=ticker, metrics=metrics, cap=cap)
                if result.score < self.min_score:
                    failed = [i.name for i in result.items if not i.passed]
                    reasons.append(f"score {result.score}/{result.max_score} < min {self.min_score} "
                                   f"(failed: {', '.join(failed)})")
                elif self.min_bounce > 0:
                    bounce_ltc = metrics.get('bounce_low_to_close')
                    if bounce_ltc is None or bounce_ltc < self.min_bounce:
                        reasons.append(f"bounce low-to-close {bounce_ltc*100:.1f}% < min {self.min_bounce*100:.0f}%"
                                       if bounce_ltc is not None else "no bounce day data")
                    else:
                        reasons.append("passed all checks — should have been caught (bug?)")
                else:
                    reasons.append("passed all checks — should have been caught (bug?)")

        return "; ".join(reasons) if reasons else "unknown"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Historical backscanner for capitulation bounce setups'
    )
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output CSV path (default: data/bounce_backscanner_{start}_{end}.csv)')
    parser.add_argument('--lookback', type=int, default=310,
                        help='Calendar days of lookback for data fetch (default: 310)')
    parser.add_argument('--cap', default='Large,ETF,Medium',
                        help='Filter to specific cap(s), comma-separated (default: "Large,ETF,Medium")')
    parser.add_argument('--min-score', type=int, default=3,
                        help='Minimum score to include in results (default: 3)')
    parser.add_argument('--min-bounce', type=float, default=2.0,
                        help='Minimum low-to-close bounce %% to confirm setup (e.g. 2 for 2%%). Default: 2')
    parser.add_argument('--verify', action='store_true',
                        help='Verify results against bounce_data.csv')
    parser.add_argument('--verify-csv', default=None,
                        help='Custom CSV path for verification (default: data/bounce_data.csv)')

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
            f"bounce_backscanner_{args.start}_{args.end}.csv"
        )

    cap_filter = [c.strip() for c in args.cap.split(',')] if args.cap else None
    # Convert min_bounce from percentage to decimal (e.g. 2 -> 0.02)
    min_bounce = args.min_bounce / 100.0 if args.min_bounce > 0 else 0.0
    scanner = BounceBackscanner(cap_filter=cap_filter, min_score=args.min_score,
                                min_bounce=min_bounce)

    # Phase 1: Fetch all market data
    print(f"\n{'=' * 80}")
    print(f"PHASE 1: Fetching market data")
    print(f"  Fetch range: {fetch_start} to {args.end}")
    print(f"  Scan range:  {args.start} to {args.end}")
    print(f"  Strategy:    Capitulation Bounce")
    if cap_filter:
        print(f"  Cap filter:  {', '.join(cap_filter)}")
    print(f"  Min score:   {args.min_score}")
    if min_bounce > 0:
        print(f"  Min bounce:  {args.min_bounce:.0f}% (low-to-close)")
    print(f"{'=' * 80}\n")

    t0 = time.time()
    scanner.fetch_market_data(fetch_start, args.end)
    fetch_time = time.time() - t0
    print(f"\nData fetch complete: {len(scanner.daily_data)} days in {fetch_time:.1f}s")

    # Phase 2: Scan
    print(f"\n{'=' * 80}")
    print(f"PHASE 2: Scanning for bounce setups")
    print(f"{'=' * 80}\n")

    t1 = time.time()
    results = scanner.scan_range(args.start, args.end, output_csv=args.output)
    scan_time = time.time() - t1
    print(f"Scan complete: {len(results)} matches in {scan_time:.1f}s")
    print(f"Total time: {fetch_time + scan_time:.1f}s")

    # Phase 3: Verification (optional)
    if args.verify or args.verify_csv:
        verify_csv = args.verify_csv
        if verify_csv is None:
            verify_csv = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data', 'bounce_data.csv'
            )
        scanner.verify_against_known_trades(results, verify_csv, args.start, args.end)


if __name__ == '__main__':
    main()
