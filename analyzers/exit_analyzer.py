"""
Exit Strategy Analyzer for Parabolic Short Reversals

Phase 1: Data Collection & MFE/MAE Analysis
- Fetches 1-min intraday data for historical Grade A trades
- Calculates Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE)
- Tracks ATR multiple hit rates
- Analyzes technical level targets
- Measures time-based exit effectiveness

Usage:
    from analyzers.exit_analyzer import ExitAnalyzer
    analyzer = ExitAnalyzer()
    results = analyzer.analyze_all_trades()
    analyzer.save_results()
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import json
import os
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / 'data'

# Ensure project root is on sys.path for imports
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from data_queries.polygon_queries import get_levels_data, get_atr, get_daily

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class ExitMetrics:
    """Comprehensive exit metrics for a single trade."""
    # Trade identification
    ticker: str
    date: str
    cap: str
    trade_grade: str

    # Entry point (using open price as entry for shorts)
    entry_price: float
    entry_time: str

    # MFE/MAE (for shorts: MFE is how low it went, MAE is how high before low)
    mfe_price: float  # Lowest price reached (best for shorts)
    mfe_pct: float    # MFE as % from entry
    mfe_time: str     # Time of MFE
    mfe_minutes: int  # Minutes from entry to MFE

    mae_price: float  # Highest price before MFE (worst for shorts before best)
    mae_pct: float    # MAE as % from entry (adverse move)
    mae_time: str     # Time of MAE

    # Close metrics
    close_price: float
    close_pct: float  # Actual captured (open to close)

    # Efficiency
    capture_efficiency: float  # close_pct / mfe_pct (how much of available move captured)

    # ATR-based targets (did price reach these levels?)
    atr: float
    hit_1x_atr: bool
    hit_1_5x_atr: bool
    hit_2x_atr: bool
    hit_2_5x_atr: bool
    hit_3x_atr: bool

    time_to_1x_atr: Optional[int]   # Minutes to reach 1x ATR
    time_to_1_5x_atr: Optional[int]
    time_to_2x_atr: Optional[int]

    # Technical level targets
    prior_day_low: Optional[float]
    hit_prior_day_low: bool
    time_to_prior_day_low: Optional[int]

    prior_day_close: Optional[float]
    hit_prior_day_close: bool

    vwap_at_entry: Optional[float]
    hit_vwap: bool

    ema_9_at_entry: Optional[float]
    hit_9ema: bool
    time_to_9ema: Optional[int]

    # Time-based analysis
    pnl_at_10am: Optional[float]
    pnl_at_1030am: Optional[float]
    pnl_at_11am: Optional[float]
    pnl_at_1130am: Optional[float]
    pnl_at_12pm: Optional[float]
    pnl_at_1pm: Optional[float]
    pnl_at_2pm: Optional[float]
    pnl_at_close: float

    # Pullback analysis (after hitting MFE)
    max_giveback_pct: float  # Max pullback from MFE before close

    # Volume analysis
    volume_at_low_bucket: Optional[str]  # Which volume quintile was the low in
    cumulative_vol_at_low_pct: Optional[float]  # % of daily volume at time of low


class ExitAnalyzer:
    """Analyzes historical trades to optimize exit strategies."""

    def __init__(self, data_path: str = None):
        self.data_path = data_path or str(_DATA_DIR / 'reversal_data.csv')
        self.results: List[ExitMetrics] = []
        self.df = None

    def load_trades(self, grade: str = 'A') -> pd.DataFrame:
        """Load trades from CSV, optionally filtering by grade."""
        df = pd.read_csv(self.data_path)
        if grade:
            df = df[df['trade_grade'] == grade].copy()

        # Convert date format
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')
        self.df = df
        logging.info(f"Loaded {len(df)} Grade {grade} trades")
        return df

    def _get_intraday_data(self, ticker: str, date: str) -> Optional[pd.DataFrame]:
        """Fetch 1-minute intraday data for a specific date."""
        try:
            # Get intraday bars
            df = get_levels_data(ticker, date, window=1, multiplier=1, timespan='minute')
            if df is None or len(df) == 0:
                logging.warning(f"No intraday data for {ticker} on {date}")
                return None

            # Filter to market hours (9:30 AM - 4:00 PM ET)
            df = df.reset_index()
            if 'close-time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['close-time'])
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                logging.warning(f"No timestamp column found for {ticker}")
                return None

            # Filter to trading day
            target_date = pd.to_datetime(date).date()
            df = df[df['timestamp'].dt.date == target_date]

            # Filter to market hours
            df = df[(df['timestamp'].dt.hour >= 9) |
                    ((df['timestamp'].dt.hour == 9) & (df['timestamp'].dt.minute >= 30))]
            df = df[df['timestamp'].dt.hour < 16]

            if len(df) == 0:
                logging.warning(f"No market hours data for {ticker} on {date}")
                return None

            return df.sort_values('timestamp').reset_index(drop=True)

        except Exception as e:
            logging.error(f"Error fetching intraday data for {ticker} on {date}: {e}")
            return None

    def _get_prior_day_data(self, ticker: str, date: str) -> Dict:
        """Get prior day's OHLC data."""
        try:
            # Get a few days of daily data to find prior day
            df = get_levels_data(ticker, date, window=5, multiplier=1, timespan='day')
            if df is None or len(df) < 2:
                return {}

            df = df.reset_index()
            target_date = pd.to_datetime(date).date()

            # Find the row before target date
            if 'close-time' in df.columns:
                df['date'] = pd.to_datetime(df['close-time']).dt.date
            else:
                # Handle case where index is datetime
                df['date'] = pd.to_datetime(df.iloc[:, 0]).dt.date

            prior_rows = df[df['date'] < target_date]

            if len(prior_rows) == 0:
                return {}

            prior = prior_rows.iloc[-1]
            return {
                'prior_high': prior['high'],
                'prior_low': prior['low'],
                'prior_close': prior['close'],
                'prior_open': prior['open']
            }
        except Exception as e:
            logging.warning(f"Error getting prior day data for {ticker}: {e}")
            return {}

    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate running VWAP from intraday data."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cumulative_tp_vol = (typical_price * df['volume']).cumsum()
        cumulative_vol = df['volume'].cumsum()
        return cumulative_tp_vol / cumulative_vol

    def _calculate_ema(self, prices: pd.Series, span: int = 9) -> float:
        """Calculate EMA at start of day using prior closes."""
        return prices.ewm(span=span, adjust=False).mean().iloc[-1]

    def analyze_trade(self, row: pd.Series) -> Optional[ExitMetrics]:
        """Analyze a single trade and calculate all exit metrics."""
        ticker = row['ticker']
        date = row['date']
        cap = row.get('cap', 'Medium')
        grade = row.get('trade_grade', 'A')

        logging.info(f"Analyzing {ticker} on {date}")

        # Get intraday data
        intraday = self._get_intraday_data(ticker, date)
        if intraday is None or len(intraday) < 10:
            logging.warning(f"Insufficient intraday data for {ticker} on {date}")
            return None

        # Get ATR
        atr = get_atr(ticker, date)
        if atr is None or atr == 0:
            atr = (intraday['high'].max() - intraday['low'].min()) / 3  # Rough estimate

        # Get prior day data
        prior_day = self._get_prior_day_data(ticker, date)

        # Entry is at open (for short reversals)
        entry_price = intraday['open'].iloc[0]
        entry_time = intraday['timestamp'].iloc[0]

        # Calculate VWAP
        intraday['vwap'] = self._calculate_vwap(intraday)
        vwap_at_entry = intraday['vwap'].iloc[0] if len(intraday) > 0 else None

        # For shorts: MFE is the LOW (best exit), MAE is the HIGH before low
        # Find the lowest point of the day
        mfe_idx = intraday['low'].idxmin()
        mfe_price = intraday['low'].iloc[mfe_idx]
        mfe_time = intraday['timestamp'].iloc[mfe_idx]
        mfe_pct = (entry_price - mfe_price) / entry_price  # Positive = profit for short
        mfe_minutes = int((mfe_time - entry_time).total_seconds() / 60)

        # MAE: highest price BEFORE the MFE (adverse excursion before best point)
        if mfe_idx > 0:
            pre_mfe = intraday.iloc[:mfe_idx+1]
            mae_idx = pre_mfe['high'].idxmax()
            mae_price = pre_mfe['high'].iloc[mae_idx]
            mae_time = pre_mfe['timestamp'].iloc[mae_idx]
            mae_pct = (mae_price - entry_price) / entry_price  # Positive = loss for short
        else:
            mae_price = entry_price
            mae_time = entry_time
            mae_pct = 0.0

        # Close metrics
        close_price = intraday['close'].iloc[-1]
        close_pct = (entry_price - close_price) / entry_price

        # Capture efficiency
        capture_efficiency = close_pct / mfe_pct if mfe_pct > 0 else 0

        # ATR-based targets (price levels below entry for shorts)
        atr_targets = {
            '1x': entry_price - (1.0 * atr),
            '1.5x': entry_price - (1.5 * atr),
            '2x': entry_price - (2.0 * atr),
            '2.5x': entry_price - (2.5 * atr),
            '3x': entry_price - (3.0 * atr),
        }

        def time_to_target(target_price: float) -> Optional[int]:
            """Find minutes to reach a target price (for shorts, looking for lows)."""
            hits = intraday[intraday['low'] <= target_price]
            if len(hits) > 0:
                hit_time = hits['timestamp'].iloc[0]
                return int((hit_time - entry_time).total_seconds() / 60)
            return None

        hit_1x_atr = mfe_price <= atr_targets['1x']
        hit_1_5x_atr = mfe_price <= atr_targets['1.5x']
        hit_2x_atr = mfe_price <= atr_targets['2x']
        hit_2_5x_atr = mfe_price <= atr_targets['2.5x']
        hit_3x_atr = mfe_price <= atr_targets['3x']

        time_to_1x = time_to_target(atr_targets['1x'])
        time_to_1_5x = time_to_target(atr_targets['1.5x'])
        time_to_2x = time_to_target(atr_targets['2x'])

        # Technical level targets
        prior_day_low = prior_day.get('prior_low')
        prior_day_close = prior_day.get('prior_close')

        hit_prior_day_low = mfe_price <= prior_day_low if prior_day_low else False
        time_to_pdl = time_to_target(prior_day_low) if prior_day_low else None

        hit_prior_day_close = mfe_price <= prior_day_close if prior_day_close else False

        # VWAP hit (for shorts, hitting below VWAP)
        hit_vwap = False
        if vwap_at_entry:
            # Check if price went below opening VWAP at any point
            hit_vwap = intraday['low'].min() < vwap_at_entry

        # 9 EMA (would need historical data to calculate properly)
        # For now, approximate using prior day close adjusted
        ema_9_at_entry = None
        hit_9ema = False
        time_to_9ema = None

        # Time-based P&L snapshots
        def pnl_at_time(hour: int, minute: int = 0) -> Optional[float]:
            """Get P&L at specific time."""
            target_time = entry_time.replace(hour=hour, minute=minute)
            mask = intraday['timestamp'] <= target_time
            if mask.any():
                price_at_time = intraday.loc[mask, 'close'].iloc[-1]
                return (entry_price - price_at_time) / entry_price
            return None

        pnl_10am = pnl_at_time(10, 0)
        pnl_1030am = pnl_at_time(10, 30)
        pnl_11am = pnl_at_time(11, 0)
        pnl_1130am = pnl_at_time(11, 30)
        pnl_12pm = pnl_at_time(12, 0)
        pnl_1pm = pnl_at_time(13, 0)
        pnl_2pm = pnl_at_time(14, 0)

        # Max giveback after MFE
        if mfe_idx < len(intraday) - 1:
            post_mfe = intraday.iloc[mfe_idx:]
            max_high_after_mfe = post_mfe['high'].max()
            max_giveback_pct = (max_high_after_mfe - mfe_price) / entry_price
        else:
            max_giveback_pct = 0.0

        # Volume analysis at low
        total_volume = intraday['volume'].sum()
        cumulative_vol_at_low = intraday.iloc[:mfe_idx+1]['volume'].sum()
        cumulative_vol_at_low_pct = cumulative_vol_at_low / total_volume if total_volume > 0 else None

        # Volume bucket at low (which quintile of day's volume)
        volume_at_low_bucket = None
        if cumulative_vol_at_low_pct:
            if cumulative_vol_at_low_pct < 0.2:
                volume_at_low_bucket = 'Q1 (0-20%)'
            elif cumulative_vol_at_low_pct < 0.4:
                volume_at_low_bucket = 'Q2 (20-40%)'
            elif cumulative_vol_at_low_pct < 0.6:
                volume_at_low_bucket = 'Q3 (40-60%)'
            elif cumulative_vol_at_low_pct < 0.8:
                volume_at_low_bucket = 'Q4 (60-80%)'
            else:
                volume_at_low_bucket = 'Q5 (80-100%)'

        return ExitMetrics(
            ticker=ticker,
            date=date,
            cap=cap,
            trade_grade=grade,
            entry_price=entry_price,
            entry_time=str(entry_time),
            mfe_price=mfe_price,
            mfe_pct=mfe_pct,
            mfe_time=str(mfe_time),
            mfe_minutes=mfe_minutes,
            mae_price=mae_price,
            mae_pct=mae_pct,
            mae_time=str(mae_time),
            close_price=close_price,
            close_pct=close_pct,
            capture_efficiency=capture_efficiency,
            atr=atr,
            hit_1x_atr=hit_1x_atr,
            hit_1_5x_atr=hit_1_5x_atr,
            hit_2x_atr=hit_2x_atr,
            hit_2_5x_atr=hit_2_5x_atr,
            hit_3x_atr=hit_3x_atr,
            time_to_1x_atr=time_to_1x,
            time_to_1_5x_atr=time_to_1_5x,
            time_to_2x_atr=time_to_2x,
            prior_day_low=prior_day_low,
            hit_prior_day_low=hit_prior_day_low,
            time_to_prior_day_low=time_to_pdl,
            prior_day_close=prior_day_close,
            hit_prior_day_close=hit_prior_day_close,
            vwap_at_entry=vwap_at_entry,
            hit_vwap=hit_vwap,
            ema_9_at_entry=ema_9_at_entry,
            hit_9ema=hit_9ema,
            time_to_9ema=time_to_9ema,
            pnl_at_10am=pnl_10am,
            pnl_at_1030am=pnl_1030am,
            pnl_at_11am=pnl_11am,
            pnl_at_1130am=pnl_1130am,
            pnl_at_12pm=pnl_12pm,
            pnl_at_1pm=pnl_1pm,
            pnl_at_2pm=pnl_2pm,
            pnl_at_close=close_pct,
            max_giveback_pct=max_giveback_pct,
            volume_at_low_bucket=volume_at_low_bucket,
            cumulative_vol_at_low_pct=cumulative_vol_at_low_pct,
        )

    def analyze_all_trades(self, grade: str = 'A', limit: int = None) -> List[ExitMetrics]:
        """Analyze all trades and collect exit metrics."""
        if self.df is None:
            self.load_trades(grade)

        df = self.df
        if limit:
            df = df.head(limit)

        self.results = []
        failed = []

        for idx, row in df.iterrows():
            try:
                metrics = self.analyze_trade(row)
                if metrics:
                    self.results.append(metrics)
                else:
                    failed.append(f"{row['ticker']} {row['date']}")
            except Exception as e:
                logging.error(f"Error analyzing {row['ticker']} {row['date']}: {e}")
                failed.append(f"{row['ticker']} {row['date']}: {e}")

        logging.info(f"Successfully analyzed {len(self.results)} trades")
        if failed:
            logging.warning(f"Failed to analyze {len(failed)} trades: {failed[:5]}...")

        return self.results

    def save_results(self, output_path: str = None):
        """Save results to CSV."""
        if not self.results:
            logging.warning("No results to save")
            return

        output_path = output_path or str(_DATA_DIR / 'exit_analysis_results.csv')

        # Convert to DataFrame
        results_df = pd.DataFrame([asdict(r) for r in self.results])
        results_df.to_csv(output_path, index=False)
        logging.info(f"Results saved to {output_path}")

        return results_df

    def generate_summary_report(self) -> Dict:
        """Generate summary statistics from results."""
        if not self.results:
            return {}

        df = pd.DataFrame([asdict(r) for r in self.results])

        summary = {
            'total_trades': len(df),

            # MFE/MAE Summary
            'avg_mfe_pct': df['mfe_pct'].mean() * 100,
            'median_mfe_pct': df['mfe_pct'].median() * 100,
            'avg_mae_pct': df['mae_pct'].mean() * 100,
            'avg_close_pct': df['close_pct'].mean() * 100,
            'avg_capture_efficiency': df['capture_efficiency'].mean() * 100,
            'avg_mfe_minutes': df['mfe_minutes'].mean(),
            'median_mfe_minutes': df['mfe_minutes'].median(),

            # ATR Hit Rates
            'hit_1x_atr_pct': df['hit_1x_atr'].mean() * 100,
            'hit_1_5x_atr_pct': df['hit_1_5x_atr'].mean() * 100,
            'hit_2x_atr_pct': df['hit_2x_atr'].mean() * 100,
            'hit_2_5x_atr_pct': df['hit_2_5x_atr'].mean() * 100,
            'hit_3x_atr_pct': df['hit_3x_atr'].mean() * 100,

            # Avg time to ATR targets (for those that hit)
            'avg_time_to_1x_atr': df[df['time_to_1x_atr'].notna()]['time_to_1x_atr'].mean(),
            'avg_time_to_1_5x_atr': df[df['time_to_1_5x_atr'].notna()]['time_to_1_5x_atr'].mean(),
            'avg_time_to_2x_atr': df[df['time_to_2x_atr'].notna()]['time_to_2x_atr'].mean(),

            # Technical level hit rates
            'hit_prior_day_low_pct': df['hit_prior_day_low'].mean() * 100,
            'hit_vwap_pct': df['hit_vwap'].mean() * 100,

            # Giveback
            'avg_max_giveback_pct': df['max_giveback_pct'].mean() * 100,

            # Time-based P&L
            'avg_pnl_10am': df['pnl_at_10am'].mean() * 100 if df['pnl_at_10am'].notna().any() else None,
            'avg_pnl_11am': df['pnl_at_11am'].mean() * 100 if df['pnl_at_11am'].notna().any() else None,
            'avg_pnl_12pm': df['pnl_at_12pm'].mean() * 100 if df['pnl_at_12pm'].notna().any() else None,
        }

        return summary

    def print_summary(self):
        """Print formatted summary report."""
        summary = self.generate_summary_report()
        if not summary:
            print("No results to summarize")
            return

        print("\n" + "=" * 70)
        print("EXIT ANALYSIS SUMMARY - Grade A Trades")
        print("=" * 70)

        print(f"\nTotal Trades Analyzed: {summary['total_trades']}")

        print("\n--- MFE/MAE Analysis ---")
        print(f"Average MFE (max available profit):     {summary['avg_mfe_pct']:+.1f}%")
        print(f"Median MFE:                             {summary['median_mfe_pct']:+.1f}%")
        print(f"Average MAE (max adverse before MFE):   {summary['avg_mae_pct']:+.1f}%")
        print(f"Average Captured (open to close):       {summary['avg_close_pct']:+.1f}%")
        print(f"Capture Efficiency:                     {summary['avg_capture_efficiency']:.1f}%")
        print(f"Average Time to MFE:                    {summary['avg_mfe_minutes']:.0f} mins")
        print(f"Median Time to MFE:                     {summary['median_mfe_minutes']:.0f} mins")

        print("\n--- ATR Target Hit Rates ---")
        print(f"Hit 1.0x ATR:   {summary['hit_1x_atr_pct']:5.1f}%  (avg {summary['avg_time_to_1x_atr']:.0f} mins)")
        print(f"Hit 1.5x ATR:   {summary['hit_1_5x_atr_pct']:5.1f}%  (avg {summary['avg_time_to_1_5x_atr']:.0f} mins)" if summary['avg_time_to_1_5x_atr'] else f"Hit 1.5x ATR:   {summary['hit_1_5x_atr_pct']:5.1f}%")
        print(f"Hit 2.0x ATR:   {summary['hit_2x_atr_pct']:5.1f}%  (avg {summary['avg_time_to_2x_atr']:.0f} mins)" if summary['avg_time_to_2x_atr'] else f"Hit 2.0x ATR:   {summary['hit_2x_atr_pct']:5.1f}%")
        print(f"Hit 2.5x ATR:   {summary['hit_2_5x_atr_pct']:5.1f}%")
        print(f"Hit 3.0x ATR:   {summary['hit_3x_atr_pct']:5.1f}%")

        print("\n--- Technical Level Hit Rates ---")
        print(f"Hit Prior Day Low:   {summary['hit_prior_day_low_pct']:5.1f}%")
        print(f"Hit VWAP:            {summary['hit_vwap_pct']:5.1f}%")

        print("\n--- Giveback Analysis ---")
        print(f"Avg Max Giveback after MFE: {summary['avg_max_giveback_pct']:.1f}%")

        print("\n--- Time-Based P&L ---")
        if summary['avg_pnl_10am']:
            print(f"Avg P&L at 10:00 AM:  {summary['avg_pnl_10am']:+.1f}%")
        if summary['avg_pnl_11am']:
            print(f"Avg P&L at 11:00 AM:  {summary['avg_pnl_11am']:+.1f}%")
        if summary['avg_pnl_12pm']:
            print(f"Avg P&L at 12:00 PM:  {summary['avg_pnl_12pm']:+.1f}%")

        print()


def run_phase1_analysis():
    """Run Phase 1: Data Collection and MFE/MAE Analysis."""
    print("\n" + "=" * 70)
    print("PHASE 1: EXIT STRATEGY DATA COLLECTION")
    print("=" * 70)

    analyzer = ExitAnalyzer()

    # Analyze all Grade A trades
    results = analyzer.analyze_all_trades(grade='A')

    # Save results
    results_df = analyzer.save_results()

    # Print summary
    analyzer.print_summary()

    return analyzer, results_df


if __name__ == '__main__':
    analyzer, results_df = run_phase1_analysis()
