"""
Fetch next-day price action for capitulation setups.

Queries Polygon API for OHLC data on the trading day following each capitulation day,
then calculates gap %, max bounce %, close %, and win/loss metrics.

Usage:
    python scripts/fetch_capitulation_nextday.py           # Fetch all missing data
    python scripts/fetch_capitulation_nextday.py --dry-run # Preview without API calls
    python scripts/fetch_capitulation_nextday.py --stats   # Show current stats
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from time import sleep

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_queries.polygon_queries import poly_client, get_daily

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
CAPITULATION_CSV = DATA_DIR / "capitulation_analysis.csv"
OUTPUT_CSV = DATA_DIR / "capitulation_nextday.csv"


def get_next_trading_day(date_str: str) -> str:
    """Get the next trading day after the given date (skips weekends)."""
    date = pd.to_datetime(date_str)
    
    # Simple approach: add days until we get past the weekend
    # This handles 99% of cases (most non-trading days are weekends)
    next_date = date + timedelta(days=1)
    
    # Skip Saturday (5) and Sunday (6)
    while next_date.weekday() >= 5:  # 5=Sat, 6=Sun
        next_date = next_date + timedelta(days=1)
    
    return next_date.strftime('%Y-%m-%d')


def fetch_next_day_data(ticker: str, capitulation_date: str) -> dict:
    """
    Fetch next-day OHLC data for a capitulation setup.
    
    Returns dict with:
        - next_day_date
        - next_day_open, next_day_high, next_day_low, next_day_close
        - gap_pct (from prior close to next open)
        - max_bounce_pct (from next open to next high)
        - close_pct (from next open to next close)
        - is_win (close > open)
    """
    next_day = get_next_trading_day(capitulation_date)
    if not next_day:
        return None
    
    try:
        # Get next day data
        next_data = get_daily(ticker, next_day)
        if next_data is None:
            return None
        
        # Get capitulation day close for gap calculation
        cap_data = get_daily(ticker, capitulation_date)
        if cap_data is None:
            return None
        
        prior_close = cap_data.close
        next_open = next_data.open
        next_high = next_data.high
        next_low = next_data.low
        next_close = next_data.close
        
        # Calculate metrics
        gap_pct = (next_open - prior_close) / prior_close if prior_close else np.nan
        max_bounce_pct = (next_high - next_open) / next_open if next_open else np.nan
        max_decline_pct = (next_open - next_low) / next_open if next_open else np.nan
        close_pct = (next_close - next_open) / next_open if next_open else np.nan
        
        return {
            'next_day_date': next_day,
            'next_day_open': next_open,
            'next_day_high': next_high,
            'next_day_low': next_low,
            'next_day_close': next_close,
            'prior_close': prior_close,
            'gap_pct': gap_pct,
            'max_bounce_pct': max_bounce_pct,
            'max_decline_pct': max_decline_pct,
            'close_pct': close_pct,
            'is_win': close_pct > 0 if not np.isnan(close_pct) else False,
            'is_gap_up': gap_pct > 0 if not np.isnan(gap_pct) else False,
            'bounce_gt_2pct': max_bounce_pct > 0.02 if not np.isnan(max_bounce_pct) else False,
        }
        
    except Exception as e:
        print(f"Error fetching {ticker} on {next_day}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Fetch next-day data for capitulation setups')
    parser.add_argument('--dry-run', action='store_true', help='Preview without API calls')
    parser.add_argument('--stats', action='store_true', help='Show stats only')
    parser.add_argument('--limit', type=int, help='Limit number of records to fetch')
    args = parser.parse_args()
    
    # Load capitulation data
    if not CAPITULATION_CSV.exists():
        print(f"Error: {CAPITULATION_CSV} not found. Run analyze_capitulation.py first.")
        return
    
    df = pd.read_csv(CAPITULATION_CSV)
    
    # Filter to capitulation days only
    cap_df = df[df['is_capitulation'] == True].copy()
    print(f"Total capitulation records: {len(cap_df)}")
    
    # Check for existing next-day data
    if OUTPUT_CSV.exists() and not args.dry_run and not args.stats:
        existing = pd.read_csv(OUTPUT_CSV)
        print(f"Existing next-day records: {len(existing)}")
        
        # Merge to avoid refetching
        existing_tickers_dates = set(zip(existing['date'], existing['ticker']))
        cap_df = cap_df[~cap_df.apply(lambda r: (r['date'], r['ticker']) in existing_tickers_dates, axis=1)]
        print(f"Records to fetch: {len(cap_df)}")
    
    if args.stats:
        if OUTPUT_CSV.exists():
            results = pd.read_csv(OUTPUT_CSV)
            print(f"\n=== Next-Day Statistics ({len(results)} records) ===")
            print(f"Gap Up: {(results['gap_pct'] > 0).sum()} ({(results['gap_pct'] > 0).mean()*100:.1f}%)")
            print(f"Avg Gap: {results['gap_pct'].mean()*100:.2f}%")
            print(f"Bounce >2%: {results['bounce_gt_2pct'].sum()} ({results['bounce_gt_2pct'].mean()*100:.1f}%)")
            print(f"Win Rate (close > open): {results['is_win'].mean()*100:.1f}%")
            print(f"Avg Close %: {results['close_pct'].mean()*100:.2f}%")
        else:
            print("No next-day data fetched yet.")
        return
    
    if len(cap_df) == 0:
        print("All capitulation records already have next-day data.")
        return
    
    if args.limit:
        cap_df = cap_df.head(args.limit)
        print(f"Limited to {args.limit} records for this run")
    
    if args.dry_run:
        print("\n=== DRY RUN - Would fetch ===")
        for _, row in cap_df.head(10).iterrows():
            next_day = get_next_trading_day(row['date'])
            print(f"  {row['ticker']} on {row['date']} -> next day: {next_day}")
        if len(cap_df) > 10:
            print(f"  ... and {len(cap_df) - 10} more")
        return
    
    # Fetch next-day data
    results = []
    print(f"\n=== Fetching next-day data for {len(cap_df)} records ===")
    
    for idx, row in cap_df.iterrows():
        ticker = row['ticker']
        date = row['date']
        
        print(f"[{idx+1}/{len(cap_df)}] {ticker} on {date}...", end=' ')
        
        data = fetch_next_day_data(ticker, date)
        if data:
            # Merge with original row data
            result = {**row.to_dict(), **data}
            results.append(result)
            print(f"OK (gap: {data['gap_pct']*100:+.1f}%, close: {data['close_pct']*100:+.1f}%)")
        else:
            print("SKIP (no data)")
        
        # Rate limiting - be nice to Polygon
        sleep(0.2)
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        
        # Merge with existing if present
        if OUTPUT_CSV.exists():
            existing = pd.read_csv(OUTPUT_CSV)
            results_df = pd.concat([existing, results_df], ignore_index=True)
        
        results_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Saved {len(results)} new records to {OUTPUT_CSV}")
        print(f"   Total records: {len(results_df)}")
    else:
        print("\n⚠️ No new data fetched")


if __name__ == '__main__':
    main()
