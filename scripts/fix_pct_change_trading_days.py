"""
Fix pct_change_3/15/30/90/120 to use TRADING days instead of calendar days.

The pipeline's get_price_with_fallback() uses calendar day lookback, which means
pct_change_3 on a Tuesday only goes back to Friday (1 trading day). This script
recalculates all pct_change columns using actual trading day counts via Polygon
daily aggs.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
API_KEY = os.getenv('POLYGON_API_KEY')

PCT_COLS = {
    'pct_change_3': 3,
    'pct_change_15': 15,
    'pct_change_30': 30,
    'pct_change_90': 90,
    'pct_change_120': 120,
}


def parse_date(d):
    for fmt in ['%m/%d/%Y', '%Y-%m-%d']:
        try:
            return datetime.strptime(d, fmt)
        except ValueError:
            continue
    return None


def get_trading_day_close(daily_bars, n_trading_days):
    """Get close price N trading days back from the end of daily_bars.

    daily_bars is sorted ascending (oldest first). The last bar is the day
    before the trade date. So index [-n] gives N trading days back.
    """
    if len(daily_bars) < n_trading_days:
        return None
    return daily_bars[-(n_trading_days)]['c']


def fetch_daily_bars(ticker, end_date_str, lookback_calendar_days=200):
    """Fetch daily bars from Polygon ending before the trade date."""
    end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
    # End at the day before the trade
    end = (end_dt - timedelta(days=1)).strftime('%Y-%m-%d')
    start = (end_dt - timedelta(days=lookback_calendar_days)).strftime('%Y-%m-%d')

    url = (f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/'
           f'{start}/{end}?adjusted=true&sort=asc&limit=200&apiKey={API_KEY}')
    r = requests.get(url)
    time.sleep(0.15)

    if r.status_code != 200:
        return []
    data = r.json()
    return data.get('results', [])


def get_trade_open(ticker, trade_date_str):
    """Get the opening price on the trade date."""
    url = (f'https://api.polygon.io/v1/open-close/{ticker}/{trade_date_str}'
           f'?adjusted=true&apiKey={API_KEY}')
    r = requests.get(url)
    time.sleep(0.15)
    if r.status_code == 200 and r.json().get('status') == 'OK':
        return r.json()['open']
    return None


def main():
    csv_path = DATA_DIR / 'reversal_data.csv'
    backup_path = DATA_DIR / 'reversal_data_pre_trading_day_fix.csv'

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from reversal_data.csv")

    # Backup
    df.to_csv(backup_path, index=False)
    print(f"Backup saved to {backup_path}")

    changes = {col: 0 for col in PCT_COLS}
    errors = []

    for idx, row in df.iterrows():
        ticker = row['ticker']
        date_str = row['date']
        trade_dt = parse_date(date_str)
        if trade_dt is None:
            errors.append(f"  {ticker} {date_str}: can't parse date")
            continue

        trade_api_date = trade_dt.strftime('%Y-%m-%d')

        # Get trade day open price
        trade_open = get_trade_open(ticker, trade_api_date)
        if trade_open is None:
            errors.append(f"  {ticker} {date_str}: can't get trade day open")
            continue

        # Fetch daily bars (enough for 120 trading days lookback)
        bars = fetch_daily_bars(ticker, trade_api_date, lookback_calendar_days=200)
        if len(bars) < 3:
            errors.append(f"  {ticker} {date_str}: only {len(bars)} bars")
            continue

        row_changes = []
        for col, n_days in PCT_COLS.items():
            old_val = row[col]
            close_n = get_trading_day_close(bars, n_days)
            if close_n is None or close_n == 0:
                continue

            new_val = (trade_open - close_n) / close_n

            if pd.notna(old_val) and old_val != 0:
                diff_pct = abs(new_val / old_val - 1.0) * 100
                if diff_pct > 1.0:  # More than 1% difference
                    row_changes.append(
                        f"    {col}: {old_val:.6f} -> {new_val:.6f} "
                        f"({diff_pct:+.1f}% diff)")
                    changes[col] += 1

            df.at[idx, col] = new_val

        if row_changes:
            print(f"\n  {ticker} {date_str}:")
            for c in row_changes:
                print(c)

    # Save
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print("SUMMARY OF CHANGES (>1% diff from old value):")
    for col, count in changes.items():
        print(f"  {col}: {count} / {len(df)} rows changed")
    print(f"\nErrors: {len(errors)}")
    for e in errors[:10]:
        print(e)
    if len(errors) > 10:
        print(f"  ... and {len(errors)-10} more")
    print(f"\nSaved to {csv_path}")
    print(f"Backup at {backup_path}")


if __name__ == '__main__':
    main()
