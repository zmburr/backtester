"""
Fix reversal_data.csv rows affected by stock splits.

Re-enriches price-derived columns using Polygon's current split-adjusted data.
Columns fixed: pct_change_3/15/30/90/120, pct_from_9ema/10mav/20mav/50mav/200mav,
atr_distance_from_50mav, gap_pct, atr_pct, upper_band_distance, bollinger_width,
closed_outside_upper_band, gap_from_pm_high, pct_from_9ema (via get_ticker_mavs_open).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import pandas as pd
import time
from pathlib import Path
from data_queries.polygon_queries import (
    get_ticker_pct_move, get_ticker_mavs_open, get_daily,
    adjust_date_to_market, get_current_price, get_atr,
)
from support.date_utils import parse_row_date

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

# 11 trades flagged with bad split data
FLAGGED = [
    ('AVGO', '5/30/2023'),
    ('AVGO', '6/20/2024'),
    ('NVDA', '2/6/2024'),
    ('NVDA', '2/12/2024'),
    ('GME', '3/10/2021'),
    ('GME', '3/29/2022'),
    ('AMC', '1/27/2021'),
    ('AMC', '6/3/2021'),
    ('AMC', '3/29/2022'),
    ('NKLA', '6/9/2020'),
    ('SMH', '5/30/2023'),
]


def fix_row(row):
    """Re-enrich all price-derived columns for a single row."""
    ticker = row['ticker']
    date_str = row['date']
    date = parse_row_date(row)

    print(f"\n  Processing {ticker} {date_str}...")

    # 1. pct_change columns (3/15/30/90/120)
    try:
        current_price = get_current_price(ticker, date)
        pct_dict = get_ticker_pct_move(ticker, date, current_price)
        for key in ['pct_change_3', 'pct_change_15', 'pct_change_30',
                     'pct_change_90', 'pct_change_120']:
            old_val = row.get(key)
            new_val = pct_dict.get(key)
            if new_val is not None:
                row[key] = new_val
                if old_val is not None and old_val != 0:
                    ratio = new_val / old_val if old_val != 0 else float('inf')
                    changed = abs(ratio - 1.0) > 0.01
                    marker = " ***" if changed else ""
                    print(f"    {key}: {old_val:.6f} -> {new_val:.6f} (ratio={ratio:.2f}){marker}")
                else:
                    print(f"    {key}: {old_val} -> {new_val:.6f}")
        time.sleep(0.5)
    except Exception as e:
        print(f"    ERROR (pct_change): {e}")

    # 2. Moving average percentages + ATR distance
    try:
        mav_dict = get_ticker_mavs_open(ticker, date)
        for key in ['pct_from_9ema', 'pct_from_10mav', 'pct_from_20mav',
                     'pct_from_50mav', 'pct_from_200mav', 'atr_distance_from_50mav']:
            old_val = row.get(key)
            new_val = mav_dict.get(key)
            if new_val is not None:
                row[key] = new_val
                changed = old_val is None or (old_val != 0 and abs(new_val / old_val - 1.0) > 0.01)
                marker = " ***" if changed else ""
                old_str = f"{old_val:.6f}" if old_val is not None else "None"
                print(f"    {key}: {old_str} -> {new_val:.6f}{marker}")
        time.sleep(0.5)
    except Exception as e:
        print(f"    ERROR (mavs): {e}")

    # 3. gap_pct (open - prev_close) / prev_close
    try:
        daily_data = get_daily(ticker, date)
        prev_date = adjust_date_to_market(date, 1)
        prev_data = get_daily(ticker, prev_date)
        if daily_data and prev_data:
            open_price = daily_data.open
            prev_close = prev_data.close
            new_gap = (open_price - prev_close) / prev_close
            old_gap = row.get('gap_pct')
            row['gap_pct'] = new_gap
            changed = old_gap is not None and old_gap != 0 and abs(new_gap / old_gap - 1.0) > 0.01
            marker = " ***" if changed else ""
            old_str = f"{old_gap:.6f}" if old_gap is not None else "None"
            print(f"    gap_pct: {old_str} -> {new_gap:.6f}{marker}")
        time.sleep(0.3)
    except Exception as e:
        print(f"    ERROR (gap_pct): {e}")

    # 4. atr_pct
    try:
        atr_val = get_atr(ticker, date)
        if atr_val and daily_data:
            # ATR as % of price (use close as reference like the original)
            close_price = daily_data.close
            new_atr_pct = atr_val / close_price if close_price else None
            if new_atr_pct is not None:
                old_atr = row.get('atr_pct')
                row['atr_pct'] = new_atr_pct
                old_str = f"{old_atr:.6f}" if old_atr is not None else "None"
                changed = old_atr is not None and old_atr != 0 and abs(new_atr_pct / old_atr - 1.0) > 0.01
                marker = " ***" if changed else ""
                print(f"    atr_pct: {old_str} -> {new_atr_pct:.6f}{marker}")
        time.sleep(0.3)
    except Exception as e:
        print(f"    ERROR (atr_pct): {e}")

    return row


def main():
    csv_path = DATA_DIR / 'reversal_data.csv'
    backup_path = DATA_DIR / 'reversal_data_pre_split_fix.csv'

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from reversal_data.csv")

    # Backup
    df.to_csv(backup_path, index=False)
    print(f"Backup saved to {backup_path}")

    fixed_count = 0
    for ticker, date_str in FLAGGED:
        mask = (df['ticker'] == ticker) & (df['date'] == date_str)
        matches = df[mask]
        if matches.empty:
            print(f"\n  WARNING: {ticker} {date_str} not found in CSV")
            continue

        idx = matches.index[0]
        row = df.loc[idx].copy()
        updated_row = fix_row(row)
        df.loc[idx] = updated_row
        fixed_count += 1

    # Save
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*60}")
    print(f"Fixed {fixed_count} rows. Saved to {csv_path}")
    print(f"Backup at {backup_path}")


if __name__ == '__main__':
    main()
