"""Verify pct_change_3 for tickers with post-trade stock splits."""
import pandas as pd
import os
import requests
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv('C:/Users/zmbur/PycharmProjects/backtester/.env')
key = os.getenv('POLYGON_API_KEY')

df = pd.read_csv('data/reversal_data.csv')

to_check = [
    ('AVGO', '5/30/2023'), ('AVGO', '6/20/2024'),
    ('NVDA', '2/6/2024'), ('NVDA', '2/12/2024'), ('NVDA', '3/8/2024'), ('NVDA', '6/6/2024'),
    ('SMCI', '2/6/2024'), ('SMCI', '2/12/2024'), ('SMCI', '2/16/2024'),
    ('MSTR', '12/29/2023'),
    ('GME', '1/28/2021'), ('GME', '3/10/2021'), ('GME', '3/29/2022'),
    ('AMC', '1/27/2021'), ('AMC', '1/28/2021'), ('AMC', '6/3/2021'), ('AMC', '3/29/2022'),
    ('NKLA', '6/9/2020'), ('NKLA', '7/14/2023'),
    ('SMH', '5/30/2023'),
]


def parse_date(d):
    for fmt in ['%m/%d/%Y', '%Y-%m-%d']:
        try:
            return datetime.strptime(d, fmt)
        except ValueError:
            continue
    return None


print(f"{'Ticker':<8} {'Date':<12} {'CSV_pct3':<12} {'Calc_pct3':<12} {'Diff':<10} {'Match?':<10} {'Open':<10} {'3d_Close':<10} {'3d_Date':<12}")
print('-' * 100)

issues = []
for ticker, date_str in to_check:
    row = df[(df['ticker'] == ticker) & (df['date'] == date_str)]
    if row.empty:
        print(f"{ticker:<8} {date_str:<12} NOT FOUND IN CSV")
        continue
    row = row.iloc[0]
    csv_pct3 = row['pct_change_3']

    trade_dt = parse_date(date_str)
    trade_api_date = trade_dt.strftime('%Y-%m-%d')

    # Get trade day open
    url = f'https://api.polygon.io/v1/open-close/{ticker}/{trade_api_date}?adjusted=true&apiKey={key}'
    r = requests.get(url)
    time.sleep(0.15)
    if r.status_code != 200 or r.json().get('status') != 'OK':
        print(f"{ticker:<8} {date_str:<12} {csv_pct3:<12.6f} {'API_FAIL':<12}")
        continue
    trade_open = r.json()['open']

    # Fetch daily bars for prior ~10 calendar days to find 3rd trading day back
    range_start = (trade_dt - timedelta(days=12)).strftime('%Y-%m-%d')
    range_end = (trade_dt - timedelta(days=1)).strftime('%Y-%m-%d')
    url2 = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{range_start}/{range_end}?adjusted=true&sort=desc&limit=5&apiKey={key}'
    r2 = requests.get(url2)
    time.sleep(0.15)

    if r2.status_code != 200:
        print(f"{ticker:<8} {date_str:<12} {csv_pct3:<12.6f} {'RANGE_FAIL':<12}")
        continue

    results = r2.json().get('results', [])
    if len(results) < 3:
        print(f"{ticker:<8} {date_str:<12} {csv_pct3:<12.6f} {'<3_BARS':<12}")
        continue

    close_3d = results[2]['c']
    close_3d_ts = results[2]['t']
    close_3d_date = datetime.utcfromtimestamp(close_3d_ts / 1000).strftime('%Y-%m-%d')

    calc_pct3 = (trade_open - close_3d) / close_3d
    diff = abs(calc_pct3 - csv_pct3)
    match = diff < 0.005
    flag = 'OK' if match else 'MISMATCH'

    print(f"{ticker:<8} {date_str:<12} {csv_pct3:<12.6f} {calc_pct3:<12.6f} {diff:<10.6f} {flag:<10} {trade_open:<10.2f} {close_3d:<10.2f} {close_3d_date:<12}")

    if not match:
        issues.append({
            'ticker': ticker, 'date': date_str,
            'csv_pct3': csv_pct3, 'calc_pct3': calc_pct3,
            'trade_open': trade_open, 'close_3d': close_3d,
            'close_3d_date': close_3d_date,
        })

print(f"\n{'='*80}")
print(f"FLAGGED MISMATCHES: {len(issues)}")
print(f"{'='*80}")
for i in issues:
    ratio = i['calc_pct3'] / i['csv_pct3'] if i['csv_pct3'] != 0 else float('inf')
    print(f"  {i['ticker']} {i['date']}:")
    print(f"    CSV pct_change_3 = {i['csv_pct3']:.6f} ({i['csv_pct3']*100:.2f}%)")
    print(f"    Actual (Polygon) = {i['calc_pct3']:.6f} ({i['calc_pct3']*100:.2f}%)")
    print(f"    Trade open = {i['trade_open']:.2f}, 3d-prior close = {i['close_3d']:.2f} ({i['close_3d_date']})")
    print(f"    Ratio (actual/csv) = {ratio:.2f}x")
    print()
