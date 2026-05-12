"""SNDK first-15-minute volume pacing analysis.

Question: how much volume needs to print in the first 15 min for the day to
be on pace to match/exceed yesterday's full-session volume?

Approach:
- Pull intraday 1-min bars for the last N trading days
- For each day, compute first-15-min volume and full-session (regular hours) volume
- Compute the ratio first15 / full_session per day
- Apply the median ratio to yesterday's full volume to derive a threshold
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
from datetime import datetime, timedelta
from data_queries.polygon_queries import get_intraday, get_levels_data, adjust_date_to_market

TICKER = 'SNDK'
N_DAYS = 15  # how many sessions to sample

# Pull last N trading days from daily bars
END = '2026-05-05'
levels = get_levels_data(TICKER, END, 60, 1, 'day')
last_days = levels.tail(N_DAYS)

print(f'{TICKER} — first-15-min volume pacing  (sample: last {len(last_days)} sessions)')
print('=' * 95)
print(f'{"Date":<12} {"Open":>9} {"Close":>9} {"FullVol":>13} {"First15Vol":>13} '
      f'{"First15$Vol":>13} {"Pct":>7}')
print('-' * 95)

rows = []
for ts in last_days.index:
    d = ts.strftime('%Y-%m-%d')
    intra = get_intraday(TICKER, d, 1, 'minute')
    if intra is None or intra.empty:
        continue
    # Regular session: 09:30 - 16:00 ET
    rth = intra.between_time('09:30', '15:59')
    if rth.empty:
        continue
    full_vol = rth['volume'].sum()
    first15 = rth.between_time('09:30', '09:44')
    first15_vol = first15['volume'].sum()
    if full_vol == 0:
        continue
    pct = first15_vol / full_vol * 100
    open_px = rth.iloc[0]['open']
    close_px = rth.iloc[-1]['close']
    first15_high = first15['high'].max() if not first15.empty else 0
    first15_low = first15['low'].min() if not first15.empty else 0
    first15_mid = (first15_high + first15_low) / 2 if first15_high else 0
    first15_dvol = first15_vol * first15_mid

    rows.append({
        'date': d,
        'open': open_px,
        'close': close_px,
        'full_vol': full_vol,
        'first15_vol': first15_vol,
        'first15_dvol': first15_dvol,
        'pct': pct,
    })
    print(f'{d:<12} {open_px:>9.2f} {close_px:>9.2f} {int(full_vol):>13,} '
          f'{int(first15_vol):>13,} ${first15_dvol/1e9:>11,.2f}B {pct:>6.2f}%')

if not rows:
    print('No data')
    sys.exit(0)

df = pd.DataFrame(rows)

print()
print('--- Distribution of first-15-min share of full-session volume ---')
print(f'  Mean:    {df["pct"].mean():.2f}%')
print(f'  Median:  {df["pct"].median():.2f}%')
print(f'  Min:     {df["pct"].min():.2f}%')
print(f'  Max:     {df["pct"].max():.2f}%')
print(f'  P25:     {df["pct"].quantile(0.25):.2f}%')
print(f'  P75:     {df["pct"].quantile(0.75):.2f}%')

# Yesterday's full-day volume (5/5)
yest_full_vol = int(levels.iloc[-1]['volume'])
yest_full_high = float(levels.iloc[-1]['high'])

print()
print(f'--- Yesterday (5/5) ---')
print(f'  Full-day volume: {yest_full_vol:,} shares')
print(f'  Day high:        ${yest_full_high:,.2f}')

# Compute thresholds
median_pct = df['pct'].median() / 100
mean_pct = df['pct'].mean() / 100
p75_pct = df['pct'].quantile(0.75) / 100

print()
print('--- First-15-min volume threshold analysis ---')
print('"On pace to match yesterday\'s full-day volume" =')
print(f'   yesterday full-vol * (typical first-15 share)')
print()
for label, ratio in [
    ('median pace (50th pctile)', median_pct),
    ('mean pace', mean_pct),
    ('heavier pace (P75 — busy open)', p75_pct),
]:
    threshold_match = yest_full_vol * ratio
    threshold_beat_20pct = yest_full_vol * 1.20 * ratio
    threshold_beat_50pct = yest_full_vol * 1.50 * ratio
    print(f'  {label} ({ratio*100:.2f}% share):')
    print(f'    To match yesterday\'s full vol:        {int(threshold_match):>12,} shares')
    print(f'    To exceed by 20%:                     {int(threshold_beat_20pct):>12,} shares')
    print(f'    To exceed by 50% (climactic day):     {int(threshold_beat_50pct):>12,} shares')

# What does this look like at yesterday's mid-price
mid_5_5 = (float(levels.iloc[-1]['high']) + float(levels.iloc[-1]['low'])) / 2
print()
print(f'--- For reference: at yesterday\'s mid-price ${mid_5_5:,.2f} ---')
match_dvol = yest_full_vol * median_pct * mid_5_5 / 1e9
beat20_dvol = yest_full_vol * 1.20 * median_pct * mid_5_5 / 1e9
print(f'  First-15 $-vol to match yesterday pace:   ${match_dvol:.2f}B')
print(f'  First-15 $-vol to exceed by 20%:          ${beat20_dvol:.2f}B')
