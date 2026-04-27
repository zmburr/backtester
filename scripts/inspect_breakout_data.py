"""One-shot inspection of breakout_data.csv after the data expansion.

Run: python scripts/inspect_breakout_data.py

Reports:
  - Column counts (total, X features, Y labels)
  - NaN-rate per column
  - setup_type tag distribution
  - pivot_source distribution (how many rows fell back to prior_close)
  - Spot summary of new key columns
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd

CSV = REPO / 'data' / 'breakout_data.csv'

df = pd.read_csv(CSV)
n = len(df)
print(f'Rows: {n}, Columns: {len(df.columns)}')
print()

# NaN rate per column
print('=== NaN RATE BY COLUMN (sorted, only non-zero) ===')
nan_rate = df.isna().sum() / n
nz = nan_rate[nan_rate > 0].sort_values(ascending=False)
for col, rate in nz.items():
    flag = ''
    if 'd1_' in col or 'overnight_gap_d1' in col or 'pm_d2_' in col:
        flag = ' (conditional: only t==1)'
    elif 'ipo' in col or col == 'days_since_ipo':
        flag = ' (conditional: only days_since_ipo<365)'
    elif col in ('float_shares', 'short_interest_pct', 'days_to_cover'):
        flag = ' (Bloomberg — separate fill)'
    print(f'  {col:<40} {rate:>6.1%}{flag}')

print()
print('=== setup_type DISTRIBUTION ===')
if 'setup_type' in df.columns:
    counts = df['setup_type'].value_counts(dropna=False)
    for tag, c in counts.items():
        print(f'  {c:>3}  {tag}')

print()
print('=== pivot_source DISTRIBUTION ===')
if 'pivot_source' in df.columns:
    for src, c in df['pivot_source'].value_counts(dropna=False).items():
        print(f'  {c:>3}  {src}')

print()
print('=== KEY COLUMN SPOT CHECK (first 5 rows) ===')
key_cols = [c for c in [
    'date', 'ticker', 't', 'setup_type', 'pivot_source',
    'pct_to_ath', 'pct_to_52wk_high', 'days_since_ath',
    'consolidation_days', 'rs_vs_spy_30d', 'ma_stack_aligned',
    'pivot_to_high_pct', 'atr_max_extension', 'held_above_pivot_at_close',
    'd1_close_at_high_pct', 'days_since_ipo'
] if c in df.columns]
print(df[key_cols].head().to_string())

print()
print('=== ROWS WITH STALE BREAKS_ATH (pivot fell back) ===')
if 'pivot_source' in df.columns:
    stale = df[df['pivot_source'].astype(str).str.contains('fallback', na=False)]
    if len(stale):
        print(stale[['date', 'ticker', 'setup_type', 'pct_to_ath', 'pivot_source']].to_string())
    else:
        print('  (none)')
