"""Audit: per-setup feature distributions to see if classes are actually separable."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analyzers.setup_matcher import PRE_DECISION_FEATURES, TOP_4_SETUPS, load_training_data

KEY_FEATS = [
    'gap_pct', 'consecutive_up_days', 'pct_change_3', 'pct_change_30',
    'pct_from_50mav', 'percent_of_premarket_vol', 'atr_pct',
    'prior_day_range_atr', 'gap_from_pm_high', 'prior_day_close_vs_high_pct',
    'rvol_score',
]

df = load_training_data()
print(f"Top-4 rows total: {len(df)}")
print()

# Per-setup means
print(f"{'feature':35s} | " + ' | '.join(f"{s:>20s}" for s in TOP_4_SETUPS))
print('-' * 130)
for f in KEY_FEATS:
    line = f"{f:35s} | "
    pieces = []
    for s in TOP_4_SETUPS:
        sub = df[df['setup'] == s]
        col = pd.to_numeric(sub[f], errors='coerce').dropna()
        if len(col):
            mean = col.mean()
            std = col.std()
            pieces.append(f"{mean:+8.3f}±{std:6.3f} (n={len(col):2d})")
        else:
            pieces.append(f"{'(no data)':>20s}")
    print(line + ' | '.join(pieces))

print()
print('=== Specific check: gap_pct distribution per setup ===')
for s in TOP_4_SETUPS:
    sub = df[df['setup'] == s]
    gp = pd.to_numeric(sub['gap_pct'], errors='coerce').dropna()
    print(f"{s:25s} n={len(gp):2d}  min={gp.min():+.3f}  max={gp.max():+.3f}  median={gp.median():+.3f}  "
          f"% positive={(gp>0).mean()*100:.0f}%  % gap>2%={(gp>0.02).mean()*100:.0f}%")

print()
print('=== consecutive_up_days distribution ===')
for s in TOP_4_SETUPS:
    sub = df[df['setup'] == s]
    cud = pd.to_numeric(sub['consecutive_up_days'], errors='coerce').dropna()
    print(f"{s:25s} n={len(cud):2d}  min={cud.min():.0f}  max={cud.max():.0f}  median={cud.median():.0f}  "
          f"mean={cud.mean():.2f}")

print()
print('=== Direct check: how many neighbors of each setup label fall within k=10 of EACH MSTR/GME/OPEN/QS-style test pattern? ===')
print('(skipping — we already have the per-test neighbor info)')

# Detailed look at the labeled examples that match test-case profiles
print()
print('=== GapDownTrendBreak examples (the setup QS expects) — what do THEY look like? ===')
gdtb = df[df['setup'] == 'GapDownTrendBreak'][
    ['date', 'ticker', 'cap', 'gap_pct', 'consecutive_up_days', 'pct_change_3', 'pct_change_30',
     'pct_from_50mav', 'prior_day_range_atr']
].copy()
# Convert to numeric for printing
for c in ['gap_pct', 'consecutive_up_days', 'pct_change_3', 'pct_change_30', 'pct_from_50mav', 'prior_day_range_atr']:
    gdtb[c] = pd.to_numeric(gdtb[c], errors='coerce')
print(gdtb.to_string(index=False))

print()
print('=== 2DBreakoutIB examples (the setup OPEN expects) ===')
ib = df[df['setup'] == '2DBreakoutIB'][
    ['date', 'ticker', 'cap', 'gap_pct', 'consecutive_up_days', 'pct_change_3', 'pct_change_30',
     'pct_from_50mav', 'prior_day_range_atr']
].copy()
for c in ['gap_pct', 'consecutive_up_days', 'pct_change_3', 'pct_change_30', 'pct_from_50mav', 'prior_day_range_atr']:
    ib[c] = pd.to_numeric(ib[c], errors='coerce')
print(ib.to_string(index=False))
