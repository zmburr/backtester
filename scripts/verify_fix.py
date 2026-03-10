"""Verify the trading-day fix results."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import pandas as pd
from analyzers.reversal_scorer import ReversalScorer

df = pd.read_csv('data/reversal_data.csv')
grade_a = df[df['trade_grade'] == 'A'].copy()
scorer = ReversalScorer()
scored = scorer.score_dataframe(grade_a)
scored['pnl'] = -scored['reversal_open_close_pct'] * 100

# AVGO
avgo = scored[scored['ticker'] == 'AVGO']
print('=== AVGO AFTER FIX ===')
for _, r in avgo.iterrows():
    print(f"  {r['date']} | pct_change_3={r['pct_change_3']:.4f} | "
          f"mom_pctile_3={r['momentum_pctile_3']} | "
          f"rec={r['pretrade_recommendation']} | intensity={r['intensity']}")

go = scored[scored['pretrade_recommendation'] == 'GO']
notgo = scored[~(scored['pretrade_recommendation'] == 'GO')]
print(f"\nGO: {len(go)} trades | Win: {(go['pnl']>0).mean()*100:.1f}% | "
      f"Avg PnL: {go['pnl'].mean():+.1f}%")
print(f"Not-GO: {len(notgo)} trades | Win: {(notgo['pnl']>0).mean()*100:.1f}% | "
      f"Avg PnL: {notgo['pnl'].mean():+.1f}%")

# Compare with old data
old = pd.read_csv('data/reversal_data_pre_trading_day_fix.csv')
old_a = old[old['trade_grade'] == 'A'].copy()
old_scored = scorer.score_dataframe(old_a)
old_scored['pnl'] = -old_scored['reversal_open_close_pct'] * 100

flipped = []
for i in range(min(len(scored), len(old_scored))):
    old_rec = old_scored.iloc[i]['pretrade_recommendation']
    new_rec = scored.iloc[i]['pretrade_recommendation']
    if old_rec != new_rec:
        r = scored.iloc[i]
        flipped.append((r['ticker'], r['date'], old_rec, new_rec,
                        r['pct_change_3'], r['pnl']))

print(f"\nRecommendation flips: {len(flipped)}")
for t, d, orec, nrec, p3, pnl in flipped:
    print(f"  {t:<8} {d:<12} {orec:<10} -> {nrec:<10} pct3={p3:.4f} pnl={pnl:+.1f}%")

# Show new percentile distribution
print("\n=== Updated Large cap pct_change_3 percentiles (Grade A) ===")
import numpy as np
from analyzers.reversal_scorer import _CAP_GROUPS, _ref_by_cap_group
# Need to recompute ref with new data since module loaded old data
ref = df[df['trade_grade'] == 'A'].copy()
ref['cap_group'] = ref['cap'].map(_CAP_GROUPS).fillna('large')
large = ref[ref['cap_group'] == 'large']
vals = large['pct_change_3'].dropna().values
for p in [5, 10, 15, 20, 25, 50, 75]:
    print(f"  p{p}: {np.percentile(vals, p):.4f} ({np.percentile(vals, p)*100:.2f}%)")
