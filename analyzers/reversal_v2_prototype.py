"""
Reversal Scorer V2 Prototype — Side-by-Side Comparison

ORIGINAL pre-trade (5 criteria):
  pct_from_9ema, prior_day_range_atr, rvol_score, consecutive_up_days, gap_pct

PROPOSED V2 pre-trade (5 criteria):
  pct_from_9ema, prior_day_range_atr, rvol_score, pct_change_3, gap_pct
  (Replaced consecutive_up_days rho=0.086 with pct_change_3 rho=+0.546)

Scores every trade in reversal_data.csv with BOTH systems, compares separation.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

from analyzers.reversal_scorer import ReversalScorer, CAP_THRESHOLDS


def score_v2(row, cap):
    """
    Score a single trade with the V2 pre-trade criteria.
    Same thresholds as original for shared criteria; pct_change_3 replaces consecutive_up_days.

    For pct_change_3 thresholds: use cap-adjusted values derived from the data.
    """
    if cap not in CAP_THRESHOLDS:
        cap = 'Medium'
    t = CAP_THRESHOLDS[cap]

    # V2 pct_change_3 thresholds — set at p25 of winning trades per cap
    # (lenient: ~75% of winners pass)
    pct_change_3_thresholds = {
        'Micro': 0.50,    # Micro caps move huge — 50%+ 3-day run
        'Small': 0.25,    # Small caps — 25%+ 3-day run
        'Medium': 0.10,   # Medium — 10%+ 3-day run
        'Large': 0.05,    # Large — 5%+ 3-day run
        'ETF': 0.03,      # ETF — 3%+ 3-day run
    }

    score = 0
    passed = []
    failed = []

    # 1. pct_from_9ema (same as original)
    val = row.get('pct_from_9ema')
    if pd.notna(val) and val >= t.pct_from_9ema:
        score += 1; passed.append('pct_from_9ema')
    else:
        failed.append('pct_from_9ema')

    # 2. prior_day_range_atr (same as original)
    val = row.get('prior_day_range_atr')
    if pd.isna(val):
        val = row.get('one_day_before_range_pct')
    if pd.notna(val) and val >= t.prior_day_range_atr:
        score += 1; passed.append('prior_day_range_atr')
    else:
        failed.append('prior_day_range_atr')

    # 3. rvol_score (same as original)
    val = row.get('rvol_score')
    if pd.notna(val) and val >= t.rvol_score:
        score += 1; passed.append('rvol_score')
    else:
        failed.append('rvol_score')

    # 4. pct_change_3 (NEW — replaces consecutive_up_days)
    val = row.get('pct_change_3')
    thresh = pct_change_3_thresholds.get(cap, 0.10)
    if pd.notna(val) and val >= thresh:
        score += 1; passed.append('pct_change_3')
    else:
        failed.append('pct_change_3')

    # 5. gap_pct (same as original)
    val = row.get('gap_pct')
    if pd.notna(val) and val >= t.gap_pct:
        score += 1; passed.append('gap_pct')
    else:
        failed.append('gap_pct')

    return score, passed, failed


def run_prototype():
    csv_path = os.path.join(DATA_PATH, 'reversal_data.csv')
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} reversal trades")

    # Outcome
    df['pnl'] = -df['reversal_open_close_pct'] * 100
    df['win'] = (df['pnl'] > 0).astype(int)

    # =====================================================================
    # Score with ORIGINAL (generic reversal_scorer.py, 6 criteria including reversal_pct)
    # =====================================================================
    scorer = ReversalScorer()
    scored = scorer.score_dataframe(df)

    # Also compute ORIGINAL pre-trade score (5 criteria, no reversal_pct)
    orig_pretrade_scores = []
    for _, row in df.iterrows():
        cap = row.get('cap', 'Medium')
        if cap not in CAP_THRESHOLDS:
            cap = 'Medium'
        t = CAP_THRESHOLDS[cap]
        s = 0
        if pd.notna(row.get('pct_from_9ema')) and row['pct_from_9ema'] >= t.pct_from_9ema: s += 1
        pdr = row.get('prior_day_range_atr')
        if pd.isna(pdr): pdr = row.get('one_day_before_range_pct')
        if pd.notna(pdr) and pdr >= t.prior_day_range_atr: s += 1
        if pd.notna(row.get('rvol_score')) and row['rvol_score'] >= t.rvol_score: s += 1
        if pd.notna(row.get('consecutive_up_days')) and row['consecutive_up_days'] >= t.consecutive_up_days: s += 1
        if pd.notna(row.get('gap_pct')) and row['gap_pct'] >= t.gap_pct: s += 1
        orig_pretrade_scores.append(s)

    df['orig_pretrade_score'] = orig_pretrade_scores

    # =====================================================================
    # Score with V2 pre-trade (5 criteria, pct_change_3 replaces consecutive_up_days)
    # =====================================================================
    v2_scores = []
    for _, row in df.iterrows():
        cap = row.get('cap', 'Medium')
        if cap not in CAP_THRESHOLDS:
            cap = 'Medium'
        s, passed, failed = score_v2(row, cap)
        v2_scores.append(s)

    df['v2_pretrade_score'] = v2_scores

    # =====================================================================
    # COMPARISON: Original pre-trade vs V2 pre-trade
    # =====================================================================
    print("\n" + "=" * 80)
    print("ORIGINAL PRE-TRADE SCORING (5 criteria, includes consecutive_up_days)")
    print("=" * 80)

    print("\nPERFORMANCE BY SCORE:")
    print("-" * 70)
    for s in range(5, -1, -1):
        subset = df[df['orig_pretrade_score'] == s]
        if len(subset) > 0:
            wr = (subset['pnl'] > 0).mean() * 100
            avg = subset['pnl'].mean()
            print(f"  Score {s}/5: {len(subset):3d} trades | Win: {wr:5.1f}% | Avg P&L: {avg:+6.1f}%")

    # Recommendation mapping: >=4 GO, 3 CAUTION, <3 NO-GO
    def orig_rec(s):
        if s >= 4: return 'GO'
        elif s == 3: return 'CAUTION'
        return 'NO-GO'

    df['orig_rec'] = df['orig_pretrade_score'].apply(orig_rec)

    print("\nBY RECOMMENDATION:")
    print("-" * 70)
    for rec in ['GO', 'CAUTION', 'NO-GO']:
        subset = df[df['orig_rec'] == rec]
        if len(subset) > 0:
            wr = (subset['pnl'] > 0).mean() * 100
            avg = subset['pnl'].mean()
            print(f"  {rec:8s}: {len(subset):3d} trades | Win: {wr:5.1f}% | Avg P&L: {avg:+6.1f}%")

    rho_orig, p_orig = stats.spearmanr(df['orig_pretrade_score'], df['pnl'])
    print(f"\n  Spearman rho (score vs P&L): {rho_orig:+.3f}  (p={p_orig:.4f})")

    # =====================================================================
    print("\n" + "=" * 80)
    print("V2 PRE-TRADE SCORING (5 criteria, pct_change_3 replaces consecutive_up_days)")
    print("=" * 80)

    print("\nPERFORMANCE BY SCORE:")
    print("-" * 70)
    for s in range(5, -1, -1):
        subset = df[df['v2_pretrade_score'] == s]
        if len(subset) > 0:
            wr = (subset['pnl'] > 0).mean() * 100
            avg = subset['pnl'].mean()
            print(f"  Score {s}/5: {len(subset):3d} trades | Win: {wr:5.1f}% | Avg P&L: {avg:+6.1f}%")

    def v2_rec(s):
        if s >= 4: return 'GO'
        elif s == 3: return 'CAUTION'
        return 'NO-GO'

    df['v2_rec'] = df['v2_pretrade_score'].apply(v2_rec)

    print("\nBY RECOMMENDATION:")
    print("-" * 70)
    for rec in ['GO', 'CAUTION', 'NO-GO']:
        subset = df[df['v2_rec'] == rec]
        if len(subset) > 0:
            wr = (subset['pnl'] > 0).mean() * 100
            avg = subset['pnl'].mean()
            print(f"  {rec:8s}: {len(subset):3d} trades | Win: {wr:5.1f}% | Avg P&L: {avg:+6.1f}%")

    rho_v2, p_v2 = stats.spearmanr(df['v2_pretrade_score'], df['pnl'])
    print(f"\n  Spearman rho (score vs P&L): {rho_v2:+.3f}  (p={p_v2:.4f})")

    # =====================================================================
    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 80)

    print(f"\n  Spearman rho:  Original={rho_orig:+.3f}  V2={rho_v2:+.3f}  Delta={rho_v2-rho_orig:+.3f}")

    # GO bucket comparison
    orig_go = df[df['orig_rec'] == 'GO']
    v2_go = df[df['v2_rec'] == 'GO']
    orig_nogo = df[df['orig_rec'] == 'NO-GO']
    v2_nogo = df[df['v2_rec'] == 'NO-GO']

    print(f"\n  GO trades:     Original={len(orig_go)} ({(orig_go['pnl']>0).mean()*100:.0f}% WR, {orig_go['pnl'].mean():+.1f}%)   V2={len(v2_go)} ({(v2_go['pnl']>0).mean()*100:.0f}% WR, {v2_go['pnl'].mean():+.1f}%)")
    print(f"  NO-GO trades:  Original={len(orig_nogo)} ({(orig_nogo['pnl']>0).mean()*100:.0f}% WR, {orig_nogo['pnl'].mean():+.1f}%)   V2={len(v2_nogo)} ({(v2_nogo['pnl']>0).mean()*100:.0f}% WR, {v2_nogo['pnl'].mean():+.1f}%)")

    # Separation (GO avg - NO-GO avg)
    orig_sep = orig_go['pnl'].mean() - orig_nogo['pnl'].mean()
    v2_sep = v2_go['pnl'].mean() - v2_nogo['pnl'].mean()
    print(f"\n  GO-NOGO P&L spread:  Original={orig_sep:+.1f}pp   V2={v2_sep:+.1f}pp")

    # =====================================================================
    # Threshold sensitivity for pct_change_3
    # =====================================================================
    print("\n" + "=" * 80)
    print("THRESHOLD SENSITIVITY: pct_change_3")
    print("=" * 80)
    print("(Testing different threshold multipliers to find optimal cutoffs)")

    # Test various flat thresholds across all trades
    for thresh in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
        above = df[df['pct_change_3'] >= thresh]
        below = df[df['pct_change_3'] < thresh]
        if len(above) > 0 and len(below) > 0:
            a_wr = (above['pnl'] > 0).mean() * 100
            a_avg = above['pnl'].mean()
            b_wr = (below['pnl'] > 0).mean() * 100
            b_avg = below['pnl'].mean()
            print(f"  >= {thresh*100:5.0f}%: {len(above):3d} trades ({a_wr:.0f}% WR, {a_avg:+.1f}%) | < {thresh*100:.0f}%: {len(below):3d} ({b_wr:.0f}% WR, {b_avg:+.1f}%)")

    # =====================================================================
    # By cap — V2 thresholds effectiveness
    # =====================================================================
    print("\n" + "=" * 80)
    print("V2 SCORING BY CAP")
    print("=" * 80)

    for cap in ['Medium', 'Small', 'Micro', 'Large', 'ETF']:
        cap_df = df[df['cap'] == cap]
        if len(cap_df) < 3:
            continue
        v2_go = cap_df[cap_df['v2_rec'] == 'GO']
        v2_nogo = cap_df[cap_df['v2_rec'] == 'NO-GO']
        orig_go = cap_df[cap_df['orig_rec'] == 'GO']
        orig_nogo = cap_df[cap_df['orig_rec'] == 'NO-GO']

        print(f"\n  {cap} ({len(cap_df)} trades):")
        if len(orig_go) > 0:
            print(f"    Original GO:    {len(orig_go):2d} | Win: {(orig_go['pnl']>0).mean()*100:.0f}% | Avg: {orig_go['pnl'].mean():+.1f}%")
        if len(orig_nogo) > 0:
            print(f"    Original NO-GO: {len(orig_nogo):2d} | Win: {(orig_nogo['pnl']>0).mean()*100:.0f}% | Avg: {orig_nogo['pnl'].mean():+.1f}%")
        if len(v2_go) > 0:
            print(f"    V2 GO:          {len(v2_go):2d} | Win: {(v2_go['pnl']>0).mean()*100:.0f}% | Avg: {v2_go['pnl'].mean():+.1f}%")
        if len(v2_nogo) > 0:
            print(f"    V2 NO-GO:       {len(v2_nogo):2d} | Win: {(v2_nogo['pnl']>0).mean()*100:.0f}% | Avg: {v2_nogo['pnl'].mean():+.1f}%")

    # =====================================================================
    # By setup type — V2 effectiveness
    # =====================================================================
    print("\n" + "=" * 80)
    print("V2 SCORING BY SETUP TYPE")
    print("=" * 80)

    for setup in df['setup'].value_counts().index:
        sub = df[df['setup'] == setup]
        if len(sub) < 3:
            continue
        v2_go = sub[sub['v2_rec'] == 'GO']
        v2_nogo = sub[sub['v2_rec'] == 'NO-GO']
        orig_go = sub[sub['orig_rec'] == 'GO']
        orig_nogo = sub[sub['orig_rec'] == 'NO-GO']

        print(f"\n  {setup} ({len(sub)} trades, {(sub['pnl']>0).mean()*100:.0f}% WR, {sub['pnl'].mean():+.1f}%):")
        if len(orig_go) > 0:
            print(f"    Orig GO:    {len(orig_go):2d} | Win: {(orig_go['pnl']>0).mean()*100:.0f}% | Avg: {orig_go['pnl'].mean():+.1f}%")
        if len(orig_nogo) > 0:
            print(f"    Orig NO-GO: {len(orig_nogo):2d} | Win: {(orig_nogo['pnl']>0).mean()*100:.0f}% | Avg: {orig_nogo['pnl'].mean():+.1f}%")
        if len(v2_go) > 0:
            print(f"    V2 GO:      {len(v2_go):2d} | Win: {(v2_go['pnl']>0).mean()*100:.0f}% | Avg: {v2_go['pnl'].mean():+.1f}%")
        if len(v2_nogo) > 0:
            print(f"    V2 NO-GO:   {len(v2_nogo):2d} | Win: {(v2_nogo['pnl']>0).mean()*100:.0f}% | Avg: {v2_nogo['pnl'].mean():+.1f}%")

    # =====================================================================
    # Trades that flipped recommendation
    # =====================================================================
    print("\n" + "=" * 80)
    print("TRADES THAT FLIPPED RECOMMENDATION")
    print("=" * 80)

    flipped = df[df['orig_rec'] != df['v2_rec']]
    print(f"\n  {len(flipped)} trades changed recommendation")

    if len(flipped) > 0:
        # Upgraded (NO-GO/CAUTION -> GO)
        upgraded = flipped[(flipped['orig_rec'].isin(['NO-GO', 'CAUTION'])) & (flipped['v2_rec'] == 'GO')]
        if len(upgraded) > 0:
            print(f"\n  UPGRADED to GO: {len(upgraded)} trades")
            print(f"    Win: {(upgraded['pnl']>0).mean()*100:.0f}% | Avg: {upgraded['pnl'].mean():+.1f}%")
            for _, r in upgraded.iterrows():
                print(f"      {r['date']} {r['ticker']:6s} {r['cap']:8s} {r['setup']:20s} P&L: {r['pnl']:+.1f}%  ({r['orig_rec']} -> {r['v2_rec']})")

        # Downgraded (GO -> CAUTION/NO-GO)
        downgraded = flipped[(flipped['orig_rec'] == 'GO') & (flipped['v2_rec'].isin(['NO-GO', 'CAUTION']))]
        if len(downgraded) > 0:
            print(f"\n  DOWNGRADED from GO: {len(downgraded)} trades")
            print(f"    Win: {(downgraded['pnl']>0).mean()*100:.0f}% | Avg: {downgraded['pnl'].mean():+.1f}%")
            for _, r in downgraded.iterrows():
                print(f"      {r['date']} {r['ticker']:6s} {r['cap']:8s} {r['setup']:20s} P&L: {r['pnl']:+.1f}%  ({r['orig_rec']} -> {r['v2_rec']})")

        # Other shifts
        other = flipped[~flipped.index.isin(upgraded.index) & ~flipped.index.isin(downgraded.index)] if len(upgraded) > 0 or len(downgraded) > 0 else flipped
        if len(other) > 0:
            print(f"\n  OTHER SHIFTS: {len(other)} trades")
            for _, r in other.iterrows():
                print(f"      {r['date']} {r['ticker']:6s} {r['cap']:8s} {r['setup']:20s} P&L: {r['pnl']:+.1f}%  ({r['orig_rec']} -> {r['v2_rec']})")


if __name__ == '__main__':
    run_prototype()
