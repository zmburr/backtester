"""
Validate compute_reversal_intensity() against historical reversal_data.csv grades.

Pass criterion:
  - Median(A) > Median(B) > Median(C)
  - IQR of A and IQR of C do not overlap

Also reports Spearman rho vs reversal_open_close_pct (a P&L proxy).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from analyzers.reversal_scorer import compute_reversal_intensity

INPUT_COLS = ['atr_pct', 'pct_from_9ema', 'pct_change_3', 'gap_pct',
              'prior_day_range_atr', 'rvol_score', 'pct_from_50mav']


def main():
    csv_path = Path(__file__).resolve().parent.parent / 'data' / 'reversal_data.csv'
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    df['intensity'] = df.apply(_score_row, axis=1)
    scored = df[df['intensity'].notna()].copy()
    print(f"Scoreable rows: {len(scored)} / {len(df)}")

    print("\n=== Intensity by trade_grade ===")
    summary = scored.groupby('trade_grade')['intensity'].describe()[
        ['count', 'mean', '25%', '50%', '75%', 'std']
    ].round(1)
    print(summary.to_string())

    print("\n=== Pass criterion check ===")
    medians = scored.groupby('trade_grade')['intensity'].median()
    iqrs = scored.groupby('trade_grade')['intensity'].quantile([0.25, 0.75]).unstack()
    iqrs.columns = ['p25', 'p75']
    grades_present = [g for g in ['A', 'B', 'C', 'F'] if g in medians.index]
    print(f"Grades present: {grades_present}")
    for g in grades_present:
        print(f"  {g}: median={medians[g]:.1f}, IQR=[{iqrs.loc[g,'p25']:.1f}, {iqrs.loc[g,'p75']:.1f}]")

    if 'A' in medians.index and 'B' in medians.index:
        gap_ab = medians['A'] - medians['B']
        print(f"  A median - B median = {gap_ab:+.1f}")
    if 'B' in medians.index and 'C' in medians.index:
        gap_bc = medians['B'] - medians['C']
        print(f"  B median - C median = {gap_bc:+.1f}")
    if 'A' in iqrs.index and 'C' in iqrs.index:
        a_p25 = iqrs.loc['A', 'p25']
        c_p75 = iqrs.loc['C', 'p75']
        overlap = c_p75 >= a_p25
        print(f"  A p25={a_p25:.1f} vs C p75={c_p75:.1f} -> "
              f"{'OVERLAP (FAIL)' if overlap else 'CLEAN SEPARATION (PASS)'}")

    print("\n=== Top-half A vs bottom-half A (within-A discrimination) ===")
    if 'A' in scored['trade_grade'].values and 'reversal_open_close_pct' in scored.columns:
        a_only = scored[scored['trade_grade'] == 'A'].dropna(subset=['reversal_open_close_pct']).copy()
        if len(a_only) >= 6:
            # More-negative reversal_open_close_pct = bigger short reversal = "better" A trade
            median_pnl = a_only['reversal_open_close_pct'].median()
            top_a = a_only[a_only['reversal_open_close_pct'] <= median_pnl]
            bot_a = a_only[a_only['reversal_open_close_pct'] > median_pnl]
            print(f"  Top-half A intensity median: {top_a['intensity'].median():.1f} (n={len(top_a)})")
            print(f"  Bot-half A intensity median: {bot_a['intensity'].median():.1f} (n={len(bot_a)})")
        else:
            print(f"  Skipped — only {len(a_only)} A trades with P&L data")

    print("\n=== Spearman rho: intensity vs reversal_open_close_pct ===")
    if 'reversal_open_close_pct' in scored.columns:
        x = scored.dropna(subset=['reversal_open_close_pct'])
        # Negate P&L so "more reversal" = higher number, easier to read
        rho, p = spearmanr(x['intensity'], -x['reversal_open_close_pct'])
        print(f"  rho = {rho:+.3f} (p = {p:.4f}, n = {len(x)})")
        print(f"  (positive rho = higher intensity -> bigger short reversal)")

    print("\n=== Distribution buckets ===")
    bins = [0, 25, 50, 75, 100]
    labels = ['0-25', '25-50', '50-75', '75-100']
    scored['bucket'] = pd.cut(scored['intensity'], bins=bins, labels=labels, include_lowest=True)
    cross = pd.crosstab(scored['bucket'], scored['trade_grade'])
    print(cross.to_string())


def _score_row(row):
    metrics = {col: row.get(col) for col in INPUT_COLS}
    if any(pd.isna(metrics[c]) for c in INPUT_COLS):
        return None
    cap = row.get('cap')
    if pd.isna(cap):
        cap = None
    result = compute_reversal_intensity(metrics, cap=cap)
    return result.get('composite')


if __name__ == '__main__':
    main()
