"""
Test: Does ATR-adjusting the scorer metrics fix the cross-cap comparability problem?

Idea: Instead of raw % (where 10% on NVDA != 10% on a micro cap), express
everything in ATR multiples so setups are directly comparable regardless of cap.

Metrics to ATR-adjust:
  - pct_from_9ema / atr_pct  = distance from 9EMA in ATRs
  - pct_change_3 / atr_pct   = 3-day move in ATRs
  - gap_pct / atr_pct         = gap in ATRs
  - prior_day_range_atr       = ALREADY ATR-adjusted
  - rvol_score                = relative volume, no ATR needed

Outcome also ATR-adjusted: reversal_open_low_pct / atr_pct
"""

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('data/reversal_data.csv')
a = df[df['trade_grade'] == 'A'].copy()

print(f"Grade A trades: {len(a)}")
print(f"atr_pct range: {a['atr_pct'].min():.4f} to {a['atr_pct'].max():.4f}, nulls: {a['atr_pct'].isna().sum()}")
print()

# ATR-adjust the 3 raw-pct metrics
a['ema9_atr'] = a['pct_from_9ema'] / a['atr_pct']
a['mom3_atr'] = a['pct_change_3'] / a['atr_pct']
a['gap_atr']  = a['gap_pct'] / a['atr_pct']

# Outcomes
a['rev_mag_atr'] = -a['reversal_open_low_pct'] / a['atr_pct']
a['rev_close_atr'] = -a['reversal_open_close_pct'] / a['atr_pct']
a['rev_mag_pct'] = -a['reversal_open_low_pct'] * 100

# ============================================================
print("=" * 70)
print("1. SANITY CHECK: Do ATR-adjusted metrics compress the cap spread?")
print("=" * 70)
print()

for raw, adj, label in [
    ('pct_from_9ema', 'ema9_atr', '9EMA distance'),
    ('pct_change_3',  'mom3_atr', '3-day momentum'),
    ('gap_pct',       'gap_atr',  'Gap'),
]:
    print(f"  {label}:")
    print(f"    {'Cap':>8s}  {'Raw mean':>10s}  {'ATR-adj mean':>13s}  n")
    print(f"    {'-'*40}")
    for cap in ['ETF', 'Large', 'Medium', 'Small', 'Micro']:
        sub = a[a['cap'] == cap]
        if len(sub) > 0:
            print(f"    {cap:>8s}  {sub[raw].mean():+10.3f}  {sub[adj].mean():+13.2f}  {len(sub)}")
    # Coefficient of variation across cap means
    cap_means_raw = [a[a['cap']==c][raw].mean() for c in ['ETF','Large','Medium','Small','Micro'] if len(a[a['cap']==c]) > 0]
    cap_means_adj = [a[a['cap']==c][adj].mean() for c in ['ETF','Large','Medium','Small','Micro'] if len(a[a['cap']==c]) > 0]
    cv_raw = np.std(cap_means_raw) / np.mean(cap_means_raw) if np.mean(cap_means_raw) != 0 else 999
    cv_adj = np.std(cap_means_adj) / np.mean(cap_means_adj) if np.mean(cap_means_adj) != 0 else 999
    print(f"    Cross-cap CV: raw={cv_raw:.2f}  ATR-adj={cv_adj:.2f}  {'COMPRESSED' if cv_adj < cv_raw else 'worse'}")
    print()

# Also check the outcome variable
print(f"  Reversal magnitude:")
print(f"    {'Cap':>8s}  {'Raw % mean':>11s}  {'ATR-adj mean':>13s}  n")
print(f"    {'-'*40}")
for cap in ['ETF', 'Large', 'Medium', 'Small', 'Micro']:
    sub = a[a['cap'] == cap]
    if len(sub) > 0:
        print(f"    {cap:>8s}  {sub['rev_mag_pct'].mean():+11.1f}  {sub['rev_mag_atr'].mean():+13.2f}  {len(sub)}")
print()

# ============================================================
print("=" * 70)
print("2. INDIVIDUAL CRITERIA: Raw vs ATR-adjusted correlation with reversal")
print("=" * 70)
print("  Outcome: reversal open-to-low in ATR multiples")
print()

comparisons = [
    ('pct_from_9ema',      'ema9_atr',           '9EMA distance'),
    ('pct_change_3',       'mom3_atr',           '3-day momentum'),
    ('gap_pct',            'gap_atr',            'Gap'),
    ('prior_day_range_atr','prior_day_range_atr','Range/ATR (already adj)'),
    ('rvol_score',         'rvol_score',         'RVOL (already adj)'),
]

print(f"  {'Criterion':<25s} {'Raw rho':>8s} {'p':>7s} {'ATR rho':>8s} {'p':>7s} {'Better?':>8s}")
print("  " + "-" * 65)
for raw_col, adj_col, label in comparisons:
    v1 = a[[raw_col, 'rev_mag_atr']].dropna()
    rho_raw, p_raw = stats.spearmanr(v1[raw_col], v1['rev_mag_atr'])
    v2 = a[[adj_col, 'rev_mag_atr']].dropna()
    rho_adj, p_adj = stats.spearmanr(v2[adj_col], v2['rev_mag_atr'])
    better = "YES" if abs(rho_adj) > abs(rho_raw) + 0.01 else ("same" if raw_col == adj_col else "no")
    print(f"  {label:<25s} {rho_raw:+.3f}  {p_raw:.4f} {rho_adj:+.3f}  {p_adj:.4f}  {better:>6s}")

# ============================================================
print()
print("=" * 70)
print("3. COMPOSITE SCORE COMPARISON")
print("=" * 70)

# Raw continuous composite
for col in ['pct_from_9ema', 'prior_day_range_atr', 'rvol_score', 'pct_change_3', 'gap_pct']:
    a[f'raw_{col}_pctile'] = a[col].rank(pct=True)
a['raw_composite'] = (a['raw_pct_from_9ema_pctile'] + a['raw_prior_day_range_atr_pctile'] +
                       a['raw_rvol_score_pctile'] + a['raw_pct_change_3_pctile'] + a['raw_gap_pct_pctile'])

# ATR-adjusted continuous composite
for col in ['ema9_atr', 'prior_day_range_atr', 'rvol_score', 'mom3_atr', 'gap_atr']:
    a[f'atr_{col}_pctile'] = a[col].rank(pct=True)
a['atr_composite'] = (a['atr_ema9_atr_pctile'] + a['atr_prior_day_range_atr_pctile'] +
                       a['atr_rvol_score_pctile'] + a['atr_mom3_atr_pctile'] + a['atr_gap_atr_pctile'])

print()
print("  Correlation with ATR-adjusted reversal magnitude:")
for score_col, label in [('raw_composite', 'Raw % composite'),
                          ('atr_composite', 'ATR-adjusted composite')]:
    v = a[[score_col, 'rev_mag_atr']].dropna()
    rho, p = stats.spearmanr(v[score_col], v['rev_mag_atr'])
    r, rp = stats.pearsonr(v[score_col], v['rev_mag_atr'])
    print(f"    {label:<25s}: rho={rho:+.3f} (p={p:.4f})  r={r:+.3f}")

print()
print("  Correlation with raw % reversal magnitude:")
for score_col, label in [('raw_composite', 'Raw % composite'),
                          ('atr_composite', 'ATR-adjusted composite')]:
    v = a[[score_col, 'rev_mag_pct']].dropna()
    rho, p = stats.spearmanr(v[score_col], v['rev_mag_pct'])
    print(f"    {label:<25s}: rho={rho:+.3f} (p={p:.4f})")

# ============================================================
print()
print("=" * 70)
print("4. THE KEY TEST: Does ATR-adjustment fix the cap bias?")
print("=" * 70)

for score_col, label in [('raw_composite', 'RAW composite'), ('atr_composite', 'ATR-ADJUSTED composite')]:
    a['tercile'] = pd.qcut(a[score_col], q=3, labels=['Bottom', 'Middle', 'Top'])
    print(f"\n  {label}:")
    ct = pd.crosstab(a['tercile'], a['cap'])
    for tercile in ['Top', 'Middle', 'Bottom']:
        row = ct.loc[tercile]
        caps_str = "  ".join([f"{c}:{row[c]}" for c in ['ETF','Large','Medium','Small','Micro'] if c in row.index])
        print(f"    {tercile:>8s}: {caps_str}")

# ============================================================
print()
print("=" * 70)
print("5. PERFORMANCE BY ATR-ADJUSTED COMPOSITE TERCILE")
print("=" * 70)

a['atr_tercile'] = pd.qcut(a['atr_composite'], q=3, labels=['Bottom', 'Middle', 'Top'])

print(f"\n  {'Tercile':<10s} {'N':>4s} {'Avg(ATR)':>10s} {'Med(ATR)':>10s} {'Avg(%)':>10s} {'Med(%)':>10s} {'WR':>6s}")
print("  " + "-" * 55)
for t in ['Top', 'Middle', 'Bottom']:
    sub = a[a['atr_tercile'] == t]
    wr = (sub['rev_close_atr'] > 0).mean() * 100
    print(f"  {t:<10s} {len(sub):4d} {sub['rev_mag_atr'].mean():+10.2f} {sub['rev_mag_atr'].median():+10.2f} "
          f"{sub['rev_mag_pct'].mean():+10.1f} {sub['rev_mag_pct'].median():+10.1f} {wr:5.0f}%")

print(f"\n  Cap breakdown per ATR-adjusted tercile:")
ct2 = pd.crosstab(a['atr_tercile'], a['cap'])
for t in ['Top', 'Middle', 'Bottom']:
    row = ct2.loc[t]
    caps_str = "  ".join([f"{c}:{row[c]}" for c in ['ETF','Large','Medium','Small','Micro'] if c in row.index and row[c] > 0])
    print(f"    {t:>8s}: {caps_str}")

# ============================================================
print()
print("=" * 70)
print("6. LARGEST RE-RANKINGS: Who moved between raw and ATR-adjusted?")
print("=" * 70)

a['raw_rank'] = a['raw_composite'].rank(ascending=False)
a['atr_rank'] = a['atr_composite'].rank(ascending=False)
a['rank_change'] = a['raw_rank'] - a['atr_rank']

print("\n  Biggest MOVERS UP (ATR-adjustment PROMOTED these):")
movers_up = a.nlargest(8, 'rank_change')
for _, r in movers_up.iterrows():
    print(f"    {r['ticker']:>6s} {r['cap']:>8s} {r['date']:>12s}  "
          f"raw_rank:{int(r['raw_rank']):>3d} -> atr_rank:{int(r['atr_rank']):>3d} ({r['rank_change']:+.0f})  "
          f"rev:{r['rev_mag_pct']:+.1f}%  rev_atr:{r['rev_mag_atr']:+.1f}")

print("\n  Biggest MOVERS DOWN (ATR-adjustment DEMOTED these):")
movers_dn = a.nsmallest(8, 'rank_change')
for _, r in movers_dn.iterrows():
    print(f"    {r['ticker']:>6s} {r['cap']:>8s} {r['date']:>12s}  "
          f"raw_rank:{int(r['raw_rank']):>3d} -> atr_rank:{int(r['atr_rank']):>3d} ({r['rank_change']:+.0f})  "
          f"rev:{r['rev_mag_pct']:+.1f}%  rev_atr:{r['rev_mag_atr']:+.1f}")

# ============================================================
print()
print("=" * 70)
print("7. BOTTOM TERCILE ATR-ADJUSTED: Are these still worth taking?")
print("=" * 70)

bottom = a[a['atr_tercile'] == 'Bottom'].sort_values('rev_mag_atr', ascending=False)
print(f"\n  Bottom tercile by ATR-adjusted score (n={len(bottom)}):")
for _, r in bottom.iterrows():
    won = 'W' if r['rev_close_atr'] > 0 else 'L'
    print(f"  {won} {r['date']:>12s} {r['ticker']:>6s} {r['cap']:>8s}  "
          f"rev:{r['rev_mag_pct']:+5.1f}%({r['rev_mag_atr']:+.1f} ATR)  "
          f"9ema:{r['ema9_atr']:+.1f}ATR  3d:{r['mom3_atr']:+.1f}ATR  gap:{r['gap_atr']:+.1f}ATR")

# ============================================================
print()
print("=" * 70)
print("8. SUMMARY")
print("=" * 70)

v_raw = a[['raw_composite', 'rev_mag_atr']].dropna()
rho_raw, p_raw = stats.spearmanr(v_raw['raw_composite'], v_raw['rev_mag_atr'])
v_atr = a[['atr_composite', 'rev_mag_atr']].dropna()
rho_atr, p_atr = stats.spearmanr(v_atr['atr_composite'], v_atr['rev_mag_atr'])

print(f"""
  Raw composite vs ATR-adj reversal:     rho = {rho_raw:+.3f} (p={p_raw:.4f})
  ATR-adj composite vs ATR-adj reversal: rho = {rho_atr:+.3f} (p={p_atr:.4f})
""")
