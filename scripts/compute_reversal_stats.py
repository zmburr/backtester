"""Compute comprehensive stats for the reversal dataset (110 trades, cleaned)."""
import pandas as pd
import numpy as np
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzers.reversal_scorer import ReversalScorer, compute_reversal_intensity
from analyzers.reversal_pretrade import ReversalPretrade, classify_reversal_setup

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
df = pd.read_csv(os.path.join(DATA_DIR, 'reversal_data.csv'))
print(f'Total trades: {len(df)}')

# P&L: short trade, negate open-to-close
df['pnl'] = -df['reversal_open_close_pct'] * 100
# MFE: max favorable excursion (open to low for shorts)
df['mfe_pct'] = -df['reversal_open_low_pct'] * 100
# MFE in ATRs
df['mfe_atrs'] = df['mfe_pct'] / (df['atr_pct'] * 100)
# Captured in ATRs
df['captured_atrs'] = df['pnl'] / (df['atr_pct'] * 100)

# =====================================================================
# 1. SETUP TYPE DISTRIBUTION
# =====================================================================
print('\n' + '=' * 70)
print('1. SETUP TYPE DISTRIBUTION')
print('=' * 70)

setup_counts = df['setup'].value_counts()
print(f'\nSetup types ({len(setup_counts)}):')
for setup, count in setup_counts.items():
    subset = df[df['setup'] == setup]
    wr = (subset['pnl'] > 0).mean() * 100
    avg = subset['pnl'].mean()
    grades = subset['trade_grade'].value_counts().to_dict()
    grade_str = ', '.join(f'{g}:{c}' for g, c in sorted(grades.items()))
    print(f'  {setup:30s}: {count:3d} trades | WR: {wr:5.1f}% | Avg: {avg:+6.1f}% | Grades: {grade_str}')

# Grade distribution overall
print(f'\nOverall grade distribution:')
for grade in ['A', 'B', 'C']:
    subset = df[df['trade_grade'] == grade]
    if len(subset) > 0:
        wr = (subset['pnl'] > 0).mean() * 100
        avg = subset['pnl'].mean()
        print(f'  Grade {grade}: {len(subset):3d} trades | WR: {wr:5.1f}% | Avg: {avg:+6.1f}%')

# Cap distribution
print(f'\nCap distribution:')
for cap in ['ETF', 'Large', 'Medium', 'Small', 'Micro']:
    subset = df[df['cap'] == cap]
    if len(subset) > 0:
        wr = (subset['pnl'] > 0).mean() * 100
        avg = subset['pnl'].mean()
        print(f'  {cap:8s}: {len(subset):3d} trades | WR: {wr:5.1f}% | Avg: {avg:+6.1f}%')

# =====================================================================
# 2. 3DGapFade PROFILE STATS
# =====================================================================
print('\n' + '=' * 70)
print('2. 3DGapFade PROFILE STATS')
print('=' * 70)

gf = df[df['setup'] == '3DGapFade'].copy()
print(f'\n3DGapFade total: {len(gf)}')
print(f'Grade distribution: {gf["trade_grade"].value_counts().to_dict()}')
print(f'Cap distribution: {gf["cap"].value_counts().to_dict()}')

gf_ab = gf[gf['trade_grade'].isin(['A', 'B'])].copy()
print(f'\n3DGapFade A+B: {len(gf_ab)}')
if len(gf_ab) > 0:
    wr = (gf_ab['pnl'] > 0).mean() * 100
    avg = gf_ab['pnl'].mean()
    print(f'  Win rate: {wr:.1f}%')
    print(f'  Avg P&L: {avg:+.1f}%')
    print(f'  Median P&L: {gf_ab["pnl"].median():+.1f}%')

# =====================================================================
# 3. PER-CAP 3DGapFade 25TH PERCENTILE THRESHOLDS (A+B trades)
# =====================================================================
print('\n' + '=' * 70)
print('3. PER-CAP 3DGapFade THRESHOLDS (25th percentile of A+B trades)')
print('=' * 70)

criteria_cols = {
    'pct_from_9ema': 'pct_from_9ema',
    'prior_day_range_atr': 'prior_day_range_atr',
    'rvol_score': 'rvol_score',
    'pct_change_3': 'pct_change_3',
    'gap_pct': 'gap_pct',
}

threshold_output = {}
for cap in ['ETF', 'Large', 'Medium', 'Small', 'Micro']:
    cap_data = gf_ab[gf_ab['cap'] == cap]
    if len(cap_data) == 0:
        print(f'\n  {cap}: NO A+B trades')
        continue
    print(f'\n  {cap} (n={len(cap_data)}):')
    cap_thresholds = {}
    for crit_name, col in criteria_cols.items():
        vals = cap_data[col].dropna()
        if len(vals) > 0:
            p25 = vals.quantile(0.25)
            p50 = vals.median()
            mn = vals.min()
            print(f'    {crit_name:25s}: min={mn:.4f}, p25={p25:.4f}, median={p50:.4f}')
            cap_thresholds[crit_name] = round(p25, 4)
        else:
            print(f'    {crit_name:25s}: NO DATA')
            cap_thresholds[crit_name] = None
    threshold_output[cap] = cap_thresholds

# Print copy-pasteable threshold dicts
print('\n--- Copy-paste for reversal_pretrade.py ---')
for crit_name in criteria_cols:
    parts = []
    for cap in ['ETF', 'Large', 'Medium', 'Small', 'Micro']:
        if cap in threshold_output and threshold_output[cap].get(crit_name) is not None:
            val = threshold_output[cap][crit_name]
            if crit_name in ('pct_from_9ema', 'gap_pct', 'pct_change_3'):
                parts.append(f"'{cap}': {val:.2f}")
            else:
                parts.append(f"'{cap}': {val:.2f}")
        else:
            parts.append(f"'{cap}': ???")
    default_val = threshold_output.get('Medium', {}).get(crit_name, '???')
    if default_val is not None and default_val != '???':
        if crit_name in ('pct_from_9ema', 'gap_pct', 'pct_change_3'):
            parts.append(f"'_default': {default_val:.2f}")
        else:
            parts.append(f"'_default': {default_val:.2f}")
    print(f'    {crit_name}={{')
    print(f'        {", ".join(parts)},')
    print(f'    }},')

# =====================================================================
# 4. GENERIC PRETRADE SCORE DISTRIBUTION (ALL trades, reversal_scorer)
# =====================================================================
print('\n' + '=' * 70)
print('4. GENERIC PRETRADE SCORE DISTRIBUTION (All trades)')
print('=' * 70)

scorer = ReversalScorer()
scored = scorer.score_dataframe(df)
scored['pnl'] = df['pnl'].values

# Score distribution - ALL trades
print('\nALL TRADES - by pretrade_score:')
for sc in range(5, -1, -1):
    s = scored[scored['pretrade_score'] == sc]
    if len(s) > 0:
        wr = (s['pnl'] > 0).mean() * 100
        avg = s['pnl'].mean()
        print(f'  Score {sc}/5: {len(s):3d} trades | WR: {wr:5.1f}% | Avg: {avg:+6.1f}%')

# By pretrade recommendation
print('\nALL TRADES - by recommendation:')
for rec in ['GO', 'CAUTION', 'NO-GO']:
    s = scored[scored['pretrade_recommendation'] == rec]
    if len(s) > 0:
        wr = (s['pnl'] > 0).mean() * 100
        avg = s['pnl'].mean()
        print(f'  {rec:8s}: {len(s):3d} trades | WR: {wr:5.1f}% | Avg: {avg:+6.1f}%')

# =====================================================================
# 5. SCORE_STATISTICS for generate_report (Grade A trades only)
# =====================================================================
print('\n' + '=' * 70)
print('5. SCORE_STATISTICS (Grade A trades, for generate_report.py)')
print('=' * 70)

grade_a = scored[scored['trade_grade'] == 'A'].copy()
print(f'\nGrade A trades: {len(grade_a)}')

print('\nGrade A - by pretrade_score:')
print('--- Copy-paste for generate_report.py SCORE_STATISTICS ---')
print('SCORE_STATISTICS = {')
for sc in range(5, -1, -1):
    s = grade_a[grade_a['pretrade_score'] == sc]
    if len(s) > 0:
        wr = (s['pnl'] > 0).mean() * 100
        avg = s['pnl'].mean()
        print(f"    {sc}: {{'trades': {len(s)}, 'win_rate': {wr:.1f}, 'avg_pnl': {avg:.1f}}},")
    else:
        print(f"    {sc}: {{'trades': 0, 'win_rate': 0.0, 'avg_pnl': 0.0}},")
print('}')

# GO/CAUTION/NO-GO summary for Grade A
print('\nGrade A - GO/CAUTION/NO-GO summary:')
for rec in ['GO', 'CAUTION', 'NO-GO']:
    s = grade_a[grade_a['pretrade_recommendation'] == rec]
    if len(s) > 0:
        wr = (s['pnl'] > 0).mean() * 100
        avg = s['pnl'].mean()
        print(f'  {rec:8s}: {len(s):3d} trades | WR: {wr:5.1f}% | Avg: {avg:+6.1f}%')

# =====================================================================
# 6. EXIT TARGET HIT RATES BY CAP (Grade A trades)
# =====================================================================
print('\n' + '=' * 70)
print('6. EXIT TARGET HIT RATES BY CAP (Grade A trades)')
print('=' * 70)

for cap in ['ETF', 'Large', 'Medium', 'Small', 'Micro']:
    cap_a = grade_a[grade_a['cap'] == cap].copy()
    if len(cap_a) == 0:
        print(f'\n  {cap}: NO Grade A trades')
        continue

    mfe_atrs = cap_a['mfe_atrs'].dropna()
    captured_atrs = cap_a['captured_atrs'].dropna()
    mfe_pct = cap_a['mfe_pct'].dropna()
    captured_pct = cap_a['pnl'].dropna()

    avg_mfe = mfe_pct.mean()
    avg_cap = captured_pct.mean()
    efficiency = (avg_cap / avg_mfe * 100) if avg_mfe > 0 else 0

    # ATR-based hit rates
    hit_1x = (mfe_atrs >= 1.0).mean() * 100
    hit_1_5x = (mfe_atrs >= 1.5).mean() * 100
    hit_2x = (mfe_atrs >= 2.0).mean() * 100

    # Gap fill: stock drops enough from open to fill the gap
    gap_data = cap_a[['mfe_pct', 'gap_pct']].dropna()
    if len(gap_data) > 0:
        gap_filled = (gap_data['mfe_pct'] / 100 >= gap_data['gap_pct']).mean() * 100
    else:
        gap_filled = 0

    # Time to MFE (from reversal_duration if available)
    # reversal_duration is in timedelta format, try to parse
    avg_time = None
    if 'reversal_duration' in cap_a.columns:
        try:
            durations = pd.to_timedelta(cap_a['reversal_duration'].dropna())
            avg_time = durations.mean().total_seconds() / 60
        except Exception:
            pass

    print(f'\n  {cap} (n={len(cap_a)}):')
    print(f'    Avg MFE: {avg_mfe:.1f}% | Avg Captured: {avg_cap:.1f}% | Efficiency: {efficiency:.1f}%')
    print(f'    MFE ATRs: p25={mfe_atrs.quantile(0.25):.2f}, med={mfe_atrs.median():.2f}, p75={mfe_atrs.quantile(0.75):.2f}')
    print(f'    Hit 1.0x ATR: {hit_1x:.0f}% | Hit 1.5x ATR: {hit_1_5x:.0f}% | Hit 2.0x ATR: {hit_2x:.0f}%')
    print(f'    Hit Gap Fill: {gap_filled:.0f}%')
    if avg_time:
        print(f'    Avg Time to MFE: {avg_time:.0f} min')

# =====================================================================
# 7. CAP_STATISTICS for exit_targets.py (Grade A trades)
# =====================================================================
print('\n' + '=' * 70)
print('7. CAP_STATISTICS (for exit_targets.py)')
print('=' * 70)

print('\n--- Copy-paste for exit_targets.py CAP_STATISTICS ---')
print('CAP_STATISTICS = {')
for cap in ['Large', 'ETF', 'Medium', 'Small', 'Micro']:
    cap_a = grade_a[grade_a['cap'] == cap].copy()
    if len(cap_a) == 0:
        print(f"    '{cap}': {{  # NO Grade A trades  }},")
        continue

    mfe_atrs = cap_a['mfe_atrs'].dropna()
    mfe_pct = cap_a['mfe_pct'].dropna()
    captured_pct = cap_a['pnl'].dropna()

    avg_mfe = mfe_pct.mean()
    avg_cap = captured_pct.mean()
    efficiency = (avg_cap / avg_mfe * 100) if avg_mfe > 0 else 0

    hit_1x = (mfe_atrs >= 1.0).mean() * 100
    hit_1_5x = (mfe_atrs >= 1.5).mean() * 100
    hit_2x = (mfe_atrs >= 2.0).mean() * 100

    gap_data = cap_a[['mfe_pct', 'gap_pct']].dropna()
    gap_filled = (gap_data['mfe_pct'] / 100 >= gap_data['gap_pct']).mean() * 100 if len(gap_data) > 0 else 0

    avg_time = None
    if 'reversal_duration' in cap_a.columns:
        try:
            durations = pd.to_timedelta(cap_a['reversal_duration'].dropna())
            avg_time = int(durations.mean().total_seconds() / 60)
        except Exception:
            avg_time = 0

    print(f"    '{cap}': {{")
    print(f"        'avg_mfe': {avg_mfe:.1f},")
    print(f"        'avg_captured': {avg_cap:.1f},")
    print(f"        'efficiency': {efficiency:.1f},")
    print(f"        'hit_1x_atr': {hit_1x:.0f},")
    print(f"        'hit_1_5x_atr': {hit_1_5x:.0f},")
    print(f"        'hit_2x_atr': {hit_2x:.0f},")
    print(f"        'hit_gap_fill': {gap_filled:.0f},")
    if avg_time is not None:
        print(f"        'avg_time_to_mfe': {avg_time},")
    print(f"    }},")
print('}')

# =====================================================================
# 8. EXIT FRAMEWORK HIT RATES (for exit_targets.py tier updates)
# =====================================================================
print('\n' + '=' * 70)
print('8. EXIT FRAMEWORK HIT RATES (Grade A trades)')
print('=' * 70)

for cap in ['Large', 'ETF', 'Medium', 'Small', 'Micro']:
    cap_a = grade_a[grade_a['cap'] == cap].copy()
    if len(cap_a) == 0:
        continue

    mfe_atrs = cap_a['mfe_atrs'].dropna()
    gap_data = cap_a[['mfe_pct', 'gap_pct']].dropna()
    gap_filled = (gap_data['mfe_pct'] / 100 >= gap_data['gap_pct']).mean() * 100 if len(gap_data) > 0 else 0

    print(f'\n  {cap} (n={len(cap_a)}):')
    for mult in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        hit = (mfe_atrs >= mult).mean() * 100
        print(f'    {mult:.1f}x ATR: {hit:.0f}%')
    print(f'    Gap Fill: {gap_filled:.0f}%')

# =====================================================================
# 9. INTENSITY CORRELATION
# =====================================================================
print('\n' + '=' * 70)
print('9. INTENSITY CORRELATION')
print('=' * 70)

try:
    from scipy.stats import spearmanr

    has_intensity = scored['intensity'].notna()
    intensity_df = scored[has_intensity].copy()
    print(f'\nTrades with intensity: {len(intensity_df)} / {len(scored)}')

    if len(intensity_df) > 0:
        rho, p = spearmanr(intensity_df['intensity'], intensity_df['pnl'])
        print(f'Spearman rho (intensity vs P&L): {rho:+.3f} (p={p:.4f})')
        print(f'Intensity range: {intensity_df["intensity"].min():.0f} - {intensity_df["intensity"].max():.0f}')
        print(f'Mean: {intensity_df["intensity"].mean():.1f} | Median: {intensity_df["intensity"].median():.1f}')

    # Pretrade score vs P&L
    rho_pt, p_pt = spearmanr(scored['pretrade_score'], scored['pnl'])
    print(f'Spearman rho (pretrade_score vs P&L): {rho_pt:+.3f} (p={p_pt:.4f})')
except ImportError:
    print('scipy not available, skipping correlation')

# =====================================================================
# 10. 3DGapFade PRETRADE SCORING
# =====================================================================
print('\n' + '=' * 70)
print('10. 3DGapFade PRETRADE SCORING (reversal_pretrade thresholds)')
print('=' * 70)

pretrade = ReversalPretrade()
gf_results = []

for _, row in gf.iterrows():
    metrics = row.to_dict()
    if 'one_day_before_range_pct' in metrics and pd.notna(metrics.get('one_day_before_range_pct')):
        if 'prior_day_range_atr' not in metrics or pd.isna(metrics.get('prior_day_range_atr')):
            metrics['prior_day_range_atr'] = metrics['one_day_before_range_pct']

    cap = row.get('cap', 'Medium')
    if cap is None or (isinstance(cap, float) and pd.isna(cap)):
        cap = 'Medium'

    result = pretrade.validate(
        ticker=row['ticker'],
        metrics=metrics,
        setup_type='3DGapFade',
        cap=cap,
    )
    pnl = -row.get('reversal_open_close_pct', 0) * 100

    gf_results.append({
        'date': row['date'],
        'ticker': row['ticker'],
        'trade_grade': row['trade_grade'],
        'cap': cap,
        'score': result.score,
        'recommendation': result.recommendation,
        'pnl': pnl,
    })

gf_df = pd.DataFrame(gf_results)
print(f'\n3DGapFade trades scored: {len(gf_df)}')

print('\nBy score:')
for sc in range(5, -1, -1):
    s = gf_df[gf_df['score'] == sc]
    if len(s) > 0:
        wr = (s['pnl'] > 0).mean() * 100
        avg = s['pnl'].mean()
        print(f'  Score {sc}/5: {len(s):3d} trades | WR: {wr:5.1f}% | Avg: {avg:+6.1f}%')

print('\nBy recommendation:')
for rec in ['GO', 'CAUTION', 'NO-GO']:
    s = gf_df[gf_df['recommendation'] == rec]
    if len(s) > 0:
        wr = (s['pnl'] > 0).mean() * 100
        avg = s['pnl'].mean()
        print(f'  {rec:8s}: {len(s):3d} trades | WR: {wr:5.1f}% | Avg: {avg:+6.1f}%')

# A+B only
gf_ab_scored = gf_df[gf_df['trade_grade'].isin(['A', 'B'])]
print(f'\n3DGapFade A+B scored: {len(gf_ab_scored)}')
print('\nA+B by recommendation:')
for rec in ['GO', 'CAUTION', 'NO-GO']:
    s = gf_ab_scored[gf_ab_scored['recommendation'] == rec]
    if len(s) > 0:
        wr = (s['pnl'] > 0).mean() * 100
        avg = s['pnl'].mean()
        print(f'  {rec:8s}: {len(s):3d} trades | WR: {wr:5.1f}% | Avg: {avg:+6.1f}%')

# =====================================================================
# 11. BACKSCANNER PRE-FILTER VALIDATION
# =====================================================================
print('\n' + '=' * 70)
print('11. BACKSCANNER PRE-FILTER VALIDATION')
print('=' * 70)

# Check if any A+B 3DGapFade trades would be filtered out by backscanner gates
print('\n3DGapFade A+B gate check:')
print(f'  consecutive_up_days >= 2: {(gf_ab["consecutive_up_days"] >= 2).sum()}/{len(gf_ab)} pass')
print(f'  gap_pct >= 0.04: {(gf_ab["gap_pct"] >= 0.04).sum()}/{len(gf_ab)} pass')
print(f'  pct_from_9ema >= 0.25 (pre-filter): {(gf_ab["pct_from_9ema"] >= 0.25).sum()}/{len(gf_ab)} pass')
print(f'  pct_from_9ema min: {gf_ab["pct_from_9ema"].min():.4f}')
print(f'  gap_pct min: {gf_ab["gap_pct"].min():.4f}')
print(f'  consecutive_up_days min: {gf_ab["consecutive_up_days"].min()}')

# =====================================================================
# 12. HEADER HTML NUMBERS
# =====================================================================
print('\n' + '=' * 70)
print('12. HEADER HTML NUMBERS (for generate_report.py)')
print('=' * 70)

# Score table rows for HTML
print(f'\nGrade A count: {len(grade_a)}')

# GO = 4-5, CAUTION = 3, NO-GO = <3
go = grade_a[grade_a['pretrade_recommendation'] == 'GO']
caution = grade_a[grade_a['pretrade_recommendation'] == 'CAUTION']
nogo = grade_a[grade_a['pretrade_recommendation'] == 'NO-GO']

if len(go) > 0:
    go_wr = (go['pnl'] > 0).mean() * 100
    go_avg = go['pnl'].mean()
    print(f'GO (4-5/5): {len(go)} trades, {go_wr:.0f}% win rate, {go_avg:+.1f}% avg')
if len(caution) > 0:
    c_wr = (caution['pnl'] > 0).mean() * 100
    c_avg = caution['pnl'].mean()
    print(f'CAUTION (3/5): {len(caution)} trades, {c_wr:.0f}% win, {c_avg:+.1f}% avg')
if len(nogo) > 0:
    n_wr = (nogo['pnl'] > 0).mean() * 100
    n_avg = nogo['pnl'].mean()
    print(f'NO-GO (<3): {len(nogo)} trades, {n_wr:.0f}% win, {n_avg:+.1f}% avg')

# Exit target hit rates by cap for HTML table (Grade A trades)
print('\nExit target table (Grade A):')
for cap in ['Large', 'ETF', 'Medium', 'Small', 'Micro']:
    cap_a = grade_a[grade_a['cap'] == cap].copy()
    if len(cap_a) == 0:
        continue
    mfe_atrs = cap_a['mfe_atrs'].dropna()
    gap_data = cap_a[['mfe_pct', 'gap_pct']].dropna()
    gap_hit = int((gap_data['mfe_pct'] / 100 >= gap_data['gap_pct']).mean() * 100) if len(gap_data) > 0 else 0
    h1 = int((mfe_atrs >= 1.0).mean() * 100)
    h15 = int((mfe_atrs >= 1.5).mean() * 100)
    h2 = int((mfe_atrs >= 2.0).mean() * 100)
    h25 = int((mfe_atrs >= 2.5).mean() * 100)
    print(f'  {cap}: n={len(cap_a)}, 1.0x={h1}%, 1.5x={h15}%, 2.0x={h2}%, 2.5x={h25}%, GapFill={gap_hit}%')

# =====================================================================
# SAVE JSON
# =====================================================================
stats_json = {
    'total_trades': len(df),
    'grade_a_count': int(len(grade_a)),
    'setup_distribution': setup_counts.to_dict(),
    '3dgapfade': {
        'total': int(len(gf)),
        'ab_count': int(len(gf_ab)),
        'ab_win_rate': round((gf_ab['pnl'] > 0).mean() * 100, 1) if len(gf_ab) > 0 else 0,
        'ab_avg_pnl': round(gf_ab['pnl'].mean(), 1) if len(gf_ab) > 0 else 0,
    },
    'thresholds': threshold_output,
}

json_path = os.path.join(DATA_DIR, 'reversal_stats.json')
with open(json_path, 'w') as f:
    json.dump(stats_json, f, indent=2, default=str)
print(f'\nStats saved to {json_path}')
