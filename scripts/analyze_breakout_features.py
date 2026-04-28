"""
Phase A — Statistical analysis of breakout features.

Goal: identify which forward-looking features predict breakout magnitude
(measured by `breakout_open_high_pct`) within each setup profile.

Splits trades into two profiles by `t`:
  - D1_news_break    : t == 0 (~14 trades)
  - D2_continuation  : t == 1 (~33 trades)

For each profile, computes Spearman rho + p-value of every forward-looking
feature against `breakout_open_high_pct` (primary Y) and `atr_max_extension`
(secondary Y). Prints sorted tables of significant features (p<0.10) and
saves a CSV summary for review.

Reference distribution for threshold derivation = trade_grade in {'A', 'B'}
(per user direction).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

CSV = REPO / 'data' / 'breakout_data.csv'
OUTPUT = REPO / 'data' / 'breakout_feature_analysis.csv'

# Forward-looking features only (X). EXCLUDES outcome metrics by design.
FORWARD_LOOKING_FEATURES = [
    # Proximity
    'pct_to_52wk_high', 'pct_to_ath',
    'pct_to_30d_high', 'pct_to_60d_high', 'pct_to_90d_high',
    'days_since_52wk_high', 'days_since_ath',

    # Consolidation / squeeze
    'consolidation_days', 'range_contraction_atr', 'bb_width_percentile_6mo',
    'bollinger_width', 'upper_band_distance',
    'vol_dry_up_ratio', 'up_down_vol_ratio_30d',
    'prior_day_range_atr', 'prior_day_close_vs_high_pct', 'consecutive_up_days',

    # Trend
    'pct_from_9ema', 'pct_from_10mav', 'pct_from_20mav', 'pct_from_50mav', 'pct_from_200mav',
    'atr_distance_from_50mav',
    'ma_50_slope_5d', 'consecutive_days_above_50ma',

    # Relative strength
    'rs_vs_spy_30d', 'rs_vs_spy_90d', 'rs_vs_spy_252d',
    'rs_vs_qqq_30d', 'rs_vs_qqq_90d',

    # Premarket / intraday signals
    'gap_pct', 'gap_from_pm_high',
    'percent_of_premarket_vol', 'percent_of_vol_in_first_5_min',
    'percent_of_vol_in_first_10_min', 'percent_of_vol_in_first_15_min',
    'percent_of_vol_in_first_30_min',
    'rvol_score', 'vol_ratio_5min_to_pm',
    'time_of_high_bucket',

    # ATR
    'atr_pct',

    # Pct change (multi-horizon momentum)
    'pct_change_3', 'pct_change_15', 'pct_change_30', 'pct_change_90', 'pct_change_120',

    # Range expansion / contraction (multi-day)
    'day_of_range_pct', 'one_day_before_range_pct',
    'two_day_before_range_pct', 'three_day_before_range_pct',
    'percent_of_vol_one_day_before', 'percent_of_vol_two_day_before',
    'percent_of_vol_three_day_before',

    # Market context
    'spy_5day_return', 'uvxy_close', 'spy_open_close_pct',
]

# D2-only features (only meaningful when t==1)
D2_ONLY_FEATURES = [
    'd1_close_at_high_pct', 'd1_rvol', 'd1_range_atr', 'd1_thrust_atr',
    'overnight_gap_d1_to_d2_pct',
]

# Boolean features (Spearman handles them but we'll point-biserial in tables)
BOOL_FEATURES = ['ma_stack_aligned', 'pm_d2_holds_above_d1_high']

# Setup-tag flags (binary derived from setup_type)
TAG_FEATURES = ['has_ATH_breakout', 'has_52wk_breakout', 'has_IPO_high_breakout']


# ---------------------------------------------------------------------------

def add_setup_flags(df):
    """Derive binary tag columns from comma-joined setup_type."""
    s = df['setup_type'].fillna('').astype(str)
    df['has_ATH_breakout'] = s.str.contains('ATH_breakout').astype(int)
    df['has_52wk_breakout'] = (s.str.contains('52wk_breakout') & ~s.str.contains('IPO_high')).astype(int)
    df['has_IPO_high_breakout'] = s.str.contains('IPO_high_breakout').astype(int)
    return df


def coerce_bool(s):
    """Coerce mixed True/False/'True'/'False'/1/0 to int. NaN stays NaN."""
    return s.map(lambda v: 1 if (v is True or str(v).strip().lower() in ('true', '1'))
                 else (0 if (v is False or str(v).strip().lower() in ('false', '0'))
                       else np.nan))


def spearman_table(df, features, y_col, label):
    """Compute Spearman rho + p-value for each feature vs y_col."""
    rows = []
    y = df[y_col].astype(float)

    for feat in features:
        if feat not in df.columns:
            continue
        x = df[feat]
        if x.dtype == bool or feat in BOOL_FEATURES:
            x = coerce_bool(x)
        try:
            x = pd.to_numeric(x, errors='coerce')
        except Exception:
            continue

        valid = x.notna() & y.notna()
        n = valid.sum()
        if n < 5:
            continue

        rho, p = spearmanr(x[valid], y[valid])
        if pd.isna(rho):
            continue

        rows.append({
            'feature': feat,
            'n': n,
            'rho': round(rho, 3),
            'p': round(p, 4),
            'abs_rho': round(abs(rho), 3),
            'profile': label,
            'y': y_col,
            'sig_p10': p < 0.10,
            'sig_p05': p < 0.05,
        })

    out = pd.DataFrame(rows).sort_values('abs_rho', ascending=False).reset_index(drop=True)
    return out


def reference_thresholds(df_ref, features):
    """For each feature, compute p25 / p50 / p75 over the A+B reference set.

    p25 used as the threshold for "higher is better" criteria (so ~75% of A+B pass).
    p75 used for "lower is better" (so ~75% of A+B pass).
    """
    rows = []
    for feat in features:
        if feat not in df_ref.columns:
            continue
        x = df_ref[feat]
        if x.dtype == bool or feat in BOOL_FEATURES:
            x = coerce_bool(x)
        try:
            x = pd.to_numeric(x, errors='coerce').dropna()
        except Exception:
            continue
        if len(x) < 3:
            continue
        rows.append({
            'feature': feat,
            'n': len(x),
            'p25': round(x.quantile(0.25), 4),
            'p50': round(x.quantile(0.50), 4),
            'p75': round(x.quantile(0.75), 4),
            'mean': round(x.mean(), 4),
        })
    return pd.DataFrame(rows)


def print_table(df, title, top_n=20):
    print(f'\n=== {title} ===')
    if df.empty:
        print('  (no rows)')
        return
    show = df.head(top_n).copy()
    show['rho'] = show['rho'].apply(lambda v: f'{v:+.3f}')
    show['p'] = show['p'].apply(lambda v: f'{v:.4f}')
    show['sig'] = show.apply(
        lambda r: '**' if r['sig_p05'] else ('*' if r['sig_p10'] else ' '), axis=1
    )
    print(show[['feature', 'n', 'rho', 'p', 'sig']].to_string(index=False))


# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv(CSV)
    print(f'Loaded {len(df)} rows from {CSV.name}')

    df = add_setup_flags(df)

    # Coerce trade_grade
    df['trade_grade'] = df['trade_grade'].astype(str).str.strip()

    # Profile split
    d1 = df[df['t'] == 0].copy()
    d2 = df[df['t'] == 1].copy()
    print(f'D1_news_break (t==0): {len(d1)} rows')
    print(f'D2_continuation (t==1): {len(d2)} rows')

    # Reference set: A+B grades only (per user direction)
    ref_grades = ('A', 'B')
    d1_ref = d1[d1['trade_grade'].isin(ref_grades)]
    d2_ref = d2[d2['trade_grade'].isin(ref_grades)]
    print(f'D1_news_break A+B: {len(d1_ref)} rows')
    print(f'D2_continuation A+B: {len(d2_ref)} rows')

    # Features per profile
    d1_feats = FORWARD_LOOKING_FEATURES + TAG_FEATURES + ['ma_stack_aligned']
    d2_feats = FORWARD_LOOKING_FEATURES + TAG_FEATURES + ['ma_stack_aligned'] + D2_ONLY_FEATURES + ['pm_d2_holds_above_d1_high']

    all_results = []

    for profile_label, profile_df, profile_feats, ref_df in [
        ('D1_news_break', d1, d1_feats, d1_ref),
        ('D2_continuation', d2, d2_feats, d2_ref),
    ]:
        if len(profile_df) < 5:
            print(f'\n[skipped] {profile_label}: only {len(profile_df)} rows - too few')
            continue

        # Y1: breakout_open_high_pct (primary)
        t1 = spearman_table(profile_df, profile_feats, 'breakout_open_high_pct', profile_label)
        all_results.append(t1)
        print_table(t1, f'{profile_label} vs breakout_open_high_pct (PRIMARY Y) - n={len(profile_df)}')
        sig = t1[t1['sig_p10']]
        print(f'  Significant at p<0.10: {len(sig)} features')

        # Y2: atr_max_extension (secondary)
        t2 = spearman_table(profile_df, profile_feats, 'atr_max_extension', profile_label)
        all_results.append(t2)
        print_table(t2, f'{profile_label} vs atr_max_extension (SECONDARY Y)')
        sig2 = t2[t2['sig_p10']]
        print(f'  Significant at p<0.10: {len(sig2)} features')

        # Reference thresholds for top-significant features
        top_features = t1[t1['sig_p10']]['feature'].tolist()[:10]
        if top_features:
            ref_table = reference_thresholds(ref_df, top_features)
            if not ref_table.empty:
                print(f'\n  REFERENCE PERCENTILES (A+B grade trades only) - top significant features:')
                print('  ' + ref_table.to_string(index=False).replace('\n', '\n  '))

    # Save combined output
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(OUTPUT, index=False)
        print(f'\nFull analysis saved to {OUTPUT.name}')


if __name__ == '__main__':
    main()
