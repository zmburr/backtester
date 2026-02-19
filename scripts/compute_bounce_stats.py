"""Compute comprehensive bounce stats from the expanded 123-trade dataset."""
import pandas as pd
import numpy as np
import json
import sys

df = pd.read_csv('data/bounce_data.csv')

# Convert numeric columns
num_cols = ['bounce_open_close_pct','bounce_open_high_pct','bounce_open_low_pct','atr_pct',
    'selloff_total_pct','consecutive_down_days','pct_off_30d_high','gap_pct',
    'one_day_before_range_pct','pct_change_3','pct_off_52wk_high','pct_change_30',
    'pct_from_50mav','pct_from_200mav','pct_change_15','prior_day_close_vs_low_pct',
    'lower_band_distance','closed_outside_lower_band','bollinger_width',
    'spy_open_close_pct','percent_of_vol_on_breakout_day','time_of_low_bucket',
    'day_of_range_pct']
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

stats = {'total_trades': len(df)}

# === BY SETUP TYPE (GapFade only) ===
for setup in ['GapFade_weakstock', 'GapFade_strongstock']:
    s = df[df['Setup'] == setup]
    oc = s['bounce_open_close_pct'].dropna()
    oh = s['bounce_open_high_pct'].dropna()
    ol = s['bounce_open_low_pct'].dropna()

    valid_atr = s['bounce_open_high_pct'].notna() & s['atr_pct'].notna() & (s['atr_pct'] > 0)
    move_h = s.loc[valid_atr, 'bounce_open_high_pct'] / s.loc[valid_atr, 'atr_pct']
    move_l = s.loc[valid_atr, 'bounce_open_low_pct'] / s.loc[valid_atr, 'atr_pct']

    key = setup.replace('GapFade_', '')
    stats[key] = {
        'n': len(s),
        'wr': round((oc > 0).sum() / len(oc) * 100, 1) if len(oc) > 0 else 0,
        'avg_pnl': round(oc.mean() * 100, 1),
        'med_oc': round(oc.median() * 100, 1),
        'med_oh': round(oh.median() * 100, 1),
        'med_ol': round(ol.median() * 100, 1),
        'avg_dip': round(ol.mean() * 100, 1),
        'med_high_atrs': round(move_h.median(), 2),
        'p75_high_atrs': round(move_h.quantile(0.75), 2),
        'med_dip_atrs': round(move_l.median(), 2),
        'p75_dip_atrs': round(move_l.quantile(0.25), 2),
    }
    # Reference medians for A-grade
    a = s[s['trade_grade'] == 'A']
    meds = {}
    for m in ['selloff_total_pct', 'consecutive_down_days', 'pct_off_30d_high', 'gap_pct',
              'one_day_before_range_pct', 'pct_change_3', 'pct_off_52wk_high', 'bounce_open_close_pct']:
        vals = pd.to_numeric(a[m], errors='coerce').dropna()
        if len(vals) > 0:
            meds[m] = round(vals.median(), 3)
    stats[key]['a_medians'] = meds
    stats[key]['a_count'] = len(a)

# === BY CAP ===
for cap in ['Large', 'Medium', 'ETF', 'Small']:
    s = df[df['cap'] == cap]
    if len(s) == 0:
        continue
    oc = s['bounce_open_close_pct'].dropna()
    oh = s['bounce_open_high_pct'].dropna()
    ol = s['bounce_open_low_pct'].dropna()
    gap = s['gap_pct'].abs()

    valid = oh.notna() & s['atr_pct'].notna() & (s['atr_pct'] > 0)
    move_h = s.loc[valid, 'bounce_open_high_pct'] / s.loc[valid, 'atr_pct']
    move_l = s.loc[valid, 'bounce_open_low_pct'] / s.loc[valid, 'atr_pct']

    valid_gap = gap.notna() & oh.notna() & (gap > 0)
    gf = (oh[valid_gap] >= gap[valid_gap]) if valid_gap.sum() > 0 else pd.Series(dtype=float)

    stats[f'cap_{cap}'] = {
        'n': len(s),
        'wr': round((oc > 0).sum() / len(oc) * 100, 1),
        'avg_pnl': round(oc.mean() * 100, 1),
        'hit_0_5x': round((move_h >= 0.5).sum() / len(move_h) * 100, 0) if len(move_h) > 0 else 0,
        'hit_1_0x': round((move_h >= 1.0).sum() / len(move_h) * 100, 0) if len(move_h) > 0 else 0,
        'hit_1_5x': round((move_h >= 1.5).sum() / len(move_h) * 100, 0) if len(move_h) > 0 else 0,
        'hit_2_0x': round((move_h >= 2.0).sum() / len(move_h) * 100, 0) if len(move_h) > 0 else 0,
        'hit_gap_fill': round(gf.mean() * 100, 0) if len(gf) > 0 else 0,
        'avg_dip_pct': round(ol.mean() * 100, 1),
        'med_dip_pct': round(ol.median() * 100, 1),
        'avg_dip_atrs': round(move_l.mean(), 2) if len(move_l) > 0 else 0,
        'med_dip_atrs': round(move_l.median(), 2) if len(move_l) > 0 else 0,
        'max_dip_pct': round(ol.min() * 100, 1),
        'p75_dip_atrs': round(move_l.quantile(0.25), 2) if len(move_l) > 0 else 0,
    }

# === OVERALL EXIT TARGETS ===
valid = df['bounce_open_high_pct'].notna() & df['atr_pct'].notna() & (df['atr_pct'] > 0)
move_h = df.loc[valid, 'bounce_open_high_pct'] / df.loc[valid, 'atr_pct']
move_c = df.loc[valid, 'bounce_open_close_pct'] / df.loc[valid, 'atr_pct']
move_l = df.loc[valid, 'bounce_open_low_pct'] / df.loc[valid, 'atr_pct']
gap = df['gap_pct'].abs()
valid_gap = gap.notna() & df['bounce_open_high_pct'].notna() & (gap > 0)

stats['overall'] = {
    'n': int(valid.sum()),
    'high_med_atrs': round(move_h.median(), 2),
    'high_p25_atrs': round(move_h.quantile(0.25), 2),
    'high_p75_atrs': round(move_h.quantile(0.75), 2),
    'high_p90_atrs': round(move_h.quantile(0.90), 2),
    'close_med_atrs': round(move_c.median(), 2),
    'close_p25_atrs': round(move_c.quantile(0.25), 2),
    'close_p75_atrs': round(move_c.quantile(0.75), 2),
    'dip_med_atrs': round(move_l.median(), 2),
    'dip_p25_atrs': round(move_l.quantile(0.25), 2),
    'dip_p75_atrs': round(move_l.quantile(0.75), 2),
    'gap_fill_pct': round((df.loc[valid_gap, 'bounce_open_high_pct'] >= gap[valid_gap]).mean() * 100, 0),
    'half_gap_fill_pct': round((df.loc[valid_gap, 'bounce_open_high_pct'] >= gap[valid_gap] * 0.5).mean() * 100, 0),
    'close_above_gap_pct': round((df.loc[valid_gap, 'bounce_open_close_pct'] >= gap[valid_gap]).mean() * 100, 0),
    'open_to_high_retained': round((move_c / move_h).median() * 100, 0),
}

# === INTENSITY SCORING CORRELATIONS ===
corrs = {}
for m in ['selloff_total_pct', 'pct_change_3', 'gap_pct', 'pct_off_30d_high',
          'pct_off_52wk_high', 'consecutive_down_days', 'pct_change_15']:
    c = df[m].corr(df['bounce_open_close_pct'])
    if not np.isnan(c):
        corrs[m] = round(c, 3)
stats['correlations'] = corrs

# === TIME OF LOW ===
tol = df['time_of_low_bucket'].dropna()
early = (tol == 1).sum()
early_df = df[df['time_of_low_bucket'] == 1]
early_wr = (early_df['bounce_open_close_pct'] > 0).sum() / len(early_df) * 100
stats['time_of_low'] = {
    'early_30min': int(early),
    'early_30min_pct': round(early / len(tol) * 100, 0),
    'total_with_data': int(len(tol)),
    'early_wr': round(early_wr, 0),
}

# === SPY CONTEXT ===
spy = df['spy_open_close_pct'].dropna()
boc_spy = df.loc[spy.index, 'bounce_open_close_pct']
spy_strong = spy > 0.02
spy_weak = spy < -0.005
stats['spy'] = {
    'avg_spy': round(spy.mean() * 100, 2),
    'corr': round(spy.corr(boc_spy), 3),
    'strong_n': int(spy_strong.sum()),
    'strong_avg': round(boc_spy[spy_strong].mean() * 100, 1),
    'weak_n': int(spy_weak.sum()),
    'weak_avg': round(boc_spy[spy_weak].mean() * 100, 1),
}

# === BOLLINGER / NEAR LOWS ===
bb_true = df[df['closed_outside_lower_band'] == 1]
bb_false = df[df['closed_outside_lower_band'] == 0]
if len(bb_true) > 0 and len(bb_false) > 0:
    stats['bollinger'] = {
        'outside_n': len(bb_true),
        'outside_wr': round((bb_true['bounce_open_close_pct'] > 0).sum() / len(bb_true) * 100, 0),
        'outside_avg': round(bb_true['bounce_open_close_pct'].mean() * 100, 1),
        'inside_n': len(bb_false),
        'inside_wr': round((bb_false['bounce_open_close_pct'] > 0).sum() / len(bb_false) * 100, 0),
        'inside_avg': round(bb_false['bounce_open_close_pct'].mean() * 100, 1),
    }

near_lows = df[df['prior_day_close_vs_low_pct'] <= 0.15]
not_near = df[df['prior_day_close_vs_low_pct'] > 0.15]
if len(near_lows) > 0:
    stats['near_lows'] = {
        'near_n': len(near_lows),
        'near_wr': round((near_lows['bounce_open_close_pct'] > 0).sum() / len(near_lows) * 100, 0),
        'near_avg': round(near_lows['bounce_open_close_pct'].mean() * 100, 1),
        'not_n': len(not_near),
        'not_wr': round((not_near['bounce_open_close_pct'] > 0).sum() / len(not_near) * 100, 0),
        'not_avg': round(not_near['bounce_open_close_pct'].mean() * 100, 1),
    }

# === EXHAUSTION GAP ===
exh_yes = df[df['gap_pct'].abs() >= 0.05]  # 5%+ gap = exhaustion
exh_no = df[df['gap_pct'].abs() < 0.05]
if len(exh_yes) > 0:
    stats['exhaustion_gap'] = {
        'yes_n': len(exh_yes),
        'yes_wr': round((exh_yes['bounce_open_close_pct'] > 0).sum() / len(exh_yes) * 100, 0),
        'yes_avg': round(exh_yes['bounce_open_close_pct'].mean() * 100, 1),
        'no_n': len(exh_no),
        'no_wr': round((exh_no['bounce_open_close_pct'] > 0).sum() / len(exh_no) * 100, 0),
        'no_avg': round(exh_no['bounce_open_close_pct'].mean() * 100, 1),
    }

# === CONSECUTIVE DOWN DAYS ===
for dd in [2, 3, 4, 5]:
    sub = df[df['consecutive_down_days'] >= dd]
    if len(sub) > 0:
        oc_sub = sub['bounce_open_close_pct'].dropna()
        stats[f'down_days_gte_{dd}'] = {
            'n': len(sub),
            'wr': round((oc_sub > 0).sum() / len(oc_sub) * 100, 0),
            'avg': round(oc_sub.mean() * 100, 1),
        }

with open('data/bounce_stats_123.json', 'w') as f:
    json.dump(stats, f, indent=2)
print('Stats saved to data/bounce_stats_123.json')
print(json.dumps(stats, indent=2))
