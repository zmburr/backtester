"""Compute comprehensive stats for the 93-trade bounce dataset."""
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzers.bounce_scorer import BounceScorer, BouncePretrade, classify_from_setup_column
from scanners.bounce_trader import compute_bounce_intensity

df = pd.read_csv('data/bounce_data.csv')
print(f'Total trades: {len(df)}')

scorer = BounceScorer()
scored = scorer.score_dataframe(df)
scored['pnl'] = scored['bounce_open_close_pct'] * 100

# Recommendation breakdown (8-criteria historical)
print('\n=== RECOMMENDATION (8-criteria historical) ===')
for rec in ['GO', 'CAUTION', 'NO-GO']:
    s = scored[scored['recommendation'] == rec]
    if len(s) > 0:
        wr = (s['pnl'] > 0).mean() * 100
        avg = s['pnl'].mean()
        print(f'{rec}: {len(s)} trades, {wr:.1f}% WR, {avg:+.1f}% avg P&L')

# Score distribution
print('\n=== SCORE DISTRIBUTION ===')
for sc in range(8, -1, -1):
    s = scored[scored['criteria_score'] == sc]
    if len(s) > 0:
        wr = (s['pnl'] > 0).mean() * 100
        avg = s['pnl'].mean()
        print(f'Score {sc}/8: {len(s)} trades, {wr:.1f}% WR, {avg:+.1f}% avg P&L')

# Pre-trade (7 criteria) stats
print('\n=== PRE-TRADE 7-CRITERIA ===')
checker = BouncePretrade()
pretrade_results = []
for _, row in df.iterrows():
    metrics = row.to_dict()
    setup = classify_from_setup_column(row.get('Setup', ''))
    cap = row.get('cap', 'Medium')
    if cap is None or (isinstance(cap, float) and pd.isna(cap)):
        cap = 'Medium'
    r = checker.validate(row['ticker'], metrics, force_setup=setup, cap=cap)
    pretrade_results.append({'rec': r.recommendation, 'score': r.score, 'pnl': row.get('bounce_open_close_pct', 0)})

pr_df = pd.DataFrame(pretrade_results)
for rec in ['GO', 'CAUTION', 'NO-GO']:
    s = pr_df[pr_df['rec'] == rec]
    if len(s) > 0:
        pnl = s['pnl'].dropna() * 100
        wr = (pnl > 0).sum() / len(pnl) * 100
        avg = pnl.mean()
        print(f'{rec}: {len(s)} trades, {wr:.1f}% WR, {avg:+.1f}% avg P&L')

# Profile stats (Grade A only)
print('\n=== PROFILE STATS (Grade A only) ===')
for setup_key in ['weakstock', 'strongstock']:
    s = df[df['Setup'].str.contains(setup_key, case=False, na=False)]
    a = s[s['trade_grade'] == 'A']
    oc = a['bounce_open_close_pct'].dropna()
    wr = (oc > 0).sum() / len(oc) * 100 if len(oc) > 0 else 0
    avg = oc.mean() * 100 if len(oc) > 0 else 0
    print(f'GapFade_{setup_key}: n={len(a)}, WR={wr:.0f}%, avg={avg:+.1f}%')

# IntradayCapitch
ic = df[df['Setup'].str.contains('IntradayCapitch', case=False, na=False)]
if len(ic) > 0:
    ic_oc = ic['bounce_open_close_pct'].dropna()
    ic_wr = (ic_oc > 0).sum() / len(ic_oc) * 100
    ic_avg = ic_oc.mean() * 100
    print(f'IntradayCapitch: n={len(ic)}, WR={ic_wr:.0f}%, avg={ic_avg:+.1f}%')

# GapFade overall
gf = df[df['Setup'].str.contains('GapFade', case=False, na=False)]
gf_oc = gf['bounce_open_close_pct'].dropna()
gf_wr = (gf_oc > 0).sum() / len(gf_oc) * 100
gf_avg = gf_oc.mean() * 100
print(f'All GapFade: n={len(gf)}, WR={gf_wr:.0f}%, avg={gf_avg:+.1f}%')

# Intensity thresholds
print('\n=== INTENSITY THRESHOLDS ===')
scores = []
for _, row in df.iterrows():
    metrics = row.to_dict()
    r = compute_bounce_intensity(metrics)
    scores.append({'score': r['composite'], 'pnl': row.get('bounce_open_close_pct', 0) * 100})
int_df = pd.DataFrame(scores)
for thresh in [50, 65, 30]:
    above = int_df[int_df['score'] >= thresh]
    below = int_df[int_df['score'] < thresh]
    if len(above) > 0:
        wr = (above['pnl'] > 0).sum() / len(above) * 100
        avg = above['pnl'].mean()
        print(f'>={thresh}: {len(above)} trades, {wr:.0f}% WR, {avg:+.1f}% avg')
    if len(below) > 0:
        wr = (below['pnl'] > 0).sum() / len(below) * 100
        avg = below['pnl'].mean()
        print(f'<{thresh}: {len(below)} trades, {wr:.0f}% WR, {avg:+.1f}% avg')

# Cluster day stats
print('\n=== CLUSTER DAY STATS ===')
date_counts = df['date'].value_counts()
cluster_dates = date_counts[date_counts >= 3].index
cluster = df[df['date'].isin(cluster_dates)]
solo = df[~df['date'].isin(cluster_dates)]
c_oc = cluster['bounce_open_close_pct'].dropna() * 100
s_oc = solo['bounce_open_close_pct'].dropna() * 100
if len(c_oc) > 0:
    print(f'Cluster: n={len(cluster)}, WR={(c_oc>0).sum()/len(c_oc)*100:.0f}%, avg={c_oc.mean():+.1f}%')
if len(s_oc) > 0:
    print(f'Solo: n={len(solo)}, WR={(s_oc>0).sum()/len(s_oc)*100:.0f}%, avg={s_oc.mean():+.1f}%')

# Overnight
print('\n=== OVERNIGHT ===')
nxt = df['bounce_open_to_day_after_open_pct'].dropna() * 100
pos = (nxt > 0).sum() / len(nxt) * 100
print(f'All: {pos:.0f}% positive overnight (n={len(nxt)}), median={nxt.median():+.1f}%')
if len(cluster) > 0:
    c_nxt = cluster['bounce_open_to_day_after_open_pct'].dropna() * 100
    if len(c_nxt) > 0:
        c_pos = (c_nxt > 0).sum() / len(c_nxt) * 100
        print(f'Cluster: {c_pos:.0f}% positive, median={c_nxt.median():+.1f}%')
    n_cluster = len(cluster_dates)
    print(f'Cluster days count: n={n_cluster}')

# Exhaustion gap
print('\n=== EXHAUSTION GAP ===')
exh_yes = df[df['gap_pct'].abs() >= 0.05]
exh_no = df[df['gap_pct'].abs() < 0.05]
ey_oc = exh_yes['bounce_open_close_pct'].dropna() * 100
en_oc = exh_no['bounce_open_close_pct'].dropna() * 100
if len(ey_oc) > 0:
    print(f'With: n={len(ey_oc)}, WR={(ey_oc>0).sum()/len(ey_oc)*100:.0f}%, avg={ey_oc.mean():+.1f}%')
if len(en_oc) > 0:
    print(f'Without: n={len(en_oc)}, WR={(en_oc>0).sum()/len(en_oc)*100:.0f}%, avg={en_oc.mean():+.1f}%')

# By cap
print('\n=== BY CAP ===')
for cap in ['ETF', 'Large', 'Medium', 'Small']:
    s = df[df['cap'] == cap]
    oc = s['bounce_open_close_pct'].dropna() * 100
    if len(oc) > 0:
        print(f'{cap}: n={len(s)}, WR={(oc>0).sum()/len(oc)*100:.0f}%, avg={oc.mean():+.1f}%')

# Consecutive down days
print('\n=== DOWN DAYS ===')
for dd in [5, 4, 3, 2]:
    s = df[df['consecutive_down_days'] >= dd]
    oc = s['bounce_open_close_pct'].dropna() * 100
    if len(oc) > 0:
        print(f'{dd}+ days: n={len(s)}, WR={(oc>0).sum()/len(oc)*100:.0f}%, avg={oc.mean():+.1f}%')

# Setup type medians
print('\n=== SETUP TYPE MEDIANS ===')
for setup_key in ['weakstock', 'strongstock']:
    s = df[df['Setup'].str.contains(setup_key, case=False, na=False)]
    oh = s['bounce_open_high_pct'].dropna() * 100
    oc = s['bounce_open_close_pct'].dropna() * 100
    if len(oh) > 0:
        print(f'{setup_key}: med_high={oh.median():+.1f}%, med_close={oc.median():+.1f}%')

# BB stats
print('\n=== BOLLINGER BAND ===')
bb_true = df[df['closed_outside_lower_band'] == 1]
bb_false = df[df['closed_outside_lower_band'] == 0]
bt_oc = bb_true['bounce_open_close_pct'].dropna() * 100
bf_oc = bb_false['bounce_open_close_pct'].dropna() * 100
if len(bt_oc) > 0:
    print(f'Outside: n={len(bb_true)}, WR={(bt_oc>0).sum()/len(bt_oc)*100:.0f}%, avg={bt_oc.mean():+.1f}%')
if len(bf_oc) > 0:
    print(f'Inside: n={len(bb_false)}, WR={(bf_oc>0).sum()/len(bf_oc)*100:.0f}%, avg={bf_oc.mean():+.1f}%')

# Near lows
print('\n=== NEAR LOWS ===')
nl = df[df['prior_day_close_vs_low_pct'] <= 0.15]
nnl = df[df['prior_day_close_vs_low_pct'] > 0.15]
nl_oc = nl['bounce_open_close_pct'].dropna() * 100
nnl_oc = nnl['bounce_open_close_pct'].dropna() * 100
if len(nl_oc) > 0:
    print(f'Near: n={len(nl)}, WR={(nl_oc>0).sum()/len(nl_oc)*100:.0f}%, avg={nl_oc.mean():+.1f}%')
if len(nnl_oc) > 0:
    print(f'Not near: n={len(nnl)}, WR={(nnl_oc>0).sum()/len(nnl_oc)*100:.0f}%, avg={nnl_oc.mean():+.1f}%')

# ATR stats
print('\n=== ATR STATS (overall) ===')
valid = df['bounce_open_high_pct'].notna() & df['atr_pct'].notna() & (df['atr_pct'] > 0)
move_h = df.loc[valid, 'bounce_open_high_pct'] / df.loc[valid, 'atr_pct']
move_c = df.loc[valid, 'bounce_open_close_pct'] / df.loc[valid, 'atr_pct']
move_l = df.loc[valid, 'bounce_open_low_pct'] / df.loc[valid, 'atr_pct']
print(f'High: p25={move_h.quantile(0.25):.2f}, med={move_h.median():.2f}, p75={move_h.quantile(0.75):.2f}')
print(f'Close: p25={move_c.quantile(0.25):.2f}, med={move_c.median():.2f}, p75={move_c.quantile(0.75):.2f}')
print(f'Dip: p25={move_l.quantile(0.25):.2f}, med={move_l.median():.2f}, p75={move_l.quantile(0.75):.2f}')

# Pre-trade score breakdown (5/7)
print('\n=== PRE-TRADE SCORE DISTRIBUTION ===')
for sc in range(7, -1, -1):
    s = pr_df[pr_df['score'] == sc]
    if len(s) > 0:
        pnl = s['pnl'].dropna() * 100
        wr = (pnl > 0).sum() / len(pnl) * 100
        avg = pnl.mean()
        print(f'Score {sc}/7: {len(s)} trades, {wr:.0f}% WR, {avg:+.1f}% avg')

# Near 52wk low
print('\n=== NEAR 52WK LOW ===')
n52 = df[df['near_52wk_low'] == True]
nn52 = df[df['near_52wk_low'] == False]
if len(n52) > 0:
    oc = n52['bounce_open_close_pct'].dropna() * 100
    print(f'Near: n={len(n52)}, avg={oc.mean():+.1f}%')
if len(nn52) > 0:
    oc = nn52['bounce_open_close_pct'].dropna() * 100
    print(f'Not near: n={len(nn52)}, avg={oc.mean():+.1f}%')

# Time of low - early 30 min
print('\n=== EARLY LOW ===')
tol = df['time_of_low_bucket'].dropna()
early = df[df['time_of_low_bucket'] == 1]
late = df[(df['time_of_low_bucket'] > 1) & df['time_of_low_bucket'].notna()]
if len(early) > 0:
    oc = early['bounce_open_close_pct'].dropna() * 100
    print(f'Early (first 30 min): n={len(early)}, WR={(oc>0).sum()/len(oc)*100:.0f}%, avg={oc.mean():+.1f}%')
if len(late) > 0:
    oc = late['bounce_open_close_pct'].dropna() * 100
    print(f'Late (after 30 min): n={len(late)}, WR={(oc>0).sum()/len(oc)*100:.0f}%, avg={oc.mean():+.1f}%')
    # After 12 PM
    after12 = df[(df['time_of_low_bucket'] >= 4) & df['time_of_low_bucket'].notna()]
    if len(after12) > 0:
        oc = after12['bounce_open_close_pct'].dropna() * 100
        print(f'After 12 PM: n={len(after12)}, WR={(oc>0).sum()/len(oc)*100:.0f}%, avg={oc.mean():+.1f}%')
