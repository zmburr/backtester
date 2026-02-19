"""
Bounce Intensity vs Outcome Analysis — V2 Comparison
=====================================================
Computes ORIGINAL and IMPROVED intensity scores for every historical trade,
then compares predictive power head-to-head.

ORIGINAL composite (5 metrics):
  selloff_total_pct (30%), gap_pct (25%), pct_off_30d_high (20%),
  percent_of_vol_on_breakout_day (15%), consecutive_down_days (10%)

IMPROVED composite (7 metrics) — drops dead-weight volume, adds proven predictors:
  selloff_total_pct (25%), pct_change_3 (20%), gap_pct (15%),
  pct_off_30d_high (15%), pct_off_52wk_high (10%),
  consecutive_down_days (10%), pct_change_15 (5%)
"""

import sys
import os
from pathlib import Path

os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scanners.bounce_trader import compute_bounce_intensity, _pctrank

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTCOME_COL = 'bounce_open_close_pct'
OUTCOME_HIGH = 'bounce_open_high_pct'
OUTCOME_LOW = 'bounce_open_low_pct'
OUTCOME_NEXT = 'bounce_open_to_day_after_open_pct'

# Original spec (mirrors bounce_trader.py)
ORIGINAL_SPEC = [
    ('selloff_total_pct',              False, 0.30),
    ('consecutive_down_days',          True,  0.10),
    ('percent_of_vol_on_breakout_day', True,  0.15),
    ('pct_off_30d_high',               False, 0.20),
    ('gap_pct',                        False, 0.25),
]

# Improved spec — swap out volume (rho=0.04), add pct_change_3 (rho=-0.70)
# and pct_off_52wk_high (rho=-0.49), pct_change_15 (rho=-0.57)
IMPROVED_SPEC = [
    ('selloff_total_pct',    False, 0.25),   # was 0.30 — still #1 predictor
    ('pct_change_3',         False, 0.20),   # NEW — rho=-0.700, #2 predictor
    ('gap_pct',              False, 0.15),   # was 0.25
    ('pct_off_30d_high',     False, 0.15),   # was 0.20
    ('pct_off_52wk_high',    False, 0.10),   # NEW — rho=-0.487
    ('consecutive_down_days', True, 0.10),   # unchanged
    ('pct_change_15',        False, 0.05),   # NEW — rho=-0.570
]


def compute_intensity_custom(metrics, ref_df, spec):
    """Generic intensity scorer using any spec."""
    details = {}
    weighted_sum = 0.0
    total_weight = 0.0
    for col, higher_is_better, weight in spec:
        actual = metrics.get(col)
        ref_vals = ref_df[col].dropna().values if col in ref_df.columns else []
        if actual is None or pd.isna(actual) or len(ref_vals) == 0:
            details[col] = {'pctile': None, 'weight': weight, 'actual': actual}
            continue
        raw_pctile = _pctrank(ref_vals, actual, kind='rank')
        pctile = raw_pctile if higher_is_better else 100.0 - raw_pctile
        details[col] = {'pctile': round(pctile, 1), 'weight': weight, 'actual': actual}
        weighted_sum += pctile * weight
        total_weight += weight
    composite = round(weighted_sum / total_weight, 1) if total_weight > 0 else 0.0
    return {'composite': composite, 'details': details}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
bounce_csv = PROJECT_ROOT / 'data' / 'bounce_data.csv'
df = pd.read_csv(bounce_csv).dropna(subset=['ticker', 'date'])

# Compute both scores for every trade
orig_scores, impr_scores = [], []
orig_details, impr_details = [], []

for _, row in df.iterrows():
    m = row.to_dict()
    o = compute_intensity_custom(m, df, ORIGINAL_SPEC)
    i = compute_intensity_custom(m, df, IMPROVED_SPEC)
    orig_scores.append(o['composite'])
    impr_scores.append(i['composite'])
    orig_details.append(o['details'])
    impr_details.append(i['details'])

df['orig_score'] = orig_scores
df['impr_score'] = impr_scores

valid = df.dropna(subset=[OUTCOME_COL]).copy()
valid['win'] = valid[OUTCOME_COL] > 0
valid['pnl_pct'] = valid[OUTCOME_COL] * 100

# ===========================================================================
# REPORT
# ===========================================================================
SEP = "=" * 90
THIN = "-" * 90

print(SEP)
print("BOUNCE INTENSITY V2: ORIGINAL vs IMPROVED COMPOSITE COMPARISON")
print(SEP)
print(f"\nTrades analyzed: {len(valid)}")


# --- Helper: print bucket table for a score column ---
def print_bucket_table(score_col, label):
    bins = [0, 50, 60, 70, 80, 100]
    labels_b = ['<50', '50-60', '60-70', '70-80', '80+']
    valid['_bucket'] = pd.cut(valid[score_col], bins=bins, labels=labels_b, include_lowest=True)

    print(f"\n  {label}:")
    print(f"  {'Bucket':<10} {'N':>4} {'Win%':>7} {'AvgP&L':>8} {'MedP&L':>8} {'AvgHigh':>9} {'AvgLow':>8} {'ATR':>7}")
    print(f"  {THIN[:70]}")

    for bucket_label in labels_b:
        subset = valid[valid['_bucket'] == bucket_label]
        if len(subset) == 0:
            print(f"  {bucket_label:<10} {'--':>4}")
            continue
        wr = subset['win'].mean() * 100
        avg = subset['pnl_pct'].mean()
        med = subset['pnl_pct'].median()
        high = subset[OUTCOME_HIGH].mean() * 100
        low = subset[OUTCOME_LOW].mean() * 100
        atr = subset['atr_pct_move'].mean()
        print(f"  {bucket_label:<10} {len(subset):>4} {wr:>6.1f}% {avg:>+7.1f}% {med:>+7.1f}% {high:>+8.1f}% {low:>+7.1f}% {atr:>+6.2f}x")

    valid.drop(columns=['_bucket'], inplace=True)


# --- Helper: print median split ---
def print_median_split(score_col, label):
    med = valid[score_col].median()
    above = valid[valid[score_col] >= med]
    below = valid[valid[score_col] < med]
    print(f"\n  {label} -- Median = {med:.1f}")
    for tag, subset in [("Below Median", below), ("Above Median", above)]:
        wr = subset['win'].mean() * 100
        avg = subset['pnl_pct'].mean()
        mdp = subset['pnl_pct'].median()
        high = subset[OUTCOME_HIGH].mean() * 100
        print(f"    {tag:<20} (n={len(subset):>2})  Win={wr:>5.1f}%  AvgP&L={avg:>+6.1f}%  MedP&L={mdp:>+6.1f}%  AvgHigh={high:>+6.1f}%")


# ---------------------------------------------------------------------------
# 1. HEAD-TO-HEAD CORRELATION
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("1. HEAD-TO-HEAD CORRELATION WITH OUTCOMES")
print(SEP)

for outcome_label, oc in [("Open-to-Close %", OUTCOME_COL),
                            ("Open-to-High %", OUTCOME_HIGH),
                            ("ATR Move", 'atr_pct_move')]:
    sub = valid.dropna(subset=[oc])
    if len(sub) < 5:
        continue
    r_o, p_o = stats.pearsonr(sub['orig_score'], sub[oc])
    rho_o, rhop_o = stats.spearmanr(sub['orig_score'], sub[oc])
    r_i, p_i = stats.pearsonr(sub['impr_score'], sub[oc])
    rho_i, rhop_i = stats.spearmanr(sub['impr_score'], sub[oc])

    delta_r = r_i - r_o
    delta_rho = rho_i - rho_o

    print(f"\n  {outcome_label}:")
    print(f"    {'':20} {'Pearson r':>10} {'p-val':>8} {'Spearman':>10} {'p-val':>8}")
    print(f"    {'ORIGINAL':20} {r_o:>+10.3f} {p_o:>8.4f} {rho_o:>+10.3f} {rhop_o:>8.4f}")
    print(f"    {'IMPROVED':20} {r_i:>+10.3f} {p_i:>8.4f} {rho_i:>+10.3f} {rhop_i:>8.4f}")
    print(f"    {'DELTA':20} {delta_r:>+10.3f} {'':8} {delta_rho:>+10.3f}")

# ---------------------------------------------------------------------------
# 2. BUCKET TABLES SIDE-BY-SIDE
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("2. BUCKET BREAKDOWN (50 / 60 / 70 / 80 splits)")
print(SEP)

print_bucket_table('orig_score', 'ORIGINAL Composite')
print_bucket_table('impr_score', 'IMPROVED Composite')

# ---------------------------------------------------------------------------
# 3. MEDIAN SPLITS
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("3. ABOVE vs BELOW MEDIAN")
print(SEP)

print_median_split('orig_score', 'ORIGINAL')
print_median_split('impr_score', 'IMPROVED')

# ---------------------------------------------------------------------------
# 4. TOP-QUARTILE vs BOTTOM-QUARTILE
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("4. TOP QUARTILE vs BOTTOM QUARTILE")
print(SEP)

for label, score_col in [("ORIGINAL", 'orig_score'), ("IMPROVED", 'impr_score')]:
    q25 = valid[score_col].quantile(0.25)
    q75 = valid[score_col].quantile(0.75)
    top = valid[valid[score_col] >= q75]
    bot = valid[valid[score_col] <= q25]
    print(f"\n  {label}  (Q25={q25:.1f}, Q75={q75:.1f})")
    for tag, subset in [("Bottom 25%", bot), ("Top 25%", top)]:
        wr = subset['win'].mean() * 100
        avg = subset['pnl_pct'].mean()
        mdp = subset['pnl_pct'].median()
        high = subset[OUTCOME_HIGH].mean() * 100
        low = subset[OUTCOME_LOW].mean() * 100
        print(f"    {tag:<15} (n={len(subset):>2})  Win={wr:>5.1f}%  AvgP&L={avg:>+7.1f}%  MedP&L={mdp:>+7.1f}%  AvgHigh={high:>+7.1f}%  AvgLow={low:>+7.1f}%")

# ---------------------------------------------------------------------------
# 5. SCORE AGREEMENT / DISAGREEMENT
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("5. WHERE SCORES DISAGREE (>15 pt difference)")
print(SEP)

valid['score_diff'] = valid['impr_score'] - valid['orig_score']
disagree = valid[valid['score_diff'].abs() > 15].sort_values('score_diff', ascending=False)

print(f"\n  {'Date':<12} {'Ticker':<8} {'Cap':<8} {'Orig':>6} {'Impr':>6} {'Diff':>6} {'P&L':>8} {'Setup':<30}")
print(f"  {THIN[:90]}")
for _, row in disagree.iterrows():
    print(f"  {row['date']:<10} {row['ticker']:<8} {str(row.get('cap','')):<8} "
          f"{row['orig_score']:>5.1f} {row['impr_score']:>5.1f} {row['score_diff']:>+5.1f} "
          f"{row['pnl_pct']:>+7.1f}% {str(row.get('Setup',''))[:28]:<30}")

# ---------------------------------------------------------------------------
# 6. PER-METRIC CONTRIBUTION ANALYSIS (IMPROVED)
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("6. IMPROVED COMPOSITE: PER-METRIC PERCENTILE CONTRIBUTION")
print(SEP)

print(f"\n  {'Metric':<35} {'Weight':>6} {'AvgPctile':>10} {'Corr w/ P&L':>12}")
print(f"  {THIN[:65]}")

for col, hib, weight in IMPROVED_SPEC:
    pctiles = []
    for d in impr_details:
        p = d.get(col, {}).get('pctile')
        if p is not None:
            pctiles.append(p)
    avg_p = np.mean(pctiles) if pctiles else 0

    sub = valid.dropna(subset=[col, OUTCOME_COL])
    if len(sub) >= 5:
        rho, _ = stats.spearmanr(sub[col], sub[OUTCOME_COL])
    else:
        rho = 0
    print(f"  {col:<35} {weight:>5.0%} {avg_p:>9.1f} {rho:>+11.3f}")

# ---------------------------------------------------------------------------
# 7. TRADE-LEVEL COMPARISON (all trades, sorted by improved score)
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("7. ALL TRADES RANKED BY IMPROVED SCORE")
print(SEP)

ranked = valid.sort_values('impr_score', ascending=False)
print(f"\n  {'Date':<12} {'Ticker':<8} {'Cap':<8} {'Orig':>6} {'Impr':>6} {'P&L':>8} {'High':>8} {'Win':>4} {'Setup':<28}")
print(f"  {THIN}")
for _, row in ranked.iterrows():
    high = row.get(OUTCOME_HIGH, 0) * 100 if pd.notna(row.get(OUTCOME_HIGH)) else 0
    w = 'Y' if row['win'] else 'N'
    print(f"  {row['date']:<10} {row['ticker']:<8} {str(row.get('cap','')):<8} "
          f"{row['orig_score']:>5.1f} {row['impr_score']:>5.1f} "
          f"{row['pnl_pct']:>+7.1f}% {high:>+7.1f}% {w:>4} {str(row.get('Setup',''))[:26]:<28}")

# ---------------------------------------------------------------------------
# 8. THRESHOLD SWEEP: OPTIMAL CUT FOR EACH SCORE
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("8. THRESHOLD SWEEP: WHERE IS THE BEST CUT?")
print(SEP)

for label, score_col in [("ORIGINAL", 'orig_score'), ("IMPROVED", 'impr_score')]:
    print(f"\n  {label}:")
    print(f"    {'Cut':>6} {'N>=':>5} {'Win%':>7} {'AvgP&L':>8} {'MedP&L':>8} {'AvgHigh':>9}")
    print(f"    {THIN[:50]}")
    for cut in [40, 45, 50, 55, 60, 65, 70, 75, 80]:
        above = valid[valid[score_col] >= cut]
        if len(above) < 2:
            continue
        wr = above['win'].mean() * 100
        avg = above['pnl_pct'].mean()
        med = above['pnl_pct'].median()
        high = above[OUTCOME_HIGH].mean() * 100
        print(f"    {cut:>5.0f} {len(above):>5} {wr:>6.1f}% {avg:>+7.1f}% {med:>+7.1f}% {high:>+8.1f}%")

# ---------------------------------------------------------------------------
# 9. KEY FINDINGS SUMMARY
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("9. KEY FINDINGS SUMMARY")
print(SEP)

# Overall correlations
sub = valid.dropna(subset=[OUTCOME_COL])
r_o, _ = stats.pearsonr(sub['orig_score'], sub[OUTCOME_COL])
rho_o, _ = stats.spearmanr(sub['orig_score'], sub[OUTCOME_COL])
r_i, _ = stats.pearsonr(sub['impr_score'], sub[OUTCOME_COL])
rho_i, _ = stats.spearmanr(sub['impr_score'], sub[OUTCOME_COL])

orig_med = valid['orig_score'].median()
impr_med = valid['impr_score'].median()

orig_above = valid[valid['orig_score'] >= orig_med]
orig_below = valid[valid['orig_score'] < orig_med]
impr_above = valid[valid['impr_score'] >= impr_med]
impr_below = valid[valid['impr_score'] < impr_med]

print(f"""
  CORRELATION COMPARISON (Open-to-Close P&L):
                   Pearson r    Spearman rho
    ORIGINAL:      {r_o:>+.3f}        {rho_o:>+.3f}
    IMPROVED:      {r_i:>+.3f}        {rho_i:>+.3f}
    DELTA:         {r_i-r_o:>+.3f}        {rho_i-rho_o:>+.3f}

  ABOVE-MEDIAN SPLIT COMPARISON:
                   Win%     AvgP&L    MedP&L
    ORIG above:    {orig_above['win'].mean()*100:>5.1f}%   {orig_above['pnl_pct'].mean():>+6.1f}%   {orig_above['pnl_pct'].median():>+6.1f}%
    ORIG below:    {orig_below['win'].mean()*100:>5.1f}%   {orig_below['pnl_pct'].mean():>+6.1f}%   {orig_below['pnl_pct'].median():>+6.1f}%
    IMPR above:    {impr_above['win'].mean()*100:>5.1f}%   {impr_above['pnl_pct'].mean():>+6.1f}%   {impr_above['pnl_pct'].median():>+6.1f}%
    IMPR below:    {impr_below['win'].mean()*100:>5.1f}%   {impr_below['pnl_pct'].mean():>+6.1f}%   {impr_below['pnl_pct'].median():>+6.1f}%
""")

# Improvement narrative
if rho_i > rho_o:
    delta = rho_i - rho_o
    print(f"  IMPROVED composite outperforms ORIGINAL by +{delta:.3f} Spearman rho.")
    print("  The new metrics (pct_change_3, pct_off_52wk_high, pct_change_15)")
    print("  add meaningful signal that volume was not providing.")
else:
    print("  ORIGINAL composite performed as well or better than IMPROVED.")
    print("  Volume metric may carry signal not captured by correlation alone.")

# Spec comparison
print(f"\n  SPEC COMPARISON:")
print(f"  {'ORIGINAL':<45} {'IMPROVED':<45}")
print(f"  {THIN[:90]}")
o_items = [(col, w) for col, _, w in ORIGINAL_SPEC]
i_items = [(col, w) for col, _, w in IMPROVED_SPEC]
for idx in range(max(len(o_items), len(i_items))):
    o_str = f"{o_items[idx][0]} ({o_items[idx][1]:.0%})" if idx < len(o_items) else ""
    i_str = f"{i_items[idx][0]} ({i_items[idx][1]:.0%})" if idx < len(i_items) else ""
    changed = ""
    if idx < len(i_items):
        ic = i_items[idx][0]
        if ic not in [x[0] for x in o_items]:
            changed = " <-- NEW"
    print(f"  {o_str:<45} {i_str:<45}{changed}")

print(f"\n{SEP}")
