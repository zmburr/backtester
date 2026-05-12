"""
Reversal/Parabolic Short Scorer — Predictive Analysis

Analyzes which metrics in reversal_data.csv actually predict reversal magnitude.
Uses Spearman rank correlation to identify:
  1. Which current 6 criteria are truly predictive
  2. Which available metrics SHOULD be in the scoring but aren't
  3. Whether we can improve the scorer like we did for bounces (V2)

Mirrors the approach from bounce_intensity_analysis.py.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def run_analysis():
    # Load data
    csv_path = os.path.join(DATA_PATH, 'reversal_data.csv')
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} total reversal trades")
    print(f"Grade distribution: {df['trade_grade'].value_counts().to_dict()}")
    print(f"Setup distribution: {df['setup'].value_counts().to_dict()}")
    print(f"Cap distribution: {df['cap'].value_counts().to_dict()}")

    # Outcome metric: reversal magnitude (negative = good for shorts)
    # For shorts, a bigger negative reversal_open_close_pct = better outcome
    # Also check day-after follow-through
    df['reversal_magnitude'] = -df['reversal_open_close_pct']  # positive = good short
    df['day_after_gain'] = -df['reversal_open_to_day_after_open_pct']  # positive = continued down next day
    df['win'] = (df['reversal_magnitude'] > 0).astype(int)

    print(f"\nOverall: {len(df)} trades")
    print(f"  Win Rate: {df['win'].mean()*100:.1f}%")
    print(f"  Avg reversal magnitude: {df['reversal_magnitude'].mean()*100:+.1f}%")
    print(f"  Avg day-after gain: {df['day_after_gain'].mean()*100:+.1f}%")

    # =====================================================================
    # SECTION 1: Correlation of ALL available metrics vs outcome
    # =====================================================================
    print("\n" + "=" * 80)
    print("SECTION 1: SPEARMAN CORRELATIONS — ALL METRICS vs REVERSAL MAGNITUDE")
    print("=" * 80)
    print("(Positive rho = higher metric -> bigger reversal = MORE PREDICTIVE)")
    print("(Negative rho = higher metric -> smaller reversal)")

    # Define candidate metrics — everything numeric in the CSV
    candidate_metrics = [
        'pct_from_9ema', 'pct_from_10mav', 'pct_from_20mav', 'pct_from_50mav',
        'pct_from_200mav', 'atr_distance_from_50mav',
        'gap_pct', 'atr_pct', 'atr_pct_move',
        'one_day_before_range_pct', 'two_day_before_range_pct', 'three_day_before_range_pct',
        'day_of_range_pct',
        'pct_change_3', 'pct_change_15', 'pct_change_30', 'pct_change_90', 'pct_change_120',
        'consecutive_up_days',
        'rvol_score', 'percent_of_vol_on_breakout_day', 'percent_of_vol_one_day_before',
        'percent_of_vol_two_day_before', 'percent_of_vol_three_day_before',
        'percent_of_premarket_vol',
        'vol_ratio_5min_to_pm',
        'prior_day_close_vs_high_pct',
        'prior_day_range_atr',
        'upper_band_distance', 'bollinger_width',
        'spy_open_close_pct', 'spy_5day_return',
        'avg_daily_vol', 'premarket_vol',
        'gap_from_pm_high',
    ]

    results = []
    for metric in candidate_metrics:
        if metric not in df.columns:
            continue
        valid = df[[metric, 'reversal_magnitude']].dropna()
        if len(valid) < 10:
            continue
        rho, pval = stats.spearmanr(valid[metric], valid['reversal_magnitude'])
        results.append({
            'metric': metric,
            'rho': rho,
            'p_value': pval,
            'n': len(valid),
            'significant': pval < 0.05,
        })

    results_df = pd.DataFrame(results).sort_values('rho', ascending=False)

    print(f"\n{'Metric':<35s} {'Rho':>8s} {'p-value':>10s} {'n':>5s} {'Sig?':>5s}")
    print("-" * 68)
    for _, r in results_df.iterrows():
        sig = "***" if r['p_value'] < 0.01 else ("**" if r['p_value'] < 0.05 else "")
        marker = ""
        if r['metric'] in ['pct_from_9ema', 'prior_day_range_atr', 'rvol_score',
                           'consecutive_up_days', 'gap_pct']:
            marker = " <-- CURRENT CRITERION"
        print(f"  {r['metric']:<33s} {r['rho']:+.3f}   {r['p_value']:.4f}  {int(r['n']):>4d}  {sig}{marker}")

    # =====================================================================
    # SECTION 2: Same analysis for day-after follow-through
    # =====================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: SPEARMAN CORRELATIONS — ALL METRICS vs DAY-AFTER GAIN")
    print("=" * 80)

    results_da = []
    for metric in candidate_metrics:
        if metric not in df.columns:
            continue
        valid = df[[metric, 'day_after_gain']].dropna()
        if len(valid) < 10:
            continue
        rho, pval = stats.spearmanr(valid[metric], valid['day_after_gain'])
        results_da.append({
            'metric': metric,
            'rho': rho,
            'p_value': pval,
            'n': len(valid),
        })

    results_da_df = pd.DataFrame(results_da).sort_values('rho', ascending=False)

    print(f"\n{'Metric':<35s} {'Rho':>8s} {'p-value':>10s} {'n':>5s}")
    print("-" * 60)
    for _, r in results_da_df.iterrows():
        sig = "***" if r['p_value'] < 0.01 else ("**" if r['p_value'] < 0.05 else "")
        print(f"  {r['metric']:<33s} {r['rho']:+.3f}   {r['p_value']:.4f}  {int(r['n']):>4d}  {sig}")

    # =====================================================================
    # SECTION 3: Current 6-criterion scorer performance
    # =====================================================================
    print("\n" + "=" * 80)
    print("SECTION 3: CURRENT SCORER PERFORMANCE (reversal_scorer.py)")
    print("=" * 80)

    from analyzers.reversal_scorer import ReversalScorer
    scorer = ReversalScorer()
    scored = scorer.score_dataframe(df)
    scored['pnl'] = -scored['reversal_open_close_pct'] * 100

    print("\nPERFORMANCE BY SCORE (ALL trades, all grades):")
    print("-" * 65)
    for score_val in range(6, -1, -1):
        subset = scored[scored['criteria_score'] == score_val]
        if len(subset) > 0:
            wr = (subset['pnl'] > 0).mean() * 100
            avg = subset['pnl'].mean()
            print(f"  Score {score_val}/6: {len(subset):3d} trades | Win: {wr:5.1f}% | Avg P&L: {avg:+6.1f}%")

    print("\nBY RECOMMENDATION:")
    print("-" * 65)
    for rec in ['GO', 'CAUTION', 'NO-GO']:
        subset = scored[scored['recommendation'] == rec]
        if len(subset) > 0:
            wr = (subset['pnl'] > 0).mean() * 100
            avg = subset['pnl'].mean()
            print(f"  {rec:8s}: {len(subset):3d} trades | Win: {wr:5.1f}% | Avg P&L: {avg:+6.1f}%")

    # Correlation of current score vs outcome
    valid_scored = scored[['criteria_score', 'pnl']].dropna()
    rho, pval = stats.spearmanr(valid_scored['criteria_score'], valid_scored['pnl'])
    print(f"\n  Current score vs P&L: Spearman rho={rho:+.3f}, p={pval:.4f}")

    # =====================================================================
    # SECTION 4: Breakdown by setup type
    # =====================================================================
    print("\n" + "=" * 80)
    print("SECTION 4: PERFORMANCE BY SETUP TYPE")
    print("=" * 80)

    for setup in scored['setup'].unique():
        subset = scored[scored['setup'] == setup]
        if len(subset) < 3:
            continue
        wr = (subset['pnl'] > 0).mean() * 100
        avg = subset['pnl'].mean()
        go = subset[subset['recommendation'] == 'GO']
        nogo = subset[subset['recommendation'] == 'NO-GO']
        go_wr = (go['pnl'] > 0).mean() * 100 if len(go) > 0 else 0
        go_avg = go['pnl'].mean() if len(go) > 0 else 0
        nogo_wr = (nogo['pnl'] > 0).mean() * 100 if len(nogo) > 0 else 0
        nogo_avg = nogo['pnl'].mean() if len(nogo) > 0 else 0

        print(f"\n  {setup}: {len(subset)} trades | Win: {wr:.0f}% | Avg: {avg:+.1f}%")
        if len(go) > 0:
            print(f"    GO:    {len(go):2d} | Win: {go_wr:.0f}% | Avg: {go_avg:+.1f}%")
        if len(nogo) > 0:
            print(f"    NO-GO: {len(nogo):2d} | Win: {nogo_wr:.0f}% | Avg: {nogo_avg:+.1f}%")

    # =====================================================================
    # SECTION 5: Per-criterion pass rate vs outcome
    # =====================================================================
    print("\n" + "=" * 80)
    print("SECTION 5: INDIVIDUAL CRITERION — PASS vs FAIL PERFORMANCE")
    print("=" * 80)
    print("(Tests whether passing each criterion actually predicts better outcomes)")

    criteria_metrics_map = {
        'pct_from_9ema': ('pct_from_9ema', 'gte'),
        'prior_day_range_atr': ('one_day_before_range_pct', 'gte'),
        'rvol_score': ('rvol_score', 'gte'),
        'consecutive_up_days': ('consecutive_up_days', 'gte'),
        'gap_pct': ('gap_pct', 'gte'),
        'reversal_pct': ('reversal_open_close_pct', 'lte'),
    }

    from analyzers.reversal_scorer import CAP_THRESHOLDS

    for criterion, (metric_col, direction) in criteria_metrics_map.items():
        pass_mask = pd.Series(False, index=df.index)
        fail_mask = pd.Series(False, index=df.index)

        for idx, row in df.iterrows():
            cap = row.get('cap', 'Medium')
            if cap not in CAP_THRESHOLDS:
                cap = 'Medium'
            thresh = getattr(CAP_THRESHOLDS[cap], criterion)
            val = row.get(metric_col)
            if pd.isna(val):
                fail_mask.iloc[idx] = True
                continue
            if direction == 'gte':
                if val >= thresh:
                    pass_mask.iloc[idx] = True
                else:
                    fail_mask.iloc[idx] = True
            else:
                if val <= thresh:
                    pass_mask.iloc[idx] = True
                else:
                    fail_mask.iloc[idx] = True

        passed = df[pass_mask]
        failed = df[fail_mask]
        p_wr = ((-passed['reversal_open_close_pct']) > 0).mean() * 100 if len(passed) > 0 else 0
        p_avg = (-passed['reversal_open_close_pct']).mean() * 100 if len(passed) > 0 else 0
        f_wr = ((-failed['reversal_open_close_pct']) > 0).mean() * 100 if len(failed) > 0 else 0
        f_avg = (-failed['reversal_open_close_pct']).mean() * 100 if len(failed) > 0 else 0

        delta_wr = p_wr - f_wr
        print(f"\n  {criterion}:")
        print(f"    PASS: {len(passed):3d} trades | Win: {p_wr:5.1f}% | Avg P&L: {p_avg:+6.1f}%")
        print(f"    FAIL: {len(failed):3d} trades | Win: {f_wr:5.1f}% | Avg P&L: {f_avg:+6.1f}%")
        print(f"    Delta WR: {delta_wr:+.1f}pp  {'<-- PREDICTIVE' if abs(delta_wr) > 5 else '<-- WEAK'}")

    # =====================================================================
    # SECTION 6: Correlation by cap bucket
    # =====================================================================
    print("\n" + "=" * 80)
    print("SECTION 6: TOP METRICS BY CAP BUCKET")
    print("=" * 80)

    for cap in ['Medium', 'Small', 'Micro', 'Large', 'ETF']:
        cap_df = df[df['cap'] == cap]
        if len(cap_df) < 8:
            continue
        print(f"\n  {cap} ({len(cap_df)} trades):")
        cap_results = []
        for metric in candidate_metrics:
            if metric not in cap_df.columns:
                continue
            valid = cap_df[[metric, 'reversal_magnitude']].dropna()
            if len(valid) < 5:
                continue
            rho, pval = stats.spearmanr(valid[metric], valid['reversal_magnitude'])
            cap_results.append({'metric': metric, 'rho': rho, 'p': pval})
        cap_results_df = pd.DataFrame(cap_results).sort_values('rho', ascending=False)
        for _, r in cap_results_df.head(5).iterrows():
            sig = "*" if r['p'] < 0.05 else ""
            print(f"    {r['metric']:<33s} rho={r['rho']:+.3f} {sig}")
        print("    ...")
        for _, r in cap_results_df.tail(3).iterrows():
            sig = "*" if r['p'] < 0.05 else ""
            print(f"    {r['metric']:<33s} rho={r['rho']:+.3f} {sig}")

    # =====================================================================
    # SECTION 7: Median split analysis for top non-criteria metrics
    # =====================================================================
    print("\n" + "=" * 80)
    print("SECTION 7: MEDIAN SPLIT — TOP NON-CRITERIA METRICS")
    print("=" * 80)
    print("(Checking if above-median metric value predicts better reversal outcomes)")

    # Top non-criteria metrics from Section 1
    non_criteria = [m for m in results_df['metric'].values
                    if m not in ['pct_from_9ema', 'prior_day_range_atr', 'rvol_score',
                                 'consecutive_up_days', 'gap_pct', 'reversal_open_close_pct']]

    for metric in non_criteria[:10]:  # Top 10 non-criteria by rho
        valid = df[[metric, 'reversal_magnitude', 'win']].dropna()
        if len(valid) < 10:
            continue
        median_val = valid[metric].median()
        above = valid[valid[metric] >= median_val]
        below = valid[valid[metric] < median_val]

        above_wr = above['win'].mean() * 100
        below_wr = below['win'].mean() * 100
        above_avg = above['reversal_magnitude'].mean() * 100
        below_avg = below['reversal_magnitude'].mean() * 100

        # Get the rho for this metric
        rho_val = results_df[results_df['metric'] == metric]['rho'].values[0]

        print(f"\n  {metric} (rho={rho_val:+.3f}, median={median_val:.3f}):")
        print(f"    Above median: {len(above):3d} trades | Win: {above_wr:5.1f}% | Avg: {above_avg:+.1f}%")
        print(f"    Below median: {len(below):3d} trades | Win: {below_wr:5.1f}% | Avg: {below_avg:+.1f}%")

    # =====================================================================
    # SECTION 8: 3DGapFade specific analysis
    # =====================================================================
    print("\n" + "=" * 80)
    print("SECTION 8: 3DGapFade SPECIFIC CORRELATIONS")
    print("=" * 80)

    gf = df[df['setup'] == '3DGapFade'].copy()
    if len(gf) >= 8:
        print(f"3DGapFade: {len(gf)} trades | Win: {(gf['reversal_magnitude']>0).mean()*100:.0f}% | Avg: {gf['reversal_magnitude'].mean()*100:+.1f}%")

        gf_results = []
        for metric in candidate_metrics:
            if metric not in gf.columns:
                continue
            valid = gf[[metric, 'reversal_magnitude']].dropna()
            if len(valid) < 5:
                continue
            rho, pval = stats.spearmanr(valid[metric], valid['reversal_magnitude'])
            gf_results.append({'metric': metric, 'rho': rho, 'p': pval, 'n': len(valid)})

        gf_results_df = pd.DataFrame(gf_results).sort_values('rho', ascending=False)

        print(f"\n{'Metric':<35s} {'Rho':>8s} {'p-value':>10s} {'n':>5s}")
        print("-" * 60)
        for _, r in gf_results_df.iterrows():
            sig = "***" if r['p'] < 0.01 else ("**" if r['p'] < 0.05 else "")
            marker = ""
            if r['metric'] in ['pct_from_9ema', 'prior_day_range_atr', 'rvol_score',
                               'consecutive_up_days', 'gap_pct']:
                marker = " <-- CURRENT"
            print(f"  {r['metric']:<33s} {r['rho']:+.3f}   {r['p']:.4f}  {int(r['n']):>4d}  {sig}{marker}")

    # =====================================================================
    # SECTION 9: reversal_pct as outcome vs criterion
    # =====================================================================
    print("\n" + "=" * 80)
    print("SECTION 9: NOTE ON reversal_pct AS CRITERION")
    print("=" * 80)
    print("reversal_pct (reversal_open_close_pct) IS the outcome metric.")
    print("Using it as a criterion is circular — it can only be evaluated AFTER the trade.")
    print("For the generic scorer (reversal_scorer.py) this is fine for historical grading,")
    print("but for pre-trade validation (reversal_pretrade.py) it must be excluded.")
    print()
    print("The 5 pre-trade criteria in reversal_pretrade.py are:")
    print("  1. pct_from_9ema  2. prior_day_range_atr  3. rvol_score")
    print("  4. consecutive_up_days  5. gap_pct")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: KEY FINDINGS")
    print("=" * 80)

    # Top 5 predictive (positive rho = higher metric -> bigger reversal)
    print("\nTOP 5 METRICS FOR PREDICTING LARGER REVERSALS (positive rho):")
    for i, (_, r) in enumerate(results_df.head(5).iterrows()):
        in_current = r['metric'] in ['pct_from_9ema', 'prior_day_range_atr', 'rvol_score',
                                      'consecutive_up_days', 'gap_pct']
        status = "IN SCORER" if in_current else "NOT IN SCORER"
        print(f"  {i+1}. {r['metric']:<30s} rho={r['rho']:+.3f}  [{status}]")

    print("\nBOTTOM 5 METRICS (negative rho = higher metric -> SMALLER reversal):")
    for i, (_, r) in enumerate(results_df.tail(5).iterrows()):
        in_current = r['metric'] in ['pct_from_9ema', 'prior_day_range_atr', 'rvol_score',
                                      'consecutive_up_days', 'gap_pct']
        status = "IN SCORER" if in_current else "NOT IN SCORER"
        print(f"  {i+1}. {r['metric']:<30s} rho={r['rho']:+.3f}  [{status}]")

    # Current criteria summary
    print("\nCURRENT 5 PRE-TRADE CRITERIA PREDICTIVE POWER:")
    current_criteria = ['pct_from_9ema', 'prior_day_range_atr', 'rvol_score',
                        'consecutive_up_days', 'gap_pct']
    for c in current_criteria:
        row = results_df[results_df['metric'] == c]
        if not row.empty:
            r = row.iloc[0]
            verdict = "STRONG" if abs(r['rho']) > 0.2 and r['p_value'] < 0.05 else \
                      "MODERATE" if abs(r['rho']) > 0.1 else "WEAK"
            print(f"  {c:<30s} rho={r['rho']:+.3f}  p={r['p_value']:.4f}  [{verdict}]")


if __name__ == '__main__':
    run_analysis()
