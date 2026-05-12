"""Console report output for walk-forward validation results.

Prints a structured report with 7 sections covering data quality,
threshold comparison, performance degradation, and statistical significance.
"""

from typing import Dict, List, Optional
from validation.walk_forward_engine import WalkForwardResult
from validation.metrics import PeriodMetrics, DegradationReport
from analyzers.bootstrap import format_ci

from analyzers.reversal_scorer import CAP_THRESHOLDS, CriteriaThresholds, READINESS_THRESHOLDS
from analyzers.reversal_pretrade import REVERSAL_SETUP_PROFILES
from analyzers.bounce_scorer import SETUP_PROFILES


def _header(title: str, width: int = 80):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def _subheader(title: str, width: int = 70):
    print(f"\n  {title}")
    print(f"  {'-' * (width - 4)}")


def _fmt_pct(val, signed=False):
    if val is None:
        return 'N/A'
    if signed:
        return f"{val:+.1f}%"
    return f"{val:.1f}%"


def _fmt_pp(val):
    """Format percentage point change."""
    if val is None:
        return 'N/A'
    return f"{val:+.1f}pp"


# ---------------------------------------------------------------------------
# Section 1: Data Split Summary
# ---------------------------------------------------------------------------

def print_split_summary(result: WalkForwardResult):
    _header(f"1. DATA SPLIT SUMMARY — {result.strategy.upper()}")

    d = result.diagnostics
    print(f"\n  Total trades: {d.total_trades}")
    print(f"  Train  (<=  {result.split.train_end}):  {d.train_n:3d} trades  {d.train_date_range}")
    print(f"  Valid  ({result.split.train_end[:4]}-{result.split.validate_end[:4]}):  {d.validate_n:3d} trades  {d.validate_date_range}")
    print(f"  Test   (>   {result.split.validate_end}):  {d.test_n:3d} trades  {d.test_date_range}")

    if d.per_cap:
        _subheader("Cap Distribution")
        print(f"  {'Cap':<10} {'Train':>6} {'Valid':>6} {'Test':>6}")
        for cap, counts in sorted(d.per_cap.items()):
            print(f"  {cap:<10} {counts.get('train', 0):>6} {counts.get('validate', 0):>6} {counts.get('test', 0):>6}")

    if d.per_setup:
        _subheader("Setup Distribution")
        print(f"  {'Setup':<30} {'Train':>6} {'Valid':>6} {'Test':>6}")
        for setup, counts in sorted(d.per_setup.items()):
            print(f"  {setup:<30} {counts.get('train', 0):>6} {counts.get('validate', 0):>6} {counts.get('test', 0):>6}")

    if d.sparse_cells:
        _subheader("Sparse Cell Warnings (<5 trades)")
        for warning in d.sparse_cells[:15]:
            print(f"  ! {warning}")
        if len(d.sparse_cells) > 15:
            print(f"  ... and {len(d.sparse_cells) - 15} more")


# ---------------------------------------------------------------------------
# Section 2: Threshold Comparison
# ---------------------------------------------------------------------------

def print_threshold_comparison(result: WalkForwardResult):
    _header(f"2. THRESHOLD COMPARISON — {result.strategy.upper()}")

    if result.strategy == 'reversal':
        _print_reversal_threshold_comparison(result)
    elif result.strategy == 'bounce':
        _print_bounce_threshold_comparison(result)
    else:
        print("\n  No scoring system for breakout — skipping threshold comparison")


def _print_reversal_threshold_comparison(result: WalkForwardResult):
    derived = result.derived.reversal_cap_thresholds
    if not derived:
        print("\n  No derived thresholds (insufficient training data)")
        return

    criteria = ['pct_from_9ema', 'prior_day_range_atr', 'rvol_score',
                'pct_change_3', 'gap_pct', 'reversal_pct']

    _subheader("Generic ReversalScorer (CAP_THRESHOLDS)")
    for cap in ['ETF', 'Large', 'Medium', 'Small', 'Micro']:
        if cap not in derived and cap not in CAP_THRESHOLDS:
            continue
        prod = CAP_THRESHOLDS.get(cap)
        train = derived.get(cap)
        if not prod or not train:
            continue

        print(f"\n  {cap}:")
        print(f"    {'Criterion':<25} {'Production':>12} {'Training':>12} {'Delta':>10}")
        for crit in criteria:
            p_val = getattr(prod, crit, None)
            t_val = getattr(train, crit, None)
            if p_val is not None and t_val is not None:
                delta = t_val - p_val
                if crit in ('pct_from_9ema', 'gap_pct', 'pct_change_3', 'reversal_pct'):
                    print(f"    {crit:<25} {p_val*100:>11.1f}% {t_val*100:>11.1f}% {delta*100:>+9.1f}%")
                else:
                    print(f"    {crit:<25} {p_val:>11.2f}x {t_val:>11.2f}x {delta:>+9.2f}x")

    # 3DGapFade profile comparison
    derived_profiles = result.derived.reversal_setup_profiles
    if derived_profiles and '3DGapFade' in derived_profiles:
        _subheader("3DGapFade Profile (per-cap pretrade thresholds)")
        prod_profile = REVERSAL_SETUP_PROFILES.get('3DGapFade')
        train_profile = derived_profiles['3DGapFade']

        if prod_profile:
            pretrade_criteria = ['pct_from_9ema', 'prior_day_range_atr', 'rvol_score',
                                 'pct_change_3', 'gap_pct']
            for cap in ['ETF', 'Large', 'Medium', 'Small', 'Micro']:
                print(f"\n  {cap}:")
                print(f"    {'Criterion':<25} {'Production':>12} {'Training':>12} {'Delta':>10}")
                for crit in pretrade_criteria:
                    p_val = prod_profile.get_threshold(crit, cap)
                    t_val = train_profile.get_threshold(crit, cap)
                    if p_val is not None and t_val is not None:
                        delta = t_val - p_val
                        if crit in ('pct_from_9ema', 'gap_pct', 'pct_change_3'):
                            print(f"    {crit:<25} {p_val*100:>11.1f}% {t_val*100:>11.1f}% {delta*100:>+9.1f}%")
                        else:
                            print(f"    {crit:<25} {p_val:>11.2f}x {t_val:>11.2f}x {delta:>+9.2f}x")

    # Readiness thresholds comparison
    derived_readiness = result.derived.reversal_readiness_thresholds
    if derived_readiness:
        _subheader("Readiness Gate (euphoric 3D momentum minimums)")
        print(f"  {'Cap':<10} {'Production':>12} {'Training':>12} {'Delta':>10}")
        for cap in ['ETF', 'Large', 'Medium', 'Small', 'Micro']:
            p_val = READINESS_THRESHOLDS.get(cap)
            t_val = derived_readiness.get(cap)
            if p_val is not None and t_val is not None:
                delta = t_val - p_val
                print(f"  {cap:<10} {p_val*100:>11.1f}% {t_val*100:>11.1f}% {delta*100:>+9.1f}%")


def _print_bounce_threshold_comparison(result: WalkForwardResult):
    derived = result.derived.bounce_setup_profiles
    if not derived:
        print("\n  No derived thresholds (insufficient training data)")
        return

    criteria = ['selloff_total_pct', 'pct_off_30d_high', 'gap_pct',
                'prior_day_range_atr', 'pct_change_3', 'pct_off_52wk_high']

    for setup_type in ['GapFade_weakstock', 'GapFade_strongstock']:
        prod = SETUP_PROFILES.get(setup_type)
        train = derived.get(setup_type)
        if not prod or not train:
            continue

        _subheader(f"{setup_type}")
        for cap in ['ETF', 'Large', 'Medium', 'Small', 'Micro']:
            print(f"\n  {cap}:")
            print(f"    {'Criterion':<25} {'Production':>12} {'Training':>12} {'Delta':>10}")
            for crit in criteria:
                p_val = prod.get_threshold(crit, cap)
                t_val = train.get_threshold(crit, cap)
                if p_val is not None and t_val is not None:
                    delta = t_val - p_val
                    if crit == 'prior_day_range_atr':
                        print(f"    {crit:<25} {p_val:>11.2f}x {t_val:>11.2f}x {delta:>+9.2f}x")
                    else:
                        print(f"    {crit:<25} {p_val*100:>11.1f}% {t_val*100:>11.1f}% {delta*100:>+9.1f}%")


# ---------------------------------------------------------------------------
# Section 3: In-Sample vs OOS Performance
# ---------------------------------------------------------------------------

def print_performance_comparison(result: WalkForwardResult):
    _header(f"3. IN-SAMPLE vs OOS PERFORMANCE — {result.strategy.upper()}")

    metrics_list = [
        ('Train (IS)', result.train_metrics),
        ('Validate (OOS)', result.validate_metrics),
        ('Validate (Prod)', result.validate_production_metrics),
        ('Test (Blind)', result.test_metrics),
    ]

    _subheader("Overall Performance")
    print(f"  {'Period':<20} {'N':>5} {'Win Rate':>12} {'Avg P&L':>12} {'Med P&L':>10} {'PF':>6}")
    for name, m in metrics_list:
        if m and m.n > 0:
            wr_str = format_ci(m.win_rate_ci) if m.win_rate_ci else _fmt_pct(m.win_rate)
            avg_str = format_ci(m.avg_pnl_ci, is_pnl=True) if m.avg_pnl_ci else _fmt_pct(m.avg_pnl, signed=True)
            print(f"  {name:<20} {m.n:>5} {wr_str:>12} {avg_str:>12} {m.median_pnl:>+9.1f}% {m.profit_factor:>6.2f}")

    _subheader("By Grade")
    for name, m in metrics_list:
        if m and m.by_grade:
            print(f"\n  {name}:")
            for grade in ['A', 'B', 'C']:
                g = m.by_grade.get(grade)
                if g:
                    print(f"    Grade {grade}: {g['n']:3d} trades | WR: {g['win_rate']:.1f}% | Avg: {g['avg_pnl']:+.1f}%")

    # Degradation summary
    if result.train_vs_validate:
        _print_degradation("Train vs Validate", result.train_vs_validate)
    if result.train_vs_test:
        _print_degradation("Train vs Test", result.train_vs_test)


def _print_degradation(label: str, deg: DegradationReport):
    _subheader(f"Degradation: {label}")
    print(f"  Win Rate Change: {_fmt_pp(deg.win_rate_change_pp)}")
    print(f"  Avg P&L Change:  {deg.avg_pnl_change_pct:+.1f}%")
    if deg.go_edge_retained is not None:
        print(f"  GO Edge Retained: {deg.go_edge_retained:.0f}%")
    print(f"  Verdict: {deg.verdict.upper()}")


# ---------------------------------------------------------------------------
# Section 4: Classification Power
# ---------------------------------------------------------------------------

def print_classification_power(result: WalkForwardResult):
    _header(f"4. CLASSIFICATION POWER — {result.strategy.upper()}")

    if result.strategy == 'breakout':
        print("\n  No scoring system — skipping")
        return

    metrics_list = [
        ('Train (IS)', result.train_metrics),
        ('Validate (OOS)', result.validate_metrics),
        ('Test (Blind)', result.test_metrics),
    ]

    _subheader("GO vs NO-GO Win Rate")
    print(f"  {'Period':<20} {'GO WR':>10} {'NO-GO WR':>10} {'Delta':>10} {'Edge?':>8}")
    for name, m in metrics_list:
        if m and m.n > 0:
            go = m.by_recommendation.get('GO', {})
            nogo = m.by_recommendation.get('NO-GO', {})
            go_wr = go.get('win_rate', None)
            nogo_wr = nogo.get('win_rate', None)
            delta = m.go_nogo_wr_delta
            edge = 'YES' if delta and delta > 0 else 'NO' if delta is not None else '?'
            print(f"  {name:<20} {_fmt_pct(go_wr):>10} {_fmt_pct(nogo_wr):>10} {_fmt_pp(delta):>10} {edge:>8}")

    _subheader("GO vs NO-GO Avg P&L")
    print(f"  {'Period':<20} {'GO Avg':>10} {'NO-GO Avg':>10} {'GO N':>6} {'NOGO N':>7}")
    for name, m in metrics_list:
        if m and m.n > 0:
            go = m.by_recommendation.get('GO', {})
            nogo = m.by_recommendation.get('NO-GO', {})
            print(f"  {name:<20} {_fmt_pct(go.get('avg_pnl'), signed=True):>10} "
                  f"{_fmt_pct(nogo.get('avg_pnl'), signed=True):>10} "
                  f"{go.get('n', 0):>6} {nogo.get('n', 0):>7}")

    _subheader("Score-P&L Correlation (Spearman)")
    for name, m in metrics_list:
        if m and m.score_pnl_correlation is not None:
            sig = '*' if m.score_pnl_pvalue and m.score_pnl_pvalue < 0.05 else ''
            print(f"  {name:<20} rho={m.score_pnl_correlation:+.3f} (p={m.score_pnl_pvalue:.4f}){sig}")


# ---------------------------------------------------------------------------
# Section 5: Blind Test
# ---------------------------------------------------------------------------

def print_blind_test(result: WalkForwardResult):
    _header(f"5. BLIND TEST (2025) — {result.strategy.upper()}")

    m = result.test_metrics
    if not m or m.n == 0:
        print("\n  No test data available")
        return

    print(f"\n  Trades: {m.n}")
    print(f"  Win Rate: {format_ci(m.win_rate_ci)}")
    print(f"  Avg P&L:  {format_ci(m.avg_pnl_ci, is_pnl=True)}")
    print(f"  Profit Factor: {m.profit_factor:.2f}")

    if m.by_recommendation:
        _subheader("By Recommendation")
        for rec in ['GO', 'CAUTION', 'NO-GO']:
            r = m.by_recommendation.get(rec)
            if r:
                print(f"  {rec:8s}: {r['n']:3d} trades | WR: {r['win_rate']:.1f}% | Avg: {r['avg_pnl']:+.1f}%")


# ---------------------------------------------------------------------------
# Section 6: Statistical Significance
# ---------------------------------------------------------------------------

def print_statistical_significance(result: WalkForwardResult):
    _header(f"6. STATISTICAL SIGNIFICANCE — {result.strategy.upper()}")

    tests = [
        ('Train vs Validate', result.train_vs_validate),
        ('Train vs Test', result.train_vs_test),
    ]

    _subheader("Fisher Exact Test (win rate comparison)")
    for label, deg in tests:
        if deg and deg.fisher_p_value is not None:
            sig = 'SIGNIFICANT' if deg.fisher_significant else 'not significant'
            print(f"  {label:<25} p={deg.fisher_p_value:.4f} ({sig})")
        elif deg:
            print(f"  {label:<25} N/A (insufficient data)")

    _subheader("Bootstrap 95% Confidence Intervals")
    for name, m in [('Train', result.train_metrics),
                     ('Validate', result.validate_metrics),
                     ('Test', result.test_metrics)]:
        if m and m.n > 0:
            wr_str = format_ci(m.win_rate_ci)
            avg_str = format_ci(m.avg_pnl_ci, is_pnl=True)
            dagger = ' (small sample)' if m.n < 10 else ''
            print(f"  {name:<12} WR: {wr_str:<25} Avg P&L: {avg_str}{dagger}")

    # CI overlap check
    if result.train_metrics and result.validate_metrics:
        t_ci = result.train_metrics.win_rate_ci
        v_ci = result.validate_metrics.win_rate_ci
        if t_ci and v_ci:
            overlap = t_ci.ci_lower <= v_ci.ci_upper and v_ci.ci_lower <= t_ci.ci_upper
            status = 'OVERLAP (consistent)' if overlap else 'NO OVERLAP (significant difference)'
            print(f"\n  Train vs Validate WR CI: {status}")


# ---------------------------------------------------------------------------
# Section 7: Verdict
# ---------------------------------------------------------------------------

def print_verdict(result: WalkForwardResult):
    _header(f"7. VERDICT — {result.strategy.upper()}")

    if result.strategy == 'breakout':
        m_train = result.train_metrics
        m_val = result.validate_metrics
        m_test = result.test_metrics
        if m_train and m_val:
            print(f"\n  Breakout (no scoring system):")
            print(f"  Train WR: {m_train.win_rate:.1f}% (n={m_train.n})")
            print(f"  Validate WR: {m_val.win_rate:.1f}% (n={m_val.n})")
            if m_test and m_test.n > 0:
                print(f"  Test WR: {m_test.win_rate:.1f}% (n={m_test.n})")
        return

    verdicts = []

    # Overall edge
    deg = result.train_vs_validate
    if deg:
        if deg.verdict == 'held':
            verdicts.append("Overall edge HELD out-of-sample")
        elif deg.verdict == 'degraded':
            verdicts.append(f"Overall edge DEGRADED ({_fmt_pp(deg.win_rate_change_pp)} win rate)")
        else:
            verdicts.append(f"Overall edge COLLAPSED ({_fmt_pp(deg.win_rate_change_pp)} win rate)")

    # GO classification power
    if result.validate_metrics and result.validate_metrics.go_nogo_wr_delta is not None:
        delta = result.validate_metrics.go_nogo_wr_delta
        if delta > 10:
            verdicts.append(f"GO classification WORKS OOS (GO beats NO-GO by {delta:.0f}pp)")
        elif delta > 0:
            verdicts.append(f"GO classification has WEAK edge OOS ({delta:.0f}pp)")
        else:
            verdicts.append("GO classification FAILED OOS (NO-GO beats GO)")

    # Production vs derived comparison
    if result.production_vs_derived_validate:
        pdeg = result.production_vs_derived_validate
        if abs(pdeg.win_rate_change_pp) < 3:
            verdicts.append("Training-derived thresholds match production (threshold stability)")
        else:
            verdicts.append(f"Training-derived thresholds differ from production by {_fmt_pp(pdeg.win_rate_change_pp)} WR")

    # Blind test
    if result.test_metrics and result.test_metrics.n >= 3:
        t_wr = result.test_metrics.win_rate
        if t_wr >= 60:
            verdicts.append(f"Blind test (2025): {t_wr:.0f}% WR on {result.test_metrics.n} trades — PROMISING")
        elif t_wr >= 40:
            verdicts.append(f"Blind test (2025): {t_wr:.0f}% WR on {result.test_metrics.n} trades — INCONCLUSIVE")
        else:
            verdicts.append(f"Blind test (2025): {t_wr:.0f}% WR on {result.test_metrics.n} trades — CONCERNING")

    # Small sample warning
    if result.validate_metrics and result.validate_metrics.n < 15:
        verdicts.append(f"WARNING: Only {result.validate_metrics.n} validation trades — interpret with caution")

    # Fisher test result
    if deg and deg.fisher_p_value is not None:
        if deg.fisher_significant:
            verdicts.append(f"Fisher exact test: SIGNIFICANT difference (p={deg.fisher_p_value:.4f})")
        else:
            verdicts.append(f"Fisher exact test: no significant difference (p={deg.fisher_p_value:.4f})")

    print()
    for i, v in enumerate(verdicts, 1):
        print(f"  {i}. {v}")


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def print_full_report(result: WalkForwardResult):
    """Print the complete walk-forward validation report."""
    banner = f"WALK-FORWARD VALIDATION: {result.strategy.upper()}"
    print(f"\n{'#' * 80}")
    print(f"#  {banner:<74}  #")
    print(f"{'#' * 80}")

    print_split_summary(result)
    print_threshold_comparison(result)
    print_performance_comparison(result)
    print_classification_power(result)
    print_blind_test(result)
    print_statistical_significance(result)
    print_verdict(result)

    print(f"\n{'#' * 80}")
    print(f"#  END OF REPORT: {result.strategy.upper():<58}  #")
    print(f"{'#' * 80}\n")
