"""
Reversal Setup Scorer (V3)

Scores parabolic short reversal setups based on 6 cap-adjusted criteria.
Returns a score (0-6), grade (A+, A, B, C, F), and GO/NO-GO recommendation.

V3 adds:
  - Pre-trade score (criteria 1-5 only, no outcome leakage from criterion 6)
  - Continuous ATR-adjusted intensity score (0-100) for magnitude prediction
    within GO trades. Binary score -> GO/NO-GO gate; intensity -> sizing signal.

The 6 criteria for a TRUE parabolic reversal:
1. Extended above 9EMA - Price significantly elevated from 9-day EMA
2. Range expansion - Prior day range >= threshold x ATR
3. Volume expansion - RVOL >= threshold
4. 3-day momentum run-up - Short-term price acceleration (rho=+0.546)
5. Euphoric gap up - Gap up on reversal day
6. Large reversal - Actual reversal size on the day (outcome — NOT pre-trade)

V2 update (Feb 2026):
  Replaced consecutive_up_days (rho=+0.086, not significant p=0.35) with
  pct_change_3 (rho=+0.546, p<0.0001). The old criterion had zero predictive
  power for reversal magnitude. V2 pre-trade score correlates with P&L at
  rho=+0.211 (p=0.02) vs original rho=+0.049 (p=0.59).

Usage:
    from analyzers.reversal_scorer import ReversalScorer, compute_reversal_intensity
    scorer = ReversalScorer()
    result = scorer.score_setup(ticker, date, cap, metrics)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

_DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ---------------------------------------------------------------------------
# Percentile ranking helper (mirrors bounce_trader.py)
# ---------------------------------------------------------------------------
try:
    from scipy.stats import percentileofscore as _pctrank
except Exception:
    def _pctrank(a, score, kind='rank'):
        """Minimal percentile-of-score: mean of strict and weak ranks."""
        try:
            arr = np.asarray(a, dtype=float)
            if arr.size == 0:
                return np.nan
            s = float(score)
        except Exception:
            return np.nan
        if kind in ('rank', 'mean'):
            return 100.0 * (np.sum(arr < s) + 0.5 * np.sum(arr == s)) / arr.size
        if kind == 'weak':
            return 100.0 * np.sum(arr <= s) / arr.size
        if kind == 'strict':
            return 100.0 * np.sum(arr < s) / arr.size
        return 100.0 * np.sum(arr <= s) / arr.size


# ---------------------------------------------------------------------------
# Reversal intensity spec: (metric, higher_is_better, weight)
# Cap-stratified percentile ranking for cross-cap comparability.
# Composite Spearman rho vs P&L: +0.444 (cap-stratified), clean Q1→Q4 monotonic.
# ---------------------------------------------------------------------------
_REVERSAL_INTENSITY_SPEC = [
    ('ema9_atr',            True, 0.20),  # ATR-adj 9EMA distance
    ('mom3_atr',            True, 0.20),  # ATR-adj 3-day momentum
    ('pct_from_50mav',      True, 0.15),  # raw % above 50MA (longer-term extension)
    ('prior_day_range_atr', True, 0.15),  # range/ATR, already adjusted
    ('gap_atr',             True, 0.15),  # ATR-adj gap
    ('rvol_score',          True, 0.15),  # relative volume
]

# Cap groups for stratified percentile ranking.
# Micro/Small have similar ATR profiles; Medium/Large/ETF are another group.
_CAP_GROUPS = {
    'Micro': 'small', 'Small': 'small',
    'Medium': 'large', 'Large': 'large', 'ETF': 'large',
}

# ---------------------------------------------------------------------------
# Load Grade-A reference data for intensity percentile ranking
# ---------------------------------------------------------------------------
_reversal_csv = _DATA_DIR / 'reversal_data.csv'
try:
    _reversal_df_all = pd.read_csv(_reversal_csv).dropna(subset=['ticker', 'date'])
    _reversal_ref = _reversal_df_all[_reversal_df_all['trade_grade'] == 'A'].copy()
    # Precompute ATR-adjusted columns for the reference set
    _ref_atr = _reversal_ref['atr_pct'].replace(0, np.nan)  # Series.replace
    _reversal_ref['ema9_atr'] = _reversal_ref['pct_from_9ema'] / _ref_atr
    _reversal_ref['mom3_atr'] = _reversal_ref['pct_change_3'] / _ref_atr
    _reversal_ref['gap_atr'] = _reversal_ref['gap_pct'] / _ref_atr
    _reversal_ref['cap_group'] = _reversal_ref['cap'].map(_CAP_GROUPS).fillna('large')
    # Pre-split by cap group for fast lookup
    _ref_by_cap_group = {g: sub for g, sub in _reversal_ref.groupby('cap_group')}
except Exception:
    _reversal_df_all = pd.DataFrame()
    _reversal_ref = pd.DataFrame()
    _ref_by_cap_group = {}


# ---------------------------------------------------------------------------
# Continuous intensity scoring (cap-stratified)
# ---------------------------------------------------------------------------

def compute_reversal_intensity(metrics: Dict, cap: str = None,
                               ref_df: pd.DataFrame = None) -> Dict:
    """
    Compute ATR-adjusted intensity score (0-100) for a reversal setup.

    Percentile-ranks each metric against the Grade-A reference distribution
    for the same cap group (Micro/Small or Medium/Large/ETF).

    Args:
        metrics: dict with pct_from_9ema, pct_change_3, gap_pct, atr_pct,
                 prior_day_range_atr, rvol_score, pct_from_50mav
        cap: market cap category (used for cap-stratified ranking)
        ref_df: override reference DataFrame (skips cap stratification)

    Returns:
        {'composite': 0-100, 'details': {col: {pctile, weight, actual}}}
    """
    # Select reference set: cap-stratified by default
    if ref_df is None:
        cap_group = _CAP_GROUPS.get(cap, 'large') if cap else 'large'
        ref_df = _ref_by_cap_group.get(cap_group, _reversal_ref)

    atr_pct = metrics.get('atr_pct')
    if atr_pct is None or pd.isna(atr_pct) or atr_pct == 0:
        return {'composite': None, 'details': {}}

    # Compute ATR-adjusted values for this setup
    derived = {
        'ema9_atr': metrics.get('pct_from_9ema', 0) / atr_pct,
        'mom3_atr': metrics.get('pct_change_3', 0) / atr_pct,
        'gap_atr': metrics.get('gap_pct', 0) / atr_pct,
        'prior_day_range_atr': metrics.get('prior_day_range_atr'),
        'rvol_score': metrics.get('rvol_score'),
        'pct_from_50mav': metrics.get('pct_from_50mav'),
    }

    details = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for col, higher_is_better, weight in _REVERSAL_INTENSITY_SPEC:
        actual = derived.get(col)
        ref_vals = ref_df[col].dropna().values if col in ref_df.columns else []
        if actual is None or pd.isna(actual) or len(ref_vals) == 0:
            details[col] = {'pctile': None, 'weight': weight, 'actual': actual}
            continue
        raw_pctile = _pctrank(ref_vals, actual, kind='rank')
        pctile = raw_pctile if higher_is_better else 100.0 - raw_pctile
        details[col] = {'pctile': round(pctile, 1), 'weight': weight,
                        'actual': round(actual, 3) if isinstance(actual, float) else actual}
        weighted_sum += pctile * weight
        total_weight += weight

    composite = round(weighted_sum / total_weight, 1) if total_weight > 0 else None
    return {'composite': composite, 'details': details}


@dataclass
class CriteriaThresholds:
    """Cap-adjusted thresholds for the 6 criteria."""
    pct_from_9ema: float      # Minimum % above 9EMA
    prior_day_range_atr: float  # Minimum range as multiple of ATR
    rvol_score: float          # Minimum relative volume
    pct_change_3: float        # Minimum 3-day price change (V2: rho=+0.546)
    gap_pct: float             # Minimum gap % (can be 0 for large/ETF)
    reversal_pct: float        # Minimum reversal % (negative value)


# Cap-specific thresholds - validated against 110 trades (V2)
# TRUE A requires 5/6 or 6/6 criteria passed
CAP_THRESHOLDS = {
    'Micro': CriteriaThresholds(
        pct_from_9ema=0.80,       # >= 80% above 9EMA
        prior_day_range_atr=3.0,  # >= 3x ATR
        rvol_score=2.0,           # >= 2x RVOL
        pct_change_3=0.50,        # >= 50% 3-day run (V2: rho=+0.546)
        gap_pct=0.15,             # >= 15% gap
        reversal_pct=-0.20,       # >= 20% reversal (stored as negative)
    ),
    'Small': CriteriaThresholds(
        pct_from_9ema=0.40,       # >= 40% above 9EMA
        prior_day_range_atr=2.0,  # >= 2x ATR
        rvol_score=2.0,           # >= 2x RVOL
        pct_change_3=0.25,        # >= 25% 3-day run
        gap_pct=0.10,             # >= 10% gap
        reversal_pct=-0.10,       # >= 10% reversal
    ),
    'Medium': CriteriaThresholds(
        pct_from_9ema=0.15,       # >= 15% above 9EMA
        prior_day_range_atr=1.0,  # >= 1x ATR
        rvol_score=1.5,           # >= 1.5x RVOL
        pct_change_3=0.10,        # >= 10% 3-day run
        gap_pct=0.05,             # >= 5% gap
        reversal_pct=-0.05,       # >= 5% reversal
    ),
    'Large': CriteriaThresholds(
        pct_from_9ema=0.08,       # >= 8% above 9EMA
        prior_day_range_atr=0.8,  # >= 0.8x ATR
        rvol_score=1.0,           # >= 1x RVOL
        pct_change_3=0.05,        # >= 5% 3-day run
        gap_pct=0.01,             # >= 1% gap (filters flat opens)
        reversal_pct=-0.03,       # >= 3% reversal
    ),
    'ETF': CriteriaThresholds(
        pct_from_9ema=0.04,       # >= 4% above 9EMA
        prior_day_range_atr=1.0,  # >= 1x ATR
        rvol_score=1.5,           # >= 1.5x RVOL
        pct_change_3=0.03,        # >= 3% 3-day run
        gap_pct=0.005,            # >= 0.5% gap (filters flat opens)
        reversal_pct=-0.015,      # >= 1.5% reversal
    ),
}

# Criteria descriptions for reporting
CRITERIA_NAMES = {
    'pct_from_9ema': 'Extended above 9EMA',
    'prior_day_range_atr': 'Range expansion (ATR)',
    'rvol_score': 'Volume expansion (RVOL)',
    'pct_change_3': '3-day momentum run-up',
    'gap_pct': 'Euphoric gap up',
    'reversal_pct': 'Large reversal',
}


class ReversalScorer:
    """Scores reversal setups based on 6 cap-adjusted criteria."""

    def __init__(self):
        self.thresholds = CAP_THRESHOLDS

    def _get_thresholds(self, cap: str) -> CriteriaThresholds:
        """Get thresholds for market cap."""
        if cap not in self.thresholds:
            logging.warning(f"Unknown cap '{cap}' (type={type(cap).__name__}), defaulting to Medium")
            return self.thresholds['Medium']
        return self.thresholds[cap]

    def _check_criterion(self, value, threshold, is_reversal: bool = False) -> bool:
        """Check if a single criterion passes."""
        if pd.isna(value) or value is None:
            return False

        if is_reversal:
            # Reversal is stored as negative, so we check <=
            return value <= threshold
        else:
            return value >= threshold

    def _evaluate_criteria(self, metrics: Dict, thresholds: CriteriaThresholds) -> Tuple[int, List[str], List[str]]:
        """
        Evaluate all 6 criteria and return score with details.

        Returns:
            Tuple of (score, passed_criteria, failed_criteria)
        """
        passed = []
        failed = []

        # 1. Extended above 9EMA
        if self._check_criterion(metrics.get('pct_from_9ema'), thresholds.pct_from_9ema):
            passed.append('pct_from_9ema')
        else:
            failed.append('pct_from_9ema')

        # 2. Range expansion
        if self._check_criterion(metrics.get('prior_day_range_atr'), thresholds.prior_day_range_atr):
            passed.append('prior_day_range_atr')
        else:
            failed.append('prior_day_range_atr')

        # 3. Volume expansion
        if self._check_criterion(metrics.get('rvol_score'), thresholds.rvol_score):
            passed.append('rvol_score')
        else:
            failed.append('rvol_score')

        # 4. 3-day momentum run-up (V2: replaces consecutive_up_days)
        if self._check_criterion(metrics.get('pct_change_3'), thresholds.pct_change_3):
            passed.append('pct_change_3')
        else:
            failed.append('pct_change_3')

        # 5. Gap up
        if self._check_criterion(metrics.get('gap_pct'), thresholds.gap_pct):
            passed.append('gap_pct')
        else:
            failed.append('gap_pct')

        # 6. Reversal size (check against negative threshold)
        if self._check_criterion(metrics.get('reversal_open_close_pct'), thresholds.reversal_pct, is_reversal=True):
            passed.append('reversal_pct')
        else:
            failed.append('reversal_pct')

        return len(passed), passed, failed

    def _score_to_grade(self, score: int) -> str:
        """Convert criteria score (0-6) to letter grade."""
        if score == 6:
            return 'A+'
        elif score == 5:
            return 'A'
        elif score == 4:
            return 'B'
        elif score == 3:
            return 'C'
        else:
            return 'F'

    @staticmethod
    def _pretrade_grade(score: int) -> str:
        """Convert pre-trade score (0-5, criteria 1-5 only) to letter grade."""
        if score == 5:
            return 'A+'
        elif score == 4:
            return 'A'
        elif score == 3:
            return 'B'
        elif score == 2:
            return 'C'
        else:
            return 'F'

    @staticmethod
    def _pretrade_recommendation(score: int) -> str:
        """Get GO/NO-GO recommendation from pre-trade score (0-5)."""
        if score >= 4:
            return 'GO'
        elif score == 3:
            return 'CAUTION'
        else:
            return 'NO-GO'

    def _get_recommendation(self, score: int) -> str:
        """Get GO/NO-GO recommendation based on score."""
        if score >= 5:
            return 'GO'
        elif score == 4:
            return 'CAUTION'
        else:
            return 'NO-GO'

    def score_setup(self, ticker: str, date: str, cap: str, metrics: Dict) -> Dict:
        """
        Score a reversal setup against 6 cap-adjusted criteria.

        Returns dict with:
          - Full 6-criterion score/grade/recommendation (includes outcome)
          - Pre-trade score (criteria 1-5 only, no outcome leakage)
          - Intensity (0-100 continuous ATR-adjusted score) when pretrade is GO
        """
        thresholds = self._get_thresholds(cap)
        score, passed, failed = self._evaluate_criteria(metrics, thresholds)
        grade = self._score_to_grade(score)
        recommendation = self._get_recommendation(score)

        # Pre-trade score: criteria 1-5 only (exclude reversal_pct / criterion 6)
        pretrade_passed = [c for c in passed if c != 'reversal_pct']
        pretrade_score = len(pretrade_passed)
        pretrade_grade = self._pretrade_grade(pretrade_score)
        pretrade_rec = self._pretrade_recommendation(pretrade_score)

        # Build detailed breakdown
        criteria_details = {}
        for criterion in ['pct_from_9ema', 'prior_day_range_atr', 'rvol_score',
                          'pct_change_3', 'gap_pct', 'reversal_pct']:
            if criterion == 'reversal_pct':
                actual = metrics.get('reversal_open_close_pct')
                threshold = getattr(thresholds, criterion)
            else:
                actual = metrics.get(criterion)
                threshold = getattr(thresholds, criterion)

            criteria_details[criterion] = {
                'name': CRITERIA_NAMES[criterion],
                'threshold': threshold,
                'actual': actual,
                'passed': criterion in passed,
            }

        # Intensity: only compute when pretrade is GO and atr_pct is available
        intensity = None
        if pretrade_rec == 'GO' and metrics.get('atr_pct'):
            intensity_result = compute_reversal_intensity(metrics, cap=cap)
            intensity = intensity_result.get('composite')

        return {
            'ticker': ticker,
            'date': date,
            'cap': cap,
            'score': score,
            'max_score': 6,
            'grade': grade,
            'recommendation': recommendation,
            'passed_criteria': passed,
            'failed_criteria': failed,
            'criteria_details': criteria_details,
            'is_true_a': score >= 5,
            # V3: pre-trade score (no outcome leakage)
            'pretrade_score': pretrade_score,
            'pretrade_max': 5,
            'pretrade_grade': pretrade_grade,
            'pretrade_recommendation': pretrade_rec,
            # V3: continuous intensity (None if not GO or no atr_pct)
            'intensity': intensity,
        }

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score all setups in a dataframe.

        Args:
            df: DataFrame with reversal data

        Returns:
            DataFrame with added score, grade, and recommendation columns
        """
        results = []

        for idx, row in df.iterrows():
            metrics = row.to_dict()
            cap = row.get('cap', 'Medium')
            if cap is None or (isinstance(cap, float) and pd.isna(cap)):
                cap = 'Medium'
            result = self.score_setup(
                ticker=row.get('ticker', ''),
                date=row.get('date', ''),
                cap=cap,
                metrics=metrics
            )
            results.append({
                'criteria_score': result['score'],
                'criteria_grade': result['grade'],
                'recommendation': result['recommendation'],
                'is_true_a': result['is_true_a'],
                'failed_criteria': ', '.join(result['failed_criteria']) if result['failed_criteria'] else 'PERFECT',
                'pretrade_score': result['pretrade_score'],
                'pretrade_grade': result['pretrade_grade'],
                'pretrade_recommendation': result['pretrade_recommendation'],
                'intensity': result['intensity'],
            })

        score_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), score_df], axis=1)


def print_score_report(result: Dict):
    """Print a formatted score report."""
    print(f"\n{'='*70}")
    print(f"REVERSAL SETUP SCORE: {result['ticker']} ({result['date']})")
    print(f"{'='*70}")
    print(f"Market Cap: {result['cap']}")
    print(f"Full Score: {result['score']}/{result['max_score']} ({result['grade']}) — {result['recommendation']}")
    print(f"Pre-Trade:  {result['pretrade_score']}/{result['pretrade_max']} ({result['pretrade_grade']}) — {result['pretrade_recommendation']}")
    intensity = result.get('intensity')
    print(f"Intensity:  {intensity:.0f}/100" if intensity is not None else "Intensity:  N/A")
    print()

    print("CRITERIA BREAKDOWN:")
    print("-" * 60)
    for criterion, details in result['criteria_details'].items():
        is_pretrade = criterion != 'reversal_pct'
        tag = "PRE" if is_pretrade else "OUT"
        status = "[PASS]" if details['passed'] else "[FAIL]"
        actual = details['actual']
        threshold = details['threshold']

        if criterion in ['pct_from_9ema', 'gap_pct', 'reversal_pct', 'pct_change_3']:
            actual_str = f"{actual*100:.1f}%" if actual is not None else "N/A"
            threshold_str = f"{threshold*100:.1f}%"
        else:
            actual_str = f"{actual:.1f}x" if actual is not None else "N/A"
            threshold_str = f"{threshold:.1f}x"

        print(f"  {status} [{tag}] {details['name']}")
        print(f"              Required: >= {threshold_str} | Actual: {actual_str}")
    print()


# Example usage and testing
if __name__ == '__main__':
    from scipy.stats import spearmanr

    df = pd.read_csv(_DATA_DIR / 'reversal_data.csv')
    grade_a = df[df['trade_grade'] == 'A'].copy()

    scorer = ReversalScorer()
    scored_df = scorer.score_dataframe(grade_a)

    # Calculate P&L (short trade: negate the reversal pct)
    scored_df['pnl'] = -scored_df['reversal_open_close_pct'] * 100

    # =================================================================
    # FULL 6-CRITERION SCORE (existing)
    # =================================================================
    print("\n" + "="*70)
    print("REVERSAL SETUP SCORING SUMMARY - Grade A Trades")
    print("="*70)

    print("\nFULL SCORE DISTRIBUTION (6 criteria):")
    print(scored_df['criteria_score'].value_counts().sort_index(ascending=False))

    print("\nPERFORMANCE BY FULL SCORE:")
    print("-" * 50)
    for score in [6, 5, 4, 3, 2, 1, 0]:
        subset = scored_df[scored_df['criteria_score'] == score]
        if len(subset) > 0:
            win_rate = (subset['pnl'] > 0).mean() * 100
            avg_pnl = subset['pnl'].mean()
            print(f"Score {score}/6: {len(subset):2d} trades | Win: {win_rate:5.1f}% | Avg P&L: {avg_pnl:+6.1f}%")

    # =================================================================
    # PRE-TRADE SCORE (criteria 1-5 only, no outcome leakage)
    # =================================================================
    print("\n" + "="*70)
    print("PRE-TRADE SCORE (criteria 1-5, no outcome leakage)")
    print("="*70)

    print("\nPRE-TRADE SCORE DISTRIBUTION:")
    print(scored_df['pretrade_score'].value_counts().sort_index(ascending=False))

    print("\nPRE-TRADE GRADE DISTRIBUTION:")
    print(scored_df['pretrade_grade'].value_counts())

    print("\nPRE-TRADE RECOMMENDATION DISTRIBUTION:")
    print(scored_df['pretrade_recommendation'].value_counts())

    print("\nPERFORMANCE BY PRE-TRADE SCORE:")
    print("-" * 50)
    for score in [5, 4, 3, 2, 1, 0]:
        subset = scored_df[scored_df['pretrade_score'] == score]
        if len(subset) > 0:
            win_rate = (subset['pnl'] > 0).mean() * 100
            avg_pnl = subset['pnl'].mean()
            print(f"Score {score}/5: {len(subset):2d} trades | Win: {win_rate:5.1f}% | Avg P&L: {avg_pnl:+6.1f}%")

    # Pre-trade GO vs NO-GO
    go = scored_df[scored_df['pretrade_recommendation'] == 'GO']
    no_go = scored_df[scored_df['pretrade_recommendation'] != 'GO']
    print(f"\nPre-Trade GO ({len(go)}): Win {(go['pnl'] > 0).mean()*100:.1f}% | Avg P&L {go['pnl'].mean():+.1f}%")
    if len(no_go) > 0:
        print(f"Pre-Trade !GO ({len(no_go)}): Win {(no_go['pnl'] > 0).mean()*100:.1f}% | Avg P&L {no_go['pnl'].mean():+.1f}%")

    # =================================================================
    # INTENSITY SCORE (continuous, ATR-adjusted)
    # =================================================================
    print("\n" + "="*70)
    print("INTENSITY SCORE (ATR-adjusted, 0-100)")
    print("="*70)

    has_intensity = scored_df['intensity'].notna()
    intensity_df = scored_df[has_intensity].copy()
    print(f"\nTrades with intensity: {len(intensity_df)} / {len(scored_df)}")

    if len(intensity_df) > 0:
        print(f"Intensity range: {intensity_df['intensity'].min():.0f} — {intensity_df['intensity'].max():.0f}")
        print(f"Mean: {intensity_df['intensity'].mean():.1f} | Median: {intensity_df['intensity'].median():.1f}")

        # Intensity quartiles
        print("\nPERFORMANCE BY INTENSITY QUARTILE (GO trades only):")
        print("-" * 60)
        quartile_labels = ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']
        try:
            intensity_df['iq'] = pd.qcut(intensity_df['intensity'], 4, labels=False)
            for q in range(4):
                subset = intensity_df[intensity_df['iq'] == q]
                if len(subset) > 0:
                    lo = subset['intensity'].min()
                    hi = subset['intensity'].max()
                    wr = (subset['pnl'] > 0).mean() * 100
                    avg = subset['pnl'].mean()
                    print(f"  {quartile_labels[q]:12s} ({lo:4.0f}-{hi:4.0f}): {len(subset):2d} trades | Win: {wr:5.1f}% | Avg P&L: {avg:+6.1f}%")
        except ValueError:
            print("  (not enough trades for quartile analysis)")

        # Correlation: intensity vs P&L
        rho, p = spearmanr(intensity_df['intensity'], intensity_df['pnl'])
        print(f"\nSpearman rho (intensity vs P&L): {rho:+.3f} (p={p:.4f})")

    # Correlation: pretrade_score vs P&L
    rho_pt, p_pt = spearmanr(scored_df['pretrade_score'], scored_df['pnl'])
    print(f"Spearman rho (pretrade_score vs P&L): {rho_pt:+.3f} (p={p_pt:.4f})")

    # =================================================================
    # SAMPLE REPORTS
    # =================================================================
    print("\n" + "="*70)
    print("SAMPLE SCORE REPORTS")
    print("="*70)

    for score_val in [6, 5, 4]:
        sample = scored_df[scored_df['criteria_score'] == score_val]
        if len(sample) > 0:
            row = sample.iloc[0]
            result = scorer.score_setup(
                ticker=row['ticker'],
                date=row['date'],
                cap=row['cap'],
                metrics=row.to_dict()
            )
            print_score_report(result)

    # Save scored data
    scored_df.to_csv(_DATA_DIR / 'reversal_scored.csv', index=False)
    print("\nScored data saved to data/reversal_scored.csv")
