"""
Reversal Setup Scorer

Scores parabolic short reversal setups based on 6 cap-adjusted criteria.
Returns a score (0-6), grade (A+, A, B, C, F), and GO/NO-GO recommendation.

The 6 criteria for a TRUE parabolic reversal:
1. Extended above 9EMA - Price significantly elevated from 9-day EMA
2. Range expansion - Prior day range >= threshold x ATR
3. Volume expansion - RVOL >= threshold
4. Consecutive up days - Sustained momentum into the top
5. Euphoric gap up - Gap up on reversal day
6. Large reversal - Actual reversal size on the day

Usage:
    from analyzers.reversal_scorer import ReversalScorer
    scorer = ReversalScorer()
    result = scorer.score_setup(ticker, date, cap, metrics)
"""

import pandas as pd
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class CriteriaThresholds:
    """Cap-adjusted thresholds for the 6 criteria."""
    pct_from_9ema: float      # Minimum % above 9EMA
    prior_day_range_atr: float  # Minimum range as multiple of ATR
    rvol_score: float          # Minimum relative volume
    consecutive_up_days: int   # Minimum consecutive up days
    gap_pct: float             # Minimum gap % (can be 0 for large/ETF)
    reversal_pct: float        # Minimum reversal % (negative value)


# Cap-specific thresholds - validated against 61 Grade A trades
# TRUE A requires 5/6 or 6/6 criteria passed
CAP_THRESHOLDS = {
    'Micro': CriteriaThresholds(
        pct_from_9ema=0.80,       # >= 80% above 9EMA
        prior_day_range_atr=3.0,  # >= 3x ATR
        rvol_score=2.0,           # >= 2x RVOL
        consecutive_up_days=3,    # >= 3 days
        gap_pct=0.15,             # >= 15% gap
        reversal_pct=-0.20,       # >= 20% reversal (stored as negative)
    ),
    'Small': CriteriaThresholds(
        pct_from_9ema=0.40,       # >= 40% above 9EMA
        prior_day_range_atr=2.0,  # >= 2x ATR
        rvol_score=2.0,           # >= 2x RVOL
        consecutive_up_days=2,    # >= 2 days
        gap_pct=0.10,             # >= 10% gap
        reversal_pct=-0.10,       # >= 10% reversal
    ),
    'Medium': CriteriaThresholds(
        pct_from_9ema=0.15,       # >= 15% above 9EMA
        prior_day_range_atr=1.0,  # >= 1x ATR
        rvol_score=1.5,           # >= 1.5x RVOL
        consecutive_up_days=2,    # >= 2 days
        gap_pct=0.05,             # >= 5% gap
        reversal_pct=-0.05,       # >= 5% reversal
    ),
    'Large': CriteriaThresholds(
        pct_from_9ema=0.08,       # >= 8% above 9EMA
        prior_day_range_atr=0.8,  # >= 0.8x ATR
        rvol_score=1.0,           # >= 1x RVOL
        consecutive_up_days=1,    # >= 1 day
        gap_pct=0.00,             # any gap OK
        reversal_pct=-0.03,       # >= 3% reversal
    ),
    'ETF': CriteriaThresholds(
        pct_from_9ema=0.04,       # >= 4% above 9EMA
        prior_day_range_atr=1.0,  # >= 1x ATR
        rvol_score=1.5,           # >= 1.5x RVOL
        consecutive_up_days=1,    # >= 1 day
        gap_pct=0.00,             # any gap OK
        reversal_pct=-0.015,      # >= 1.5% reversal
    ),
}

# Criteria descriptions for reporting
CRITERIA_NAMES = {
    'pct_from_9ema': 'Extended above 9EMA',
    'prior_day_range_atr': 'Range expansion (ATR)',
    'rvol_score': 'Volume expansion (RVOL)',
    'consecutive_up_days': 'Consecutive up days',
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
            logging.warning(f"Unknown cap '{cap}', defaulting to Medium")
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

        # 4. Consecutive up days
        if self._check_criterion(metrics.get('consecutive_up_days'), thresholds.consecutive_up_days):
            passed.append('consecutive_up_days')
        else:
            failed.append('consecutive_up_days')

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

        Args:
            ticker: Stock ticker symbol
            date: Trade date
            cap: Market cap category (Micro, Small, Medium, Large, ETF)
            metrics: Dictionary of indicator values

        Returns:
            Dictionary with score, grade, recommendation, and detailed breakdown
        """
        thresholds = self._get_thresholds(cap)
        score, passed, failed = self._evaluate_criteria(metrics, thresholds)
        grade = self._score_to_grade(score)
        recommendation = self._get_recommendation(score)

        # Build detailed breakdown
        criteria_details = {}
        for criterion in ['pct_from_9ema', 'prior_day_range_atr', 'rvol_score',
                          'consecutive_up_days', 'gap_pct', 'reversal_pct']:
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
            result = self.score_setup(
                ticker=row.get('ticker', ''),
                date=row.get('date', ''),
                cap=row.get('cap', 'Medium'),
                metrics=metrics
            )
            results.append({
                'criteria_score': result['score'],
                'criteria_grade': result['grade'],
                'recommendation': result['recommendation'],
                'is_true_a': result['is_true_a'],
                'failed_criteria': ', '.join(result['failed_criteria']) if result['failed_criteria'] else 'PERFECT'
            })

        score_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), score_df], axis=1)


def print_score_report(result: Dict):
    """Print a formatted score report."""
    print(f"\n{'='*70}")
    print(f"REVERSAL SETUP SCORE: {result['ticker']} ({result['date']})")
    print(f"{'='*70}")
    print(f"Market Cap: {result['cap']}")
    print(f"Score: {result['score']}/{result['max_score']}")
    print(f"Grade: {result['grade']}")
    print(f"Recommendation: {result['recommendation']}")
    print(f"True Grade A: {'YES' if result['is_true_a'] else 'NO'}")
    print()

    print("CRITERIA BREAKDOWN:")
    print("-" * 60)
    for criterion, details in result['criteria_details'].items():
        status = "[PASS]" if details['passed'] else "[FAIL]"
        actual = details['actual']
        threshold = details['threshold']

        # Format values for display
        if criterion in ['pct_from_9ema', 'gap_pct', 'reversal_pct']:
            actual_str = f"{actual*100:.1f}%" if actual is not None else "N/A"
            threshold_str = f"{threshold*100:.1f}%"
        elif criterion == 'consecutive_up_days':
            actual_str = f"{int(actual)}" if actual is not None else "N/A"
            threshold_str = f"{int(threshold)}"
        else:
            actual_str = f"{actual:.1f}x" if actual is not None else "N/A"
            threshold_str = f"{threshold:.1f}x"

        print(f"  {status} {details['name']}")
        print(f"         Required: >= {threshold_str} | Actual: {actual_str}")
    print()


# Example usage and testing
if __name__ == '__main__':
    df = pd.read_csv('C:\\Users\\zmbur\\PycharmProjects\\backtester\\data\\reversal_data.csv')
    grade_a = df[df['trade_grade'] == 'A'].copy()

    scorer = ReversalScorer()
    scored_df = scorer.score_dataframe(grade_a)

    # Calculate P&L
    scored_df['pnl'] = -scored_df['reversal_open_close_pct'] * 100

    print("\n" + "="*70)
    print("REVERSAL SETUP SCORING SUMMARY - Grade A Trades")
    print("="*70)

    # Score distribution
    print("\nSCORE DISTRIBUTION:")
    print(scored_df['criteria_score'].value_counts().sort_index(ascending=False))

    print("\nGRADE DISTRIBUTION:")
    print(scored_df['criteria_grade'].value_counts())

    print("\nRECOMMENDATION DISTRIBUTION:")
    print(scored_df['recommendation'].value_counts())

    # Performance by score
    print("\nPERFORMANCE BY SCORE:")
    print("-" * 50)
    for score in [6, 5, 4, 3, 2, 1, 0]:
        subset = scored_df[scored_df['criteria_score'] == score]
        if len(subset) > 0:
            win_rate = (subset['pnl'] > 0).mean() * 100
            avg_pnl = subset['pnl'].mean()
            print(f"Score {score}/6: {len(subset):2d} trades | Win: {win_rate:5.1f}% | Avg P&L: {avg_pnl:+6.1f}%")

    # True A vs Not True A
    true_a = scored_df[scored_df['is_true_a'] == True]
    not_true_a = scored_df[scored_df['is_true_a'] == False]

    print("\n" + "="*70)
    print("TRUE A vs NOT TRUE A COMPARISON")
    print("="*70)
    print(f"\nTRUE A (score >= 5): {len(true_a)} trades")
    print(f"  Win Rate: {(true_a['pnl'] > 0).mean()*100:.1f}%")
    print(f"  Avg P&L: {true_a['pnl'].mean():+.1f}%")

    print(f"\nNOT TRUE A (score < 5): {len(not_true_a)} trades")
    print(f"  Win Rate: {(not_true_a['pnl'] > 0).mean()*100:.1f}%")
    print(f"  Avg P&L: {not_true_a['pnl'].mean():+.1f}%")

    # Show sample reports
    print("\n" + "="*70)
    print("SAMPLE SCORE REPORTS")
    print("="*70)

    # Show one 6/6, one 5/6, one 4/6
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
    scored_df.to_csv('C:\\Users\\zmbur\\PycharmProjects\\backtester\\data\\reversal_scored.csv', index=False)
    print("\nScored data saved to data/reversal_scored.csv")
