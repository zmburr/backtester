"""
Pre-Trade Checklist for Parabolic Short Reversals

Validates if a potential reversal setup meets cap-specific criteria.
Uses 6 core criteria with cap-adjusted thresholds.
Outputs GO (5-6/6), CAUTION (4/6), or NO-GO (<4/6) recommendation.

The 6 criteria for a TRUE parabolic reversal:
1. Extended above 9EMA - Price significantly elevated from 9-day EMA
2. Range expansion - Prior day range >= threshold x ATR
3. Volume expansion - RVOL >= threshold
4. Consecutive up days - Sustained momentum into the top
5. Euphoric gap up - Gap up on reversal day
6. Large reversal - Actual reversal size on the day

Usage:
    from analyzers.pretrade_checklist import PreTradeChecklist
    checker = PreTradeChecklist()
    result = checker.validate(ticker, cap, metrics)
    checker.print_checklist(result)
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)


@dataclass
class ChecklistItem:
    """Single checklist item result."""
    name: str
    description: str
    threshold: float
    actual: float
    passed: bool
    threshold_display: str
    actual_display: str


@dataclass
class ChecklistResult:
    """Complete checklist result."""
    ticker: str
    cap: str
    timestamp: str
    items: List[ChecklistItem]
    score: int
    max_score: int
    recommendation: str  # 'GO', 'CAUTION', 'NO-GO'
    is_true_a: bool
    summary: str
    # Bonus factors (not part of core 6)
    bonuses: List[str]
    warnings: List[str]


# Cap-specific thresholds for the 6 core criteria
# Validated against 61 Grade A trades - TRUE A = score >= 5/6
CAP_THRESHOLDS = {
    'Micro': {
        'pct_from_9ema': (0.80, 'Extended above 9EMA >= 80%'),
        'prior_day_range_atr': (3.0, 'Prior day range >= 3x ATR'),
        'rvol_score': (2.0, 'RVOL >= 2x'),
        'consecutive_up_days': (3, 'Consecutive up days >= 3'),
        'gap_pct': (0.15, 'Gap up >= 15%'),
        'reversal_pct': (-0.20, 'Reversal >= 20%'),
    },
    'Small': {
        'pct_from_9ema': (0.40, 'Extended above 9EMA >= 40%'),
        'prior_day_range_atr': (2.0, 'Prior day range >= 2x ATR'),
        'rvol_score': (2.0, 'RVOL >= 2x'),
        'consecutive_up_days': (2, 'Consecutive up days >= 2'),
        'gap_pct': (0.10, 'Gap up >= 10%'),
        'reversal_pct': (-0.10, 'Reversal >= 10%'),
    },
    'Medium': {
        'pct_from_9ema': (0.15, 'Extended above 9EMA >= 15%'),
        'prior_day_range_atr': (1.0, 'Prior day range >= 1x ATR'),
        'rvol_score': (1.5, 'RVOL >= 1.5x'),
        'consecutive_up_days': (2, 'Consecutive up days >= 2'),
        'gap_pct': (0.05, 'Gap up >= 5%'),
        'reversal_pct': (-0.05, 'Reversal >= 5%'),
    },
    'Large': {
        'pct_from_9ema': (0.08, 'Extended above 9EMA >= 8%'),
        'prior_day_range_atr': (0.8, 'Prior day range >= 0.8x ATR'),
        'rvol_score': (1.0, 'RVOL >= 1x'),
        'consecutive_up_days': (1, 'Consecutive up days >= 1'),
        'gap_pct': (0.00, 'Gap up (any)'),
        'reversal_pct': (-0.03, 'Reversal >= 3%'),
    },
    'ETF': {
        'pct_from_9ema': (0.04, 'Extended above 9EMA >= 4%'),
        'prior_day_range_atr': (1.0, 'Prior day range >= 1x ATR'),
        'rvol_score': (1.5, 'RVOL >= 1.5x'),
        'consecutive_up_days': (1, 'Consecutive up days >= 1'),
        'gap_pct': (0.00, 'Gap up (any)'),
        'reversal_pct': (-0.015, 'Reversal >= 1.5%'),
    },
}

# Bonus factors that improve setup quality (not part of core 6)
BONUS_CHECKS = {
    'spy_5day_return': (-0.01, '<=', 'SPY 5-day <= -1% (weak market = better)'),
    'closed_outside_upper_band': (True, '==', 'Closed outside upper Bollinger Band'),
    'close_green_red': (True, '==', 'Prior day closed green-to-red'),
}

# Warning conditions
WARNING_CHECKS = {
    'spy_5day_return': (0.02, '>=', 'SPY 5-day >= +2% (strong market = worse odds)'),
}


class PreTradeChecklist:
    """Validates reversal setups against 6 cap-adjusted criteria."""

    def __init__(self):
        self.thresholds = CAP_THRESHOLDS

    def _check_criterion(self, value, threshold, is_reversal: bool = False) -> bool:
        """Check if a single criterion passes."""
        if value is None or pd.isna(value):
            return False

        if is_reversal:
            return value <= threshold
        else:
            return value >= threshold

    def _check_condition(self, value, threshold, operator: str) -> bool:
        """Check a condition with operator."""
        if value is None or pd.isna(value):
            return False

        if operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return value == threshold
        return False

    def _format_value(self, value, criterion: str) -> str:
        """Format value for display."""
        if value is None or pd.isna(value):
            return 'N/A'

        if criterion in ['pct_from_9ema', 'gap_pct', 'reversal_pct']:
            return f"{value*100:.1f}%"
        elif criterion == 'consecutive_up_days':
            return f"{int(value)}"
        elif criterion in ['prior_day_range_atr', 'rvol_score']:
            return f"{value:.1f}x"
        elif isinstance(value, bool):
            return 'Yes' if value else 'No'
        elif criterion == 'spy_5day_return':
            return f"{value*100:+.1f}%"
        else:
            return f"{value:.2f}"

    def _format_threshold(self, threshold, criterion: str) -> str:
        """Format threshold for display."""
        if criterion in ['pct_from_9ema', 'gap_pct', 'reversal_pct']:
            return f"{abs(threshold)*100:.0f}%"
        elif criterion == 'consecutive_up_days':
            return f"{int(threshold)}"
        elif criterion in ['prior_day_range_atr', 'rvol_score']:
            return f"{threshold:.1f}x"
        elif isinstance(threshold, bool):
            return 'Yes'
        elif criterion == 'spy_5day_return':
            return f"{threshold*100:+.0f}%"
        else:
            return f"{threshold}"

    def validate(self, ticker: str, cap: str, metrics: Dict) -> ChecklistResult:
        """
        Validate a setup against 6 cap-specific criteria.

        Args:
            ticker: Stock symbol
            cap: Market cap category
            metrics: Dictionary of indicator values

        Returns:
            ChecklistResult with score and recommendation
        """
        if cap not in self.thresholds:
            cap = 'Medium'

        config = self.thresholds[cap]
        items = []
        score = 0

        # Check each of the 6 criteria
        criteria_order = ['pct_from_9ema', 'prior_day_range_atr', 'rvol_score',
                          'consecutive_up_days', 'gap_pct', 'reversal_pct']

        for criterion in criteria_order:
            threshold, description = config[criterion]

            if criterion == 'reversal_pct':
                actual = metrics.get('reversal_open_close_pct')
                passed = self._check_criterion(actual, threshold, is_reversal=True)
            else:
                actual = metrics.get(criterion)
                passed = self._check_criterion(actual, threshold)

            if passed:
                score += 1

            items.append(ChecklistItem(
                name=criterion,
                description=description,
                threshold=threshold,
                actual=actual,
                passed=passed,
                threshold_display=self._format_threshold(threshold, criterion),
                actual_display=self._format_value(actual, criterion),
            ))

        # Check bonus factors
        bonuses = []
        for indicator, (threshold, operator, description) in BONUS_CHECKS.items():
            value = metrics.get(indicator)
            if self._check_condition(value, threshold, operator):
                bonuses.append(description)

        # Check warnings
        warnings = []
        for indicator, (threshold, operator, description) in WARNING_CHECKS.items():
            value = metrics.get(indicator)
            if self._check_condition(value, threshold, operator):
                warnings.append(description)

        # Determine recommendation
        if score >= 5:
            recommendation = 'GO'
            summary = f"PASS: {score}/6 criteria met - TRUE GRADE A setup"
        elif score == 4:
            recommendation = 'CAUTION'
            failed = [i.name for i in items if not i.passed]
            summary = f"MARGINAL: {score}/6 criteria - Missing: {', '.join(failed)}"
        else:
            recommendation = 'NO-GO'
            failed = [i.name for i in items if not i.passed]
            summary = f"FAIL: Only {score}/6 criteria - Missing: {', '.join(failed)}"

        return ChecklistResult(
            ticker=ticker,
            cap=cap,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            items=items,
            score=score,
            max_score=6,
            recommendation=recommendation,
            is_true_a=score >= 5,
            summary=summary,
            bonuses=bonuses,
            warnings=warnings,
        )

    def print_checklist(self, result: ChecklistResult):
        """Print formatted checklist to console."""
        print()
        print("=" * 70)
        print(f"PRE-TRADE CHECKLIST: {result.ticker} ({result.cap} Cap)")
        print(f"Generated: {result.timestamp}")
        print("=" * 70)

        # Recommendation banner
        if result.recommendation == 'GO':
            print(f"\n  >>> {result.recommendation} - TRADE IT <<<")
        elif result.recommendation == 'CAUTION':
            print(f"\n  >>> {result.recommendation} - REDUCED SIZE <<<")
        else:
            print(f"\n  >>> {result.recommendation} - DO NOT TRADE <<<")

        print(f"\nScore: {result.score}/{result.max_score}")
        print(f"{result.summary}")

        # Core criteria
        print(f"\nCORE CRITERIA ({result.score}/{result.max_score}):")
        print("-" * 60)
        for item in result.items:
            status = "[PASS]" if item.passed else "[FAIL]"
            print(f"  {status} {item.description}")
            print(f"         Required: {item.threshold_display} | Actual: {item.actual_display}")

        # Bonuses
        if result.bonuses:
            print(f"\nBONUS FACTORS:")
            print("-" * 60)
            for bonus in result.bonuses:
                print(f"  [+] {bonus}")

        # Warnings
        if result.warnings:
            print(f"\nWARNINGS:")
            print("-" * 60)
            for warning in result.warnings:
                print(f"  [!] {warning}")

        print()

    def validate_from_row(self, row: pd.Series) -> ChecklistResult:
        """Validate a setup from a DataFrame row."""
        return self.validate(
            ticker=row.get('ticker', 'UNKNOWN'),
            cap=row.get('cap', 'Medium'),
            metrics=row.to_dict()
        )


def run_checklist_demo():
    """Run checklist on sample data."""
    df = pd.read_csv('C:\\Users\\zmbur\\PycharmProjects\\backtester\\data\\reversal_data.csv')
    grade_a = df[df['trade_grade'] == 'A'].copy()
    grade_a['pnl'] = -grade_a['reversal_open_close_pct'] * 100

    checker = PreTradeChecklist()

    print("\n" + "=" * 70)
    print("PRE-TRADE CHECKLIST VALIDATION - Grade A Setups")
    print("=" * 70)

    # Validate all and collect results
    results = []
    for _, row in grade_a.iterrows():
        result = checker.validate_from_row(row)
        results.append({
            'ticker': result.ticker,
            'date': row.get('date', ''),
            'cap': result.cap,
            'score': result.score,
            'recommendation': result.recommendation,
            'is_true_a': result.is_true_a,
            'pnl': row['pnl'],
        })

    results_df = pd.DataFrame(results)

    # Summary statistics
    print("\nRECOMMENDATION DISTRIBUTION:")
    print(results_df['recommendation'].value_counts())

    print("\nPERFORMANCE BY RECOMMENDATION:")
    print("-" * 50)
    for rec in ['GO', 'CAUTION', 'NO-GO']:
        subset = results_df[results_df['recommendation'] == rec]
        if len(subset) > 0:
            win_rate = (subset['pnl'] > 0).mean() * 100
            avg_pnl = subset['pnl'].mean()
            print(f"{rec:8s}: {len(subset):2d} trades | Win: {win_rate:5.1f}% | Avg P&L: {avg_pnl:+6.1f}%")

    print("\nBY CAP:")
    print(results_df.groupby(['cap', 'recommendation']).size().unstack(fill_value=0))

    # Show sample checklists
    print("\n" + "=" * 70)
    print("SAMPLE CHECKLISTS")
    print("=" * 70)

    # Show one GO, one CAUTION, one NO-GO
    for rec in ['GO', 'CAUTION', 'NO-GO']:
        sample = results_df[results_df['recommendation'] == rec]
        if len(sample) > 0:
            row_data = grade_a[
                (grade_a['ticker'] == sample.iloc[0]['ticker']) &
                (grade_a['date'] == sample.iloc[0]['date'])
            ].iloc[0]
            result = checker.validate_from_row(row_data)
            checker.print_checklist(result)


if __name__ == '__main__':
    run_checklist_demo()
