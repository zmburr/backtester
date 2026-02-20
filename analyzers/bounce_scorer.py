"""
Bounce Setup Scorer + Pre-Trade Checklist (Setup-Based)

Scores capitulation bounce setups based on SETUP TYPE, not market cap.
Setup profiles are derived from 123 historical bounce trades in bounce_data.csv.

Two setup profiles auto-detected from stock positioning:
  - GapFade_weakstock:  Stock already in downtrend, deep multi-day selloff to capitulation
                        (28 Grade A trades: 86% WR, +9.1% avg P&L)
  - GapFade_strongstock: Healthy stock hit by sudden selloff, gap down bounce
                        (40 Grade A trades: 83% WR, +7.4% avg P&L)

Classification uses pct_from_200mav (primary) or pct_change_30 (fallback).

V2 criteria update (Feb 2026):
  Removed vol_expansion (prior_day_rvol, rho=0.04 — zero predictive power).
  Added pct_change_3 (rho=-0.700, #2 predictor) and pct_off_52wk_high (rho=-0.487).
  Pre-trade: 7 criteria. Historical: 8 criteria (adds bounce_pct).

Two modes:
1. Historical: Score all rows in bounce_data.csv (8/8 criteria) -> validates setups
2. Live watchlist: Fetch data from Polygon, auto-classify, score 7 pre-trade criteria

Usage:
    from analyzers.bounce_scorer import BounceScorer, BouncePretrade, classify_stock
    scorer = BounceScorer()
    result = scorer.score_setup(ticker, date, setup_type, metrics)

    checker = BouncePretrade()
    result = checker.validate(ticker, metrics)
    checker.print_checklist(result)
"""

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SetupProfile:
    """
    Complete profile for a bounce setup type.
    Thresholds derived from Grade A trade percentiles in bounce_data.csv.

    Core criteria thresholds are Dict[str, float] keyed by market cap
    (ETF, Large, Medium, Small, Micro) with a _default fallback.
    bounce_pct remains a scalar (no cap variation).
    """
    name: str
    description: str
    sample_size: int                # How many Grade A trades this was derived from
    historical_win_rate: float      # Grade A win rate
    historical_avg_pnl: float       # Grade A avg P&L %

    # Core criteria thresholds — cap-keyed dicts
    selloff_total_pct: Dict[str, float] = field(default_factory=dict)
    consecutive_down_days: Dict[str, int] = field(default_factory=dict)
    pct_off_30d_high: Dict[str, float] = field(default_factory=dict)
    gap_pct: Dict[str, float] = field(default_factory=dict)
    prior_day_range_atr: Dict[str, float] = field(default_factory=dict)  # Range expansion
    pct_change_3: Dict[str, float] = field(default_factory=dict)         # V2: 3-day momentum (rho=-0.700)
    pct_off_52wk_high: Dict[str, float] = field(default_factory=dict)    # V2: 52wk high discount (rho=-0.487)

    # Scalars (no cap variation)
    bounce_pct: float = 0.02

    # Bonus signals specific to this setup
    bonus_checks: Dict = field(default_factory=dict)

    # Reference ranges (median of Grade A trades, for context)
    reference_medians: Dict = field(default_factory=dict)

    def get_threshold(self, criterion: str, cap: str):
        """Look up cap-specific threshold, falling back to _default then Medium."""
        d = getattr(self, criterion)
        if not isinstance(d, dict):
            return d  # scalar (vol_premarket, bounce_pct)
        return d.get(cap, d.get('_default', d.get('Medium')))


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
    reference: str = ''  # "Grade A median: X" for context


@dataclass
class ChecklistResult:
    """Complete checklist result."""
    ticker: str
    setup_type: str
    timestamp: str
    items: List[ChecklistItem]
    score: int
    max_score: int
    recommendation: str  # 'GO', 'CAUTION', 'NO-GO'
    summary: str
    bonuses: List[str]
    warnings: List[str]
    classification_details: Dict = field(default_factory=dict)
    cap: str = 'Medium'


# ---------------------------------------------------------------------------
# Setup profiles — derived from Grade A percentiles in bounce_data.csv
# Thresholds set at p75 for <= criteria (lenient) and p25 for >= criteria
# so that ~75% of historical Grade A trades pass each criterion
# ---------------------------------------------------------------------------

SETUP_PROFILES = {
    'GapFade_weakstock': SetupProfile(
        name='GapFade_weakstock',
        description='Weak stock capitulation bounce — stock already in downtrend, extended multi-day selloff',
        sample_size=28,
        historical_win_rate=0.86,
        historical_avg_pnl=9.1,

        # Core criteria — cap-keyed thresholds
        # Thresholds set at p75 of winning trades (lenient: ~75% of winners pass)
        selloff_total_pct={
            'ETF': -0.05, 'Large': -0.10, 'Medium': -0.17,
            'Small': -0.30, 'Micro': -0.30, '_default': -0.17,
        },
        consecutive_down_days={
            'ETF': 2, 'Large': 2, 'Medium': 4,
            'Small': 4, 'Micro': 4, '_default': 4,
        },
        pct_off_30d_high={
            'ETF': -0.24, 'Large': -0.30, 'Medium': -0.50,
            'Small': -0.50, 'Micro': -0.50, '_default': -0.50,
        },
        gap_pct={
            'ETF': -0.03, 'Large': -0.03, 'Medium': -0.02,
            'Small': -0.09, 'Micro': -0.09, '_default': -0.02,
        },
        prior_day_range_atr={
            'ETF': 1.0, 'Large': 1.0, 'Medium': 1.0,
            'Small': 1.0, 'Micro': 1.0, '_default': 1.0,
        },
        # V2: 3-day momentum crash (rho=-0.700, #2 predictor)
        # Thresholds from p75 of winning weak-stock trades
        pct_change_3={
            'ETF': -0.10, 'Large': -0.08, 'Medium': -0.20,
            'Small': -0.20, 'Micro': -0.20, '_default': -0.13,
        },
        # V2: discount from 52wk high (rho=-0.487)
        # Thresholds from p75 of winning weak-stock trades
        pct_off_52wk_high={
            'ETF': -0.30, 'Large': -0.30, 'Medium': -0.50,
            'Small': -0.50, 'Micro': -0.50, '_default': -0.50,
        },
        bounce_pct=0.02,

        bonus_checks={
            'closed_outside_lower_band': (True, '==', 'Closed outside lower Bollinger Band'),
            'prior_day_close_vs_low_pct': (0.10, '<=', 'Prior day closed near lows — capitulation signal'),
        },

        reference_medians={
            'selloff_total_pct': -0.334,
            'consecutive_down_days': 5.0,
            'pct_off_30d_high': -0.571,
            'gap_pct': -0.057,
            'prior_day_range_atr': 1.458,
            'pct_change_3': -0.122,
            'pct_off_52wk_high': -0.760,
            'bounce_pct': 0.090,
        },
    ),

    'GapFade_strongstock': SetupProfile(
        name='GapFade_strongstock',
        description='Strong stock pullback bounce — healthy stock hit by sudden selloff (macro, sector, earnings)',
        sample_size=40,
        historical_win_rate=0.83,
        historical_avg_pnl=7.4,

        # Core criteria — cap-keyed thresholds
        selloff_total_pct={
            'ETF': -0.05, 'Large': -0.04, 'Medium': -0.15,
            'Small': -0.10, 'Micro': -0.10, '_default': -0.15,
        },
        consecutive_down_days={
            'ETF': 2, 'Large': 2, 'Medium': 3,
            'Small': 2, 'Micro': 2, '_default': 3,
        },
        pct_off_30d_high={
            'ETF': -0.14, 'Large': -0.14, 'Medium': -0.20,
            'Small': -0.20, 'Micro': -0.20, '_default': -0.20,
        },
        gap_pct={
            'ETF': -0.02, 'Large': -0.01, 'Medium': -0.03,
            'Small': -0.10, 'Micro': -0.10, '_default': -0.03,
        },
        prior_day_range_atr={
            'ETF': 1.0, 'Large': 1.0, 'Medium': 1.0,
            'Small': 1.0, 'Micro': 1.0, '_default': 1.0,
        },
        # V2: 3-day momentum crash (rho=-0.700, #2 predictor)
        # Thresholds from p75 of winning strong-stock trades
        pct_change_3={
            'ETF': -0.08, 'Large': -0.05, 'Medium': -0.05,
            'Small': -0.15, 'Micro': -0.15, '_default': -0.08,
        },
        # V2: discount from 52wk high (rho=-0.487)
        # Thresholds from p75 of winning strong-stock trades
        pct_off_52wk_high={
            'ETF': -0.20, 'Large': -0.15, 'Medium': -0.20,
            'Small': -0.30, 'Micro': -0.30, '_default': -0.20,
        },
        bounce_pct=0.02,

        bonus_checks={
            'bollinger_width': (0.90, '>=', 'Wide Bollinger Bands — volatility expansion'),
            'prior_day_close_vs_low_pct': (0.15, '<=', 'Prior day closed near lows — capitulation signal'),
            'day_of_range_pct': (1.5, '>=', 'Large range day (>= 1.5x ATR)'),
        },

        reference_medians={
            'selloff_total_pct': -0.183,
            'consecutive_down_days': 3.0,
            'pct_off_30d_high': -0.325,
            'gap_pct': -0.061,
            'prior_day_range_atr': 1.203,
            'pct_change_3': -0.090,
            'pct_off_52wk_high': -0.416,
            'bounce_pct': 0.057,
        },
    ),
}


# Criteria display names
CRITERIA_NAMES = {
    'selloff_total_pct': 'Deep selloff',
    'consecutive_down_days': 'Consecutive down days',
    'pct_off_30d_high': 'Discount from 30d high',
    'gap_pct': 'Capitulation gap down',
    'prior_day_range_atr': 'Prior day range expansion',
    'pct_change_3': '3-day momentum crash',
    'pct_off_52wk_high': 'Discount from 52wk high',
    'bounce_pct': 'Large bounce (open-to-close)',
}


# ---------------------------------------------------------------------------
# Classification — auto-detect setup type from stock positioning
# ---------------------------------------------------------------------------

def classify_stock(metrics: Dict) -> Tuple[str, Dict]:
    """
    Classify a stock as weakstock or strongstock based on 200-day moving average.

    Below 200 MA = weakstock (stock in downtrend, capitulation bounce)
    Above 200 MA = strongstock (healthy stock, pullback bounce)

    Falls back to pct_change_30 if 200 MA not available.

    Args:
        metrics: Dictionary containing pct_from_200mav (primary) or pct_change_30 (fallback)

    Returns:
        Tuple of (setup_type_name, classification_details_dict)
    """
    pct_200 = metrics.get('pct_from_200mav')

    details = {
        'pct_from_200mav': pct_200,
        'signals': [],
    }

    if pct_200 is not None and not pd.isna(pct_200):
        if pct_200 < 0:
            setup_type = 'GapFade_weakstock'
            details['signals'].append(f'Below 200MA by {abs(pct_200)*100:.0f}% -> WEAK')
        else:
            setup_type = 'GapFade_strongstock'
            details['signals'].append(f'Above 200MA by {pct_200*100:.0f}% -> STRONG')
    else:
        # Fallback: use 30d change if 200 MA unavailable
        pct_30 = metrics.get('pct_change_30')
        if pct_30 is not None and not pd.isna(pct_30) and pct_30 <= -0.25:
            setup_type = 'GapFade_weakstock'
            details['signals'].append(f'200MA N/A, 30d change {pct_30*100:+.0f}% -> WEAK (fallback)')
        else:
            setup_type = 'GapFade_strongstock'
            pct_30_str = f'{pct_30*100:+.0f}%' if pct_30 is not None and not pd.isna(pct_30) else 'N/A'
            details['signals'].append(f'200MA N/A, 30d change {pct_30_str} -> STRONG (fallback)')

    details['classification'] = setup_type

    return setup_type, details


def classify_from_setup_column(setup_name: str) -> str:
    """Classify from the Setup column in bounce_data.csv for historical scoring."""
    if 'weakstock' in setup_name:
        return 'GapFade_weakstock'
    else:
        # strongstock, IntradayCapitch, plain GapFade -> all use strongstock profile
        return 'GapFade_strongstock'


# ---------------------------------------------------------------------------
# BounceScorer — scores historical trades (8/8 criteria)
# ---------------------------------------------------------------------------

class BounceScorer:
    """Scores bounce setups based on 8 setup-specific criteria (V2 + bounce_pct)."""

    def __init__(self):
        self.profiles = SETUP_PROFILES

    def _get_profile(self, setup_type: str) -> SetupProfile:
        if setup_type not in self.profiles:
            logging.warning(f"Unknown setup '{setup_type}', defaulting to GapFade_strongstock")
            return self.profiles['GapFade_strongstock']
        return self.profiles[setup_type]

    def _check_lte(self, value, threshold) -> bool:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return False
        return value <= threshold

    def _check_gte(self, value, threshold) -> bool:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return False
        return value >= threshold

    def _evaluate_criteria(self, metrics: Dict, profile: SetupProfile, cap: str = 'Medium') -> Tuple[int, List[str], List[str]]:
        passed = []
        failed = []

        checks = [
            ('selloff_total_pct', metrics.get('selloff_total_pct'), profile.get_threshold('selloff_total_pct', cap), 'lte'),
            ('consecutive_down_days', metrics.get('consecutive_down_days'), profile.get_threshold('consecutive_down_days', cap), 'gte'),
            ('pct_off_30d_high', metrics.get('pct_off_30d_high'), profile.get_threshold('pct_off_30d_high', cap), 'lte'),
            ('gap_pct', metrics.get('gap_pct'), profile.get_threshold('gap_pct', cap), 'lte'),
            ('prior_day_range_atr', metrics.get('one_day_before_range_pct'), profile.get_threshold('prior_day_range_atr', cap), 'gte'),
            ('pct_change_3', metrics.get('pct_change_3'), profile.get_threshold('pct_change_3', cap), 'lte'),
            ('pct_off_52wk_high', metrics.get('pct_off_52wk_high'), profile.get_threshold('pct_off_52wk_high', cap), 'lte'),
            ('bounce_pct', metrics.get('bounce_open_close_pct'), profile.get_threshold('bounce_pct', cap), 'gte'),
        ]

        for name, actual, threshold, direction in checks:
            if direction == 'lte':
                result = self._check_lte(actual, threshold)
            else:
                result = self._check_gte(actual, threshold)

            if result:
                passed.append(name)
            else:
                failed.append(name)

        return len(passed), passed, failed

    def _score_to_grade(self, score: int) -> str:
        """Grade based on 8 historical criteria (includes bounce_pct)."""
        if score >= 8:
            return 'A+'
        elif score >= 7:
            return 'A'
        elif score == 6:
            return 'B'
        elif score == 5:
            return 'C'
        else:
            return 'F'

    def _get_recommendation(self, score: int) -> str:
        """Recommendation based on 8 historical criteria.

        GO threshold is 7/8 (88%) here vs 6/7 (86%) in BouncePretrade.validate().
        This is intentional: the 8th criterion (bounce_pct) is an outcome metric
        not available pre-trade, so historical scoring has a higher absolute bar.
        """
        if score >= 7:
            return 'GO'
        elif score == 6:
            return 'CAUTION'
        else:
            return 'NO-GO'

    def score_setup(self, ticker: str, date: str, setup_type: str, metrics: Dict, cap: str = 'Medium') -> Dict:
        profile = self._get_profile(setup_type)
        score, passed, failed = self._evaluate_criteria(metrics, profile, cap)
        grade = self._score_to_grade(score)
        recommendation = self._get_recommendation(score)

        criterion_to_metric = {
            'selloff_total_pct': 'selloff_total_pct',
            'consecutive_down_days': 'consecutive_down_days',
            'pct_off_30d_high': 'pct_off_30d_high',
            'gap_pct': 'gap_pct',
            'prior_day_range_atr': 'one_day_before_range_pct',
            'pct_change_3': 'pct_change_3',
            'pct_off_52wk_high': 'pct_off_52wk_high',
            'bounce_pct': 'bounce_open_close_pct',
        }

        criteria_details = {}
        for criterion in ['selloff_total_pct', 'consecutive_down_days',
                          'pct_off_30d_high', 'gap_pct', 'prior_day_range_atr',
                          'pct_change_3', 'pct_off_52wk_high', 'bounce_pct']:
            metric_key = criterion_to_metric[criterion]
            actual = metrics.get(metric_key)
            threshold = profile.get_threshold(criterion, cap)
            ref_median = profile.reference_medians.get(criterion)

            criteria_details[criterion] = {
                'name': CRITERIA_NAMES[criterion],
                'threshold': threshold,
                'actual': actual,
                'passed': criterion in passed,
                'reference_median': ref_median,
            }

        return {
            'ticker': ticker,
            'date': date,
            'setup_type': setup_type,
            'profile_name': profile.name,
            'score': score,
            'max_score': 8,
            'grade': grade,
            'recommendation': recommendation,
            'passed_criteria': passed,
            'failed_criteria': failed,
            'criteria_details': criteria_details,
            'is_true_a': score >= 7,
            'cap': cap,
        }

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in df.iterrows():
            metrics = row.to_dict()
            # Classify from Setup column
            setup_type = classify_from_setup_column(row.get('Setup', ''))
            # Get cap from the row's cap column (default Medium)
            cap = row.get('cap', 'Medium')
            if cap is None or (isinstance(cap, float) and pd.isna(cap)):
                cap = 'Medium'

            result = self.score_setup(
                ticker=row.get('ticker', ''),
                date=row.get('date', ''),
                setup_type=setup_type,
                metrics=metrics,
                cap=cap,
            )
            results.append({
                'setup_profile': result['setup_type'],
                'criteria_score': result['score'],
                'criteria_grade': result['grade'],
                'recommendation': result['recommendation'],
                'is_true_a': result['is_true_a'],
                'failed_criteria': ', '.join(result['failed_criteria']) if result['failed_criteria'] else 'PERFECT',
                'cap_used': result['cap'],
            })

        score_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), score_df], axis=1)


# ---------------------------------------------------------------------------
# BouncePretrade — live pre-trade checklist (7/7 criteria, no bounce_pct)
# ---------------------------------------------------------------------------

class BouncePretrade:
    """
    Validates bounce setups against 7 pre-trade criteria (V2).
    Auto-classifies stock as weakstock/strongstock and applies the matching profile.

    V2 criteria: selloff depth, consecutive down days, 30d high discount,
    gap down, range expansion, 3-day momentum crash, 52wk high discount.
    (Removed vol_expansion — rho=0.04, zero predictive power.)
    """

    def __init__(self):
        self.profiles = SETUP_PROFILES

    def _check_condition(self, value, threshold, operator: str) -> bool:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return False
        if operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return value == threshold
        return False

    def _format_value(self, value, criterion: str) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return 'N/A'
        if criterion in ['selloff_total_pct', 'pct_off_30d_high', 'gap_pct',
                         'pct_change_3', 'pct_off_52wk_high']:
            return f"{value*100:.1f}%"
        elif criterion == 'consecutive_down_days':
            return f"{int(value)}"
        elif criterion == 'prior_day_range_atr':
            return f"{value:.2f}x"
        elif isinstance(value, bool):
            return 'Yes' if value else 'No'
        else:
            return f"{value:.2f}"

    def _format_threshold(self, threshold, criterion: str) -> str:
        if criterion in ['selloff_total_pct', 'pct_off_30d_high', 'gap_pct',
                         'pct_change_3', 'pct_off_52wk_high']:
            return f"{abs(threshold)*100:.0f}%"
        elif criterion == 'consecutive_down_days':
            return f"{int(threshold)}"
        elif criterion == 'prior_day_range_atr':
            return f"{threshold:.1f}x"
        else:
            return f"{threshold}"

    def validate(self, ticker: str, metrics: Dict, force_setup: Optional[str] = None, cap: str = 'Medium') -> ChecklistResult:
        """
        Validate a setup: auto-classify the stock, then score against matched profile.

        Args:
            ticker: Stock symbol
            metrics: Dictionary of indicator values (from fetch_bounce_metrics or CSV row)
            force_setup: Override auto-classification with specific setup type
            cap: Market cap category (ETF, Large, Medium, Small, Micro) for threshold lookup

        Returns:
            ChecklistResult with score and recommendation
        """
        # Step 1: Classify
        if force_setup and force_setup in self.profiles:
            setup_type = force_setup
            class_details = {'classification': force_setup, 'signals': ['Manually specified']}
        else:
            setup_type, class_details = classify_stock(metrics)

        profile = self.profiles[setup_type]

        # Step 2: Build pre-trade criteria (7 criteria, no bounce_pct)
        selloff_thresh = profile.get_threshold('selloff_total_pct', cap)
        down_days_thresh = profile.get_threshold('consecutive_down_days', cap)
        off_30d_thresh = profile.get_threshold('pct_off_30d_high', cap)
        gap_thresh = profile.get_threshold('gap_pct', cap)
        range_thresh = profile.get_threshold('prior_day_range_atr', cap)
        pct3_thresh = profile.get_threshold('pct_change_3', cap)
        off_52wk_thresh = profile.get_threshold('pct_off_52wk_high', cap)

        criteria = [
            ('selloff_total_pct', 'selloff_total_pct', selloff_thresh,
             f'Deep selloff >= {abs(selloff_thresh)*100:.0f}%', 'lte'),
            ('consecutive_down_days', 'consecutive_down_days', down_days_thresh,
             f'Consecutive down days >= {int(down_days_thresh)}', 'gte'),
            ('pct_off_30d_high', 'pct_off_30d_high', off_30d_thresh,
             f'Discount from 30d high >= {abs(off_30d_thresh)*100:.0f}%', 'lte'),
            ('gap_pct', 'gap_pct', gap_thresh,
             f'Gap down >= {abs(gap_thresh)*100:.0f}%', 'lte'),
            ('prior_day_range_atr', 'prior_day_range_atr', range_thresh,
             f'Prior day range >= {range_thresh:.1f}x ATR', 'gte'),
            ('pct_change_3', 'pct_change_3', pct3_thresh,
             f'3-day momentum crash >= {abs(pct3_thresh)*100:.0f}%', 'lte'),
            ('pct_off_52wk_high', 'pct_off_52wk_high', off_52wk_thresh,
             f'Discount from 52wk high >= {abs(off_52wk_thresh)*100:.0f}%', 'lte'),
        ]

        items = []
        score = 0

        for criterion_key, metric_key, threshold, description, direction in criteria:
            actual = metrics.get(metric_key)
            if direction == 'lte':
                passed = self._check_condition(actual, threshold, '<=')
            else:
                passed = self._check_condition(actual, threshold, '>=')

            if passed:
                score += 1

            # Reference median for context
            ref_median = profile.reference_medians.get(criterion_key)
            if ref_median is not None:
                if criterion_key in ['selloff_total_pct', 'pct_off_30d_high', 'gap_pct',
                                     'pct_change_3', 'pct_off_52wk_high']:
                    ref_str = f"A median: {ref_median*100:.0f}%"
                elif criterion_key == 'consecutive_down_days':
                    ref_str = f"A median: {ref_median:.0f}"
                else:
                    ref_str = f"A median: {ref_median:.1f}x"
            else:
                ref_str = ''

            items.append(ChecklistItem(
                name=criterion_key,
                description=description,
                threshold=threshold,
                actual=actual if actual is not None else 0.0,
                passed=passed,
                threshold_display=self._format_threshold(threshold, criterion_key),
                actual_display=self._format_value(actual, criterion_key),
                reference=ref_str,
            ))

        # Step 3: Bonus factors (setup-specific)
        bonuses = []
        for indicator, (threshold, operator, description) in profile.bonus_checks.items():
            value = metrics.get(indicator)
            if self._check_condition(value, threshold, operator):
                bonuses.append(description)

        # Step 4: Warnings
        warnings = []
        spy_val = metrics.get('spy_open_close_pct')
        if self._check_condition(spy_val, 0.02, '>='):
            warnings.append('SPY strong (+2%), bounce may fade')
        down_days = metrics.get('consecutive_down_days')
        if down_days is not None and not pd.isna(down_days) and down_days <= 1:
            warnings.append('No sustained selloff — may not be capitulation')

        # Check for IntradayCapitch pattern (big single-day drop, no multi-day setup)
        if (down_days is not None and not pd.isna(down_days) and down_days <= 1
                and metrics.get('day_of_range_pct') is not None
                and metrics.get('day_of_range_pct', 0) >= 4.0):
            warnings.append('Looks like IntradayCapitch pattern (17% WR, -13.6% avg) — AVOID')

        # Step 5: Recommendation (7 pre-trade criteria)
        max_score = 7
        if score >= 6:
            recommendation = 'GO'
            summary = f"PASS: {score}/{max_score} criteria met — matches {profile.name} profile ({cap} Cap)"
        elif score == 5:
            recommendation = 'CAUTION'
            failed_names = [i.name for i in items if not i.passed]
            summary = f"MARGINAL: {score}/{max_score} — Missing: {', '.join(failed_names)}"
        else:
            recommendation = 'NO-GO'
            failed_names = [i.name for i in items if not i.passed]
            summary = f"FAIL: Only {score}/{max_score} — Missing: {', '.join(failed_names)}"

        return ChecklistResult(
            ticker=ticker,
            setup_type=setup_type,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            items=items,
            score=score,
            max_score=max_score,
            recommendation=recommendation,
            summary=summary,
            bonuses=bonuses,
            warnings=warnings,
            classification_details=class_details,
            cap=cap,
        )

    def print_checklist(self, result: ChecklistResult):
        profile = SETUP_PROFILES[result.setup_type]

        print()
        print("=" * 70)
        print(f"BOUNCE PRE-TRADE CHECKLIST: {result.ticker}")
        print(f"Setup Profile: {profile.name}")
        print(f"  {profile.description}")
        print(f"  Historical: {profile.sample_size} A-grade trades, "
              f"{profile.historical_win_rate*100:.0f}% WR, "
              f"+{profile.historical_avg_pnl:.0f}% avg P&L")
        print(f"Generated: {result.timestamp}")
        print("=" * 70)

        # Classification reasoning
        if result.classification_details.get('signals'):
            print(f"\nCLASSIFICATION:")
            for signal in result.classification_details['signals']:
                print(f"  - {signal}")

        # Recommendation banner
        if result.recommendation == 'GO':
            print(f"\n  >>> {result.recommendation} - TRADE IT <<<")
        elif result.recommendation == 'CAUTION':
            print(f"\n  >>> {result.recommendation} - REDUCED SIZE <<<")
        else:
            print(f"\n  >>> {result.recommendation} - DO NOT TRADE <<<")

        print(f"\nScore: {result.score}/{result.max_score}")
        print(result.summary)

        # Core criteria with reference medians
        print(f"\nCORE CRITERIA ({result.score}/{result.max_score}):")
        print("-" * 60)
        for item in result.items:
            status = "[PASS]" if item.passed else "[FAIL]"
            ref = f"  ({item.reference})" if item.reference else ""
            print(f"  {status} {item.description}")
            print(f"         Required: {item.threshold_display} | Actual: {item.actual_display}{ref}")

        if result.bonuses:
            print(f"\nBONUS FACTORS:")
            print("-" * 60)
            for bonus in result.bonuses:
                print(f"  [+] {bonus}")

        if result.warnings:
            print(f"\nWARNINGS:")
            print("-" * 60)
            for warning in result.warnings:
                print(f"  [!] {warning}")

        print()


# ---------------------------------------------------------------------------
# Live data fetching from Polygon
# ---------------------------------------------------------------------------

def fetch_bounce_metrics(ticker: str, date: str) -> Dict:
    """
    Fetch all metrics needed for bounce pre-trade scoring from Polygon.
    Computes both scoring metrics AND classification metrics.

    Args:
        ticker: Stock ticker symbol
        date: Date in YYYY-MM-DD format (the potential bounce day)

    Returns:
        Dictionary of metrics for BouncePretrade.validate()
    """
    from data_queries.polygon_queries import (
        get_daily, get_levels_data, adjust_date_to_market,
        fetch_and_calculate_volumes
    )
    try:
        from data_queries.trillium_queries import get_actual_current_price_trill
    except Exception:  # pragma: no cover
        get_actual_current_price_trill = None

    metrics: Dict = {}

    # Get 310 calendar days (~200+ trading days) for SMA200 classification
    # Use `date` (today) so we can optionally include today's daily bar if it exists.
    # IMPORTANT: Many features (consecutive down days, prior-day stats, 30d high) should
    # be computed off *completed* daily bars, so we detect and exclude today's bar when present.
    prior_date = adjust_date_to_market(date, 1)
    levels = get_levels_data(ticker, date, 310, 1, 'day')

    if levels is not None and not levels.empty:
        # Detect whether Polygon returned a daily bar for `date` (during market hours)
        # so we can exclude it for "prior-day" / "completed-bar" calculations.
        has_today_bar = False
        try:
            last_bar_date = levels.index[-1].date()
            has_today_bar = (last_bar_date == pd.to_datetime(date).date())
        except Exception:
            has_today_bar = False

        hist_levels = levels.iloc[:-1] if has_today_bar and len(levels) > 1 else levels

        # Use live price as reference for all current-price calculations
        live_price = None
        if get_actual_current_price_trill is not None:
            try:
                live_price = get_actual_current_price_trill(ticker)
            except Exception:
                pass
        if live_price is not None:
            current_close = live_price
        else:
            # Fallback to the most recent daily close available (today if present, else prior day)
            current_close = levels.iloc[-1]['close']

        metrics['current_price'] = current_close

        # --- Classification metrics ---

        # pct_from_200mav (primary classifier: below = weak, above = strong)
        if len(hist_levels) >= 200:
            sma200 = hist_levels['close'].rolling(200).mean().iloc[-1]
            if not pd.isna(sma200) and sma200 != 0:
                metrics['pct_from_200mav'] = (current_close - sma200) / sma200

        # pct_change_30 (fallback classifier if 200 MA unavailable)
        if len(hist_levels) >= 30:
            close_30ago = hist_levels.iloc[-30]['close']
            if close_30ago != 0:
                metrics['pct_change_30'] = (current_close - close_30ago) / close_30ago

        # pct_change_3 (V2 criterion — rho=-0.700, #2 predictor of bounce magnitude)
        if len(hist_levels) >= 3:
            close_3ago = hist_levels.iloc[-3]['close']
            if close_3ago != 0:
                metrics['pct_change_3'] = (current_close - close_3ago) / close_3ago

        # pct_change_15 (V2 intensity metric — rho=-0.570)
        if len(hist_levels) >= 15:
            close_15ago = hist_levels.iloc[-15]['close']
            if close_15ago != 0:
                metrics['pct_change_15'] = (current_close - close_15ago) / close_15ago

        # pct_off_52wk_high (V2 criterion — rho=-0.487)
        high_all = hist_levels['high'].max() if hist_levels is not None and not hist_levels.empty else levels['high'].max()
        metrics['pct_off_52wk_high'] = (current_close - high_all) / high_all if high_all != 0 else 0

        # --- Scoring metrics ---

        # Consecutive down days — exclude today (the bounce day) from the count.
        # Definition: consecutive *down closes* (close < prior day's close),
        # not "red candles" (close < open). This better captures multi-day
        # selloffs where gap-down + green-close days are still down vs prior close.
        consecutive_down = 0
        start_idx = len(hist_levels) - 1
        for i in range(start_idx, 0, -1):
            cur_close = hist_levels.iloc[i]['close']
            prev_close = hist_levels.iloc[i - 1]['close']
            if pd.isna(cur_close) or pd.isna(prev_close):
                break
            if cur_close < prev_close:
                consecutive_down += 1
            else:
                break
        metrics['consecutive_down_days'] = consecutive_down

        # Selloff total pct — measure from first down day's open to current price
        if consecutive_down > 0:
            selloff_start = start_idx - consecutive_down + 1
            first_open = hist_levels.iloc[selloff_start]['open']
            metrics['selloff_start_open'] = first_open
            # Use live/current price for true current selloff depth
            metrics['selloff_total_pct'] = (current_close - first_open) / first_open if first_open != 0 else 0
        else:
            metrics['selloff_start_open'] = None
            metrics['selloff_total_pct'] = 0.0

        # Pct off 30d high
        high_30d = None
        if hist_levels is not None and not hist_levels.empty:
            window_df = hist_levels.iloc[-30:] if len(hist_levels) >= 30 else hist_levels
            high_30d = window_df['high'].max()
        metrics['high_30d'] = float(high_30d) if high_30d is not None and not pd.isna(high_30d) else None
        if high_30d and high_30d != 0:
            metrics['pct_off_30d_high'] = (current_close - high_30d) / high_30d
        else:
            metrics['pct_off_30d_high'] = None

        # Bollinger Bands (20-day SMA +/- 2 sigma)
        if hist_levels is not None and len(hist_levels) >= 20:
            closes = hist_levels['close'].values
            sma20 = np.mean(closes[-20:])
            std20 = np.std(closes[-20:], ddof=1)
            upper_band = sma20 + 2 * std20
            lower_band = sma20 - 2 * std20

            metrics['closed_outside_lower_band'] = bool(current_close < lower_band)
            metrics['bollinger_width'] = (upper_band - lower_band) / sma20 if sma20 != 0 else 0
        else:
            metrics['closed_outside_lower_band'] = False
            metrics['bollinger_width'] = 0

        # Prior day close vs low pct
        prior_row = hist_levels.iloc[-1] if hist_levels is not None and not hist_levels.empty else levels.iloc[-1]
        prior_range = prior_row['high'] - prior_row['low']
        if prior_range > 0:
            metrics['prior_day_close_vs_low_pct'] = (prior_row['close'] - prior_row['low']) / prior_range
        else:
            metrics['prior_day_close_vs_low_pct'] = 0.5

        # Day-of range pct (for IntradayCapitch warning) and prior day range expansion
        if hist_levels is not None and len(hist_levels) >= 15:
            tr_vals = []
            for i in range(1, len(hist_levels)):
                hl = hist_levels.iloc[i]['high'] - hist_levels.iloc[i]['low']
                hpc = abs(hist_levels.iloc[i]['high'] - hist_levels.iloc[i-1]['close'])
                lpc = abs(hist_levels.iloc[i]['low'] - hist_levels.iloc[i-1]['close'])
                tr_vals.append(max(hl, hpc, lpc))
            if len(tr_vals) >= 14:
                atr = np.mean(tr_vals[-14:])
                today_range = prior_row['high'] - prior_row['low']
                metrics['day_of_range_pct'] = today_range / atr if atr > 0 else 0
                # Prior day range as multiple of ATR (for range expansion criterion)
                metrics['prior_day_range_atr'] = prior_range / atr if atr > 0 else 0

    # Volume metrics — prior day RVOL + premarket RVOL (either can satisfy criterion)
    vol_data = fetch_and_calculate_volumes(ticker, date)
    if vol_data:
        adv = vol_data.get('avg_daily_vol', 0)
        vol_one_day_before = vol_data.get('vol_one_day_before', 0)
        premarket_vol = vol_data.get('premarket_vol', 0)
        if adv and adv > 0:
            metrics['prior_day_rvol'] = vol_one_day_before / adv if vol_one_day_before else 0
            metrics['premarket_rvol'] = premarket_vol / adv if premarket_vol else 0
        else:
            metrics['prior_day_rvol'] = None
            metrics['premarket_rvol'] = None
    else:
        metrics['prior_day_rvol'] = None
        metrics['premarket_rvol'] = None

    # Gap pct — use today's open; during premarket fall back to Trillium live price
    today_daily = get_daily(ticker, date)
    prior_daily = get_daily(ticker, prior_date)

    today_price = None
    prior_close = None

    if today_daily:
        today_price = getattr(today_daily, 'open', None)
    # Premarket fallback: open is None before market open, fetch live price from Trillium
    if not today_price and get_actual_current_price_trill is not None:
        try:
            today_price = get_actual_current_price_trill(ticker)
        except Exception:
            pass

    if prior_daily:
        prior_close = getattr(prior_daily, 'close', None)
    elif levels is not None and len(levels) >= 1:
        try:
            # Use the most recent *completed* daily close as prior_close.
            last_bar_date = levels.index[-1].date()
            has_today_bar = (last_bar_date == pd.to_datetime(date).date())
        except Exception:
            has_today_bar = False
        prior_close = levels.iloc[-2]['close'] if has_today_bar and len(levels) >= 2 else levels.iloc[-1]['close']

    if today_price and prior_close and prior_close != 0:
        metrics['gap_pct'] = (today_price - prior_close) / prior_close
    else:
        metrics['gap_pct'] = None

    metrics['prior_close'] = prior_close

    return metrics


def scan_watchlist(tickers: List[str], date: Optional[str] = None):
    """
    Batch evaluate bounce pre-trade checklist for a list of tickers.
    Auto-classifies each stock and applies the appropriate setup profile.

    Args:
        tickers: List of ticker symbols
        date: Date to evaluate (YYYY-MM-DD). Defaults to today.
    """
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    checker = BouncePretrade()

    print("\n" + "=" * 70)
    print(f"BOUNCE WATCHLIST SCAN - {date}")
    print("=" * 70)

    for ticker in tickers:
        try:
            logging.info(f"Fetching metrics for {ticker}...")
            metrics = fetch_bounce_metrics(ticker, date)
            result = checker.validate(ticker, metrics)
            checker.print_checklist(result)
        except Exception as e:
            print(f"\n  ERROR scanning {ticker}: {e}")


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_score_report(result: Dict):
    """Print a formatted score report for a historical bounce."""
    print(f"\n{'='*70}")
    print(f"BOUNCE SETUP SCORE: {result['ticker']} ({result['date']})")
    print(f"{'='*70}")
    print(f"Setup Profile: {result['setup_type']}")
    print(f"Score: {result['score']}/{result['max_score']}")
    print(f"Grade: {result['grade']}")
    print(f"Recommendation: {result['recommendation']}")
    print()

    print("CRITERIA BREAKDOWN:")
    print("-" * 60)
    for criterion, details in result['criteria_details'].items():
        status = "[PASS]" if details['passed'] else "[FAIL]"
        actual = details['actual']
        threshold = details['threshold']
        ref = details.get('reference_median')

        if criterion in ['selloff_total_pct', 'pct_off_30d_high', 'gap_pct', 'bounce_pct',
                        'pct_change_3', 'pct_off_52wk_high']:
            actual_str = f"{actual*100:.1f}%" if actual is not None and not pd.isna(actual) else "N/A"
            threshold_str = f"{threshold*100:.1f}%"
            direction = "<=" if criterion != 'bounce_pct' else ">="
            ref_str = f"  (A median: {ref*100:.0f}%)" if ref else ""
        elif criterion == 'consecutive_down_days':
            actual_str = f"{int(actual)}" if actual is not None and not pd.isna(actual) else "N/A"
            threshold_str = f"{int(threshold)}"
            direction = ">="
            ref_str = f"  (A median: {ref:.0f})" if ref else ""
        else:  # prior_day_range_atr
            actual_str = f"{actual:.2f}x" if actual is not None and not pd.isna(actual) else "N/A"
            threshold_str = f"{threshold:.1f}x"
            direction = ">="
            ref_str = f"  (A median: {ref:.1f}x)" if ref else ""

        print(f"  {status} {details['name']}")
        print(f"         Required: {direction} {threshold_str} | Actual: {actual_str}{ref_str}")
    print()


# ---------------------------------------------------------------------------
# __main__ - historical scoring + live watchlist scan
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    # ------------------------------------------------------------------
    # 1. Score historical bounce_data.csv using setup-based profiles
    # ------------------------------------------------------------------
    csv_path = os.path.join(DATA_PATH, 'bounce_data.csv')
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} trades from bounce_data.csv")

    scorer = BounceScorer()
    scored_df = scorer.score_dataframe(df)

    scored_df['pnl'] = scored_df['bounce_open_close_pct'] * 100

    print("\n" + "=" * 70)
    print("BOUNCE SETUP SCORING SUMMARY (Setup-Based Profiles)")
    print("=" * 70)

    # Profile assignment
    print("\nPROFILE ASSIGNMENT:")
    print(scored_df['setup_profile'].value_counts())

    # Score distribution
    print("\nSCORE DISTRIBUTION:")
    print(scored_df['criteria_score'].value_counts().sort_index(ascending=False))

    print("\nGRADE DISTRIBUTION:")
    print(scored_df['criteria_grade'].value_counts())

    print("\nRECOMMENDATION DISTRIBUTION:")
    print(scored_df['recommendation'].value_counts())

    # Performance by score
    print("\nPERFORMANCE BY SCORE:")
    print("-" * 60)
    for score_val in [8, 7, 6, 5, 4, 3, 2, 1, 0]:
        subset = scored_df[scored_df['criteria_score'] == score_val]
        if len(subset) > 0:
            win_rate = (subset['pnl'] > 0).mean() * 100
            avg_pnl = subset['pnl'].mean()
            print(f"Score {score_val}/8: {len(subset):2d} trades | Win: {win_rate:5.1f}% | Avg P&L: {avg_pnl:+6.1f}%")

    # ------------------------------------------------------------------
    # 2. GO vs NO-GO performance comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GO vs NO-GO PERFORMANCE COMPARISON")
    print("=" * 70)

    for rec in ['GO', 'CAUTION', 'NO-GO']:
        subset = scored_df[scored_df['recommendation'] == rec]
        if len(subset) > 0:
            win_rate = (subset['pnl'] > 0).mean() * 100
            avg_pnl = subset['pnl'].mean()
            print(f"\n{rec:8s}: {len(subset):2d} trades")
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Avg P&L: {avg_pnl:+.1f}%")

    # By profile
    print("\nBY SETUP PROFILE:")
    print("-" * 60)
    for profile_name in SETUP_PROFILES:
        subset = scored_df[scored_df['setup_profile'] == profile_name]
        if len(subset) > 0:
            go = subset[subset['recommendation'] == 'GO']
            nogo = subset[subset['recommendation'] == 'NO-GO']
            print(f"\n  {profile_name}:")
            if len(go) > 0:
                print(f"    GO:    {len(go):2d} trades | Win: {(go['pnl']>0).mean()*100:.0f}% | Avg: {go['pnl'].mean():+.1f}%")
            if len(nogo) > 0:
                print(f"    NO-GO: {len(nogo):2d} trades | Win: {(nogo['pnl']>0).mean()*100:.0f}% | Avg: {nogo['pnl'].mean():+.1f}%")

    # By original trade grade
    print("\nBY ORIGINAL TRADE GRADE:")
    print("-" * 60)
    for grade in ['A', 'B', 'C']:
        subset = scored_df[scored_df['trade_grade'] == grade]
        if len(subset) > 0:
            avg_score = subset['criteria_score'].mean()
            go_pct = (subset['recommendation'] == 'GO').mean() * 100
            print(f"Grade {grade}: {len(subset):2d} trades | Avg Score: {avg_score:.1f}/8 | GO rate: {go_pct:.0f}%")

    # Show sample reports
    print("\n" + "=" * 70)
    print("SAMPLE SCORE REPORTS")
    print("=" * 70)

    for score_val in [8, 7, 6, 5]:
        sample = scored_df[scored_df['criteria_score'] == score_val]
        if len(sample) > 0:
            row = sample.iloc[0]
            setup_type = classify_from_setup_column(row.get('Setup', ''))
            row_cap = row.get('cap', 'Medium')
            if row_cap is None or (isinstance(row_cap, float) and pd.isna(row_cap)):
                row_cap = 'Medium'
            result = scorer.score_setup(
                ticker=row['ticker'],
                date=row['date'],
                setup_type=setup_type,
                metrics=row.to_dict(),
                cap=row_cap,
            )
            print_score_report(result)

    # Save scored data
    out_path = os.path.join(DATA_PATH, 'bounce_scored.csv')
    scored_df.to_csv(out_path, index=False)
    print(f"\nScored data saved to {out_path}")

    # ------------------------------------------------------------------
    # 3. Scan watchlist tickers (live, auto-classified)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("LIVE WATCHLIST SCAN (Auto-Classified)")
    print("=" * 70)

    watchlist = ['IBIT', 'ETHE', 'GWRE', 'FIG', 'NOW']

    try:
        scan_watchlist(watchlist)
    except Exception as e:
        print(f"\nWatchlist scan error (may require market hours / Polygon access): {e}")
