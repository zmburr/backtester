"""
Bounce Setup Scorer + Pre-Trade Checklist (Setup-Based)

Scores capitulation bounce setups based on SETUP TYPE, not market cap.
Setup profiles are derived from 36 historical bounce trades in bounce_data.csv.

Two setup profiles auto-detected from stock positioning:
  - GapFade_weakstock:  Stock already in downtrend, deep multi-day selloff to capitulation
                        (10 Grade A trades: 80% WR, +18.8% avg P&L)
  - GapFade_strongstock: Healthy stock hit by sudden selloff, gap down bounce
                        (5 Grade A trades: 100% WR, +9.6% avg P&L)

Classification uses pct_from_50mav and pct_change_30 to determine stock type.

Two modes:
1. Historical: Score all rows in bounce_data.csv (6/6 criteria) -> validates setups
2. Live watchlist: Fetch data from Polygon, auto-classify, score 5 pre-trade criteria

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
    """
    name: str
    description: str
    sample_size: int                # How many Grade A trades this was derived from
    historical_win_rate: float      # Grade A win rate
    historical_avg_pnl: float       # Grade A avg P&L %

    # Classification rules (how to identify this setup type)
    classify_pct_from_50mav: Tuple[str, float]   # ('<=', -0.15) or ('>', -0.10)
    classify_pct_change_30: Tuple[str, float]     # ('<=', -0.30) or ('>', -0.15)

    # Core criteria thresholds (from Grade A percentiles)
    selloff_total_pct: float        # <= threshold (more negative = deeper)
    consecutive_down_days: int      # >= threshold
    vol_expansion: float            # >= threshold (breakout vol / ADV)
    pct_off_30d_high: float         # <= threshold (more negative = steeper)
    gap_pct: float                  # <= threshold (more negative = bigger gap down)
    bounce_pct: float               # >= threshold (post-trade only)

    # Bonus signals specific to this setup
    bonus_checks: Dict = field(default_factory=dict)

    # Reference ranges (median of Grade A trades, for context)
    reference_medians: Dict = field(default_factory=dict)


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


# ---------------------------------------------------------------------------
# Setup profiles — derived from Grade A percentiles in bounce_data.csv
# Thresholds set at p75 for <= criteria (lenient) and p25 for >= criteria
# so that ~75% of historical Grade A trades pass each criterion
# ---------------------------------------------------------------------------

SETUP_PROFILES = {
    'GapFade_weakstock': SetupProfile(
        name='GapFade_weakstock',
        description='Weak stock capitulation bounce — stock already in downtrend, extended multi-day selloff',
        sample_size=10,
        historical_win_rate=0.80,
        historical_avg_pnl=18.8,

        # Classification: below 50MA, negative 30d momentum
        classify_pct_from_50mav=('<=', -0.15),
        classify_pct_change_30=('<=', -0.25),

        # Core criteria (from 10 Grade A trades)
        selloff_total_pct=-0.18,       # p75: -0.182 (75% of A trades sold off >= 18%)
        consecutive_down_days=4,       # p25: 4.0 (75% had >= 4 down days)
        vol_expansion=2.3,             # ~p25: 3.43, using 2.3 for practical threshold
        pct_off_30d_high=-0.51,        # p75: -0.513 (75% were >= 51% off 30d high)
        gap_pct=-0.04,                 # p75: -0.043 (75% gapped down >= 4%)
        bounce_pct=0.02,              # p25: 0.024 (post-trade only)

        bonus_checks={
            'closed_outside_lower_band': (True, '==', 'Closed outside lower Bollinger Band (80% of A trades)'),
            'prior_day_close_vs_low_pct': (0.10, '<=', 'Prior day closed near lows — capitulation signal'),
            'pct_off_52wk_high': (-0.75, '<=', 'Deep off 52-week high (extreme distress)'),
        },

        reference_medians={
            'selloff_total_pct': -0.258,
            'consecutive_down_days': 4.5,
            'vol_expansion': 4.892,
            'pct_off_30d_high': -0.673,
            'gap_pct': -0.097,
            'bounce_pct': 0.142,
        },
    ),

    'GapFade_strongstock': SetupProfile(
        name='GapFade_strongstock',
        description='Strong stock pullback bounce — healthy stock hit by sudden selloff (macro, sector, earnings)',
        sample_size=5,
        historical_win_rate=1.00,
        historical_avg_pnl=9.6,

        # Classification: near/above 50MA, flat or positive 30d momentum
        classify_pct_from_50mav=('>', -0.15),
        classify_pct_change_30=('>', -0.25),

        # Core criteria (from 5 Grade A trades — smaller sample, wider thresholds)
        selloff_total_pct=-0.05,       # p75: -0.122, using -0.05 (even shallow selloffs qualify)
        consecutive_down_days=2,       # p25: 2.0
        vol_expansion=1.0,             # p25: 1.153, using 1.0
        pct_off_30d_high=-0.20,        # p75: -0.198
        gap_pct=-0.03,                 # p75: -0.044, using -0.03 (less gap required)
        bounce_pct=0.02,              # p25: 0.024 (post-trade only)

        bonus_checks={
            'bollinger_width': (0.90, '>=', 'Wide Bollinger Bands — volatility expansion (A median: 0.93)'),
            'prior_day_close_vs_low_pct': (0.15, '<=', 'Prior day closed near lows — capitulation signal'),
            'day_of_range_pct': (1.5, '>=', 'Large range day (>= 1.5x ATR)'),
        },

        reference_medians={
            'selloff_total_pct': -0.156,
            'consecutive_down_days': 3.0,
            'vol_expansion': 2.388,
            'pct_off_30d_high': -0.369,
            'gap_pct': -0.102,
            'bounce_pct': 0.059,
        },
    ),
}


# Criteria display names
CRITERIA_NAMES = {
    'selloff_total_pct': 'Deep selloff',
    'consecutive_down_days': 'Consecutive down days',
    'vol_expansion': 'Volume climax (vs ADV)',
    'pct_off_30d_high': 'Discount from 30d high',
    'gap_pct': 'Capitulation gap down',
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
# BounceScorer — scores historical trades (6/6 criteria)
# ---------------------------------------------------------------------------

class BounceScorer:
    """Scores bounce setups based on 6 setup-specific criteria."""

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

    def _evaluate_criteria(self, metrics: Dict, profile: SetupProfile) -> Tuple[int, List[str], List[str]]:
        passed = []
        failed = []

        checks = [
            ('selloff_total_pct', metrics.get('selloff_total_pct'), profile.selloff_total_pct, 'lte'),
            ('consecutive_down_days', metrics.get('consecutive_down_days'), profile.consecutive_down_days, 'gte'),
            ('vol_expansion', metrics.get('percent_of_vol_on_breakout_day'), profile.vol_expansion, 'gte'),
            ('pct_off_30d_high', metrics.get('pct_off_30d_high'), profile.pct_off_30d_high, 'lte'),
            ('gap_pct', metrics.get('gap_pct'), profile.gap_pct, 'lte'),
            ('bounce_pct', metrics.get('bounce_open_close_pct'), profile.bounce_pct, 'gte'),
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
        if score >= 5:
            return 'GO'
        elif score == 4:
            return 'CAUTION'
        else:
            return 'NO-GO'

    def score_setup(self, ticker: str, date: str, setup_type: str, metrics: Dict) -> Dict:
        profile = self._get_profile(setup_type)
        score, passed, failed = self._evaluate_criteria(metrics, profile)
        grade = self._score_to_grade(score)
        recommendation = self._get_recommendation(score)

        criterion_to_metric = {
            'selloff_total_pct': 'selloff_total_pct',
            'consecutive_down_days': 'consecutive_down_days',
            'vol_expansion': 'percent_of_vol_on_breakout_day',
            'pct_off_30d_high': 'pct_off_30d_high',
            'gap_pct': 'gap_pct',
            'bounce_pct': 'bounce_open_close_pct',
        }

        criteria_details = {}
        for criterion in ['selloff_total_pct', 'consecutive_down_days', 'vol_expansion',
                          'pct_off_30d_high', 'gap_pct', 'bounce_pct']:
            metric_key = criterion_to_metric[criterion]
            actual = metrics.get(metric_key)
            threshold = getattr(profile, criterion)
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
            'max_score': 6,
            'grade': grade,
            'recommendation': recommendation,
            'passed_criteria': passed,
            'failed_criteria': failed,
            'criteria_details': criteria_details,
            'is_true_a': score >= 5,
        }

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in df.iterrows():
            metrics = row.to_dict()
            # Classify from Setup column
            setup_type = classify_from_setup_column(row.get('Setup', ''))

            result = self.score_setup(
                ticker=row.get('ticker', ''),
                date=row.get('date', ''),
                setup_type=setup_type,
                metrics=metrics
            )
            results.append({
                'setup_profile': result['setup_type'],
                'criteria_score': result['score'],
                'criteria_grade': result['grade'],
                'recommendation': result['recommendation'],
                'is_true_a': result['is_true_a'],
                'failed_criteria': ', '.join(result['failed_criteria']) if result['failed_criteria'] else 'PERFECT'
            })

        score_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), score_df], axis=1)


# ---------------------------------------------------------------------------
# BouncePretrade — live pre-trade checklist (5/5 criteria, no bounce_pct)
# ---------------------------------------------------------------------------

class BouncePretrade:
    """
    Validates bounce setups against 5 pre-trade criteria.
    Auto-classifies stock as weakstock/strongstock and applies the matching profile.
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
        if criterion in ['selloff_total_pct', 'pct_off_30d_high', 'gap_pct']:
            return f"{value*100:.1f}%"
        elif criterion == 'consecutive_down_days':
            return f"{int(value)}"
        elif criterion == 'vol_expansion':
            return f"{value:.2f}x"
        elif isinstance(value, bool):
            return 'Yes' if value else 'No'
        else:
            return f"{value:.2f}"

    def _format_threshold(self, threshold, criterion: str) -> str:
        if criterion in ['selloff_total_pct', 'pct_off_30d_high', 'gap_pct']:
            return f"{abs(threshold)*100:.0f}%"
        elif criterion == 'consecutive_down_days':
            return f"{int(threshold)}"
        elif criterion == 'vol_expansion':
            return f"{threshold:.1f}x"
        else:
            return f"{threshold}"

    def validate(self, ticker: str, metrics: Dict, force_setup: Optional[str] = None) -> ChecklistResult:
        """
        Validate a setup: auto-classify the stock, then score against matched profile.

        Args:
            ticker: Stock symbol
            metrics: Dictionary of indicator values (from fetch_bounce_metrics or CSV row)
            force_setup: Override auto-classification with specific setup type

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

        # Step 2: Build pre-trade criteria (5 criteria, no bounce_pct)
        criteria = [
            ('selloff_total_pct', 'selloff_total_pct', profile.selloff_total_pct,
             f'Deep selloff >= {abs(profile.selloff_total_pct)*100:.0f}%', 'lte'),
            ('consecutive_down_days', 'consecutive_down_days', profile.consecutive_down_days,
             f'Consecutive down days >= {int(profile.consecutive_down_days)}', 'gte'),
            ('vol_expansion', 'percent_of_vol_on_breakout_day', profile.vol_expansion,
             f'Volume climax >= {profile.vol_expansion:.1f}x ADV', 'gte'),
            ('pct_off_30d_high', 'pct_off_30d_high', profile.pct_off_30d_high,
             f'Discount from 30d high >= {abs(profile.pct_off_30d_high)*100:.0f}%', 'lte'),
            ('gap_pct', 'gap_pct', profile.gap_pct,
             f'Gap down >= {abs(profile.gap_pct)*100:.0f}%', 'lte'),
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
                if criterion_key in ['selloff_total_pct', 'pct_off_30d_high', 'gap_pct']:
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
                actual=actual,
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

        # Step 5: Recommendation
        max_score = 5
        if score >= 4:
            recommendation = 'GO'
            summary = f"PASS: {score}/{max_score} criteria met — matches {profile.name} profile"
        elif score == 3:
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

    metrics = {}

    # Get 310 calendar days (~200+ trading days) for SMA200 classification
    # Use `date` (today) so today's bar is included in consecutive down days count
    prior_date = adjust_date_to_market(date, 1)
    levels = get_levels_data(ticker, date, 310, 1, 'day')

    if levels is not None and not levels.empty:
        current_close = levels.iloc[-1]['close']

        # --- Classification metrics ---

        # pct_from_200mav (primary classifier: below = weak, above = strong)
        if len(levels) >= 200:
            sma200 = levels['close'].rolling(200).mean().iloc[-1]
            if not pd.isna(sma200) and sma200 != 0:
                metrics['pct_from_200mav'] = (current_close - sma200) / sma200

        # pct_change_30 (fallback classifier if 200 MA unavailable)
        if len(levels) >= 30:
            close_30ago = levels.iloc[-30]['close']
            if close_30ago != 0:
                metrics['pct_change_30'] = (current_close - close_30ago) / close_30ago

        # pct_off_52wk_high (for context)
        high_all = levels['high'].max()
        metrics['pct_off_52wk_high'] = (current_close - high_all) / high_all if high_all != 0 else 0

        # --- Scoring metrics ---

        # Consecutive down days
        consecutive_down = 0
        for i in range(len(levels) - 1, -1, -1):
            row = levels.iloc[i]
            if row['close'] < row['open']:
                consecutive_down += 1
            else:
                break
        metrics['consecutive_down_days'] = consecutive_down

        # Selloff total pct
        if consecutive_down > 0:
            recent = levels.iloc[-consecutive_down:]
            first_open = recent.iloc[0]['open']
            last_close = recent.iloc[-1]['close']
            metrics['selloff_total_pct'] = (last_close - first_open) / first_open if first_open != 0 else 0
        else:
            metrics['selloff_total_pct'] = 0.0

        # Pct off 30d high
        if len(levels) >= 30:
            high_30d = levels.iloc[-30:]['high'].max()
        else:
            high_30d = levels['high'].max()
        metrics['pct_off_30d_high'] = (current_close - high_30d) / high_30d if high_30d != 0 else 0

        # Bollinger Bands (20-day SMA +/- 2 sigma)
        if len(levels) >= 20:
            closes = levels['close'].values
            sma20 = np.mean(closes[-20:])
            std20 = np.std(closes[-20:])
            upper_band = sma20 + 2 * std20
            lower_band = sma20 - 2 * std20

            metrics['closed_outside_lower_band'] = bool(current_close < lower_band)
            metrics['bollinger_width'] = (upper_band - lower_band) / sma20 if sma20 != 0 else 0
        else:
            metrics['closed_outside_lower_band'] = False
            metrics['bollinger_width'] = 0

        # Prior day close vs low pct
        prior_row = levels.iloc[-1]
        prior_range = prior_row['high'] - prior_row['low']
        if prior_range > 0:
            metrics['prior_day_close_vs_low_pct'] = (prior_row['close'] - prior_row['low']) / prior_range
        else:
            metrics['prior_day_close_vs_low_pct'] = 0.5

        # Day-of range pct (for IntradayCapitch warning)
        if len(levels) >= 15:
            tr_vals = []
            for i in range(1, len(levels)):
                hl = levels.iloc[i]['high'] - levels.iloc[i]['low']
                hpc = abs(levels.iloc[i]['high'] - levels.iloc[i-1]['close'])
                lpc = abs(levels.iloc[i]['low'] - levels.iloc[i-1]['close'])
                tr_vals.append(max(hl, hpc, lpc))
            if len(tr_vals) >= 14:
                atr = np.mean(tr_vals[-14:])
                today_range = prior_row['high'] - prior_row['low']
                metrics['day_of_range_pct'] = today_range / atr if atr > 0 else 0

    # Volume metrics
    vol_data = fetch_and_calculate_volumes(ticker, date)
    if vol_data:
        adv = vol_data.get('avg_daily_vol', 0)
        breakout_vol = vol_data.get('vol_on_breakout_day', 0)
        if adv and adv > 0 and breakout_vol:
            metrics['percent_of_vol_on_breakout_day'] = breakout_vol / adv
        else:
            metrics['percent_of_vol_on_breakout_day'] = None

    # Gap pct
    today = get_daily(ticker, date)
    prior_daily = get_daily(ticker, prior_date)
    if today and prior_daily:
        today_open = today.open if hasattr(today, 'open') else today.get('open')
        prior_close = prior_daily.close if hasattr(prior_daily, 'close') else prior_daily.get('close')
        if today_open and prior_close and prior_close != 0:
            metrics['gap_pct'] = (today_open - prior_close) / prior_close
        else:
            metrics['gap_pct'] = None
    else:
        metrics['gap_pct'] = None

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

        if criterion in ['selloff_total_pct', 'pct_off_30d_high', 'gap_pct', 'bounce_pct']:
            actual_str = f"{actual*100:.1f}%" if actual is not None and not pd.isna(actual) else "N/A"
            threshold_str = f"{threshold*100:.1f}%"
            direction = "<=" if criterion != 'bounce_pct' else ">="
            ref_str = f"  (A median: {ref*100:.0f}%)" if ref else ""
        elif criterion == 'consecutive_down_days':
            actual_str = f"{int(actual)}" if actual is not None and not pd.isna(actual) else "N/A"
            threshold_str = f"{int(threshold)}"
            direction = ">="
            ref_str = f"  (A median: {ref:.0f})" if ref else ""
        else:  # vol_expansion
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
    for score_val in [6, 5, 4, 3, 2, 1, 0]:
        subset = scored_df[scored_df['criteria_score'] == score_val]
        if len(subset) > 0:
            win_rate = (subset['pnl'] > 0).mean() * 100
            avg_pnl = subset['pnl'].mean()
            print(f"Score {score_val}/6: {len(subset):2d} trades | Win: {win_rate:5.1f}% | Avg P&L: {avg_pnl:+6.1f}%")

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
            print(f"Grade {grade}: {len(subset):2d} trades | Avg Score: {avg_score:.1f}/6 | GO rate: {go_pct:.0f}%")

    # Show sample reports
    print("\n" + "=" * 70)
    print("SAMPLE SCORE REPORTS")
    print("=" * 70)

    for score_val in [6, 5, 4]:
        sample = scored_df[scored_df['criteria_score'] == score_val]
        if len(sample) > 0:
            row = sample.iloc[0]
            setup_type = classify_from_setup_column(row.get('Setup', ''))
            result = scorer.score_setup(
                ticker=row['ticker'],
                date=row['date'],
                setup_type=setup_type,
                metrics=row.to_dict()
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
