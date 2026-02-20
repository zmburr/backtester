"""
Reversal Pre-Trade Checklist (Per-Setup-Type)

Mirrors the BouncePretrade pattern from bounce_scorer.py but for reversal
(parabolic short) setups. Replaces the generic ReversalScorer thresholds with
per-setup-type profiles derived from historical Grade A+B trades.

Starting setup type: 3DGapFade
  - Minimum 2 prior euphoric up days + euphoric gap up on the fade day
  - "Up day" = close > prior_close (a red candle that closes above prior close counts)
  - 33 historical trades (24 A, 8 B, 1 C) in reversal_data.csv

Classification gate (all must be true):
  - consecutive_up_days >= 2
  - gap_pct >= 0.04 (4%+ gap up on fade day)
  - pct_from_9ema >= 0.30 (30%+ above 9EMA — truly parabolic)
  - pct_from_50mav >= 0.4 (40%+ above 50SMA — not just grinding at ATH)
  - atr_pct between 0.04 and 0.20 (sufficient volatility, not penny-stock noise)
  - Reversal confirmation: NOT (closed 2%+ green AND top 25% of range)
    Rejects continuations where the gap-up just kept running

Thresholds derived from 2021 backscanner labeling (32 Y vs 129 N):
  - pct_from_9ema >= 0.30: keeps 88% Y, removes 50% N
  - gap_pct >= 0.04: keeps 91% Y, removes 40% N
  - pct_from_50mav >= 0.4: keeps 100% Y, removes 29% N
  - atr_pct bounds: keeps 100% Y, removes ~8% N
  - Combined: keeps ~81% Y, removes ~66% N (precision 20% -> 37%)

Usage:
    from analyzers.reversal_pretrade import ReversalPretrade, classify_reversal_setup

    pretrade = ReversalPretrade()
    result = pretrade.validate(ticker, metrics, cap='Medium')
    pretrade.print_checklist(result)
"""

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime

try:
    from analyzers.bounce_scorer import ChecklistItem, ChecklistResult
except ImportError:
    from bounce_scorer import ChecklistItem, ChecklistResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ReversalSetupProfile:
    """
    Per-setup-type profile for reversal (parabolic short) pre-trade scoring.
    Thresholds are Dict[str, float] keyed by market cap (ETF/Large/Medium/Small/Micro).
    """
    name: str
    description: str
    sample_size: int
    historical_win_rate: float
    historical_avg_pnl: float

    # Per-cap threshold dicts
    pct_from_9ema: Dict[str, float] = field(default_factory=dict)
    prior_day_range_atr: Dict[str, float] = field(default_factory=dict)
    rvol_score: Dict[str, float] = field(default_factory=dict)
    pct_change_3: Dict[str, float] = field(default_factory=dict)   # V2: 3-day momentum (rho=+0.546)
    gap_pct: Dict[str, float] = field(default_factory=dict)

    def get_threshold(self, criterion: str, cap: str):
        """Look up cap-specific threshold, falling back to _default then Medium."""
        d = getattr(self, criterion)
        if not isinstance(d, dict):
            return d
        return d.get(cap, d.get('_default', d.get('Medium')))


# ---------------------------------------------------------------------------
# Setup profiles — derived from Grade A+B trades in reversal_data.csv
# ---------------------------------------------------------------------------

REVERSAL_SETUP_PROFILES = {
    '3DGapFade': ReversalSetupProfile(
        name='3DGapFade',
        description='2+ euphoric up days + gap up on fade day — classic parabolic exhaustion',
        sample_size=32,
        historical_win_rate=0.91,
        historical_avg_pnl=14.5,

        pct_from_9ema={
            'ETF': 0.10, 'Large': 0.11, 'Medium': 0.44,
            'Small': 1.54, 'Micro': 1.27, '_default': 0.44,
        },
        prior_day_range_atr={
            'ETF': 1.60, 'Large': 1.06, 'Medium': 1.15,
            'Small': 2.50, 'Micro': 5.23, '_default': 1.15,
        },
        rvol_score={
            'ETF': 2.75, 'Large': 1.92, 'Medium': 2.80,
            'Small': 7.05, 'Micro': 1.65, '_default': 2.80,
        },
        # V2: 3-day momentum run-up (rho=+0.546, replaces consecutive_up_days rho=0.086)
        pct_change_3={
            'ETF': 0.04, 'Large': 0.03, 'Medium': 0.09,
            'Small': 0.92, 'Micro': 0.30, '_default': 0.09,
        },
        gap_pct={
            'ETF': 0.02, 'Large': 0.02, 'Medium': 0.07,
            'Small': 0.69, 'Micro': 0.30, '_default': 0.07,
        },
    ),
}

# Criteria display names
REVERSAL_CRITERIA_NAMES = {
    'pct_from_9ema': 'Extended above 9EMA',
    'prior_day_range_atr': 'Range expansion (ATR)',
    'rvol_score': 'Volume expansion (RVOL)',
    'pct_change_3': '3-day momentum run-up',
    'gap_pct': 'Euphoric gap up',
}


# ---------------------------------------------------------------------------
# Classification — auto-detect reversal setup type from metrics
# ---------------------------------------------------------------------------

def classify_reversal_setup(metrics: Dict) -> Optional[str]:
    """
    Auto-detect reversal setup type from computed metrics.

    3DGapFade classification gate (all required):
      - consecutive_up_days >= 2
      - gap_pct >= 0.04 (4%+ gap up on fade day)
      - pct_from_9ema >= 0.30 (30%+ above 9EMA)

    Supplementary filters (applied if data available):
      - pct_from_50mav >= 0.4 (must be extended above trend, not grinding ATH)
      - atr_pct between 0.04 and 0.20 (no ultra-low-vol or penny-stock noise)
      - Reversal confirmation: reject if closed 2%+ green AND in top 25% of range
        (stock just kept running — continuation, not a reversal)

    Returns:
        Setup type string (e.g. '3DGapFade') or None if no match.
    """
    consecutive_up = metrics.get('consecutive_up_days')
    gap = metrics.get('gap_pct')
    pct_9ema = metrics.get('pct_from_9ema')
    pct_50ma = metrics.get('pct_from_50mav')
    atr_pct = metrics.get('atr_pct')
    fade_return = metrics.get('fade_day_return')
    close_pos = metrics.get('fade_day_close_position')

    # Guard against None/NaN
    def _valid(val):
        return val is not None and not (isinstance(val, float) and pd.isna(val))

    # Core gate: all three required
    if _valid(consecutive_up) and _valid(gap) and _valid(pct_9ema):
        if consecutive_up >= 2 and gap >= 0.04 and pct_9ema >= 0.30:
            # Supplementary: must be extended above 50MA (not grinding at ATH)
            if _valid(pct_50ma) and pct_50ma < 0.4:
                return None
            # Supplementary: ATR volatility bounds
            if _valid(atr_pct) and (atr_pct < 0.04 or atr_pct > 0.20):
                return None
            # Reversal confirmation: reject if closed green at highs (continuation)
            if _valid(fade_return) and _valid(close_pos):
                if fade_return > 0.02 and close_pos > 0.75:
                    return None
            return '3DGapFade'

    return None


# ---------------------------------------------------------------------------
# ReversalPretrade — pre-trade checklist for typed reversal setups
# ---------------------------------------------------------------------------

class ReversalPretrade:
    """
    Validates reversal setups against per-setup-type, per-cap thresholds.
    Returns a ChecklistResult with score and recommendation.
    """

    def __init__(self):
        self.profiles = REVERSAL_SETUP_PROFILES

    def _check_gte(self, value, threshold) -> bool:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return False
        return value >= threshold

    def _format_value(self, value, criterion: str) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return 'N/A'
        if criterion in ('pct_from_9ema', 'gap_pct', 'pct_change_3'):
            return f"{value * 100:+.1f}%"
        elif criterion in ('prior_day_range_atr', 'rvol_score'):
            return f"{value:.2f}x"
        return f"{value:.2f}"

    def _format_threshold(self, threshold, criterion: str) -> str:
        if criterion in ('pct_from_9ema', 'gap_pct', 'pct_change_3'):
            return f"{threshold * 100:.0f}%"
        elif criterion in ('prior_day_range_atr', 'rvol_score'):
            return f"{threshold:.1f}x"
        return f"{threshold}"

    def validate(self, ticker: str, metrics: Dict, setup_type: str = '3DGapFade',
                 cap: str = 'Medium') -> ChecklistResult:
        """
        Validate a reversal setup against its per-cap thresholds.

        Args:
            ticker: Stock symbol
            metrics: Dictionary of computed metrics
            setup_type: Reversal setup type (must exist in REVERSAL_SETUP_PROFILES)
            cap: Market cap category

        Returns:
            ChecklistResult with score and recommendation
        """
        if setup_type not in self.profiles:
            logging.warning(f"Unknown reversal setup '{setup_type}', defaulting to 3DGapFade")
            setup_type = '3DGapFade'

        profile = self.profiles[setup_type]

        criteria = [
            ('pct_from_9ema', 'pct_from_9ema',
             f'Extended above 9EMA >= {profile.get_threshold("pct_from_9ema", cap) * 100:.0f}%'),
            ('prior_day_range_atr', 'prior_day_range_atr',
             f'Range expansion >= {profile.get_threshold("prior_day_range_atr", cap):.1f}x ATR'),
            ('rvol_score', 'rvol_score',
             f'Volume expansion >= {profile.get_threshold("rvol_score", cap):.1f}x RVOL'),
            ('pct_change_3', 'pct_change_3',
             f'3-day momentum run-up >= {profile.get_threshold("pct_change_3", cap) * 100:.0f}%'),
            ('gap_pct', 'gap_pct',
             f'Euphoric gap up >= {profile.get_threshold("gap_pct", cap) * 100:.0f}%'),
        ]

        items = []
        score = 0

        for criterion_key, metric_key, description in criteria:
            threshold = profile.get_threshold(criterion_key, cap)
            actual = metrics.get(metric_key)
            passed = self._check_gte(actual, threshold)

            if passed:
                score += 1

            items.append(ChecklistItem(
                name=criterion_key,
                description=description,
                threshold=threshold,
                actual=actual,
                passed=passed,
                threshold_display=self._format_threshold(threshold, criterion_key),
                actual_display=self._format_value(actual, criterion_key),
            ))

        # Scoring: 5/5=A+ GO, 4/5=A GO, 3/5=B CAUTION, <3=F NO-GO
        max_score = 5
        if score >= 4:
            recommendation = 'GO'
            grade = 'A+' if score == 5 else 'A'
            summary = f"PASS: {score}/{max_score} criteria met — {setup_type} ({cap} Cap)"
        elif score == 3:
            recommendation = 'CAUTION'
            grade = 'B'
            failed_names = [i.name for i in items if not i.passed]
            summary = f"MARGINAL: {score}/{max_score} — Missing: {', '.join(failed_names)}"
        else:
            recommendation = 'NO-GO'
            grade = 'F'
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
            bonuses=[],
            warnings=[],
            classification_details={'setup_type': setup_type, 'cap': cap, 'grade': grade},
            cap=cap,
        )

    def print_checklist(self, result: ChecklistResult):
        """Print a formatted checklist for a reversal setup."""
        profile = self.profiles.get(result.setup_type)
        grade = result.classification_details.get('grade', '?')

        print()
        print("=" * 70)
        print(f"REVERSAL PRE-TRADE CHECKLIST: {result.ticker}")
        print(f"Setup Type: {result.setup_type} ({result.cap} Cap)")
        if profile:
            print(f"  {profile.description}")
            print(f"  Historical: {profile.sample_size} A+B trades, "
                  f"{profile.historical_win_rate * 100:.0f}% WR, "
                  f"+{profile.historical_avg_pnl:.0f}% avg P&L")
        print(f"Generated: {result.timestamp}")
        print("=" * 70)

        # Recommendation banner
        if result.recommendation == 'GO':
            print(f"\n  >>> {grade} {result.recommendation} - TRADE IT <<<")
        elif result.recommendation == 'CAUTION':
            print(f"\n  >>> {grade} {result.recommendation} - REDUCED SIZE <<<")
        else:
            print(f"\n  >>> {grade} {result.recommendation} - DO NOT TRADE <<<")

        print(f"\nScore: {result.score}/{result.max_score}")
        print(result.summary)

        print(f"\nCORE CRITERIA ({result.score}/{result.max_score}):")
        print("-" * 60)
        for item in result.items:
            status = "[PASS]" if item.passed else "[FAIL]"
            print(f"  {status} {item.description}")
            print(f"         Required: >= {item.threshold_display} | Actual: {item.actual_display}")

        print()


# ---------------------------------------------------------------------------
# __main__ — validate thresholds against reversal_data.csv 3DGapFade trades
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    csv_path = os.path.join(DATA_PATH, 'reversal_data.csv')

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} trades from reversal_data.csv")

    # Filter to 3DGapFade setup type
    gf = df[df['setup'] == '3DGapFade'].copy()
    print(f"3DGapFade trades: {len(gf)}")
    print(f"Grade distribution: {gf['trade_grade'].value_counts().to_dict()}")
    print(f"Cap distribution: {gf['cap'].value_counts().to_dict()}")

    # Score each trade
    pretrade = ReversalPretrade()
    results = []

    for _, row in gf.iterrows():
        metrics = row.to_dict()
        # Map CSV columns to expected metric names
        if 'one_day_before_range_pct' in metrics and pd.notna(metrics['one_day_before_range_pct']):
            if 'prior_day_range_atr' not in metrics or pd.isna(metrics.get('prior_day_range_atr')):
                metrics['prior_day_range_atr'] = metrics['one_day_before_range_pct']

        cap = row.get('cap', 'Medium')
        if cap is None or (isinstance(cap, float) and pd.isna(cap)):
            cap = 'Medium'

        result = pretrade.validate(
            ticker=row['ticker'],
            metrics=metrics,
            setup_type='3DGapFade',
            cap=cap,
        )
        pnl = -row.get('reversal_open_close_pct', 0) * 100

        results.append({
            'date': row['date'],
            'ticker': row['ticker'],
            'trade_grade': row['trade_grade'],
            'cap': cap,
            'score': result.score,
            'recommendation': result.recommendation,
            'grade': result.classification_details.get('grade', '?'),
            'pnl': pnl,
            'failed': ', '.join([i.name for i in result.items if not i.passed]) or 'PERFECT',
        })

    results_df = pd.DataFrame(results)

    # --- Score distribution ---
    print("\n" + "=" * 70)
    print("3DGapFade SCORING SUMMARY (Per-Cap Thresholds)")
    print("=" * 70)

    print("\nSCORE DISTRIBUTION:")
    print(results_df['score'].value_counts().sort_index(ascending=False))

    print("\nGRADE DISTRIBUTION:")
    print(results_df['grade'].value_counts())

    print("\nRECOMMENDATION DISTRIBUTION:")
    print(results_df['recommendation'].value_counts())

    # --- Performance by score ---
    print("\nPERFORMANCE BY SCORE:")
    print("-" * 60)
    for score_val in range(5, -1, -1):
        subset = results_df[results_df['score'] == score_val]
        if len(subset) > 0:
            win_rate = (subset['pnl'] > 0).mean() * 100
            avg_pnl = subset['pnl'].mean()
            print(f"Score {score_val}/5: {len(subset):2d} trades | "
                  f"Win: {win_rate:5.1f}% | Avg P&L: {avg_pnl:+6.1f}%")

    # --- GO vs NO-GO ---
    print("\nGO vs CAUTION vs NO-GO:")
    print("-" * 60)
    for rec in ['GO', 'CAUTION', 'NO-GO']:
        subset = results_df[results_df['recommendation'] == rec]
        if len(subset) > 0:
            win_rate = (subset['pnl'] > 0).mean() * 100
            avg_pnl = subset['pnl'].mean()
            print(f"{rec:8s}: {len(subset):2d} trades | Win: {win_rate:5.1f}% | Avg P&L: {avg_pnl:+6.1f}%")

    # --- By trade grade ---
    print("\nBY ORIGINAL TRADE GRADE:")
    print("-" * 60)
    for grade in ['A', 'B', 'C']:
        subset = results_df[results_df['trade_grade'] == grade]
        if len(subset) > 0:
            avg_score = subset['score'].mean()
            go_pct = (subset['recommendation'] == 'GO').mean() * 100
            print(f"Grade {grade}: {len(subset):2d} trades | Avg Score: {avg_score:.1f}/5 | GO rate: {go_pct:.0f}%")

    # --- By cap ---
    print("\nBY CAP:")
    print("-" * 60)
    for cap in ['ETF', 'Large', 'Medium', 'Small', 'Micro']:
        subset = results_df[results_df['cap'] == cap]
        if len(subset) > 0:
            avg_score = subset['score'].mean()
            print(f"{cap:8s}: {len(subset):2d} trades | Avg Score: {avg_score:.1f}/5")

    # --- Sample reports ---
    print("\n" + "=" * 70)
    print("SAMPLE SCORE REPORTS")
    print("=" * 70)

    for score_val in [5, 4, 3]:
        sample = results_df[results_df['score'] == score_val]
        if len(sample) > 0:
            row_info = sample.iloc[0]
            row_data = gf[
                (gf['ticker'] == row_info['ticker']) &
                (gf['date'] == row_info['date'])
            ].iloc[0]
            metrics = row_data.to_dict()
            if 'one_day_before_range_pct' in metrics and pd.notna(metrics['one_day_before_range_pct']):
                if 'prior_day_range_atr' not in metrics or pd.isna(metrics.get('prior_day_range_atr')):
                    metrics['prior_day_range_atr'] = metrics['one_day_before_range_pct']
            cap = row_data.get('cap', 'Medium')
            if cap is None or (isinstance(cap, float) and pd.isna(cap)):
                cap = 'Medium'
            result = pretrade.validate(
                ticker=row_data['ticker'],
                metrics=metrics,
                setup_type='3DGapFade',
                cap=cap,
            )
            pretrade.print_checklist(result)

    # --- Most common failure modes ---
    print("\nMOST COMMON FAILURES:")
    print("-" * 60)
    all_failures = []
    for f in results_df['failed']:
        if f != 'PERFECT':
            all_failures.extend(f.split(', '))
    if all_failures:
        from collections import Counter
        for criterion, count in Counter(all_failures).most_common():
            print(f"  {criterion}: {count} trades failed")
