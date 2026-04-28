"""
Breakout Setup Scorer + Pre-Trade Checklist (Setup-Based)

Scores momentum/breakout setups based on SETUP TYPE derived from the `t`
column (days since news catalyst). Two profiles auto-detected:

  - D1_news_break   : t == 0  — breakout on the news day itself
  - D2_continuation : t == 1  — buy at prior high after a Day-1 break (user's main play)

Criteria selection methodology (Phase A analysis):
  Spearman rho of every forward-looking feature vs `breakout_open_high_pct`
  (open-to-high move). Kept features with p<0.10, dropped redundant cousins
  (e.g. pct_from_10/20mav once 9ema is in), dropped features with
  counter-intuitive signs in small samples (consolidation_days, has_ATH
  for D2). Thresholds set at p25 of A+B grade reference distribution.

Modes:
  1. Historical: Score all rows in breakout_data.csv (pretrade + outcome) -> validates setups
  2. Live watchlist: Auto-classify by `t`, score pre-trade criteria only

Usage:
    from analyzers.breakout_scorer import BreakoutScorer, BreakoutPretrade, classify_breakout_setup

    scorer = BreakoutScorer()
    result = scorer.score_setup(ticker, date, setup_type, metrics)

    checker = BreakoutPretrade()
    result = checker.validate(ticker, metrics)
    checker.print_checklist(result)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_DATA_DIR = Path(__file__).resolve().parent.parent / 'data'


# ---------------------------------------------------------------------------
# Percentile-of-score helper (same pattern as reversal_scorer)
# ---------------------------------------------------------------------------
try:
    from scipy.stats import percentileofscore as _pctrank
except Exception:
    def _pctrank(a, score, kind='rank'):
        try:
            arr = np.asarray(a, dtype=float)
            if arr.size == 0:
                return np.nan
            s = float(score)
        except Exception:
            return np.nan
        if kind in ('rank', 'mean'):
            return 100.0 * (np.sum(arr < s) + 0.5 * np.sum(arr == s)) / arr.size
        return 100.0 * np.sum(arr <= s) / arr.size


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CriterionSpec:
    """One criterion in a setup profile."""
    name: str          # column name in breakout_data.csv
    threshold: float
    direction: str     # 'gte' or 'lte'
    display: str       # human-readable description


@dataclass
class SetupProfile:
    name: str
    description: str
    sample_size: int
    historical_avg_extension: float   # avg breakout_open_high_pct of A+B reference
    historical_pct_grade_a: float     # share of trades that were Grade A

    pretrade_criteria: List[CriterionSpec] = field(default_factory=list)
    outcome_threshold: float = 0.05   # p25 of breakout_open_high_pct on A+B

    # Bonus signals (not gated, just flagged)
    bonus_checks: Dict = field(default_factory=dict)

    # Reference medians for context
    reference_medians: Dict = field(default_factory=dict)


@dataclass
class ChecklistItem:
    name: str
    description: str
    threshold: float
    actual: float
    passed: bool
    threshold_display: str
    actual_display: str
    reference: str = ''


@dataclass
class ChecklistResult:
    ticker: str
    setup_type: str
    timestamp: str
    items: List[ChecklistItem]
    score: int
    max_score: int
    recommendation: str
    summary: str
    bonuses: List[str]
    warnings: List[str]
    classification_details: Dict = field(default_factory=dict)
    intensity: Optional[float] = None


# ---------------------------------------------------------------------------
# Setup profiles — thresholds derived from p25 of A+B reference distribution
# (see scripts/analyze_breakout_features.py)
# ---------------------------------------------------------------------------

SETUP_PROFILES = {
    'D1_news_break': SetupProfile(
        name='D1_news_break',
        description='Breakout on the news day itself (t=0). Momentum-extension predicts size.',
        sample_size=12,             # A+B reference rows
        historical_avg_extension=0.2363,
        historical_pct_grade_a=0.50,
        outcome_threshold=0.1002,    # p25 of breakout_open_high_pct on A+B

        pretrade_criteria=[
            CriterionSpec('pct_from_9ema',                0.18,  'gte', 'Extended above 9 EMA (>= 18%)'),
            CriterionSpec('pct_from_50mav',               0.36,  'gte', 'Extended above 50 MA (>= 36%)'),
            CriterionSpec('range_contraction_atr',        0.90,  'gte', 'Recent range expansion (5d TR / 30d TR >= 0.90)'),
            CriterionSpec('prior_day_close_vs_high_pct',  0.27,  'gte', 'Prior day closed in upper 27% of range'),
            CriterionSpec('gap_from_pm_high',            -0.03,  'gte', 'Open within 3% of PM high (no severe gap fade)'),
        ],

        bonus_checks={
            'has_ATH_breakout':       (1, 'eq', 'ATH breakout flag set'),
            'has_52wk_breakout':      (1, 'eq', '52-week high breakout'),
            'has_IPO_high_breakout':  (1, 'eq', 'IPO high breakout'),
            'pct_change_15':          (0.15, 'gte', '15-day momentum strong (>= 15%)'),
            'consecutive_days_above_50ma': (10, 'gte', 'Trending above 50 MA (10+ days)'),
        },

        reference_medians={
            'pct_from_9ema':               0.2547,
            'pct_from_50mav':              0.5633,
            'range_contraction_atr':       0.9805,
            'prior_day_close_vs_high_pct': 0.4624,
            'gap_from_pm_high':           -0.0133,
        },
    ),

    'D2_continuation': SetupProfile(
        name='D2_continuation',
        description='Day-after continuation buy at prior high (t=1). Proximity to key levels matters.',
        sample_size=27,
        historical_avg_extension=0.1175,
        historical_pct_grade_a=0.50,
        outcome_threshold=0.0641,    # p25 of breakout_open_high_pct on A+B

        pretrade_criteria=[
            CriterionSpec('pct_from_9ema',                0.09,  'gte', 'Extended above 9 EMA (>= 9%)'),
            CriterionSpec('pct_to_52wk_high',            -0.38,  'gte', 'Within 38% of 52wk high'),
            CriterionSpec('atr_pct',                      0.025, 'gte', 'Daily ATR >= 2.5% (cap-neutral floor; mega-caps OK)'),
            CriterionSpec('gap_pct',                      0.00,  'gte', 'Positive overnight gap'),
            CriterionSpec('percent_of_premarket_vol',     0.036, 'gte', 'Premarket volume >= 3.6% of ADV'),
            CriterionSpec('overnight_gap_d1_to_d2_pct',   0.008, 'gte', 'D1->D2 overnight gap up >= 0.8%'),
        ],

        bonus_checks={
            'has_ATH_breakout':       (1, 'eq', 'ATH breakout context'),
            'has_IPO_high_breakout':  (1, 'eq', 'IPO breakout context'),
            'pct_change_3':           (0.10, 'gte', '3-day momentum strong (>= 10%)'),
            'pm_d2_holds_above_d1_high': (True, 'eq', 'Premarket held above D1 high'),
            'd1_close_at_high_pct':   (0.80, 'gte', 'D1 closed in top 20% of range'),
        },

        reference_medians={
            'pct_from_9ema':              0.1600,
            'pct_to_52wk_high':          -0.0513,
            'atr_pct':                    0.0538,
            'gap_pct':                    None,
            'percent_of_premarket_vol':   None,
            'overnight_gap_d1_to_d2_pct': 0.018,
        },
    ),
}


# ---------------------------------------------------------------------------
# Intensity spec (cap-stratification not used — small sample). Mirrors the
# reversal_scorer ATR-adjusted percentile-rank composite.
# ---------------------------------------------------------------------------
_INTENSITY_SPEC = {
    'D1_news_break': [
        # (column, derived_atr_adj?, higher_is_better, weight)
        ('pct_from_9ema',               True,  True, 0.25),
        ('pct_from_50mav',              True,  True, 0.20),
        ('range_contraction_atr',       False, True, 0.15),
        ('prior_day_close_vs_high_pct', False, True, 0.20),
        ('gap_from_pm_high',            True,  True, 0.20),
    ],
    'D2_continuation': [
        ('pct_from_9ema',              True,  True, 0.20),
        ('pct_to_52wk_high',           False, True, 0.20),
        ('atr_pct',                    False, True, 0.15),
        ('gap_pct',                    True,  True, 0.15),
        ('percent_of_premarket_vol',   False, True, 0.15),
        ('overnight_gap_d1_to_d2_pct', True,  True, 0.15),
    ],
}


# ---------------------------------------------------------------------------
# Load A+B reference for intensity percentile ranking
# ---------------------------------------------------------------------------
def _load_reference():
    csv = _DATA_DIR / 'breakout_data.csv'
    try:
        df = pd.read_csv(csv).dropna(subset=['ticker', 'date'])
        df['trade_grade'] = df['trade_grade'].astype(str).str.strip()
        ab = df[df['trade_grade'].isin(['A', 'B'])].copy()

        # Derive setup_type from t for reference set
        ab['_profile'] = ab['t'].map(lambda v: 'D1_news_break' if v == 0
                                     else 'D2_continuation' if v == 1
                                     else None)
        ab = ab[ab['_profile'].notna()].copy()

        # Pre-compute ATR-adjusted columns
        atr = pd.to_numeric(ab['atr_pct'], errors='coerce').replace(0, np.nan)
        for raw in ('pct_from_9ema', 'pct_from_50mav', 'gap_from_pm_high',
                    'gap_pct', 'overnight_gap_d1_to_d2_pct'):
            if raw in ab.columns:
                ab[f'{raw}_atr'] = pd.to_numeric(ab[raw], errors='coerce') / atr

        return {p: sub for p, sub in ab.groupby('_profile')}
    except Exception as e:
        logging.warning(f'Could not load breakout reference: {e}')
        return {}


_REF_BY_PROFILE = _load_reference()
# _INTENSITY_THRESHOLDS is populated after compute_breakout_intensity is defined (see below).


def _compute_intensity_thresholds():
    """Compute per-profile intensity thresholds from A+B reference distribution.

    Returns dict: {profile: {'tradable_min': float, 'high_conviction_min': float, 'p25': ..., 'p50': ..., 'p75': ..., 'min': ...}}

    These thresholds drive the intensity-based recommendation tiers:
      - intensity >= high_conviction_min -> FULL_SIZE
      - intensity >= tradable_min        -> REDUCED_SIZE (still tradable)
      - intensity < tradable_min         -> AVOID
    """
    out = {}
    for profile in ('D1_news_break', 'D2_continuation'):
        ref = _REF_BY_PROFILE.get(profile)
        if ref is None or ref.empty:
            continue
        # Compute intensity for each A+B reference trade (closure over module state)
        intensities = []
        for _, row in ref.iterrows():
            metrics = row.to_dict()
            r = compute_breakout_intensity(metrics, profile)
            if r['composite'] is not None:
                intensities.append(r['composite'])
        if not intensities:
            continue
        s = pd.Series(intensities)
        out[profile] = {
            'min': float(s.min()),
            'p25': float(s.quantile(0.25)),
            'p50': float(s.quantile(0.50)),
            'p75': float(s.quantile(0.75)),
            'tradable_min': float(s.min()),       # floor — anything historically tradable
            'high_conviction_min': float(s.quantile(0.50)),  # median = full-size threshold
            'n': len(intensities),
        }
    return out


# Placeholder; populated below after compute_breakout_intensity is defined.
_INTENSITY_THRESHOLDS: Dict = {}


def get_intensity_threshold(profile: str, key: str = 'tradable_min'):
    """Look up an intensity threshold for a profile."""
    return _INTENSITY_THRESHOLDS.get(profile, {}).get(key)


# ---------------------------------------------------------------------------
# Intensity computation
# ---------------------------------------------------------------------------

def compute_breakout_intensity(metrics: Dict, setup_type: str) -> Dict:
    """
    Compute ATR-adjusted intensity score (0-100) for a breakout setup.
    Percentile-ranks each metric against A+B reference distribution within
    the same profile.

    Returns:
        {'composite': 0-100, 'details': {col: {pctile, weight, actual}}}
    """
    spec = _INTENSITY_SPEC.get(setup_type)
    if not spec:
        return {'composite': None, 'details': {}}
    ref_df = _REF_BY_PROFILE.get(setup_type)
    if ref_df is None or ref_df.empty:
        return {'composite': None, 'details': {}}

    atr_pct = metrics.get('atr_pct')
    if atr_pct is None or pd.isna(atr_pct) or atr_pct == 0:
        atr_pct = None

    details = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for col, atr_adjust, higher_is_better, weight in spec:
        # Compute the actual (ATR-adjusted if specified)
        actual = metrics.get(col)
        if atr_adjust:
            if actual is None or pd.isna(actual) or atr_pct is None:
                actual_adj = None
            else:
                actual_adj = float(actual) / atr_pct
            ref_col = f'{col}_atr'
        else:
            actual_adj = float(actual) if (actual is not None and not pd.isna(actual)) else None
            ref_col = col

        ref_vals = ref_df[ref_col].dropna().values if ref_col in ref_df.columns else []
        if actual_adj is None or len(ref_vals) == 0:
            details[col] = {'pctile': None, 'weight': weight, 'actual': actual}
            continue

        raw_pctile = _pctrank(ref_vals, actual_adj, kind='rank')
        pctile = raw_pctile if higher_is_better else 100.0 - raw_pctile
        details[col] = {
            'pctile': round(pctile, 1),
            'weight': weight,
            'actual': round(actual_adj, 3) if isinstance(actual_adj, float) else actual_adj,
        }
        weighted_sum += pctile * weight
        total_weight += weight

    composite = round(weighted_sum / total_weight, 1) if total_weight > 0 else None
    return {'composite': composite, 'details': details}


# ---------------------------------------------------------------------------
# Setup classification
# ---------------------------------------------------------------------------

def classify_breakout_setup(metrics: Dict) -> Tuple[str, Dict]:
    """Auto-classify based on `t` (days since news catalyst)."""
    t_val = metrics.get('t')
    details = {'t': t_val, 'signals': []}
    try:
        t_int = int(t_val) if t_val is not None and not pd.isna(t_val) else None
    except (ValueError, TypeError):
        t_int = None

    if t_int == 0:
        details['signals'].append('t=0 -> news day breakout (D1)')
        details['classification'] = 'D1_news_break'
        return 'D1_news_break', details
    if t_int == 1:
        details['signals'].append('t=1 -> day-after continuation (D2)')
        details['classification'] = 'D2_continuation'
        return 'D2_continuation', details

    details['signals'].append(f't={t_val} (no clean classification) - defaulting to D2_continuation')
    details['classification'] = 'D2_continuation'
    return 'D2_continuation', details


# Now that compute_breakout_intensity is defined, populate the threshold table.
_INTENSITY_THRESHOLDS = _compute_intensity_thresholds()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_condition(value, threshold, direction: str) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if direction == 'gte':
        return value >= threshold
    if direction == 'lte':
        return value <= threshold
    if direction == 'eq':
        return value == threshold
    return False


def _coerce_bool(v):
    if isinstance(v, bool):
        return v
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    s = str(v).strip().lower()
    return s in ('true', '1', 'yes')


def _format_value(value, criterion: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 'N/A'
    pct_cols = {'pct_from_9ema', 'pct_from_50mav', 'pct_to_52wk_high', 'gap_pct',
                'gap_from_pm_high', 'overnight_gap_d1_to_d2_pct',
                'percent_of_premarket_vol', 'atr_pct', 'pct_change_3', 'pct_change_15',
                'breakout_open_high_pct'}
    if criterion in pct_cols:
        return f'{value*100:+.2f}%'
    if criterion in ('range_contraction_atr',):
        return f'{value:.2f}x'
    if criterion in ('prior_day_close_vs_high_pct',):
        return f'{value:.2f}'
    if criterion in ('consecutive_days_above_50ma', 't'):
        return f'{int(value)}'
    return f'{value:.3f}' if isinstance(value, float) else str(value)


def _format_threshold(threshold, criterion: str) -> str:
    pct_cols = {'pct_from_9ema', 'pct_from_50mav', 'pct_to_52wk_high', 'gap_pct',
                'gap_from_pm_high', 'overnight_gap_d1_to_d2_pct',
                'percent_of_premarket_vol', 'atr_pct', 'pct_change_3', 'pct_change_15'}
    if criterion in pct_cols:
        return f'{threshold*100:+.2f}%'
    if criterion in ('range_contraction_atr',):
        return f'{threshold:.2f}x'
    return f'{threshold:.3f}' if isinstance(threshold, float) else str(threshold)


# ---------------------------------------------------------------------------
# BreakoutScorer — historical scoring (pretrade criteria + outcome criterion)
# ---------------------------------------------------------------------------

class BreakoutScorer:
    """Scores breakout setups historically. Adds 1 outcome criterion to pretrade."""

    def __init__(self, profiles=None):
        self.profiles = profiles or SETUP_PROFILES

    def _get_profile(self, setup_type: str) -> SetupProfile:
        if setup_type not in self.profiles:
            logging.warning(f"Unknown setup '{setup_type}', defaulting to D2_continuation")
            return self.profiles['D2_continuation']
        return self.profiles[setup_type]

    def _evaluate_criteria(self, metrics: Dict, profile: SetupProfile) -> Tuple[int, int, List[str], List[str]]:
        """Returns (pretrade_score, full_score, passed_list, failed_list).

        full_score = pretrade_score + 1 if outcome criterion is met.
        """
        passed = []
        failed = []

        for crit in profile.pretrade_criteria:
            actual = metrics.get(crit.name)
            if _check_condition(actual, crit.threshold, crit.direction):
                passed.append(crit.name)
            else:
                failed.append(crit.name)

        pretrade_score = len(passed)

        # Outcome criterion (breakout_open_high_pct >= profile.outcome_threshold)
        actual_out = metrics.get('breakout_open_high_pct')
        if _check_condition(actual_out, profile.outcome_threshold, 'gte'):
            passed.append('breakout_open_high_pct')
        else:
            failed.append('breakout_open_high_pct')

        full_score = len(passed)
        return pretrade_score, full_score, passed, failed

    def _grade(self, pretrade_score: int, full_score: int, n_pretrade: int) -> str:
        """A+ = full pass, A = full pass except outcome, etc."""
        if full_score == n_pretrade + 1:
            return 'A+'
        if full_score == n_pretrade:
            return 'A'
        if full_score == n_pretrade - 1:
            return 'B'
        if full_score == n_pretrade - 2:
            return 'C'
        return 'F'

    def _recommendation(self, pretrade_score: int, n_pretrade: int) -> str:
        # Legacy binary count — retained for descriptive output only.
        # Primary recommendation comes from intensity tiers (see _intensity_recommendation).
        if pretrade_score >= n_pretrade - 1:
            return 'GO'
        if pretrade_score == n_pretrade - 2:
            return 'CAUTION'
        return 'NO-GO'

    def _intensity_recommendation(self, intensity: Optional[float], setup_type: str) -> Tuple[str, Dict]:
        """Tier the trade by intensity score against the A+B database distribution.

        Returns (tier, threshold_dict). Tiers:
          FULL_SIZE     : intensity >= median of historically-tradable A+B trades
          REDUCED_SIZE  : intensity >= minimum tradable score (still in distribution)
          AVOID         : intensity below the floor — no historical analog supports the trade
          UNRATED       : intensity could not be computed (insufficient features)

        These tiers are the PRIMARY recommendation. The legacy GO/CAUTION/NO-GO
        from binary criteria is kept for diagnostic context only.
        """
        thresholds = _INTENSITY_THRESHOLDS.get(setup_type, {})
        if intensity is None or not thresholds:
            return 'UNRATED', thresholds

        if intensity >= thresholds['high_conviction_min']:
            return 'FULL_SIZE', thresholds
        if intensity >= thresholds['tradable_min']:
            return 'REDUCED_SIZE', thresholds
        return 'AVOID', thresholds

    def score_setup(self, ticker: str, date: str, setup_type: str, metrics: Dict) -> Dict:
        profile = self._get_profile(setup_type)
        n_pretrade = len(profile.pretrade_criteria)

        pretrade_score, full_score, passed, failed = self._evaluate_criteria(metrics, profile)
        grade = self._grade(pretrade_score, full_score, n_pretrade)
        recommendation = self._recommendation(pretrade_score, n_pretrade)

        # Build criteria detail
        criteria_details = {}
        for crit in profile.pretrade_criteria:
            actual = metrics.get(crit.name)
            criteria_details[crit.name] = {
                'name': crit.display,
                'threshold': crit.threshold,
                'direction': crit.direction,
                'actual': actual,
                'passed': crit.name in passed,
                'reference_median': profile.reference_medians.get(crit.name),
            }
        # Outcome criterion
        criteria_details['breakout_open_high_pct'] = {
            'name': f'Outcome: open-to-high >= {profile.outcome_threshold*100:.1f}%',
            'threshold': profile.outcome_threshold,
            'direction': 'gte',
            'actual': metrics.get('breakout_open_high_pct'),
            'passed': 'breakout_open_high_pct' in passed,
            'reference_median': None,
        }

        # Intensity is now ALWAYS computed (not just on GO) so we can tier nuance
        # within "CAUTION-but-still-tradable" trades.
        ir = compute_breakout_intensity(metrics, setup_type)
        intensity = ir.get('composite')

        # Primary recommendation: intensity tier vs A+B database distribution
        intensity_tier, tier_thresholds = self._intensity_recommendation(intensity, setup_type)

        return {
            'ticker': ticker,
            'date': date,
            'setup_type': setup_type,
            'pretrade_score': pretrade_score,
            'pretrade_max': n_pretrade,
            'full_score': full_score,
            'full_max': n_pretrade + 1,
            'grade': grade,
            'recommendation': recommendation,         # legacy binary tier (descriptive)
            'intensity_tier': intensity_tier,         # PRIMARY tier
            'tier_thresholds': tier_thresholds,
            'passed_criteria': passed,
            'failed_criteria': failed,
            'criteria_details': criteria_details,
            'is_true_a': full_score >= n_pretrade,
            'intensity': intensity,
        }

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in df.iterrows():
            metrics = row.to_dict()
            # Derive setup tags for bonus_checks
            setup_str = str(metrics.get('setup_type', ''))
            metrics['has_ATH_breakout'] = 1 if 'ATH_breakout' in setup_str else 0
            metrics['has_52wk_breakout'] = 1 if ('52wk_breakout' in setup_str and 'IPO_high' not in setup_str) else 0
            metrics['has_IPO_high_breakout'] = 1 if 'IPO_high_breakout' in setup_str else 0

            setup_type, _ = classify_breakout_setup(metrics)
            result = self.score_setup(
                ticker=row.get('ticker', ''),
                date=row.get('date', ''),
                setup_type=setup_type,
                metrics=metrics,
            )
            results.append({
                'derived_setup': setup_type,
                'pretrade_score': result['pretrade_score'],
                'pretrade_max': result['pretrade_max'],
                'full_score': result['full_score'],
                'criteria_grade': result['grade'],
                'recommendation': result['recommendation'],
                'intensity_tier': result['intensity_tier'],
                'is_true_a': result['is_true_a'],
                'failed_criteria': ', '.join(result['failed_criteria']) if result['failed_criteria'] else 'PERFECT',
                'intensity': result['intensity'],
            })

        score_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), score_df], axis=1)


# ---------------------------------------------------------------------------
# BreakoutPretrade — live pre-trade checklist (no outcome leakage)
# ---------------------------------------------------------------------------

class BreakoutPretrade:
    """Validates breakout setups against pre-trade criteria only (no outcome).

    Auto-classifies via `t` (days since news catalyst), then scores against the
    matching profile.
    """

    def __init__(self, profiles=None):
        self.profiles = profiles or SETUP_PROFILES

    def validate(self, ticker: str, metrics: Dict, force_setup: Optional[str] = None) -> ChecklistResult:
        if force_setup and force_setup in self.profiles:
            setup_type = force_setup
            class_details = {'classification': force_setup, 'signals': ['Manually specified']}
        else:
            setup_type, class_details = classify_breakout_setup(metrics)
        profile = self.profiles[setup_type]

        # Compute setup-tag flags from setup_type column if present
        setup_str = str(metrics.get('setup_type', ''))
        metrics = dict(metrics)
        metrics.setdefault('has_ATH_breakout', 1 if 'ATH_breakout' in setup_str else 0)
        metrics.setdefault('has_52wk_breakout', 1 if ('52wk_breakout' in setup_str and 'IPO_high' not in setup_str) else 0)
        metrics.setdefault('has_IPO_high_breakout', 1 if 'IPO_high_breakout' in setup_str else 0)

        items = []
        score = 0
        for crit in profile.pretrade_criteria:
            actual = metrics.get(crit.name)
            passed = _check_condition(actual, crit.threshold, crit.direction)
            if passed:
                score += 1
            ref_median = profile.reference_medians.get(crit.name)
            ref_str = ''
            if ref_median is not None:
                ref_str = f'A+B median: {_format_value(ref_median, crit.name)}'
            items.append(ChecklistItem(
                name=crit.name,
                description=crit.display,
                threshold=crit.threshold,
                actual=actual if actual is not None else 0.0,
                passed=passed,
                threshold_display=_format_threshold(crit.threshold, crit.name),
                actual_display=_format_value(actual, crit.name),
                reference=ref_str,
            ))

        # Bonus signals
        bonuses = []
        for indicator, (threshold, op, description) in profile.bonus_checks.items():
            v = metrics.get(indicator)
            v_check = _coerce_bool(v) if isinstance(threshold, bool) else v
            if op == 'eq':
                hit = (v_check == threshold)
            elif op == 'gte':
                hit = (v_check is not None and not (isinstance(v_check, float) and pd.isna(v_check)) and v_check >= threshold)
            elif op == 'lte':
                hit = (v_check is not None and not (isinstance(v_check, float) and pd.isna(v_check)) and v_check <= threshold)
            else:
                hit = False
            if hit:
                bonuses.append(description)

        # Warnings
        warnings = []
        spy_5d = metrics.get('spy_5day_return')
        if spy_5d is not None and not pd.isna(spy_5d) and spy_5d < -0.02:
            warnings.append(f'SPY 5-day return weak ({spy_5d*100:+.1f}%) — breakouts fail in weak markets')
        gap_pm = metrics.get('gap_from_pm_high')
        if gap_pm is not None and not pd.isna(gap_pm) and gap_pm < -0.05:
            warnings.append(f'Gap fade from PM high ({gap_pm*100:+.1f}%) — momentum failing')
        gap = metrics.get('gap_pct')
        atr = metrics.get('atr_pct')
        if gap is not None and atr and atr > 0 and gap / atr > 3.0:
            warnings.append(f'Gap > 3x ATR ({gap*100:.1f}% vs {atr*100:.1f}%) — exhaustion gap risk')

        # Always compute intensity — it's the primary signal
        ir = compute_breakout_intensity(metrics, setup_type)
        intensity = ir.get('composite')

        # PRIMARY recommendation: intensity tier vs A+B database distribution
        thresholds = _INTENSITY_THRESHOLDS.get(setup_type, {})
        if intensity is None or not thresholds:
            recommendation = 'UNRATED'
            tier_summary = 'No intensity score (insufficient features)'
        elif intensity >= thresholds.get('high_conviction_min', 100):
            recommendation = 'FULL_SIZE'
            tier_summary = f'High conviction: intensity {intensity:.0f} >= median A+B ({thresholds["high_conviction_min"]:.0f})'
        elif intensity >= thresholds.get('tradable_min', 100):
            recommendation = 'REDUCED_SIZE'
            tier_summary = f'Tradable: intensity {intensity:.0f} above floor ({thresholds["tradable_min"]:.0f}) but below median'
        else:
            recommendation = 'AVOID'
            tier_summary = f'Below tradable floor: intensity {intensity:.0f} < {thresholds["tradable_min"]:.0f}'

        n_pretrade = len(profile.pretrade_criteria)
        failed_names = [i.name for i in items if not i.passed]
        summary = f'{tier_summary} | Pre-trade criteria: {score}/{n_pretrade}'
        if failed_names:
            summary += f' (missed: {", ".join(failed_names)})'

        return ChecklistResult(
            ticker=ticker,
            setup_type=setup_type,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            items=items,
            score=score,
            max_score=n_pretrade,
            recommendation=recommendation,
            summary=summary,
            bonuses=bonuses,
            warnings=warnings,
            classification_details=class_details,
            intensity=intensity,
        )

    def print_checklist(self, result: ChecklistResult):
        profile = self.profiles[result.setup_type]
        print()
        print('=' * 70)
        print(f'BREAKOUT PRE-TRADE CHECKLIST: {result.ticker}')
        print(f'Setup Profile: {profile.name}')
        print(f'  {profile.description}')
        print(f'  Reference: {profile.sample_size} A+B trades, '
              f'avg open-to-high {profile.historical_avg_extension*100:.1f}%')
        print(f'Generated: {result.timestamp}')
        print('=' * 70)

        if result.classification_details.get('signals'):
            print('\nCLASSIFICATION:')
            for sig in result.classification_details['signals']:
                print(f'  - {sig}')

        banner = {
            'FULL_SIZE': 'FULL SIZE - HIGH CONVICTION',
            'REDUCED_SIZE': 'REDUCED SIZE - TRADABLE',
            'AVOID': 'AVOID - BELOW HISTORICAL FLOOR',
            'UNRATED': 'UNRATED - INSUFFICIENT DATA',
        }
        print(f'\n  >>> {result.recommendation} - {banner.get(result.recommendation, "")} <<<')
        print(f'\nScore: {result.score}/{result.max_score}')
        print(result.summary)
        if result.intensity is not None:
            print(f'Intensity: {result.intensity:.0f}/100')

        print(f'\nCORE CRITERIA ({result.score}/{result.max_score}):')
        print('-' * 60)
        for item in result.items:
            status = '[PASS]' if item.passed else '[FAIL]'
            ref = f'  ({item.reference})' if item.reference else ''
            direction_sym = '>=' if 'gte' in item.description.lower() or item.threshold >= 0 else '<='
            print(f'  {status} {item.description}')
            print(f'         Required: >= {item.threshold_display} | Actual: {item.actual_display}{ref}')

        if result.bonuses:
            print('\nBONUS FACTORS:')
            print('-' * 60)
            for b in result.bonuses:
                print(f'  [+] {b}')

        if result.warnings:
            print('\nWARNINGS:')
            print('-' * 60)
            for w in result.warnings:
                print(f'  [!] {w}')

        print()


# ---------------------------------------------------------------------------
# Print helper for historical reports
# ---------------------------------------------------------------------------

def print_score_report(result: Dict):
    print(f'\n{"=" * 70}')
    print(f'BREAKOUT SETUP SCORE: {result["ticker"]} ({result["date"]})')
    print(f'{"=" * 70}')
    print(f'Setup Profile: {result["setup_type"]}')
    print(f'Pre-Trade Score: {result["pretrade_score"]}/{result["pretrade_max"]}')
    print(f'Full Score:      {result["full_score"]}/{result["full_max"]}')
    print(f'Grade: {result["grade"]}')
    print(f'Recommendation: {result["recommendation"]}')
    if result['intensity'] is not None:
        print(f'Intensity: {result["intensity"]:.0f}/100')
    print()
    print('CRITERIA BREAKDOWN:')
    print('-' * 60)
    for criterion, details in result['criteria_details'].items():
        status = '[PASS]' if details['passed'] else '[FAIL]'
        actual_str = _format_value(details['actual'], criterion)
        thresh_str = _format_threshold(details['threshold'], criterion)
        ref_str = ''
        if details.get('reference_median') is not None:
            ref_str = f'  (A+B median: {_format_value(details["reference_median"], criterion)})'
        print(f'  {status} {details["name"]}')
        print(f'         Required: {details["direction"]} {thresh_str} | Actual: {actual_str}{ref_str}')
    print()


# ---------------------------------------------------------------------------
# __main__ - score historical breakout_data.csv
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import os
    from scipy.stats import spearmanr

    csv_path = _DATA_DIR / 'breakout_data.csv'
    df = pd.read_csv(csv_path)
    print(f'\nLoaded {len(df)} trades from breakout_data.csv')

    scorer = BreakoutScorer()
    scored_df = scorer.score_dataframe(df)

    # Use breakout_open_high_pct as P&L proxy (the move you can capture)
    scored_df['pnl_proxy'] = pd.to_numeric(scored_df['breakout_open_high_pct'], errors='coerce') * 100

    print('\n' + '=' * 70)
    print('BREAKOUT SETUP SCORING SUMMARY')
    print('=' * 70)

    print('\nPROFILE ASSIGNMENT:')
    print(scored_df['derived_setup'].value_counts())

    print('\nINTENSITY THRESHOLDS (derived from A+B reference):')
    for prof, t in _INTENSITY_THRESHOLDS.items():
        print(f'  {prof}: tradable_min={t["tradable_min"]:.0f}, high_conviction_min={t["high_conviction_min"]:.0f} '
              f'(p25={t["p25"]:.0f}, p75={t["p75"]:.0f}, n={t["n"]})')

    print('\nINTENSITY TIER DISTRIBUTION (PRIMARY recommendation):')
    print(scored_df['intensity_tier'].value_counts())

    print('\nLEGACY BINARY DISTRIBUTION (for comparison):')
    print(scored_df['recommendation'].value_counts())

    # Performance by intensity tier (primary)
    print('\n' + '=' * 70)
    print('PERFORMANCE BY INTENSITY TIER (open-to-high % proxy)')
    print('=' * 70)
    for tier in ['FULL_SIZE', 'REDUCED_SIZE', 'AVOID', 'UNRATED']:
        sub = scored_df[scored_df['intensity_tier'] == tier]
        if len(sub) == 0:
            continue
        avg = sub['pnl_proxy'].mean()
        wr = (sub['pnl_proxy'] > 5).mean() * 100  # "made >=5%" win rate
        print(f'  {tier:13s}: {len(sub):2d} trades | Avg open-to-high: {avg:+.2f}% | %>=5%: {wr:.0f}%')

    # By profile + intensity tier
    print('\nBY PROFILE x INTENSITY TIER:')
    print('-' * 60)
    for prof in ['D1_news_break', 'D2_continuation']:
        ps = scored_df[scored_df['derived_setup'] == prof]
        if len(ps) == 0:
            continue
        print(f'\n  {prof} (n={len(ps)}):')
        for tier in ['FULL_SIZE', 'REDUCED_SIZE', 'AVOID', 'UNRATED']:
            ss = ps[ps['intensity_tier'] == tier]
            if len(ss):
                avg = ss['pnl_proxy'].mean()
                print(f'    {tier:13s}: {len(ss):2d} trades | Avg: {avg:+.2f}%')

    # By trade_grade
    print('\nBY ORIGINAL TRADE GRADE:')
    print('-' * 60)
    for grade in ['A', 'B', 'C']:
        sub = scored_df[scored_df['trade_grade'].astype(str).str.strip() == grade]
        if len(sub) == 0:
            continue
        avg_intensity = sub['intensity'].mean()
        tradable_pct = sub['intensity_tier'].isin(['FULL_SIZE', 'REDUCED_SIZE']).mean() * 100
        full_size_pct = (sub['intensity_tier'] == 'FULL_SIZE').mean() * 100
        avg_pnl = sub['pnl_proxy'].mean()
        print(f'  Grade {grade}: {len(sub):2d} trades | Avg intensity: {avg_intensity:.0f} | Tradable: {tradable_pct:.0f}% | Full-size: {full_size_pct:.0f}% | Avg P&L: {avg_pnl:+.2f}%')

    # Validation: pretrade_score vs P&L proxy
    print('\n' + '=' * 70)
    print('VALIDATION (Spearman rho)')
    print('=' * 70)
    valid = scored_df['pretrade_score'].notna() & scored_df['pnl_proxy'].notna()
    if valid.sum() > 5:
        rho, p = spearmanr(scored_df.loc[valid, 'pretrade_score'], scored_df.loc[valid, 'pnl_proxy'])
        print(f'  pretrade_score vs open-to-high: rho={rho:+.3f} (p={p:.4f}) on n={valid.sum()}')
    int_valid = scored_df['intensity'].notna() & scored_df['pnl_proxy'].notna()
    if int_valid.sum() > 5:
        rho_i, p_i = spearmanr(scored_df.loc[int_valid, 'intensity'], scored_df.loc[int_valid, 'pnl_proxy'])
        print(f'  intensity vs open-to-high:      rho={rho_i:+.3f} (p={p_i:.4f}) on n={int_valid.sum()}')

    # Sample reports for top scoring trades
    print('\n' + '=' * 70)
    print('SAMPLE SCORE REPORTS')
    print('=' * 70)
    for prof in ['D1_news_break', 'D2_continuation']:
        sub = scored_df[(scored_df['derived_setup'] == prof) & (scored_df['recommendation'] == 'GO')]
        if len(sub):
            row = sub.sort_values('pnl_proxy', ascending=False).iloc[0]
            metrics = row.to_dict()
            metrics['has_ATH_breakout'] = 1 if 'ATH_breakout' in str(metrics.get('setup_type', '')) else 0
            metrics['has_52wk_breakout'] = 1 if ('52wk_breakout' in str(metrics.get('setup_type', '')) and 'IPO_high' not in str(metrics.get('setup_type', ''))) else 0
            metrics['has_IPO_high_breakout'] = 1 if 'IPO_high_breakout' in str(metrics.get('setup_type', '')) else 0
            result = scorer.score_setup(
                ticker=row.get('ticker', ''),
                date=row.get('date', ''),
                setup_type=prof,
                metrics=metrics,
            )
            print_score_report(result)

    # Save scored data
    out_path = _DATA_DIR / 'breakout_scored.csv'
    scored_df.to_csv(out_path, index=False)
    print(f'\nScored data saved to {out_path.name}')
