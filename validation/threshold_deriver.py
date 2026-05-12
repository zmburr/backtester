"""Re-derive scoring thresholds from training data only.

This is the core walk-forward module. It mirrors the logic in
compute_reversal_stats.py and bounce_scorer.py to derive thresholds,
but restricted to a training subset to prevent look-ahead bias.
"""

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional
from copy import deepcopy

from analyzers.reversal_scorer import CriteriaThresholds, _CAP_GROUPS, _REVERSAL_INTENSITY_SPEC, EUPHORIC_SETUPS
from analyzers.reversal_pretrade import ReversalSetupProfile
from analyzers.bounce_scorer import SetupProfile, classify_stock, classify_from_setup_column

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MIN_TRADES_PER_CELL = 5

# Cap fallback hierarchy when a cap has too few training trades.
# Micro → Small → Medium (ascending liquidity)
# ETF → Large → Medium (ascending volatility)
_CAP_FALLBACK = {
    'Micro': ['Small', 'Medium'],
    'Small': ['Medium'],
    'Medium': [],
    'Large': ['Medium'],
    'ETF': ['Large', 'Medium'],
}

ALL_CAPS = ['ETF', 'Large', 'Medium', 'Small', 'Micro']


@dataclass
class DerivedThresholds:
    """All thresholds derived from training data for a single strategy."""
    strategy: str

    # Reversal-specific
    reversal_cap_thresholds: Optional[Dict[str, CriteriaThresholds]] = None
    reversal_setup_profiles: Optional[Dict[str, ReversalSetupProfile]] = None
    reversal_ref_by_cap_group: Optional[Dict[str, pd.DataFrame]] = None
    reversal_readiness_thresholds: Optional[Dict[str, float]] = None

    # Bounce-specific
    bounce_setup_profiles: Optional[Dict[str, SetupProfile]] = None

    # Metadata
    derivation_log: list = field(default_factory=list)
    cap_pooling_log: list = field(default_factory=list)


def _pool_cap_data(df: pd.DataFrame, target_cap: str, cap_col: str = 'cap') -> pd.DataFrame:
    """Pool data from adjacent caps when target cap has too few trades.

    Returns pooled DataFrame and logs the pooling decision.
    """
    target_data = df[df[cap_col] == target_cap]
    if len(target_data) >= MIN_TRADES_PER_CELL:
        return target_data

    fallbacks = _CAP_FALLBACK.get(target_cap, [])
    pooled = target_data.copy()

    for fallback_cap in fallbacks:
        fallback_data = df[df[cap_col] == fallback_cap]
        pooled = pd.concat([pooled, fallback_data], ignore_index=True)
        if len(pooled) >= MIN_TRADES_PER_CELL:
            break

    return pooled


def _safe_quantile(series: pd.Series, q: float, min_n: int = 2) -> Optional[float]:
    """Compute quantile only if enough data points."""
    vals = series.dropna()
    if len(vals) < min_n:
        return None
    return float(vals.quantile(q))


# ---------------------------------------------------------------------------
# Reversal threshold derivation
# ---------------------------------------------------------------------------

def derive_reversal_thresholds(train_df: pd.DataFrame, log: list) -> DerivedThresholds:
    """
    Derive reversal thresholds from training data.

    Mirrors the logic in compute_reversal_stats.py:
    1. Filter to Grade A+B trades
    2. For each cap: compute p25 of each criterion → CriteriaThresholds
    3. For 3DGapFade: compute per-cap p25 → ReversalSetupProfile
    4. Build intensity reference from Grade A trades
    """
    result = DerivedThresholds(strategy='reversal')
    result.derivation_log = log

    # --- 1. Generic ReversalScorer thresholds (CAP_THRESHOLDS) ---
    ab = train_df[train_df['trade_grade'].isin(['A', 'B'])].copy()
    log.append(f"Reversal training: {len(train_df)} total, {len(ab)} A+B trades")

    cap_thresholds = {}
    criteria_cols = {
        'pct_from_9ema': 'pct_from_9ema',
        'prior_day_range_atr': 'prior_day_range_atr',
        'rvol_score': 'rvol_score',
        'pct_change_3': 'pct_change_3',
        'gap_pct': 'gap_pct',
        'reversal_pct': 'reversal_open_close_pct',
    }

    for cap in ALL_CAPS:
        cap_data = _pool_cap_data(ab, cap)
        if len(cap_data) < 2:
            log.append(f"  {cap}: insufficient data ({len(cap_data)} trades), skipping")
            continue

        pooled_from = cap_data['cap'].unique().tolist()
        if len(pooled_from) > 1 or (len(pooled_from) == 1 and pooled_from[0] != cap):
            msg = f"  {cap}: pooled with {pooled_from} ({len(cap_data)} trades)"
            log.append(msg)
            result.cap_pooling_log.append(msg)

        thresholds = {}
        for crit_name, col in criteria_cols.items():
            if col not in cap_data.columns:
                thresholds[crit_name] = 0.0
                continue
            vals = cap_data[col].dropna()
            if crit_name == 'reversal_pct':
                # Reversal is negative (short trade), p25 is the "smaller" reversal
                p25 = _safe_quantile(vals, 0.75)  # p75 of negative = lenient
                thresholds[crit_name] = p25 if p25 is not None else -0.05
            else:
                p25 = _safe_quantile(vals, 0.25)
                thresholds[crit_name] = p25 if p25 is not None else 0.0

        cap_thresholds[cap] = CriteriaThresholds(
            pct_from_9ema=thresholds.get('pct_from_9ema', 0.0),
            prior_day_range_atr=thresholds.get('prior_day_range_atr', 0.0),
            rvol_score=thresholds.get('rvol_score', 0.0),
            pct_change_3=thresholds.get('pct_change_3', 0.0),
            gap_pct=thresholds.get('gap_pct', 0.0),
            reversal_pct=thresholds.get('reversal_pct', -0.05),
        )

    result.reversal_cap_thresholds = cap_thresholds

    # --- 2. 3DGapFade profile (ReversalSetupProfile) ---
    gf = train_df[train_df['setup'] == '3DGapFade'].copy() if 'setup' in train_df.columns else pd.DataFrame()
    gf_ab = gf[gf['trade_grade'].isin(['A', 'B'])] if len(gf) > 0 else pd.DataFrame()
    log.append(f"3DGapFade training: {len(gf)} total, {len(gf_ab)} A+B")

    pretrade_criteria = ['pct_from_9ema', 'prior_day_range_atr', 'rvol_score', 'pct_change_3', 'gap_pct']

    profile_thresholds = {crit: {} for crit in pretrade_criteria}

    for cap in ALL_CAPS:
        cap_data = _pool_cap_data(gf_ab, cap) if len(gf_ab) > 0 else pd.DataFrame()
        if len(cap_data) < 2:
            # Use the generic cap threshold as fallback
            if cap in cap_thresholds:
                for crit in pretrade_criteria:
                    profile_thresholds[crit][cap] = getattr(cap_thresholds[cap], crit)
            continue

        pooled_from = cap_data['cap'].unique().tolist()
        if len(pooled_from) > 1 or (len(pooled_from) == 1 and pooled_from[0] != cap):
            msg = f"  3DGapFade {cap}: pooled with {pooled_from} ({len(cap_data)} trades)"
            result.cap_pooling_log.append(msg)

        for crit in pretrade_criteria:
            if crit in cap_data.columns:
                p25 = _safe_quantile(cap_data[crit], 0.25)
                profile_thresholds[crit][cap] = p25 if p25 is not None else 0.0
            else:
                profile_thresholds[crit][cap] = 0.0

    # Add _default as Medium value
    for crit in pretrade_criteria:
        if 'Medium' in profile_thresholds[crit]:
            profile_thresholds[crit]['_default'] = profile_thresholds[crit]['Medium']

    # Compute historical win rate and avg P&L from training 3DGapFade A+B
    gf_ab_pnl = -gf_ab['reversal_open_close_pct'] * 100 if len(gf_ab) > 0 and 'reversal_open_close_pct' in gf_ab.columns else pd.Series(dtype=float)
    hist_wr = float((gf_ab_pnl > 0).mean()) if len(gf_ab_pnl) > 0 else 0.0
    hist_avg = float(gf_ab_pnl.mean()) if len(gf_ab_pnl) > 0 else 0.0

    derived_profile = ReversalSetupProfile(
        name='3DGapFade',
        description='2+ euphoric up days + gap up on fade day (training-derived)',
        sample_size=len(gf_ab),
        historical_win_rate=hist_wr,
        historical_avg_pnl=hist_avg,
        pct_from_9ema=profile_thresholds['pct_from_9ema'],
        prior_day_range_atr=profile_thresholds['prior_day_range_atr'],
        rvol_score=profile_thresholds['rvol_score'],
        pct_change_3=profile_thresholds['pct_change_3'],
        gap_pct=profile_thresholds['gap_pct'],
    )
    result.reversal_setup_profiles = {'3DGapFade': derived_profile}

    # --- 3. Intensity reference DataFrames from Grade A training trades ---
    grade_a = train_df[train_df['trade_grade'] == 'A'].copy()
    if len(grade_a) > 0:
        ref_atr = grade_a['atr_pct'].replace(0, np.nan)
        grade_a['ema9_atr'] = grade_a['pct_from_9ema'] / ref_atr
        grade_a['mom3_atr'] = grade_a['pct_change_3'] / ref_atr
        grade_a['gap_atr'] = grade_a['gap_pct'] / ref_atr
        grade_a['cap_group'] = grade_a['cap'].map(_CAP_GROUPS).fillna('large')
        result.reversal_ref_by_cap_group = {g: sub for g, sub in grade_a.groupby('cap_group')}
    else:
        result.reversal_ref_by_cap_group = {}

    # --- 4. Readiness thresholds for euphoric setups ---
    READINESS_FLOOR = 0.03
    euphoric_ab = ab[ab['setup'].isin(EUPHORIC_SETUPS)] if 'setup' in ab.columns else pd.DataFrame()
    log.append(f"Euphoric A+B for readiness: {len(euphoric_ab)} trades")

    readiness = {}
    for cap in ALL_CAPS:
        cap_data = _pool_cap_data(euphoric_ab, cap) if len(euphoric_ab) > 0 else pd.DataFrame()
        if len(cap_data) >= 2 and 'pct_change_3' in cap_data.columns:
            p25 = _safe_quantile(cap_data['pct_change_3'], 0.25)
            readiness[cap] = max(p25, READINESS_FLOOR) if p25 is not None else READINESS_FLOOR
        else:
            readiness[cap] = READINESS_FLOOR

    log.append(f"Readiness thresholds: {', '.join(f'{k}={v*100:.1f}%' for k, v in readiness.items())}")
    result.reversal_readiness_thresholds = readiness

    return result


# ---------------------------------------------------------------------------
# Bounce threshold derivation
# ---------------------------------------------------------------------------

def derive_bounce_thresholds(train_df: pd.DataFrame, log: list) -> DerivedThresholds:
    """
    Derive bounce thresholds from training data.

    Mirrors bounce_scorer.py derivation:
    1. Classify each trade by setup type (weakstock/strongstock)
    2. For each setup_type + cap: compute p75 for <= criteria, p25 for >= criteria
    3. Compute reference medians from Grade A trades
    """
    result = DerivedThresholds(strategy='bounce')
    result.derivation_log = log

    # Classify trades
    train = train_df.copy()
    setup_col = 'Setup' if 'Setup' in train.columns else 'setup'
    if setup_col in train.columns:
        train['_setup_type'] = train[setup_col].apply(classify_from_setup_column)
    else:
        # Fallback: classify from metrics
        train['_setup_type'] = train.apply(
            lambda row: classify_stock(row.to_dict())[0], axis=1
        )

    # Exclude IntradayCapitch (different pattern entirely)
    train = train[train['_setup_type'] != 'IntradayCapitch'].copy()
    log.append(f"Bounce training: {len(train)} GapFade trades (excl. IntradayCapitch)")

    # Criteria and their directions (lte = lower is better, gte = higher is better)
    criteria_spec = {
        'selloff_total_pct': ('selloff_total_pct', 'lte'),  # p75 (lenient for <= criteria)
        'pct_off_30d_high': ('pct_off_30d_high', 'lte'),
        'gap_pct': ('gap_pct', 'lte'),
        'prior_day_range_atr': ('one_day_before_range_pct', 'gte'),  # p25 for >= criteria
        'pct_change_3': ('pct_change_3', 'lte'),
        'pct_off_52wk_high': ('pct_off_52wk_high', 'lte'),
        'bounce_pct': ('bounce_open_close_pct', 'gte'),
    }

    # Reference median columns
    ref_median_cols = ['selloff_total_pct', 'pct_off_30d_high', 'gap_pct',
                       'one_day_before_range_pct', 'pct_change_3', 'pct_off_52wk_high',
                       'bounce_open_close_pct']

    setup_profiles = {}

    for setup_type in ['GapFade_weakstock', 'GapFade_strongstock']:
        setup_data = train[train['_setup_type'] == setup_type].copy()
        grade_a = setup_data[setup_data['trade_grade'] == 'A']
        grade_ab = setup_data[setup_data['trade_grade'].isin(['A', 'B'])]

        log.append(f"\n  {setup_type}: {len(setup_data)} total, {len(grade_a)} A, {len(grade_ab)} A+B")

        # Compute per-cap thresholds from A+B (or A if AB is too small)
        threshold_source = grade_ab if len(grade_ab) >= MIN_TRADES_PER_CELL else grade_a

        cap_thresholds = {crit: {} for crit in criteria_spec}

        for cap in ALL_CAPS:
            cap_data = _pool_cap_data(threshold_source, cap)
            if len(cap_data) < 2:
                log.append(f"    {cap}: insufficient data, using _default")
                continue

            pooled_from = cap_data['cap'].unique().tolist()
            if len(pooled_from) > 1 or (len(pooled_from) == 1 and pooled_from[0] != cap):
                msg = f"    {setup_type} {cap}: pooled with {pooled_from} ({len(cap_data)} trades)"
                result.cap_pooling_log.append(msg)

            for crit_name, (col, direction) in criteria_spec.items():
                if col not in cap_data.columns:
                    continue
                vals = cap_data[col].dropna()
                if len(vals) < 2:
                    continue
                if direction == 'lte':
                    # p75 for <= criteria (lenient: 75% of winners pass)
                    cap_thresholds[crit_name][cap] = float(vals.quantile(0.75))
                else:
                    # p25 for >= criteria (lenient: 75% of winners pass)
                    cap_thresholds[crit_name][cap] = float(vals.quantile(0.25))

        # Set _default as Medium or first available
        for crit_name in cap_thresholds:
            d = cap_thresholds[crit_name]
            if 'Medium' in d:
                d['_default'] = d['Medium']
            elif d:
                d['_default'] = next(iter(d.values()))

        # Compute historical stats from training A-grade
        pnl_col = 'bounce_open_close_pct'
        if len(grade_a) > 0 and pnl_col in grade_a.columns:
            pnl_vals = grade_a[pnl_col].dropna() * 100
            hist_wr = float((pnl_vals > 0).mean())
            hist_avg = float(pnl_vals.mean())
            sample_n = len(grade_a)
        else:
            hist_wr = 0.0
            hist_avg = 0.0
            sample_n = 0

        # Compute reference medians from A-grade trades
        ref_medians = {}
        ref_source = grade_a if len(grade_a) >= 3 else threshold_source
        crit_to_ref_col = {
            'selloff_total_pct': 'selloff_total_pct',
            'pct_off_30d_high': 'pct_off_30d_high',
            'gap_pct': 'gap_pct',
            'prior_day_range_atr': 'one_day_before_range_pct',
            'pct_change_3': 'pct_change_3',
            'pct_off_52wk_high': 'pct_off_52wk_high',
            'bounce_pct': 'bounce_open_close_pct',
        }
        for crit_name, col in crit_to_ref_col.items():
            if col in ref_source.columns:
                med = _safe_quantile(ref_source[col], 0.5)
                if med is not None:
                    ref_medians[crit_name] = round(med, 3)

        profile = SetupProfile(
            name=setup_type,
            description=f'{setup_type} (training-derived)',
            sample_size=sample_n,
            historical_win_rate=hist_wr,
            historical_avg_pnl=hist_avg,
            selloff_total_pct=cap_thresholds.get('selloff_total_pct', {}),
            pct_off_30d_high=cap_thresholds.get('pct_off_30d_high', {}),
            gap_pct=cap_thresholds.get('gap_pct', {}),
            prior_day_range_atr=cap_thresholds.get('prior_day_range_atr', {}),
            pct_change_3=cap_thresholds.get('pct_change_3', {}),
            pct_off_52wk_high=cap_thresholds.get('pct_off_52wk_high', {}),
            bounce_pct=cap_thresholds.get('bounce_pct', {}).get('_default', 0.02),
            reference_medians=ref_medians,
        )
        setup_profiles[setup_type] = profile

    result.bounce_setup_profiles = setup_profiles
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def derive_thresholds(train_df: pd.DataFrame, strategy: str) -> DerivedThresholds:
    """
    Derive all thresholds for a strategy from training data.

    Args:
        train_df: Training period DataFrame.
        strategy: 'reversal', 'bounce', or 'breakout'.

    Returns:
        DerivedThresholds with all necessary threshold structures.
    """
    log = []

    if strategy == 'reversal':
        return derive_reversal_thresholds(train_df, log)
    elif strategy == 'bounce':
        return derive_bounce_thresholds(train_df, log)
    elif strategy == 'breakout':
        # Breakout has no dedicated scoring system — return empty
        result = DerivedThresholds(strategy='breakout')
        result.derivation_log = [f"Breakout: no scoring system, {len(train_df)} training trades"]
        return result
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
