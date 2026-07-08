"""Walk-forward validation engine (out-of-sample).

Pipeline per strategy:

  1. Load the curated labeled CSV (reversal_data.csv / bounce_data.csv) — the
     derivation source — AND the unconditioned candidate population
     (reversal_universe / bounce_population) — the measurement source.
  2. Split BOTH by the same date cutoffs (temporal_split).
  3. Derive thresholds from the TRAIN curated trades only
     (validation.threshold_deriver.derive_thresholds) — no look-ahead.
  4. Build derived scorers from those thresholds (with a train-only reference
     distribution injected into ReversalScorer to close the intensity /
     momentum-percentile leak) and the default PRODUCTION scorers.
  5. Re-score the validate- and test-window CANDIDATES with the derived scorers,
     and the validate-window candidates ALSO with production scorers.
  6. Report outcome-conditional PeriodMetrics per window, plus degradation
     (train vs validate, train vs test) and a production-vs-derived comparison
     of the GO-conditional win rate on the validate window.

Outcome definitions (measured on the population, not the curated winners):
  - reversal: outcome column `fade_day_return` = (close-open)/open. A short
    "wins" when fade_day_return < 0. P&L (pts) = -fade_day_return*100.
  - bounce: outcome columns `bounce_day_return` (open-to-close) and
    `bounce_low_to_close`. A bounce "wins" when bounce_low_to_close >= 0.05
    (matches build_bounce_odds semantics — the setup reached a tradeable
    +5% off the low). P&L (pts) = bounce_day_return*100.

Legacy-vs-new metric semantics (IMPORTANT for continuity):
  The four legacy attributes read by research/experiments/walk_forward_sensitivity.py
  — {train,validate,test}_metrics and train_vs_validate — are now computed on the
  candidate POPULATION scored by the DERIVED scorer, NOT on the curated
  winners-only CSV as before. The PeriodMetrics.win_rate is the overall base rate
  of successful setups in each window (~50-55% for reversal), and the
  GO-conditional edge lives in PeriodMetrics.by_recommendation['GO'].
  train_vs_validate / train_vs_test degradation is measured on the GO-CONDITIONAL
  win rate (the GO bucket), because a scorer walk-forward's actual question is
  "does the GO edge hold OOS?" — the whole-population base rate is
  scorer-independent and near-flat across time, so degrading on it is
  uninformative. This is intentional: the old curated-trades measurement was
  tautological (it re-measured a winners-only file); the new measurement is a
  real, non-circular quantity the sensitivity experiment can compare across
  splits. The signature and those attribute names are preserved.

  Caveat: reversal fade returns are small intraday moves (avg |P&L| < ~1.5%),
  so the avg-P&L relative-change term in the verdict is noisy near a zero base.
  Read the win-rate change, Fisher p-value, and GO-vs-NO-GO delta as the signal.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

from validation.metrics import (
    PeriodMetrics,
    DegradationReport,
    build_period_metrics,
    compute_degradation,
    go_bucket_as_metrics,
)
from validation.temporal_split import temporal_split
from validation.threshold_deriver import derive_thresholds, DerivedThresholds

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# Strategy configuration: curated CSV (derivation) + candidate population (measurement).
_STRATEGY_CONFIG = {
    "reversal": {
        "curated_csv": "reversal_data.csv",
        "population_csv": "reversal_universe_2020-01-01_2025-12-31.csv",
    },
    "bounce": {
        "curated_csv": "bounce_data.csv",
        "population_csv": "bounce_population_2022_2026.csv",
    },
    "breakout": {
        "curated_csv": "breakout_data.csv",
        "population_csv": None,  # no scoring system / population
    },
}

ALL_CAPS = ['ETF', 'Large', 'Medium', 'Small', 'Micro']


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class Split:
    train_end: str
    validate_end: str


@dataclass
class Diagnostics:
    total_trades: int = 0
    train_n: int = 0
    validate_n: int = 0
    test_n: int = 0
    train_date_range: str = ''
    validate_date_range: str = ''
    test_date_range: str = ''
    per_cap: Dict[str, Dict[str, int]] = field(default_factory=dict)
    per_setup: Dict[str, Dict[str, int]] = field(default_factory=dict)
    sparse_cells: List[str] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """Output of a single walk-forward run.

    The first four attributes are the legacy compat surface (see module
    docstring); everything below is additive and consumed by validation/report.py.
    """

    # --- Legacy compat (candidate-population, derived-scorer metrics) ---
    train_metrics: PeriodMetrics
    validate_metrics: PeriodMetrics
    test_metrics: Optional[PeriodMetrics]
    train_vs_validate: Optional[DegradationReport]

    # --- Additive ---
    strategy: str = ''
    split: Optional[Split] = None
    diagnostics: Optional[Diagnostics] = None
    derived: Optional[DerivedThresholds] = None
    validate_production_metrics: Optional[PeriodMetrics] = None
    train_vs_test: Optional[DegradationReport] = None
    production_vs_derived_validate: Optional[DegradationReport] = None


# ---------------------------------------------------------------------------
# Grade helpers
# ---------------------------------------------------------------------------

def _bounce_grade(score: int) -> str:
    """Map a 0-6 bounce pre-trade score to a display grade."""
    if score >= 6:
        return 'A+'
    if score == 5:
        return 'A'
    if score == 4:
        return 'B'
    if score == 3:
        return 'C'
    return 'F'


# ---------------------------------------------------------------------------
# Candidate scoring — reversal
# ---------------------------------------------------------------------------

_REVERSAL_METRIC_COLS = [
    'pct_from_9ema', 'prior_day_range_atr', 'rvol_score', 'pct_change_3',
    'gap_pct', 'atr_pct', 'pct_from_50mav', 'pct_change_30',
]


def _score_reversal_candidates(pop: pd.DataFrame, scorer) -> Optional[pd.DataFrame]:
    """Score a reversal candidate population and attach outcome columns.

    Returns a DataFrame with columns: recommendation (pre-trade GO/CAUTION/NO-GO),
    score (0-5 pretrade), grade, pnl (short P&L pts), win (fade worked). Rows with
    a missing outcome are dropped. Returns None if the population is empty.
    """
    if pop is None or len(pop) == 0:
        return None

    if 'fade_day_return' not in pop.columns:
        raise ValueError("reversal population missing 'fade_day_return' outcome column")

    df = pop.dropna(subset=['fade_day_return']).copy()
    if len(df) == 0:
        return None

    recs, scores, grades = [], [], []
    caps = df['cap'].fillna('Medium').tolist()
    setups = df['setup_type'].fillna('generic').tolist() if 'setup_type' in df.columns else ['generic'] * len(df)
    tickers = df['ticker'].tolist() if 'ticker' in df.columns else [''] * len(df)
    dates = df['date'].tolist() if 'date' in df.columns else [''] * len(df)
    fade = df['fade_day_return'].to_numpy(dtype=float)

    # Pull metric columns once as records for a fast scoring loop.
    metric_cols = [c for c in _REVERSAL_METRIC_COLS if c in df.columns]
    records = df[metric_cols].to_dict('records')

    for i, metrics in enumerate(records):
        # Criterion 6 (reversal size) is the outcome; feed it so the full score
        # is coherent, but the GO gate uses the pre-trade recommendation only.
        metrics['reversal_open_close_pct'] = fade[i]
        cap = caps[i]
        setup = setups[i]
        res = scorer.score_setup(tickers[i], dates[i], cap, metrics, setup=setup)
        recs.append(res['pretrade_recommendation'])
        scores.append(res['pretrade_score'])
        grades.append(res['pretrade_grade'])

    out = pd.DataFrame({
        'recommendation': recs,
        'score': scores,
        'grade': grades,
        'pnl': -fade * 100.0,      # short P&L: fade down => positive
        'win': fade < 0,           # short works when the day fades
    })
    return out


# ---------------------------------------------------------------------------
# Candidate scoring — bounce
# ---------------------------------------------------------------------------

_BOUNCE_METRIC_COLS = [
    'selloff_total_pct', 'pct_off_30d_high', 'gap_pct', 'prior_day_range_atr',
    'pct_change_3', 'pct_off_52wk_high', 'pct_from_200mav', 'pct_from_50mav',
    'pct_change_30', 'consecutive_down_days',
]


def _score_bounce_candidates(pop: pd.DataFrame, pretrade) -> Optional[pd.DataFrame]:
    """Score a bounce candidate population and attach outcome columns.

    Win = bounce_low_to_close >= 0.05 (setup reached a tradeable +5% off the low);
    P&L (pts) = bounce_day_return*100 (open-to-close).
    """
    if pop is None or len(pop) == 0:
        return None

    needed = {'bounce_day_return', 'bounce_low_to_close'}
    missing = needed - set(pop.columns)
    if missing:
        raise ValueError(f"bounce population missing outcome columns: {missing}")

    df = pop.dropna(subset=['bounce_day_return', 'bounce_low_to_close']).copy()
    if len(df) == 0:
        return None

    caps = df['cap'].fillna('Medium').tolist()
    tickers = df['ticker'].tolist() if 'ticker' in df.columns else [''] * len(df)
    day_return = df['bounce_day_return'].to_numpy(dtype=float)
    low_to_close = df['bounce_low_to_close'].to_numpy(dtype=float)

    metric_cols = [c for c in _BOUNCE_METRIC_COLS if c in df.columns]
    records = df[metric_cols].to_dict('records')

    recs, scores, grades = [], [], []
    for i, metrics in enumerate(records):
        res = pretrade.validate(tickers[i], metrics, cap=caps[i])
        recs.append(res.recommendation)
        scores.append(res.score)
        grades.append(_bounce_grade(res.score))

    out = pd.DataFrame({
        'recommendation': recs,
        'score': scores,
        'grade': grades,
        'pnl': day_return * 100.0,
        'win': low_to_close >= 0.05,
    })
    return out


def _metrics_from_scored(scored: Optional[pd.DataFrame]) -> PeriodMetrics:
    """Build PeriodMetrics from a scored candidate frame (or empty metrics)."""
    if scored is None or len(scored) == 0:
        return PeriodMetrics(n=0, win_rate=0.0, avg_pnl=0.0)
    return build_period_metrics(
        pnl=scored['pnl'].to_numpy(),
        wins=scored['win'].to_numpy(),
        recommendations=scored['recommendation'].tolist(),
        grades=scored['grade'].tolist(),
        scores=scored['score'].to_numpy(),
    )


# ---------------------------------------------------------------------------
# Derived scorer construction
# ---------------------------------------------------------------------------

def _build_derived_scorers(strategy: str, derived: DerivedThresholds):
    """Return (derived_scorer, production_scorer) for the strategy.

    For reversal the scorer is the generic ReversalScorer (matches the universe
    backscanner's recommendation column). For bounce it is BouncePretrade.
    """
    if strategy == 'reversal':
        from analyzers.reversal_scorer import ReversalScorer
        derived_scorer = ReversalScorer(
            thresholds=derived.reversal_cap_thresholds,
            readiness_thresholds=derived.reversal_readiness_thresholds,
            ref_by_cap_group=derived.reversal_ref_by_cap_group,
        )
        production_scorer = ReversalScorer()  # defaults = production
        return derived_scorer, production_scorer

    if strategy == 'bounce':
        from analyzers.bounce_scorer import BouncePretrade
        derived_scorer = BouncePretrade(profiles=derived.bounce_setup_profiles)
        production_scorer = BouncePretrade()
        return derived_scorer, production_scorer

    return None, None


def _score(strategy: str, pop: pd.DataFrame, scorer) -> Optional[pd.DataFrame]:
    if strategy == 'reversal':
        return _score_reversal_candidates(pop, scorer)
    if strategy == 'bounce':
        return _score_bounce_candidates(pop, scorer)
    return None


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _date_range(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0 or 'date' not in df.columns:
        return 'n/a'
    d = pd.to_datetime(df['date'], format='mixed', dayfirst=False, errors='coerce').dropna()
    if len(d) == 0:
        return 'n/a'
    return f"{d.min().date()} .. {d.max().date()}"


def _build_diagnostics(strategy: str, split: Split,
                       train_c: pd.DataFrame, validate_c: pd.DataFrame,
                       test_c: pd.DataFrame, derived: DerivedThresholds) -> Diagnostics:
    """Diagnostics from the curated (labeled) split — these are the derivation trades."""
    diag = Diagnostics(
        total_trades=len(train_c) + len(validate_c) + len(test_c),
        train_n=len(train_c),
        validate_n=len(validate_c),
        test_n=len(test_c),
        train_date_range=_date_range(train_c),
        validate_date_range=_date_range(validate_c),
        test_date_range=_date_range(test_c),
    )

    windows = [('train', train_c), ('validate', validate_c), ('test', test_c)]

    if all('cap' in d.columns for _, d in windows):
        per_cap: Dict[str, Dict[str, int]] = {}
        for label, d in windows:
            for cap, cnt in d['cap'].fillna('Unknown').value_counts().items():
                per_cap.setdefault(cap, {})[label] = int(cnt)
        diag.per_cap = per_cap

    setup_col = 'setup' if strategy == 'reversal' else ('Setup' if any('Setup' in d.columns for _, d in windows) else 'setup')
    if all(setup_col in d.columns for _, d in windows):
        per_setup: Dict[str, Dict[str, int]] = {}
        for label, d in windows:
            for stp, cnt in d[setup_col].fillna('Unknown').value_counts().items():
                per_setup.setdefault(str(stp), {})[label] = int(cnt)
        diag.per_setup = per_setup

    # Sparse-cell warnings come from the derivation's cap-pooling decisions.
    if derived is not None and derived.cap_pooling_log:
        diag.sparse_cells = [m.strip() for m in derived.cap_pooling_log]

    return diag


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _load_population(strategy: str, override: Optional[str]) -> Optional[pd.DataFrame]:
    scfg = _STRATEGY_CONFIG[strategy]
    if override:
        path = override
    elif scfg.get("population_csv"):
        path = os.path.join(_DATA_DIR, scfg["population_csv"])
    else:
        return None
    if not os.path.exists(path):
        logger.warning(f"Population CSV not found for {strategy}: {path}")
        return None
    df = pd.read_csv(path).dropna(subset=["ticker", "date"])
    logger.info(f"Loaded {len(df)} candidates from {path}")
    return df


def run_walk_forward(
    strategy: str,
    train_end: str,
    validate_end: str,
    csv_path: str = None,
    population_path: str = None,
) -> WalkForwardResult:
    """Run a single walk-forward validation pass.

    Args:
        strategy: "reversal", "bounce", or "breakout".
        train_end: Last date of the training period (inclusive).
        validate_end: Last date of the validation period (inclusive).
        csv_path: Override path to the CURATED CSV (derivation source). If None,
            uses data/<curated_csv>. (This is the parameter the sensitivity
            experiment passes; kept for backward compatibility.)
        population_path: Override path to the candidate population (measurement
            source). If None, uses data/<population_csv>.

    Returns:
        WalkForwardResult (see class docstring for the legacy-vs-additive split).
    """
    if strategy not in _STRATEGY_CONFIG:
        raise ValueError(f"Unknown strategy: {strategy}. Must be one of {list(_STRATEGY_CONFIG.keys())}")

    split = Split(train_end=train_end, validate_end=validate_end)

    # --- Load curated (derivation) trades ---
    scfg = _STRATEGY_CONFIG[strategy]
    curated_path = csv_path or os.path.join(_DATA_DIR, scfg["curated_csv"])
    if not os.path.exists(curated_path):
        raise FileNotFoundError(curated_path)
    curated = pd.read_csv(curated_path).dropna(subset=["ticker", "date"])
    logger.info(f"Loaded {len(curated)} curated trades from {curated_path}")

    train_c, validate_c, test_c = temporal_split(curated, train_end, validate_end)
    logger.info(f"Curated split: train={len(train_c)}, validate={len(validate_c)}, test={len(test_c)}")

    # --- Derive thresholds on TRAIN curated only ---
    derived = derive_thresholds(train_c, strategy)

    diagnostics = _build_diagnostics(strategy, split, train_c, validate_c, test_c, derived)

    # Breakout has no scoring system / population — measure base rates on curated.
    if strategy == 'breakout' or scfg.get("population_csv") is None:
        return _breakout_result(strategy, split, curated, train_end, validate_end,
                                derived, diagnostics)

    # --- Load candidate population (measurement) and split by the same cutoffs ---
    population = _load_population(strategy, population_path)
    if population is None:
        # No population available: fall back to empty metrics but keep structure.
        empty = PeriodMetrics(n=0, win_rate=0.0, avg_pnl=0.0)
        return WalkForwardResult(
            train_metrics=empty, validate_metrics=empty, test_metrics=None,
            train_vs_validate=None, strategy=strategy, split=split,
            diagnostics=diagnostics, derived=derived,
            validate_production_metrics=empty,
        )

    train_p, validate_p, test_p = temporal_split(population, train_end, validate_end)
    logger.info(f"Population split: train={len(train_p)}, validate={len(validate_p)}, test={len(test_p)}")

    derived_scorer, production_scorer = _build_derived_scorers(strategy, derived)

    # --- Score candidates with the DERIVED scorer ---
    train_scored = _score(strategy, train_p, derived_scorer)
    validate_scored = _score(strategy, validate_p, derived_scorer)
    test_scored = _score(strategy, test_p, derived_scorer)

    train_m = _metrics_from_scored(train_scored)
    validate_m = _metrics_from_scored(validate_scored)
    test_m = _metrics_from_scored(test_scored)

    # --- Score the SAME validate candidates with the PRODUCTION scorer ---
    validate_prod_scored = _score(strategy, validate_p, production_scorer)
    validate_prod_m = _metrics_from_scored(validate_prod_scored)

    # --- Degradation (base = train), measured on the GO-CONDITIONAL win rate ---
    # A scorer walk-forward asks "does the GO edge hold out-of-sample?", so
    # degradation compares the GO bucket, not the whole-population base rate
    # (which is scorer-independent and ~flat across time). The full-population
    # base rates still live in the PeriodMetrics objects for report Sections 3/6.
    train_go = go_bucket_as_metrics(train_m)
    validate_go = go_bucket_as_metrics(validate_m)
    test_go = go_bucket_as_metrics(test_m)

    train_vs_validate = None
    if train_go is not None and validate_go is not None:
        train_vs_validate = compute_degradation(train_go, validate_go)
    train_vs_test = None
    if train_go is not None and test_go is not None:
        train_vs_test = compute_degradation(train_go, test_go)

    # --- Production vs derived, on GO-conditional win rate (the whole point) ---
    production_vs_derived_validate = None
    prod_go = go_bucket_as_metrics(validate_prod_m)
    derived_go = go_bucket_as_metrics(validate_m)
    if prod_go is not None and derived_go is not None:
        production_vs_derived_validate = compute_degradation(prod_go, derived_go)

    return WalkForwardResult(
        train_metrics=train_m,
        validate_metrics=validate_m,
        test_metrics=test_m if test_m.n > 0 else None,
        train_vs_validate=train_vs_validate,
        strategy=strategy,
        split=split,
        diagnostics=diagnostics,
        derived=derived,
        validate_production_metrics=validate_prod_m,
        train_vs_test=train_vs_test,
        production_vs_derived_validate=production_vs_derived_validate,
    )


def _breakout_result(strategy, split, curated, train_end, validate_end,
                     derived, diagnostics) -> WalkForwardResult:
    """Breakout has no scorer/population: measure base rates on curated trades."""
    train_c, validate_c, test_c = temporal_split(curated, train_end, validate_end)

    def _base(df: pd.DataFrame) -> PeriodMetrics:
        col = 'reversal_open_close_pct'  # not present for breakout; degrade gracefully
        if col not in df.columns or len(df) == 0:
            return PeriodMetrics(n=len(df), win_rate=0.0, avg_pnl=0.0)
        pnl = df[col].to_numpy(dtype=float) * 100.0
        return build_period_metrics(pnl=pnl, wins=pnl > 0)

    train_m, validate_m, test_m = _base(train_c), _base(validate_c), _base(test_c)
    deg = compute_degradation(train_m, validate_m) if train_m.n and validate_m.n else None
    return WalkForwardResult(
        train_metrics=train_m, validate_metrics=validate_m,
        test_metrics=test_m if test_m.n > 0 else None,
        train_vs_validate=deg, strategy=strategy, split=split,
        diagnostics=diagnostics, derived=derived,
        validate_production_metrics=None,
    )
