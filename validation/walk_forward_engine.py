"""Walk-forward validation engine.

Splits trade data temporally, computes per-period metrics, and measures
out-of-sample degradation.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from validation.metrics import PeriodMetrics, DegradationReport, compute_degradation
from validation.temporal_split import temporal_split

try:
    from scipy.stats import fisher_exact
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)

# Strategy-specific configuration (mirrors filter_sweep.STRATEGY_CONFIG)
_STRATEGY_CONFIG = {
    "reversal": {
        "csv_name": "reversal_data.csv",
        "target_col": "reversal_open_close_pct",
        "pnl_sign": -1,
    },
    "bounce": {
        "csv_name": "bounce_data.csv",
        "target_col": "bounce_open_close_pct",
        "pnl_sign": 1,
    },
}


@dataclass
class WalkForwardResult:
    """Output of a single walk-forward run."""

    train_metrics: PeriodMetrics
    validate_metrics: PeriodMetrics
    test_metrics: Optional[PeriodMetrics]
    train_vs_validate: Optional[DegradationReport]


def _compute_period_metrics(df: pd.DataFrame, target_col: str, pnl_sign: int) -> PeriodMetrics:
    """Compute PeriodMetrics for a subset of trades."""
    if len(df) == 0:
        return PeriodMetrics(n=0, win_rate=0.0, avg_pnl=0.0, median_pnl=0.0, wins=0, losses=0)

    pnl = df[target_col] * pnl_sign * 100  # convert to percentage
    wins = int((pnl > 0).sum())
    losses = int((pnl <= 0).sum())

    return PeriodMetrics(
        n=len(df),
        win_rate=round(wins / len(df) * 100, 1),
        avg_pnl=round(pnl.mean(), 2),
        median_pnl=round(pnl.median(), 2),
        wins=wins,
        losses=losses,
    )


def _fisher_p(train: PeriodMetrics, validate: PeriodMetrics) -> Optional[float]:
    """Compute Fisher exact test p-value comparing win rates."""
    if not HAS_SCIPY or train.n == 0 or validate.n == 0:
        return None
    try:
        table = [
            [validate.wins, validate.losses],
            [train.wins, train.losses],
        ]
        _, p = fisher_exact(table)
        return p
    except Exception:
        return None


def run_walk_forward(
    strategy: str,
    train_end: str,
    validate_end: str,
    csv_path: str = None,
) -> WalkForwardResult:
    """Run a single walk-forward validation pass.

    Args:
        strategy: "reversal" or "bounce".
        train_end: Last date for the training period (inclusive).
        validate_end: Last date for the validation period (inclusive).
        csv_path: Path to the CSV file. Required.

    Returns:
        WalkForwardResult with train, validate, test metrics and degradation report.
    """
    if strategy not in _STRATEGY_CONFIG:
        raise ValueError(f"Unknown strategy: {strategy}. Must be one of {list(_STRATEGY_CONFIG.keys())}")

    scfg = _STRATEGY_CONFIG[strategy]
    target_col = scfg["target_col"]
    pnl_sign = scfg["pnl_sign"]

    if csv_path is None:
        raise ValueError("csv_path is required — pass it from the experiment config")

    df = pd.read_csv(str(csv_path)).dropna(subset=["ticker", "date"])
    logger.info(f"Loaded {len(df)} trades from {csv_path}")

    train_df, validate_df, test_df = temporal_split(df, train_end, validate_end)
    logger.info(f"Split: train={len(train_df)}, validate={len(validate_df)}, test={len(test_df)}")

    train_m = _compute_period_metrics(train_df, target_col, pnl_sign)
    validate_m = _compute_period_metrics(validate_df, target_col, pnl_sign)
    test_m = _compute_period_metrics(test_df, target_col, pnl_sign) if len(test_df) > 0 else None

    # Degradation report
    degradation = None
    if train_m.n > 0 and validate_m.n > 0:
        p_value = _fisher_p(train_m, validate_m)
        degradation = compute_degradation(train_m, validate_m, fisher_p=p_value)

    return WalkForwardResult(
        train_metrics=train_m,
        validate_metrics=validate_m,
        test_metrics=test_m,
        train_vs_validate=degradation,
    )
