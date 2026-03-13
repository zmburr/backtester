"""Data classes for walk-forward validation metrics."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PeriodMetrics:
    """Metrics for a single time period (train, validate, or test)."""

    n: int
    win_rate: float  # percentage, e.g. 65.0
    avg_pnl: float  # percentage, e.g. +12.5
    median_pnl: float = 0.0
    wins: int = 0
    losses: int = 0


@dataclass
class DegradationReport:
    """Comparison of validate vs train performance."""

    win_rate_change_pp: float  # percentage-point change (validate - train)
    avg_pnl_change_pct: float  # relative change: (val - train) / |train| * 100
    verdict: str  # "held", "degraded", or "collapsed"
    fisher_p_value: Optional[float] = None
    go_edge_retained: Optional[float] = None  # validate_avg_pnl / train_avg_pnl * 100


def compute_degradation(train: PeriodMetrics, validate: PeriodMetrics,
                        fisher_p: Optional[float] = None) -> DegradationReport:
    """Compare validate to train and produce a degradation verdict."""
    wr_change = validate.win_rate - train.win_rate
    if abs(train.avg_pnl) > 1e-9:
        pnl_change = (validate.avg_pnl - train.avg_pnl) / abs(train.avg_pnl) * 100
        edge_retained = validate.avg_pnl / train.avg_pnl * 100
    else:
        pnl_change = 0.0
        edge_retained = 0.0

    # Verdict thresholds
    if abs(wr_change) < 5 and pnl_change > -20:
        verdict = "held"
    elif abs(wr_change) < 15 and pnl_change > -50:
        verdict = "degraded"
    else:
        verdict = "collapsed"

    return DegradationReport(
        win_rate_change_pp=wr_change,
        avg_pnl_change_pct=pnl_change,
        verdict=verdict,
        fisher_p_value=fisher_p,
        go_edge_retained=edge_retained,
    )
