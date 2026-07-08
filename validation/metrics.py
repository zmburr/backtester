"""Data classes and builders for walk-forward validation metrics.

PeriodMetrics carries the outcome-conditional statistics for one time window
(train / validate / test) of a candidate population scored by one scorer:
an overall win rate/avg P&L plus bootstrap CIs, a per-grade and
per-recommendation breakdown (this is where the GO-conditional numbers live),
the GO-vs-NO-GO win-rate delta, and the score/P&L rank correlation.

DegradationReport compares two PeriodMetrics (train vs validate, train vs test,
or production vs derived on the same window).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import numpy as np

try:
    from scipy.stats import spearmanr, fisher_exact
    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    HAS_SCIPY = False

from analyzers.bootstrap import (
    BootstrapResult,
    bootstrap_win_rate,
    bootstrap_mean_pnl,
)

# Recommendation buckets we always try to surface (in report order).
RECOMMENDATION_BUCKETS = ['GO', 'CAUTION', 'NO-GO', 'VETO']

# Minimum GO-bucket sample size for a degradation verdict to be trustworthy.
# A GO bucket with 1-9 trades produces a win rate too noisy to compare across
# windows, so go_bucket_as_metrics treats n < MIN_GO_N like an empty bucket
# (returns None -> the caller routes into the no_data path).
MIN_GO_N = 10


@dataclass
class PeriodMetrics:
    """Metrics for a single time period (train, validate, or test)."""

    n: int
    win_rate: float  # percentage, e.g. 65.0
    avg_pnl: float  # percentage, e.g. +12.5
    median_pnl: float = 0.0
    wins: int = 0
    losses: int = 0

    # Bootstrap confidence intervals (None when sample is empty).
    win_rate_ci: Optional[BootstrapResult] = None
    avg_pnl_ci: Optional[BootstrapResult] = None

    # Profit factor: gross win P&L / gross loss P&L.
    profit_factor: float = 0.0

    # Outcome-conditional breakdowns: {label: {n, wins, losses, win_rate, avg_pnl}}.
    by_grade: Dict[str, dict] = field(default_factory=dict)
    by_recommendation: Dict[str, dict] = field(default_factory=dict)

    # GO win rate minus NO-GO win rate (percentage points). None if a side is empty.
    go_nogo_wr_delta: Optional[float] = None

    # Spearman rank correlation of derived score vs P&L (classification power).
    score_pnl_correlation: Optional[float] = None
    score_pnl_pvalue: Optional[float] = None


@dataclass
class DegradationReport:
    """Comparison of a validate/test period vs a baseline (train, or production)."""

    win_rate_change_pp: float  # percentage-point change (compare - base)
    avg_pnl_change_pct: float  # relative change: (compare - base) / |base| * 100
    verdict: str  # "held", "degraded", or "collapsed"
    fisher_p_value: Optional[float] = None
    fisher_significant: Optional[bool] = None
    go_edge_retained: Optional[float] = None  # compare_avg_pnl / base_avg_pnl * 100


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _profit_factor(pnl: np.ndarray) -> float:
    """Gross winning P&L divided by absolute gross losing P&L."""
    gains = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    return float(gains / abs(losses))


def _bucket_stats(pnl: np.ndarray, wins: np.ndarray) -> dict:
    """Summary stats for one grade/recommendation bucket."""
    n = int(len(pnl))
    if n == 0:
        return {'n': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0, 'avg_pnl': 0.0}
    n_wins = int(wins.sum())
    return {
        'n': n,
        'wins': n_wins,
        'losses': n - n_wins,
        'win_rate': round(float(wins.mean()) * 100, 1),
        'avg_pnl': round(float(pnl.mean()), 2),
    }


def build_period_metrics(
    pnl: Sequence[float],
    wins: Sequence[bool],
    recommendations: Optional[Sequence[str]] = None,
    grades: Optional[Sequence[str]] = None,
    scores: Optional[Sequence[float]] = None,
) -> PeriodMetrics:
    """Assemble a full PeriodMetrics from per-candidate arrays.

    Args:
        pnl: signed P&L per candidate (percentage points, e.g. +12.5).
        wins: boolean win flag per candidate. This is the outcome definition
            (short worked / bounce reached target); it is NOT derived from pnl,
            because a bounce "win" is a low-to-close threshold, not pnl > 0.
        recommendations: derived scorer recommendation per candidate
            (GO / CAUTION / NO-GO / VETO). Enables by_recommendation + GO/NO-GO delta.
        grades: derived grade per candidate. Enables by_grade.
        scores: derived numeric score per candidate. Enables score/P&L correlation.
    """
    pnl = np.asarray(pnl, dtype=float)
    wins = np.asarray(wins, dtype=bool)
    n = int(len(pnl))

    if n == 0:
        return PeriodMetrics(n=0, win_rate=0.0, avg_pnl=0.0)

    n_wins = int(wins.sum())
    wins_float = wins.astype(float)

    pm = PeriodMetrics(
        n=n,
        win_rate=round(float(wins.mean()) * 100, 1),
        avg_pnl=round(float(pnl.mean()), 2),
        median_pnl=round(float(np.median(pnl)), 2),
        wins=n_wins,
        losses=n - n_wins,
        win_rate_ci=bootstrap_win_rate(wins_float),
        avg_pnl_ci=bootstrap_mean_pnl(pnl),
        profit_factor=_profit_factor(pnl),
    )

    # By recommendation
    if recommendations is not None:
        recs = np.asarray(recommendations, dtype=object)
        present = [b for b in RECOMMENDATION_BUCKETS if (recs == b).any()]
        for bucket in present:
            mask = recs == bucket
            pm.by_recommendation[bucket] = _bucket_stats(pnl[mask], wins[mask])
        go = pm.by_recommendation.get('GO')
        nogo = pm.by_recommendation.get('NO-GO')
        if go and nogo and go['n'] > 0 and nogo['n'] > 0:
            pm.go_nogo_wr_delta = round(go['win_rate'] - nogo['win_rate'], 1)

    # By grade
    if grades is not None:
        grds = np.asarray(grades, dtype=object)
        for grade in sorted({g for g in grds.tolist() if g}):
            mask = grds == grade
            pm.by_grade[grade] = _bucket_stats(pnl[mask], wins[mask])

    # Score / P&L rank correlation
    if scores is not None and HAS_SCIPY:
        sc = np.asarray(scores, dtype=float)
        valid = ~(np.isnan(sc) | np.isnan(pnl))
        if valid.sum() >= 3 and len(np.unique(sc[valid])) > 1:
            rho, p = spearmanr(sc[valid], pnl[valid])
            if not np.isnan(rho):
                pm.score_pnl_correlation = round(float(rho), 3)
                pm.score_pnl_pvalue = round(float(p), 4)

    return pm


def _fisher_p(base: PeriodMetrics, compare: PeriodMetrics) -> Optional[float]:
    """Fisher exact p-value comparing two periods' win/loss counts."""
    if not HAS_SCIPY or base.n == 0 or compare.n == 0:
        return None
    try:
        _, p = fisher_exact([[compare.wins, compare.losses],
                             [base.wins, base.losses]])
        return float(p)
    except Exception:
        return None


def compute_degradation(base: PeriodMetrics, compare: PeriodMetrics,
                        fisher_p: Optional[float] = None) -> DegradationReport:
    """Compare `compare` to `base` and produce a degradation verdict.

    `base` is the reference (train, or production scoring); `compare` is the
    out-of-sample / derived side. Positive win_rate_change_pp means `compare`
    won more often than `base`.
    """
    wr_change = compare.win_rate - base.win_rate
    if abs(base.avg_pnl) > 1e-9:
        pnl_change = (compare.avg_pnl - base.avg_pnl) / abs(base.avg_pnl) * 100
        edge_retained = compare.avg_pnl / base.avg_pnl * 100
    else:
        pnl_change = 0.0
        edge_retained = 0.0

    # Direction-aware: a favorable OOS change (win rate / P&L UP) is never a
    # collapse. Only degradation below the baseline downgrades the verdict.
    if wr_change >= -5 and pnl_change > -20:
        verdict = "held"
    elif wr_change >= -15 and pnl_change > -50:
        verdict = "degraded"
    else:
        verdict = "collapsed"

    if fisher_p is None:
        fisher_p = _fisher_p(base, compare)

    return DegradationReport(
        win_rate_change_pp=round(wr_change, 1),
        avg_pnl_change_pct=round(pnl_change, 1),
        verdict=verdict,
        fisher_p_value=fisher_p,
        fisher_significant=(fisher_p is not None and fisher_p < 0.05),
        go_edge_retained=round(edge_retained, 1),
    )


def go_bucket_as_metrics(pm: Optional[PeriodMetrics]) -> Optional[PeriodMetrics]:
    """Rebuild a minimal PeriodMetrics from a period's GO recommendation bucket.

    Used to compare production-vs-derived GO-conditional win rates via
    compute_degradation. Returns None if there is no GO bucket, or if the GO
    bucket has fewer than MIN_GO_N trades (too few to compare reliably).
    """
    if pm is None:
        return None
    go = pm.by_recommendation.get('GO')
    if not go or go['n'] < MIN_GO_N:
        return None
    return PeriodMetrics(
        n=go['n'],
        win_rate=go['win_rate'],
        avg_pnl=go['avg_pnl'],
        wins=go['wins'],
        losses=go['losses'],
    )
