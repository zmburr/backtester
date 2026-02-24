"""Bootstrap confidence interval utilities for small-sample trading statistics."""
import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    from scipy.stats import bootstrap as scipy_bootstrap
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class BootstrapResult:
    """Result of a bootstrap confidence interval calculation."""
    point_estimate: float
    ci_lower: float
    ci_upper: float
    n: int
    method: str  # 'BCa' or 'percentile'
    is_pct: bool = False  # True for win rates (0-100 scale)


def bootstrap_win_rate(pnl_array: np.ndarray, n_resamples: int = 10000,
                       confidence_level: float = 0.95, random_seed: int = 42) -> Optional[BootstrapResult]:
    """Bootstrap CI for win rate (% of trades with pnl > 0).

    Args:
        pnl_array: Array of P&L values.
        n_resamples: Number of bootstrap resamples.
        confidence_level: CI level (default 0.95).
        random_seed: Seed for reproducibility.

    Returns:
        BootstrapResult with point estimate and CI in percentage (0-100), or None if empty.
    """
    pnl = np.asarray(pnl_array, dtype=float)
    pnl = pnl[~np.isnan(pnl)]
    if len(pnl) == 0:
        return None

    point = (pnl > 0).mean() * 100
    n = len(pnl)

    if n < 2:
        return BootstrapResult(point_estimate=point, ci_lower=point, ci_upper=point,
                               n=n, method='percentile', is_pct=True)

    wins = (pnl > 0).astype(float)
    ci_lower, ci_upper, method = _compute_ci(wins, lambda x: np.mean(x) * 100,
                                              n_resamples, confidence_level, random_seed, n)
    return BootstrapResult(point_estimate=point, ci_lower=ci_lower, ci_upper=ci_upper,
                           n=n, method=method, is_pct=True)


def bootstrap_mean_pnl(pnl_array: np.ndarray, n_resamples: int = 10000,
                        confidence_level: float = 0.95, random_seed: int = 42) -> Optional[BootstrapResult]:
    """Bootstrap CI for mean P&L.

    Args:
        pnl_array: Array of P&L values.
        n_resamples: Number of bootstrap resamples.
        confidence_level: CI level (default 0.95).
        random_seed: Seed for reproducibility.

    Returns:
        BootstrapResult with point estimate and CI, or None if empty.
    """
    pnl = np.asarray(pnl_array, dtype=float)
    pnl = pnl[~np.isnan(pnl)]
    if len(pnl) == 0:
        return None

    point = float(np.mean(pnl))
    n = len(pnl)

    if n < 2:
        return BootstrapResult(point_estimate=point, ci_lower=point, ci_upper=point,
                               n=n, method='percentile', is_pct=False)

    ci_lower, ci_upper, method = _compute_ci(pnl, np.mean, n_resamples, confidence_level, random_seed, n)
    return BootstrapResult(point_estimate=point, ci_lower=ci_lower, ci_upper=ci_upper,
                           n=n, method=method, is_pct=False)


def _compute_ci(data, statistic_fn, n_resamples, confidence_level, random_seed, n):
    """Compute CI using BCa (n>=10) or percentile fallback (n<10)."""
    use_bca = n >= 10 and HAS_SCIPY

    if use_bca:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = scipy_bootstrap(
                    (data,),
                    statistic=lambda x, axis: np.apply_along_axis(statistic_fn, axis, x),
                    n_resamples=n_resamples,
                    confidence_level=confidence_level,
                    random_state=np.random.default_rng(random_seed),
                    method='BCa',
                )
            low, high = float(res.confidence_interval.low), float(res.confidence_interval.high)
            if not (np.isnan(low) or np.isnan(high)):
                return low, high, 'BCa'
            # BCa returned nan (degenerate distribution), fall through to percentile
        except Exception:
            pass  # Fall through to manual percentile

    # Manual percentile bootstrap
    rng = np.random.default_rng(random_seed)
    boot_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = statistic_fn(sample)

    alpha = (1 - confidence_level) / 2
    ci_lower = float(np.percentile(boot_stats, alpha * 100))
    ci_upper = float(np.percentile(boot_stats, (1 - alpha) * 100))
    return ci_lower, ci_upper, 'percentile'


def format_ci(result: Optional[BootstrapResult], is_pnl: bool = False) -> str:
    """Format a BootstrapResult as a compact string.

    Examples:
        Win rate: "90.9% [78.9-97.0]"
        P&L:     "+14.4% [+9.1 to +20.2]"
        Small n: "50.0% [16.7-83.3]†"

    Args:
        result: BootstrapResult to format.
        is_pnl: If True, use signed P&L format with "to" separator.

    Returns:
        Formatted string.
    """
    if result is None:
        return 'N/A'

    dagger = '\u2020' if result.n < 10 else ''

    if is_pnl:
        return f'{result.point_estimate:+.1f}% [{result.ci_lower:+.1f} to {result.ci_upper:+.1f}]{dagger}'
    else:
        return f'{result.point_estimate:.1f}% [{result.ci_lower:.1f}-{result.ci_upper:.1f}]{dagger}'


def bootstrap_to_dict(result: Optional[BootstrapResult]) -> Optional[dict]:
    """Convert BootstrapResult to a JSON-serializable dict."""
    if result is None:
        return None
    return {
        'point': round(result.point_estimate, 2),
        'ci_lower': round(result.ci_lower, 2),
        'ci_upper': round(result.ci_upper, 2),
        'n': result.n,
        'method': result.method,
    }
