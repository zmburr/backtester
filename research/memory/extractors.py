"""Per-experiment-type extraction functions.

Each function takes an ExperimentResult and returns structured facts
to merge into the knowledge base. Pure Python — no Claude CLI calls.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def extract_feature_importance(result) -> Optional[Dict]:
    """Extract top feature rankings from a feature_importance result."""
    metrics = result.metrics
    if not metrics:
        return None

    # Get top features from spearman results
    top_spearman = metrics.get("top_spearman", [])
    if not top_spearman:
        # Fallback to top_features (combined ranking)
        top_features = metrics.get("top_features", [])
        if not top_features:
            return None
        entries = []
        for f in top_features[:10]:
            entries.append({
                "feature": f.get("feature", ""),
                "rho": f.get("spearman_rho", 0),
            })
        return {"strategy": result.strategy, "target": "pnl", "features": entries}

    entries = []
    for s in top_spearman[:10]:
        entries.append({
            "feature": s.get("feature", ""),
            "rho": s.get("spearman_rho", 0),
        })
    return {"strategy": result.strategy, "target": "pnl", "features": entries}


def extract_filter_sweep(result) -> Optional[List[Dict]]:
    """Extract top filter discoveries from a filter_sweep result."""
    metrics = result.metrics
    if not metrics:
        return None

    discoveries = []

    # Single filters
    for f in metrics.get("top_single_filters", [])[:3]:
        if f.get("improvement", 0) > 0:
            discoveries.append({
                "strategy": result.strategy,
                "filter_desc": f.get("filter", ""),
                "improvement_pct": f.get("improvement", 0),
                "n_filtered": f.get("count", 0),
                "win_rate": f.get("win_rate", 0),
                "is_significant": result.is_significant,
                "walk_forward_tested": False,
            })

    # Combinations
    for c in metrics.get("top_combinations", [])[:2]:
        if c.get("improvement", 0) > 0:
            discoveries.append({
                "strategy": result.strategy,
                "filter_desc": c.get("filters", ""),
                "improvement_pct": c.get("improvement", 0),
                "n_filtered": c.get("count", 0),
                "win_rate": c.get("win_rate", 0),
                "is_significant": result.is_significant,
                "walk_forward_tested": False,
            })

    return discoveries if discoveries else None


def extract_regime_analysis(result) -> Optional[Dict]:
    """Extract the best regime finding from a regime_analysis result."""
    metrics = result.metrics
    if not metrics:
        return None

    best = metrics.get("best_finding")
    if not best:
        return None

    return {
        "strategy": result.strategy,
        "regime": best.get("regime", ""),
        "best_bucket": best.get("best_bucket", ""),
        "worst_bucket": best.get("worst_bucket", ""),
        "improvement_pct": best.get("improvement", 0),
        "p_value": best.get("p_value"),
        "is_significant": result.is_significant,
    }


def extract_walk_forward(result) -> Optional[Dict]:
    """Extract walk-forward stability verdict."""
    metrics = result.metrics
    if not metrics:
        return None

    return {
        "strategy": result.strategy,
        "verdict": metrics.get("stability", "UNKNOWN"),
        "avg_oos_wr": metrics.get("avg_validate_wr"),
        "avg_oos_pnl": metrics.get("avg_validate_avg_pnl"),
        "n_splits": metrics.get("n_splits_tested", 0),
    }


def extract_risk_metrics(result) -> Optional[Dict]:
    """Extract key risk numbers."""
    metrics = result.metrics
    if not metrics:
        return None

    mc = metrics.get("monte_carlo", {})
    return {
        "strategy": result.strategy,
        "kelly_fraction": metrics.get("kelly_fraction"),
        "sharpe_ratio": metrics.get("sharpe_ratio"),
        "ruin_probability": mc.get("ruin_probability"),
        "median_max_drawdown": mc.get("median_max_drawdown"),
        "p95_max_drawdown": mc.get("p95_max_drawdown"),
    }


def extract_failed_experiment(result, min_sample: int = 10) -> Optional[Dict]:
    """Check if an experiment failed due to insufficient data."""
    # Check for explicit insufficient data warnings
    metadata = result.metadata or {}
    if metadata.get("warning") == "insufficient_data":
        return {
            "experiment_type": result.experiment_type,
            "strategy": result.strategy,
            "params_key": _params_key(result.parameters),
            "failure_reason": "insufficient_data",
            "n_available": result.metrics.get("n", 0),
        }

    # Check summary for insufficient data signals
    summary = result.summary or ""
    if "Insufficient" in summary or "too few" in summary.lower():
        return {
            "experiment_type": result.experiment_type,
            "strategy": result.strategy,
            "params_key": _params_key(result.parameters),
            "failure_reason": "insufficient_data",
            "n_available": result.metrics.get("n", 0),
        }

    # Check for errors
    if "FAILED" in summary:
        return {
            "experiment_type": result.experiment_type,
            "strategy": result.strategy,
            "params_key": _params_key(result.parameters),
            "failure_reason": summary[:100],
            "n_available": 0,
        }

    return None


def _params_key(params: dict) -> str:
    """Create a short key from params for deduplication."""
    parts = []
    for k in sorted(params.keys()):
        if k == "strategy":
            continue  # already tracked separately
        v = params[k]
        if v is not None:
            parts.append(f"{k}={v}")
    return ",".join(parts) if parts else "default"
