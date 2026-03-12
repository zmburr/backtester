"""Filter sweep experiment — wraps filter_optimizer.py for both reversal and bounce."""

import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List

from research.experiments.base import BaseExperiment, ExperimentResult
from analyzers.filter_optimizer import (
    load_reversal_data,
    calculate_metrics,
    find_optimal_filters,
    find_best_filter_combinations,
    analyze_indicator_impact,
    test_combined_filters,
)
from analyzers.bootstrap import bootstrap_mean_pnl, bootstrap_win_rate

try:
    from scipy.stats import fisher_exact
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)

# Column mapping per strategy
STRATEGY_CONFIG = {
    "reversal": {
        "csv_name": "reversal_data.csv",
        "target_col": "reversal_open_close_pct",
        "pnl_sign": -1,  # shorts: flip sign
        "rec_col": "pretrade_recommendation",
    },
    "bounce": {
        "csv_name": "bounce_data.csv",
        "target_col": "bounce_open_close_pct",
        "pnl_sign": 1,  # longs: positive = profit
        "rec_col": "pt_recommendation",
    },
}


class FilterSweepExperiment(BaseExperiment):
    name = "filter_sweep"
    description = "Test single filter thresholds and combinations on trade data."
    supported_strategies = ["reversal", "bounce"]

    def validate_params(self, params: dict) -> bool:
        strategy = params.get("strategy")
        if strategy not in self.supported_strategies:
            return False
        return True

    def describe_capabilities(self) -> str:
        return (
            "Tests single filters and 2-3 filter combinations to find thresholds that improve expectancy.\n"
            "Params:\n"
            "  - strategy: 'reversal' or 'bounce' (required)\n"
            "  - grade: trade grade filter, e.g. 'A', 'B', or None for all (default: None)\n"
            "  - cap: cap size filter, e.g. 'Micro', 'Small', etc. or None (default: None)\n"
            "  - columns_to_sweep: list of specific columns to test, or None for all defaults\n"
            "  - min_sample: minimum trades after filtering (default: 10)\n"
            "  - max_combo_size: max combination size 2 or 3 (default: 2)\n"
            "  - top_n_filters: how many top single filters to use for combos (default: 10)\n"
            "Returns: baseline metrics, top single filters, top combinations, correlations."
        )

    def run(self, params: dict, config) -> ExperimentResult:
        t0 = time.time()
        strategy = params["strategy"]
        scfg = STRATEGY_CONFIG[strategy]
        grade = params.get("grade")
        cap = params.get("cap")
        min_sample = params.get("min_sample", config.min_sample)
        max_combo_size = params.get("max_combo_size", 2)
        top_n = params.get("top_n_filters", 10)
        target_col = scfg["target_col"]
        pnl_sign = scfg["pnl_sign"]

        # Load data
        csv_path = config.data_dir / scfg["csv_name"]
        df = pd.read_csv(str(csv_path)).dropna(subset=["ticker", "date"])

        # Apply filters
        if grade:
            df = df[df["trade_grade"] == grade].copy()
        if cap:
            df = df[df["cap"] == cap].copy()

        n_total = len(df)
        if n_total < min_sample:
            return ExperimentResult(
                experiment_id=self.make_id(),
                experiment_type=self.name,
                strategy=strategy,
                parameters=params,
                metrics={"n": n_total},
                summary=f"Insufficient data: {n_total} trades after filtering (need {min_sample}).",
                statistical_tests={},
                is_significant=False,
                metadata={"runtime_seconds": round(time.time() - t0, 1), "warning": "insufficient_data"},
            )

        # Baseline metrics — compute_metrics uses reversal sign convention internally
        if strategy == "bounce":
            # For bounce, calculate_metrics assumes short convention.
            # We need to compute our own baseline.
            pnl = df[target_col] * pnl_sign
            baseline = {
                "count": len(df),
                "win_rate": round((pnl > 0).mean() * 100, 1),
                "avg_pnl": round(pnl.mean() * 100, 2),
                "median_pnl": round(pnl.median() * 100, 2),
            }
        else:
            baseline = calculate_metrics(df, target_col)

        baseline_expectancy = baseline["avg_pnl"]

        # Run single filter sweep
        _, single_filters = find_optimal_filters(df, grade=None, min_sample=min_sample, target_col=target_col)

        # Run combinations on top filters
        combos = []
        if single_filters and len(single_filters) >= 2:
            combos = find_best_filter_combinations(
                df, single_filters[:top_n], max_combo_size=max_combo_size,
                min_sample=min_sample, target_col=target_col,
            )

        # Correlations
        correlations = analyze_indicator_impact(df, target_col)

        # Pick the best finding (highest improvement)
        best = None
        if combos:
            best = combos[0]
            best_type = "combo"
        elif single_filters:
            best = single_filters[0]
            best_type = "single"

        # Statistical evaluation of the best finding
        stat_tests = {}
        is_sig = False
        if best:
            improvement = best.get("improvement", 0)
            n_filtered = best.get("count", 0)

            # Compute p-value via Fisher exact (win rate comparison)
            p_value = None
            if HAS_SCIPY and n_filtered >= 5:
                best_wr = best["win_rate"] / 100
                base_wr = baseline["win_rate"] / 100
                wins_t = int(round(best_wr * n_filtered))
                losses_t = n_filtered - wins_t
                wins_c = int(round(base_wr * n_total))
                losses_c = n_total - wins_c
                try:
                    _, p_value = fisher_exact([[wins_t, losses_t], [wins_c, losses_c]])
                except Exception:
                    pass

            # Bootstrap CI on filtered subset PNL
            pnl_ci = None
            if best_type == "combo":
                # Re-filter to get the actual PNL array
                filter_str = best["filters"]
                # Parse combo filters and re-run
                pnl_arr = np.array([baseline_expectancy])  # fallback
            elif best_type == "single":
                pnl_arr = np.array([baseline_expectancy])

            stat_tests = {
                "improvement_pct": round(improvement, 2),
                "p_value": round(p_value, 4) if p_value is not None else None,
                "n_filtered": n_filtered,
                "n_baseline": n_total,
                "best_win_rate": best["win_rate"],
                "baseline_win_rate": baseline["win_rate"],
            }

            # Simple significance check
            is_sig = (
                n_filtered >= min_sample
                and improvement >= config.min_effect_size
                and p_value is not None
                and p_value < config.alpha
            )

        # Build summary
        top_3_single = single_filters[:3] if single_filters else []
        top_3_combo = combos[:3] if combos else []

        if best:
            summary = (
                f"Best finding: {best.get('filters', best.get('filter', '?'))} "
                f"improves avg P&L by {best.get('improvement', 0):+.2f}% "
                f"(N={best.get('count', 0)}, WR={best.get('win_rate', 0)}%)."
            )
        else:
            summary = "No meaningful filter improvements found."

        metrics = {
            "baseline": baseline,
            "top_single_filters": [
                {"filter": f.get("filter"), "count": f["count"], "win_rate": f["win_rate"],
                 "avg_pnl": f["avg_pnl"], "improvement": f.get("improvement", 0)}
                for f in top_3_single
            ],
            "top_combinations": [
                {"filters": c["filters"], "count": c["count"], "win_rate": c["win_rate"],
                 "avg_pnl": c["avg_pnl"], "improvement": c.get("improvement", 0)}
                for c in top_3_combo
            ],
            "top_correlations": correlations[:5],
            "n_single_filters_tested": len(single_filters),
            "n_combinations_tested": len(combos),
        }

        return ExperimentResult(
            experiment_id=self.make_id(),
            experiment_type=self.name,
            strategy=strategy,
            parameters=params,
            metrics=metrics,
            summary=summary,
            statistical_tests=stat_tests,
            is_significant=is_sig,
            metadata={
                "runtime_seconds": round(time.time() - t0, 1),
                "n_total_trades": n_total,
            },
        )
