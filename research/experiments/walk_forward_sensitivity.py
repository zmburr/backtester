"""Walk-forward sensitivity experiment — tests multiple train/validate split dates."""

import time
import logging
from typing import Dict, List

from research.experiments.base import BaseExperiment, ExperimentResult

logger = logging.getLogger(__name__)

# Default split configurations to test
DEFAULT_SPLITS = [
    {"train_end": "2023-06-30", "validate_end": "2024-06-30", "label": "mid-2023 split"},
    {"train_end": "2023-12-31", "validate_end": "2024-12-31", "label": "year-end 2023 split"},
    {"train_end": "2024-06-30", "validate_end": "2025-06-30", "label": "mid-2024 split"},
]


class WalkForwardSensitivityExperiment(BaseExperiment):
    name = "walk_forward_sensitivity"
    description = "Run walk-forward validation with multiple train/validate split dates to test edge stability."
    supported_strategies = ["reversal", "bounce"]

    def validate_params(self, params: dict) -> bool:
        strategy = params.get("strategy")
        if strategy not in self.supported_strategies:
            return False
        splits = params.get("splits", DEFAULT_SPLITS)
        return len(splits) >= 1

    def describe_capabilities(self) -> str:
        return (
            "Tests whether the trading edge is stable across different training windows.\n"
            "Runs walk-forward validation with multiple split dates and compares degradation.\n"
            "Params:\n"
            "  - strategy: 'reversal' or 'bounce' (required)\n"
            "  - splits: list of {train_end, validate_end, label} dicts (default: 3 standard splits)\n"
            "  - csv_path: path to CSV (default: auto from strategy)\n"
            "Returns: per-split metrics, degradation verdicts, stability assessment."
        )

    def run(self, params: dict, config) -> ExperimentResult:
        t0 = time.time()
        strategy = params["strategy"]
        splits = params.get("splits", DEFAULT_SPLITS)
        csv_path = params.get("csv_path")

        # Lazy import to avoid circular deps
        from validation.walk_forward_engine import run_walk_forward

        split_results = []
        verdicts = []
        errors = []

        for split_cfg in splits:
            train_end = split_cfg["train_end"]
            validate_end = split_cfg["validate_end"]
            label = split_cfg.get("label", f"{train_end}/{validate_end}")

            try:
                wf = run_walk_forward(
                    strategy=strategy,
                    train_end=train_end,
                    validate_end=validate_end,
                    csv_path=csv_path,
                )

                train_m = wf.train_metrics
                val_m = wf.validate_metrics
                deg = wf.train_vs_validate

                result = {
                    "label": label,
                    "train_end": train_end,
                    "validate_end": validate_end,
                    "train_n": train_m.n,
                    "train_wr": round(train_m.win_rate, 1),
                    "train_avg_pnl": round(train_m.avg_pnl, 2),
                    "validate_n": val_m.n,
                    "validate_wr": round(val_m.win_rate, 1),
                    "validate_avg_pnl": round(val_m.avg_pnl, 2),
                }

                if deg:
                    result["wr_change_pp"] = round(deg.win_rate_change_pp, 1)
                    result["pnl_change_pct"] = round(deg.avg_pnl_change_pct, 1)
                    result["verdict"] = deg.verdict
                    result["fisher_p"] = round(deg.fisher_p_value, 4) if deg.fisher_p_value else None
                    result["go_edge_retained"] = round(deg.go_edge_retained, 1) if deg.go_edge_retained else None
                    verdicts.append(deg.verdict)
                else:
                    result["verdict"] = "no_data"

                # Add test period if available
                if wf.test_metrics and wf.test_metrics.n > 0:
                    result["test_n"] = wf.test_metrics.n
                    result["test_wr"] = round(wf.test_metrics.win_rate, 1)
                    result["test_avg_pnl"] = round(wf.test_metrics.avg_pnl, 2)

                split_results.append(result)

            except Exception as e:
                errors.append({"label": label, "error": str(e)})
                logger.warning(f"Walk-forward failed for {label}: {e}")

        if not split_results:
            return ExperimentResult(
                experiment_id=self.make_id(),
                experiment_type=self.name,
                strategy=strategy,
                parameters=params,
                metrics={},
                summary=f"All walk-forward splits failed: {errors}",
                statistical_tests={},
                is_significant=False,
                metadata={"errors": errors, "runtime_seconds": round(time.time() - t0, 1)},
            )

        # Stability assessment
        n_held = verdicts.count("held")
        n_degraded = verdicts.count("degraded")
        n_collapsed = verdicts.count("collapsed")

        if n_collapsed == 0 and n_degraded <= 1:
            stability = "STABLE"
        elif n_collapsed == 0:
            stability = "MODERATELY_STABLE"
        elif n_collapsed <= 1:
            stability = "UNSTABLE"
        else:
            stability = "NO_EDGE"

        # Average OOS metrics
        avg_val_wr = sum(r["validate_wr"] for r in split_results) / len(split_results)
        avg_val_pnl = sum(r["validate_avg_pnl"] for r in split_results) / len(split_results)

        summary = (
            f"Walk-forward sensitivity: {stability}. "
            f"{n_held} held, {n_degraded} degraded, {n_collapsed} collapsed across {len(split_results)} splits. "
            f"Avg OOS: WR={avg_val_wr:.1f}%, Avg P&L={avg_val_pnl:+.2f}%."
        )

        metrics = {
            "split_results": split_results,
            "stability": stability,
            "n_splits_tested": len(split_results),
            "n_held": n_held,
            "n_degraded": n_degraded,
            "n_collapsed": n_collapsed,
            "avg_validate_wr": round(avg_val_wr, 1),
            "avg_validate_avg_pnl": round(avg_val_pnl, 2),
        }

        return ExperimentResult(
            experiment_id=self.make_id(),
            experiment_type=self.name,
            strategy=strategy,
            parameters=params,
            metrics=metrics,
            summary=summary,
            statistical_tests={
                "stability_verdict": stability,
                "verdicts": verdicts,
            },
            is_significant=(stability in ("STABLE", "MODERATELY_STABLE")),
            metadata={
                "runtime_seconds": round(time.time() - t0, 1),
                "errors": errors,
            },
        )
