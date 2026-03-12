"""Exit optimization experiment — wraps ExitOptimizer for ATR targets, time exits, MFE/MAE."""

import io
import sys
import time
import logging
import pandas as pd
import numpy as np
from typing import Dict

from research.experiments.base import BaseExperiment, ExperimentResult

logger = logging.getLogger(__name__)


class ExitOptimizationExperiment(BaseExperiment):
    name = "exit_optimization"
    description = "Analyze exit strategies: ATR targets, time exits, MFE/MAE, trailing stops."
    supported_strategies = ["reversal"]

    def validate_params(self, params: dict) -> bool:
        return params.get("strategy") in self.supported_strategies

    def describe_capabilities(self) -> str:
        return (
            "Analyzes exit strategy effectiveness using historical exit analysis data.\n"
            "Tests ATR-based targets, time exits, MFE/MAE patterns, and trailing stops.\n"
            "Only works for reversal (requires exit_analysis_results.csv).\n"
            "Params:\n"
            "  - strategy: 'reversal' (required)\n"
            "  - cap: filter to specific cap size (default: None for all)\n"
            "  - analysis_type: 'atr_targets', 'time_exits', 'mfe_mae', 'all' (default: 'all')\n"
            "  - trailing_percentages: list of trail % to test (default: [0.25, 0.50, 0.75, 1.0])\n"
            "Returns: optimal exit tiers by cap, capture efficiency, time-based P&L, trailing stop analysis."
        )

    def run(self, params: dict, config) -> ExperimentResult:
        t0 = time.time()
        strategy = params["strategy"]
        cap_filter = params.get("cap")
        analysis_type = params.get("analysis_type", "all")

        csv_path = str(config.exit_analysis_csv)

        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            return ExperimentResult(
                experiment_id=self.make_id(),
                experiment_type=self.name,
                strategy=strategy,
                parameters=params,
                metrics={},
                summary="exit_analysis_results.csv not found. Run exit analysis phase 1 first.",
                statistical_tests={},
                is_significant=False,
                metadata={"error": "file_not_found"},
            )

        if cap_filter:
            df = df[df["cap"] == cap_filter].copy()

        if len(df) < 10:
            return ExperimentResult(
                experiment_id=self.make_id(),
                experiment_type=self.name,
                strategy=strategy,
                parameters=params,
                metrics={"n": len(df)},
                summary=f"Insufficient exit analysis data: {len(df)} trades.",
                statistical_tests={},
                is_significant=False,
            )

        metrics = {"n": len(df)}

        # ATR target analysis
        if analysis_type in ("all", "atr_targets"):
            atr_cols = ["hit_1x_atr", "hit_1_5x_atr", "hit_2x_atr", "hit_2_5x_atr", "hit_3x_atr"]
            atr_results = {}
            for col in atr_cols:
                if col in df.columns:
                    target = col.replace("hit_", "").replace("_atr", "").replace("_", ".")
                    atr_results[f"{target}x_hit_rate"] = round(df[col].mean() * 100, 1)

            # By cap
            atr_by_cap = {}
            if "cap" in df.columns:
                for cap_val in df["cap"].dropna().unique():
                    cap_df = df[df["cap"] == cap_val]
                    if len(cap_df) >= 5:
                        atr_by_cap[cap_val] = {
                            col.replace("hit_", "").replace("_atr", "").replace("_", ".") + "x":
                            round(cap_df[col].mean() * 100, 1)
                            for col in atr_cols if col in cap_df.columns
                        }

            metrics["atr_hit_rates"] = atr_results
            metrics["atr_by_cap"] = atr_by_cap

        # Time exit analysis
        if analysis_type in ("all", "time_exits"):
            time_cols = ["pnl_at_10am", "pnl_at_1030am", "pnl_at_11am",
                         "pnl_at_12pm", "pnl_at_1pm", "pnl_at_2pm", "pnl_at_close"]
            time_results = {}
            for col in time_cols:
                if col in df.columns:
                    valid = df[df[col].notna()]
                    if len(valid) > 0:
                        time_label = col.replace("pnl_at_", "")
                        time_results[time_label] = {
                            "avg_pnl": round(valid[col].mean() * 100, 2),
                            "win_rate": round((valid[col] > 0).mean() * 100, 1),
                            "n": len(valid),
                        }
            metrics["time_exits"] = time_results

        # MFE/MAE analysis
        if analysis_type in ("all", "mfe_mae"):
            if "mfe_pct" in df.columns and "close_pct" in df.columns:
                metrics["mfe_mae"] = {
                    "avg_mfe": round(df["mfe_pct"].mean() * 100, 2),
                    "median_mfe": round(df["mfe_pct"].median() * 100, 2),
                    "avg_captured": round(df["close_pct"].mean() * 100, 2),
                    "capture_efficiency": round(df["capture_efficiency"].mean() * 100, 1)
                    if "capture_efficiency" in df.columns else None,
                    "avg_giveback": round(df["max_giveback_pct"].mean() * 100, 2)
                    if "max_giveback_pct" in df.columns else None,
                }

                # Trailing stop simulation
                trailing_pcts = params.get("trailing_percentages", [0.25, 0.50, 0.75, 1.0])
                trailing_results = {}
                if "max_giveback_pct" in df.columns:
                    for trail in trailing_pcts:
                        captured = df.apply(
                            lambda x: x["mfe_pct"] - min(trail * x["mfe_pct"], x["max_giveback_pct"]),
                            axis=1,
                        ).mean() * 100
                        trailing_results[f"{trail*100:.0f}pct"] = round(captured, 2)
                metrics["trailing_stop_results"] = trailing_results

        # Summary
        summary_parts = []
        if "atr_hit_rates" in metrics:
            hr_1x = metrics["atr_hit_rates"].get("1x_hit_rate", "?")
            summary_parts.append(f"1x ATR hit rate: {hr_1x}%")
        if "mfe_mae" in metrics and metrics["mfe_mae"].get("capture_efficiency"):
            eff = metrics["mfe_mae"]["capture_efficiency"]
            summary_parts.append(f"Capture efficiency: {eff}%")
        if "trailing_stop_results" in metrics and "50pct" in metrics.get("trailing_stop_results", {}):
            trail_50 = metrics["trailing_stop_results"]["50pct"]
            summary_parts.append(f"50% trailing stop would capture {trail_50:+.2f}%")

        summary = "Exit analysis: " + ". ".join(summary_parts) + "." if summary_parts else "Exit analysis complete."

        return ExperimentResult(
            experiment_id=self.make_id(),
            experiment_type=self.name,
            strategy=strategy,
            parameters=params,
            metrics=metrics,
            summary=summary,
            statistical_tests={},
            is_significant=False,  # exit analysis is descriptive, not hypothesis testing
            metadata={"runtime_seconds": round(time.time() - t0, 1)},
        )
