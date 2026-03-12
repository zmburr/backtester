"""Entry signal comparison experiment — compare performance by signal/setup type."""

import time
import logging
import pandas as pd
import numpy as np
from typing import Dict

from research.experiments.base import BaseExperiment, ExperimentResult
from analyzers.bootstrap import bootstrap_mean_pnl, bootstrap_win_rate

try:
    from scipy.stats import fisher_exact
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)

STRATEGY_CONFIG = {
    "reversal": {
        "csv_name": "reversal_data.csv",
        "target_col": "reversal_open_close_pct",
        "pnl_sign": -1,
        "signal_col": "intraday_setup",
    },
    "bounce": {
        "csv_name": "bounce_data.csv",
        "target_col": "bounce_open_close_pct",
        "pnl_sign": 1,
        "signal_col": "Setup",
    },
}


class EntrySignalComparisonExperiment(BaseExperiment):
    name = "entry_signal_comparison"
    description = "Compare performance across different entry signal types."
    supported_strategies = ["reversal", "bounce"]

    def validate_params(self, params: dict) -> bool:
        return params.get("strategy") in self.supported_strategies

    def describe_capabilities(self) -> str:
        return (
            "Groups trades by entry signal/setup type and compares performance.\n"
            "For reversal: groups by intraday_setup column.\n"
            "For bounce: groups by Setup column (GapFade_weakstock, GapFade_strongstock, etc.).\n"
            "Params:\n"
            "  - strategy: 'reversal' or 'bounce' (required)\n"
            "  - signal_column: column to group by (default: auto from strategy)\n"
            "  - grade: filter to specific grade (default: None)\n"
            "  - cap: filter to specific cap (default: None)\n"
            "Returns: per-signal metrics, bootstrap CIs, pairwise Fisher tests."
        )

    def run(self, params: dict, config) -> ExperimentResult:
        t0 = time.time()
        strategy = params["strategy"]
        scfg = STRATEGY_CONFIG[strategy]
        signal_col = params.get("signal_column", scfg["signal_col"])
        grade = params.get("grade")
        cap = params.get("cap")

        # Load data
        csv_path = config.data_dir / scfg["csv_name"]
        df = pd.read_csv(str(csv_path)).dropna(subset=["ticker", "date"])

        if grade:
            df = df[df["trade_grade"] == grade].copy()
        if cap:
            df = df[df["cap"] == cap].copy()

        df["_pnl"] = df[scfg["target_col"]] * scfg["pnl_sign"] * 100

        if signal_col not in df.columns:
            # Try alternate column names
            alt_cols = ["setup", "Setup", "intraday_setup", "signal_type"]
            found = None
            for alt in alt_cols:
                if alt in df.columns:
                    found = alt
                    break
            if found:
                signal_col = found
            else:
                return ExperimentResult(
                    experiment_id=self.make_id(),
                    experiment_type=self.name,
                    strategy=strategy,
                    parameters=params,
                    metrics={},
                    summary=f"Signal column '{signal_col}' not found in data.",
                    statistical_tests={},
                    is_significant=False,
                )

        # Group by signal type
        signal_results = []
        baseline_pnl = df["_pnl"].mean()

        for signal_val, group in df.groupby(signal_col):
            if pd.isna(signal_val) or len(group) < 3:
                continue

            pnl = group["_pnl"]
            wr = (pnl > 0).mean() * 100
            avg = pnl.mean()

            result = {
                "signal": str(signal_val),
                "n": len(group),
                "win_rate": round(wr, 1),
                "avg_pnl": round(avg, 2),
                "median_pnl": round(pnl.median(), 2),
                "vs_baseline": round(avg - baseline_pnl, 2),
            }

            # Bootstrap CI
            ci = bootstrap_mean_pnl(pnl.values)
            if ci:
                result["pnl_ci_lower"] = round(ci.ci_lower, 2)
                result["pnl_ci_upper"] = round(ci.ci_upper, 2)

            wr_ci = bootstrap_win_rate(pnl.values)
            if wr_ci:
                result["wr_ci_lower"] = round(wr_ci.ci_lower, 1)
                result["wr_ci_upper"] = round(wr_ci.ci_upper, 1)

            signal_results.append(result)

        if not signal_results:
            return ExperimentResult(
                experiment_id=self.make_id(),
                experiment_type=self.name,
                strategy=strategy,
                parameters=params,
                metrics={},
                summary="No signal types with sufficient data.",
                statistical_tests={},
                is_significant=False,
            )

        signal_results.sort(key=lambda x: x["avg_pnl"], reverse=True)

        # Pairwise Fisher tests between top and bottom
        pairwise = []
        if HAS_SCIPY and len(signal_results) >= 2:
            best = signal_results[0]
            worst = signal_results[-1]
            if best["n"] >= 5 and worst["n"] >= 5:
                b_wins = int(round(best["win_rate"] / 100 * best["n"]))
                b_losses = best["n"] - b_wins
                w_wins = int(round(worst["win_rate"] / 100 * worst["n"]))
                w_losses = worst["n"] - w_wins
                try:
                    _, p_val = fisher_exact([[b_wins, b_losses], [w_wins, w_losses]])
                    pairwise.append({
                        "comparison": f"{best['signal']} vs {worst['signal']}",
                        "p_value": round(p_val, 4),
                        "improvement": round(best["avg_pnl"] - worst["avg_pnl"], 2),
                    })
                except Exception:
                    pass

        # Summary
        best_signal = signal_results[0]
        summary = (
            f"Best signal: {best_signal['signal']} "
            f"(N={best_signal['n']}, WR={best_signal['win_rate']}%, "
            f"Avg P&L={best_signal['avg_pnl']:+.2f}%). "
            f"{len(signal_results)} signal types compared."
        )

        is_sig = (
            len(pairwise) > 0
            and pairwise[0].get("p_value", 1) < config.alpha
            and abs(pairwise[0].get("improvement", 0)) >= config.min_effect_size
        )

        return ExperimentResult(
            experiment_id=self.make_id(),
            experiment_type=self.name,
            strategy=strategy,
            parameters=params,
            metrics={
                "baseline_expectancy": round(baseline_pnl, 2),
                "n_total": len(df),
                "signal_results": signal_results,
                "n_signal_types": len(signal_results),
            },
            summary=summary,
            statistical_tests={
                "pairwise_tests": pairwise,
            },
            is_significant=is_sig,
            metadata={
                "runtime_seconds": round(time.time() - t0, 1),
                "signal_column_used": signal_col,
            },
        )
