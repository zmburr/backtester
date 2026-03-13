"""Regime analysis experiment — bucket trades by market conditions (SPY/UVXY)."""

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
    },
    "bounce": {
        "csv_name": "bounce_data.csv",
        "target_col": "bounce_open_close_pct",
        "pnl_sign": 1,
    },
}

# Default regime definitions using columns already in the data
DEFAULT_REGIME_DEFS = {
    "spy_trend": {
        "column": "spy_5day_return",
        "buckets": [
            {"label": "spy_down", "op": "<", "threshold": -0.01},
            {"label": "spy_flat", "op": "between", "low": -0.01, "high": 0.01},
            {"label": "spy_up", "op": ">", "threshold": 0.01},
        ],
    },
    "volatility": {
        "column": "uvxy_close",
        "buckets": [
            {"label": "low_vol", "op": "<", "threshold": 15},
            {"label": "mid_vol", "op": "between", "low": 15, "high": 25},
            {"label": "high_vol", "op": ">=", "threshold": 25},
        ],
    },
}


class RegimeAnalysisExperiment(BaseExperiment):
    name = "regime_analysis"
    description = "Bucket trades by market regime (SPY trend, volatility, etc.) and compare performance."
    supported_strategies = ["reversal", "bounce"]

    def validate_params(self, params: dict) -> bool:
        return params.get("strategy") in self.supported_strategies

    def describe_capabilities(self) -> str:
        return (
            "Buckets trades by market conditions and compares performance across regimes.\n"
            "Uses pre-trade columns (spy_5day_return, uvxy_close) in the trade CSVs.\n"
            "Params:\n"
            "  - strategy: 'reversal' or 'bounce' (required)\n"
            "  - regime: 'spy_trend', 'volatility', or 'all' (default: 'all')\n"
            "  - grade: filter to specific grade (default: None)\n"
            "  - cap: filter to specific cap (default: None)\n"
            "Returns: per-regime metrics, best/worst regimes, Fisher exact between best and worst."
        )

    def run(self, params: dict, config) -> ExperimentResult:
        t0 = time.time()
        strategy = params["strategy"]
        scfg = STRATEGY_CONFIG[strategy]
        regime_name = params.get("regime", "all")
        grade = params.get("grade")
        cap = params.get("cap")

        # Load data
        csv_path = config.data_dir / scfg["csv_name"]
        df = pd.read_csv(str(csv_path)).dropna(subset=["ticker", "date"])

        if grade:
            df = df[df["trade_grade"] == grade].copy()
        if cap:
            df = df[df["cap"] == cap].copy()

        # Compute PNL
        df["_pnl"] = df[scfg["target_col"]] * scfg["pnl_sign"] * 100

        if len(df) < 20:
            return ExperimentResult(
                experiment_id=self.make_id(),
                experiment_type=self.name,
                strategy=strategy,
                parameters=params,
                metrics={"n": len(df)},
                summary=f"Insufficient data: {len(df)} trades.",
                statistical_tests={},
                is_significant=False,
            )

        # Select regimes to analyze
        if regime_name == "all":
            regimes_to_test = DEFAULT_REGIME_DEFS
        elif regime_name in DEFAULT_REGIME_DEFS:
            regimes_to_test = {regime_name: DEFAULT_REGIME_DEFS[regime_name]}
        else:
            return ExperimentResult(
                experiment_id=self.make_id(),
                experiment_type=self.name,
                strategy=strategy,
                parameters=params,
                metrics={},
                summary=f"Unknown regime: {regime_name}",
                statistical_tests={},
                is_significant=False,
            )

        all_regime_results = {}
        best_finding = None
        best_improvement = 0
        best_p_value = None
        baseline_expectancy = df["_pnl"].mean()

        for rname, rdef in regimes_to_test.items():
            col = rdef["column"]
            if col not in df.columns:
                logger.warning(f"Column {col} not in data, skipping regime {rname}")
                continue

            bucket_results = []
            for bucket in rdef["buckets"]:
                label = bucket["label"]
                op = bucket["op"]

                if op == "<":
                    mask = df[col] < bucket["threshold"]
                elif op == "<=":
                    mask = df[col] <= bucket["threshold"]
                elif op == ">":
                    mask = df[col] > bucket["threshold"]
                elif op == ">=":
                    mask = df[col] >= bucket["threshold"]
                elif op == "between":
                    mask = (df[col] >= bucket["low"]) & (df[col] < bucket["high"])
                else:
                    continue

                subset = df[mask]
                if len(subset) < 3:
                    continue

                pnl = subset["_pnl"]
                wr = (pnl > 0).mean() * 100
                avg = pnl.mean()

                bucket_result = {
                    "label": label,
                    "n": len(subset),
                    "win_rate": round(wr, 1),
                    "avg_pnl": round(avg, 2),
                    "median_pnl": round(pnl.median(), 2),
                }

                # Bootstrap CI
                ci = bootstrap_mean_pnl(pnl.values)
                if ci:
                    bucket_result["pnl_ci_lower"] = round(ci.ci_lower, 2)
                    bucket_result["pnl_ci_upper"] = round(ci.ci_upper, 2)

                bucket_results.append(bucket_result)

            if bucket_results:
                all_regime_results[rname] = bucket_results

                # Find best and worst buckets for comparison
                sorted_buckets = sorted(bucket_results, key=lambda x: x["avg_pnl"], reverse=True)
                if len(sorted_buckets) >= 2:
                    best_bucket = sorted_buckets[0]
                    worst_bucket = sorted_buckets[-1]
                    improvement = best_bucket["avg_pnl"] - worst_bucket["avg_pnl"]

                    # Fisher exact test between best and worst
                    p_value = None
                    if HAS_SCIPY and best_bucket["n"] >= 5 and worst_bucket["n"] >= 5:
                        b_wins = int(round(best_bucket["win_rate"] / 100 * best_bucket["n"]))
                        b_losses = best_bucket["n"] - b_wins
                        w_wins = int(round(worst_bucket["win_rate"] / 100 * worst_bucket["n"]))
                        w_losses = worst_bucket["n"] - w_wins
                        try:
                            _, p_value = fisher_exact([[b_wins, b_losses], [w_wins, w_losses]])
                        except Exception:
                            pass

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_finding = {
                            "regime": rname,
                            "best_bucket": best_bucket["label"],
                            "worst_bucket": worst_bucket["label"],
                            "improvement": round(improvement, 2),
                            "p_value": round(p_value, 4) if p_value else None,
                        }
                        best_p_value = p_value

        # Determine significance
        is_sig = (
            best_finding is not None
            and best_improvement >= config.min_effect_size
            and best_p_value is not None
            and best_p_value < config.alpha
        )

        # Summary
        if best_finding:
            summary = (
                f"Best regime split: {best_finding['regime']} — "
                f"{best_finding['best_bucket']} outperforms {best_finding['worst_bucket']} "
                f"by {best_finding['improvement']:+.2f}% avg P&L"
                f" (p={best_finding['p_value']})." if best_finding.get("p_value") else "."
            )
        else:
            summary = "No significant regime differences found."

        return ExperimentResult(
            experiment_id=self.make_id(),
            experiment_type=self.name,
            strategy=strategy,
            parameters=params,
            metrics={
                "baseline_expectancy": round(baseline_expectancy, 2),
                "n_total": len(df),
                "regimes": all_regime_results,
                "best_finding": best_finding,
            },
            summary=summary,
            statistical_tests=best_finding or {},
            is_significant=is_sig,
            metadata={"runtime_seconds": round(time.time() - t0, 1)},
        )
