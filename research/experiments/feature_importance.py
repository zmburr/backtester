"""Feature importance experiment — Spearman correlation + RandomForest importance."""

import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List

from research.experiments.base import BaseExperiment, ExperimentResult
from research.config import OUTCOME_COLUMNS

try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)

# Default feature columns to analyze (pre-trade / early-session only — no outcome data)
DEFAULT_FEATURES = [
    # Price momentum (known premarket)
    "pct_change_15", "pct_change_30", "pct_change_3", "pct_change_90", "pct_change_120",
    # Distance from moving averages (known premarket)
    "pct_from_10mav", "pct_from_20mav", "pct_from_50mav", "pct_from_9ema", "pct_from_200mav",
    "atr_distance_from_50mav",
    # Volatility / range (known premarket)
    "atr_pct",
    "one_day_before_range_pct",
    # Volume (premarket + early session)
    "percent_of_premarket_vol",
    "percent_of_vol_in_first_5_min", "percent_of_vol_in_first_10_min",
    "percent_of_vol_in_first_15_min", "percent_of_vol_in_first_30_min",
    "vol_ratio_5min_to_pm", "rvol_score",
    # Gap (known at open)
    "gap_pct", "gap_from_pm_high",
    # Bollinger bands (known from prior close)
    "upper_band_distance", "bollinger_width",
    # Prior day context (known premarket)
    "prior_day_close_vs_high_pct", "consecutive_up_days", "prior_day_range_atr",
    # Market context (known premarket)
    "spy_5day_return", "uvxy_close",
]

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


class FeatureImportanceExperiment(BaseExperiment):
    name = "feature_importance"
    description = "Rank features by predictive power for trade P&L using Spearman correlation and RandomForest."
    supported_strategies = ["reversal", "bounce"]

    def validate_params(self, params: dict) -> bool:
        return params.get("strategy") in self.supported_strategies

    def describe_capabilities(self) -> str:
        return (
            "Ranks all numeric features by their ability to predict trade P&L.\n"
            "Methods: Spearman rank correlation + RandomForest feature importance.\n"
            "Params:\n"
            "  - strategy: 'reversal' or 'bounce' (required)\n"
            "  - target: 'pnl' (continuous) or 'binary_win' (classification) (default: 'pnl')\n"
            "  - features: list of column names to analyze (default: all 30+ numeric columns)\n"
            "  - grade: filter to specific grade (default: None)\n"
            "  - cap: filter to specific cap size (default: None)\n"
            "Returns: ranked features by correlation and RF importance, top predictors."
        )

    def run(self, params: dict, config) -> ExperimentResult:
        t0 = time.time()
        strategy = params["strategy"]
        scfg = STRATEGY_CONFIG[strategy]
        target_mode = params.get("target", "pnl")
        features = params.get("features", DEFAULT_FEATURES)
        # Guard: strip any outcome columns Claude may have proposed
        features = [f for f in features if f not in OUTCOME_COLUMNS]
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
        pnl = df[scfg["target_col"]] * scfg["pnl_sign"] * 100

        # Filter to features that actually exist
        available = [f for f in features if f in df.columns]
        missing = [f for f in features if f not in df.columns]

        if len(available) < 3:
            return ExperimentResult(
                experiment_id=self.make_id(),
                experiment_type=self.name,
                strategy=strategy,
                parameters=params,
                metrics={"n": len(df), "available_features": len(available)},
                summary=f"Too few features available ({len(available)}).",
                statistical_tests={},
                is_significant=False,
                metadata={"missing_features": missing},
            )

        # 1. Spearman correlations
        spearman_results = []
        if HAS_SCIPY:
            for feat in available:
                valid = df[[feat]].copy()
                valid["pnl"] = pnl
                valid = valid.dropna()
                if len(valid) >= 10:
                    rho, p = spearmanr(valid[feat], valid["pnl"])
                    spearman_results.append({
                        "feature": feat,
                        "spearman_rho": round(float(rho), 4),
                        "p_value": round(float(p), 4),
                        "abs_rho": round(abs(float(rho)), 4),
                        "direction": "higher=better" if rho > 0 else "lower=better",
                        "n": len(valid),
                    })

        spearman_results.sort(key=lambda x: x["abs_rho"], reverse=True)

        # 2. RandomForest importance
        rf_results = []
        if HAS_SKLEARN:
            # Prepare feature matrix
            feat_df = df[available].copy()
            feat_df["pnl"] = pnl.values

            # Drop rows with NaN in any feature
            feat_df = feat_df.dropna()

            if len(feat_df) >= 20:
                X = feat_df[available].values
                if target_mode == "binary_win":
                    y = (feat_df["pnl"] > 0).astype(int).values
                    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                else:
                    y = feat_df["pnl"].values
                    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

                model.fit(X, y)
                importances = model.feature_importances_

                for feat, imp in zip(available, importances):
                    rf_results.append({
                        "feature": feat,
                        "rf_importance": round(float(imp), 4),
                    })

                rf_results.sort(key=lambda x: x["rf_importance"], reverse=True)

        # Merge rankings
        combined = {}
        for i, s in enumerate(spearman_results):
            combined[s["feature"]] = {
                "feature": s["feature"],
                "spearman_rho": s["spearman_rho"],
                "spearman_rank": i + 1,
                "spearman_p": s["p_value"],
                "direction": s["direction"],
            }

        for i, r in enumerate(rf_results):
            feat = r["feature"]
            if feat not in combined:
                combined[feat] = {"feature": feat}
            combined[feat]["rf_importance"] = r["rf_importance"]
            combined[feat]["rf_rank"] = i + 1

        # Composite rank (average of spearman and RF ranks)
        for feat, data in combined.items():
            s_rank = data.get("spearman_rank", len(available))
            r_rank = data.get("rf_rank", len(available))
            data["composite_rank"] = (s_rank + r_rank) / 2

        ranked = sorted(combined.values(), key=lambda x: x["composite_rank"])

        # Summary
        top_5 = ranked[:5]
        top_names = [f["feature"] for f in top_5]
        summary = (
            f"Top 5 predictors of {strategy} P&L: {', '.join(top_names)}. "
            f"Analyzed {len(available)} features across {len(df)} trades."
        )

        metrics = {
            "n_trades": len(df),
            "n_features_analyzed": len(available),
            "top_features": ranked[:10],
            "top_spearman": spearman_results[:10],
            "top_rf": rf_results[:10],
        }

        return ExperimentResult(
            experiment_id=self.make_id(),
            experiment_type=self.name,
            strategy=strategy,
            parameters=params,
            metrics=metrics,
            summary=summary,
            statistical_tests={
                "method": "spearman + random_forest",
                "n_significant_spearman": sum(1 for s in spearman_results if s["p_value"] < 0.05),
            },
            is_significant=False,  # Feature importance is exploratory, not a significance test
            metadata={
                "runtime_seconds": round(time.time() - t0, 1),
                "missing_features": missing[:5],
                "target_mode": target_mode,
            },
        )
