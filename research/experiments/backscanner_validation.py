"""Backscanner validation experiment — cross-reference scanner output vs actual trades."""

import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

from research.experiments.base import BaseExperiment, ExperimentResult
from analyzers.bootstrap import bootstrap_mean_pnl

logger = logging.getLogger(__name__)

STRATEGY_CONFIG = {
    "reversal": {
        "csv_name": "reversal_data.csv",
        "target_col": "reversal_open_close_pct",
        "pnl_sign": -1,
        "scanner_pattern": "backscanner_*.csv",
    },
    "bounce": {
        "csv_name": "bounce_data.csv",
        "target_col": "bounce_open_close_pct",
        "pnl_sign": 1,
        "scanner_pattern": "bounce_backscanner_*.csv",
    },
}


class BackscannerValidationExperiment(BaseExperiment):
    name = "backscanner_validation"
    description = "Cross-reference backscanner candidates vs actual trade outcomes."
    supported_strategies = ["reversal", "bounce"]

    def validate_params(self, params: dict) -> bool:
        return params.get("strategy") in self.supported_strategies

    def describe_capabilities(self) -> str:
        return (
            "Cross-references backscanner CSV output against actual trade data.\n"
            "Measures: scanner accuracy, missed opportunities, score distribution for winners vs losers.\n"
            "Params:\n"
            "  - strategy: 'reversal' or 'bounce' (required)\n"
            "  - scanner_csv: path to specific scanner CSV (default: auto-find latest)\n"
            "  - grade: filter trades to grade (default: None)\n"
            "  - min_score: minimum scanner score to count as 'detected' (default: 3)\n"
            "Returns: detection rate, false positive rate, score vs P&L correlation."
        )

    def run(self, params: dict, config) -> ExperimentResult:
        t0 = time.time()
        strategy = params["strategy"]
        scfg = STRATEGY_CONFIG[strategy]
        grade = params.get("grade")
        min_score = params.get("min_score", 3)

        # Load actual trade data
        trade_csv = config.data_dir / scfg["csv_name"]
        trades = pd.read_csv(str(trade_csv)).dropna(subset=["ticker", "date"])
        if grade:
            trades = trades[trades["trade_grade"] == grade].copy()

        trades["_pnl"] = trades[scfg["target_col"]] * scfg["pnl_sign"] * 100

        # Find scanner CSVs
        scanner_csv = params.get("scanner_csv")
        if scanner_csv:
            scanner_files = [Path(scanner_csv)]
        else:
            scanner_files = sorted(config.data_dir.glob(scfg["scanner_pattern"]))

        if not scanner_files:
            return ExperimentResult(
                experiment_id=self.make_id(),
                experiment_type=self.name,
                strategy=strategy,
                parameters=params,
                metrics={},
                summary=f"No backscanner CSVs found matching {scfg['scanner_pattern']}.",
                statistical_tests={},
                is_significant=False,
            )

        # Load and combine scanner data
        scanner_dfs = []
        for f in scanner_files:
            try:
                sdf = pd.read_csv(str(f))
                sdf["_source"] = f.name
                scanner_dfs.append(sdf)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")

        if not scanner_dfs:
            return ExperimentResult(
                experiment_id=self.make_id(),
                experiment_type=self.name,
                strategy=strategy,
                parameters=params,
                metrics={},
                summary="Failed to load any backscanner CSVs.",
                statistical_tests={},
                is_significant=False,
            )

        scanner = pd.concat(scanner_dfs, ignore_index=True)

        # Normalize date formats for matching
        for df in [trades, scanner]:
            if "date" in df.columns:
                df["_date_key"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            if "ticker" in df.columns:
                df["_ticker_key"] = df["ticker"].str.upper().str.strip()

        # Match trades to scanner detections
        trades["_matched"] = False
        trades["_scanner_score"] = np.nan
        trades["_scanner_grade"] = None

        for idx, trade in trades.iterrows():
            match = scanner[
                (scanner["_ticker_key"] == trade["_ticker_key"])
                & (scanner["_date_key"] == trade["_date_key"])
            ]
            if len(match) > 0:
                best_match = match.iloc[0]
                trades.loc[idx, "_matched"] = True
                if "score" in match.columns:
                    trades.loc[idx, "_scanner_score"] = best_match["score"]
                if "grade" in match.columns:
                    trades.loc[idx, "_scanner_grade"] = best_match["grade"]

        # Metrics
        n_trades = len(trades)
        n_matched = trades["_matched"].sum()
        detection_rate = n_matched / n_trades * 100 if n_trades > 0 else 0

        # Scanner candidates not in actual trades (potential false positives)
        scanner_detected = scanner[
            scanner.get("score", pd.Series(dtype=float)).fillna(0) >= min_score
        ] if "score" in scanner.columns else scanner
        n_scanner_candidates = len(scanner_detected)

        # P&L for matched vs unmatched
        matched_trades = trades[trades["_matched"]]
        unmatched_trades = trades[~trades["_matched"]]

        matched_pnl = matched_trades["_pnl"].mean() if len(matched_trades) > 0 else 0
        unmatched_pnl = unmatched_trades["_pnl"].mean() if len(unmatched_trades) > 0 else 0

        # Score vs P&L correlation for matched trades
        score_corr = None
        if len(matched_trades) >= 5 and "_scanner_score" in matched_trades.columns:
            valid = matched_trades[["_scanner_score", "_pnl"]].dropna()
            if len(valid) >= 5:
                try:
                    from scipy.stats import spearmanr
                    rho, p = spearmanr(valid["_scanner_score"], valid["_pnl"])
                    score_corr = {"rho": round(float(rho), 4), "p_value": round(float(p), 4)}
                except Exception:
                    pass

        # Winners vs losers score distribution
        score_dist = {}
        if "_scanner_score" in matched_trades.columns:
            winners = matched_trades[matched_trades["_pnl"] > 0]["_scanner_score"].dropna()
            losers = matched_trades[matched_trades["_pnl"] <= 0]["_scanner_score"].dropna()
            if len(winners) > 0:
                score_dist["winner_avg_score"] = round(winners.mean(), 2)
            if len(losers) > 0:
                score_dist["loser_avg_score"] = round(losers.mean(), 2)

        metrics = {
            "n_actual_trades": n_trades,
            "n_scanner_matched": int(n_matched),
            "detection_rate": round(detection_rate, 1),
            "n_scanner_candidates": n_scanner_candidates,
            "matched_avg_pnl": round(matched_pnl, 2),
            "unmatched_avg_pnl": round(unmatched_pnl, 2),
            "score_pnl_correlation": score_corr,
            "score_distribution": score_dist,
            "scanner_files_used": [f.name for f in scanner_files],
        }

        summary = (
            f"Scanner detected {detection_rate:.0f}% of actual trades "
            f"({int(n_matched)}/{n_trades}). "
            f"Matched avg P&L: {matched_pnl:+.2f}%, "
            f"Unmatched: {unmatched_pnl:+.2f}%."
        )

        return ExperimentResult(
            experiment_id=self.make_id(),
            experiment_type=self.name,
            strategy=strategy,
            parameters=params,
            metrics=metrics,
            summary=summary,
            statistical_tests={"score_pnl_correlation": score_corr} if score_corr else {},
            is_significant=False,  # validation is descriptive
            metadata={
                "runtime_seconds": round(time.time() - t0, 1),
                "scanner_files": len(scanner_files),
            },
        )
