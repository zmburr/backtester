"""Risk metrics experiment — Monte Carlo risk-of-ruin, Kelly criterion, drawdown analysis."""

import time
import logging
import pandas as pd
import numpy as np
from typing import Dict

from research.experiments.base import BaseExperiment, ExperimentResult

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


def monte_carlo_drawdown(pnl_returns: np.ndarray, initial_capital: float = 100000,
                         position_size_pct: float = 10, n_simulations: int = 10000,
                         n_trades: int = 200, ruin_threshold: float = 0.5,
                         seed: int = 42) -> Dict:
    """Run Monte Carlo simulation to estimate risk metrics.

    Args:
        pnl_returns: Array of per-trade P&L percentages (e.g., +5.0 means 5% gain).
        initial_capital: Starting capital.
        position_size_pct: % of capital risked per trade.
        n_simulations: Number of simulation paths.
        n_trades: Trades per simulation path.
        ruin_threshold: Fraction of capital loss that counts as ruin (0.5 = 50% drawdown).
        seed: Random seed.

    Returns:
        Dict with max_drawdowns, ruin_probability, median_final_capital, etc.
    """
    rng = np.random.default_rng(seed)
    max_drawdowns = np.empty(n_simulations)
    final_capitals = np.empty(n_simulations)
    max_consecutive_losses = np.empty(n_simulations, dtype=int)
    ruin_count = 0

    for sim in range(n_simulations):
        # Sample trades with replacement
        sampled = rng.choice(pnl_returns, size=n_trades, replace=True)
        capital = initial_capital
        peak = capital
        max_dd = 0
        consec_loss = 0
        max_consec = 0

        for trade_pnl in sampled:
            # Position size is % of current capital
            position = capital * (position_size_pct / 100)
            pnl_dollar = position * (trade_pnl / 100)
            capital += pnl_dollar

            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

            if trade_pnl <= 0:
                consec_loss += 1
                max_consec = max(max_consec, consec_loss)
            else:
                consec_loss = 0

            if capital <= initial_capital * (1 - ruin_threshold):
                ruin_count += 1
                break

        max_drawdowns[sim] = max_dd
        final_capitals[sim] = capital
        max_consecutive_losses[sim] = max_consec

    return {
        "ruin_probability": round(ruin_count / n_simulations * 100, 2),
        "median_max_drawdown": round(np.median(max_drawdowns) * 100, 1),
        "p95_max_drawdown": round(np.percentile(max_drawdowns, 95) * 100, 1),
        "p99_max_drawdown": round(np.percentile(max_drawdowns, 99) * 100, 1),
        "median_final_capital": round(np.median(final_capitals), 0),
        "p10_final_capital": round(np.percentile(final_capitals, 10), 0),
        "p90_final_capital": round(np.percentile(final_capitals, 90), 0),
        "median_max_consecutive_losses": int(np.median(max_consecutive_losses)),
        "p95_max_consecutive_losses": int(np.percentile(max_consecutive_losses, 95)),
    }


def kelly_criterion(win_rate: float, avg_winner: float, avg_loser: float) -> float:
    """Calculate Kelly criterion fraction.

    Args:
        win_rate: Win rate as decimal (0-1).
        avg_winner: Average winning trade return (positive).
        avg_loser: Average losing trade return (positive, absolute value).

    Returns:
        Kelly fraction (0-1), capped at 0.25 for safety.
    """
    if avg_loser == 0:
        return 0.25
    b = avg_winner / avg_loser  # Win/loss ratio
    kelly = (win_rate * b - (1 - win_rate)) / b
    return max(0, min(kelly, 0.25))  # Cap at 25%


class RiskMetricsExperiment(BaseExperiment):
    name = "risk_metrics"
    description = "Monte Carlo risk-of-ruin, Kelly criterion, drawdown analysis."
    supported_strategies = ["reversal", "bounce"]

    def validate_params(self, params: dict) -> bool:
        return params.get("strategy") in self.supported_strategies

    def describe_capabilities(self) -> str:
        return (
            "Computes risk metrics for a strategy via Monte Carlo simulation.\n"
            "Includes: risk of ruin, max drawdown distribution, Kelly criterion,\n"
            "consecutive loss streaks, and final capital distribution.\n"
            "Params:\n"
            "  - strategy: 'reversal' or 'bounce' (required)\n"
            "  - grade: filter to specific grade (default: None)\n"
            "  - cap: filter to specific cap (default: None)\n"
            "  - initial_capital: starting capital (default: 100000)\n"
            "  - position_size_pct: % of capital per trade (default: 10)\n"
            "  - n_simulations: Monte Carlo paths (default: 10000)\n"
            "  - n_trades_per_sim: trades per path (default: 200)\n"
            "  - ruin_threshold: drawdown % that counts as ruin (default: 0.5)\n"
            "Returns: risk of ruin %, max drawdown distribution, Kelly %, Sharpe-like ratio."
        )

    def run(self, params: dict, config) -> ExperimentResult:
        t0 = time.time()
        strategy = params["strategy"]
        scfg = STRATEGY_CONFIG[strategy]
        grade = params.get("grade")
        cap = params.get("cap")
        initial_capital = params.get("initial_capital", 100000)
        position_size_pct = params.get("position_size_pct", 10)
        n_simulations = params.get("n_simulations", 10000)
        n_trades = params.get("n_trades_per_sim", 200)
        ruin_threshold = params.get("ruin_threshold", 0.5)

        # Load data
        csv_path = config.data_dir / scfg["csv_name"]
        df = pd.read_csv(str(csv_path)).dropna(subset=["ticker", "date"])

        if grade:
            df = df[df["trade_grade"] == grade].copy()
        if cap:
            df = df[df["cap"] == cap].copy()

        pnl = df[scfg["target_col"]] * scfg["pnl_sign"] * 100
        pnl = pnl.dropna().values

        if len(pnl) < 10:
            return ExperimentResult(
                experiment_id=self.make_id(),
                experiment_type=self.name,
                strategy=strategy,
                parameters=params,
                metrics={"n": len(pnl)},
                summary=f"Insufficient data: {len(pnl)} trades.",
                statistical_tests={},
                is_significant=False,
            )

        # Basic stats
        win_rate = (pnl > 0).mean()
        avg_pnl = pnl.mean()
        std_pnl = pnl.std()
        winners = pnl[pnl > 0]
        losers = pnl[pnl <= 0]
        avg_winner = winners.mean() if len(winners) > 0 else 0
        avg_loser = abs(losers.mean()) if len(losers) > 0 else 0

        # Kelly criterion
        kelly_frac = kelly_criterion(win_rate, avg_winner, avg_loser)

        # Sharpe-like ratio (mean / std of trade returns)
        sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0

        # Monte Carlo
        mc_results = monte_carlo_drawdown(
            pnl, initial_capital=initial_capital,
            position_size_pct=position_size_pct,
            n_simulations=n_simulations,
            n_trades=n_trades,
            ruin_threshold=ruin_threshold,
        )

        metrics = {
            "n_trades": len(pnl),
            "win_rate": round(win_rate * 100, 1),
            "avg_pnl": round(avg_pnl, 2),
            "std_pnl": round(std_pnl, 2),
            "avg_winner": round(avg_winner, 2),
            "avg_loser": round(avg_loser, 2),
            "kelly_fraction": round(kelly_frac * 100, 1),
            "sharpe_ratio": round(sharpe, 3),
            "monte_carlo": mc_results,
        }

        # Summary
        summary = (
            f"Risk analysis ({strategy}): "
            f"Kelly={kelly_frac*100:.1f}%, "
            f"Ruin prob={mc_results['ruin_probability']}%, "
            f"Median max DD={mc_results['median_max_drawdown']}%, "
            f"P95 max DD={mc_results['p95_max_drawdown']}%. "
            f"Sharpe-like ratio={sharpe:.3f}."
        )

        return ExperimentResult(
            experiment_id=self.make_id(),
            experiment_type=self.name,
            strategy=strategy,
            parameters=params,
            metrics=metrics,
            summary=summary,
            statistical_tests={
                "kelly_criterion": round(kelly_frac * 100, 1),
                "ruin_probability": mc_results["ruin_probability"],
            },
            is_significant=False,  # risk metrics are descriptive
            metadata={
                "runtime_seconds": round(time.time() - t0, 1),
                "n_simulations": n_simulations,
            },
        )
