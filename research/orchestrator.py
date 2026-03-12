"""Research orchestrator — main loop that drives the autonomous experiment cycle."""

import logging
import time
import traceback
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from research.config import ResearchConfig
from research.experiments.registry import ExperimentRegistry, build_default_registry
from research.experiments.base import ExperimentResult
from research.claude_researcher import ClaudeResearcher
from research.result_store import ResultStore, get_session_dir
from research.statistical_guard import StatisticalGuard

logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    """Main research loop: Claude proposes -> experiments run -> Claude adapts."""

    def __init__(self, config: ResearchConfig = None, session_name: str = None):
        self.config = config or ResearchConfig()
        self.registry: ExperimentRegistry = build_default_registry()
        self.researcher = ClaudeResearcher(
            model=self.config.claude_model,
            timeout=self.config.claude_timeout,
        )
        self.guard = StatisticalGuard(
            alpha=self.config.alpha,
            min_sample=self.config.min_sample,
            min_effect_size=self.config.min_effect_size,
            correction=self.config.correction,
        )

        session_dir = get_session_dir(self.config.results_dir, session_name)
        self.store = ResultStore(session_dir)
        self.results: List[ExperimentResult] = self.store.load_all_results()
        self.start_time = time.time()

    def run(self) -> List[ExperimentResult]:
        """Execute the full research loop. Returns all results."""
        logger.info("=" * 60)
        logger.info("OVERNIGHT RESEARCH SESSION STARTING")
        logger.info(f"Strategies: {self.config.strategies}")
        logger.info(f"Budget: {self.config.max_iterations} iterations, {self.config.max_runtime_seconds}s")
        logger.info("=" * 60)

        try:
            # Build data overview for Claude
            data_overview = self._build_data_overview()
            catalog = self.registry.catalog()

            # Phase 1: Get initial experiment proposals from Claude
            iteration = self.store.get_completed_count()

            if iteration == 0:
                logger.info("Asking Claude for initial experiment proposals...")
                proposals = self.researcher.propose_initial_experiments(data_overview, catalog)
            else:
                logger.info(f"Resuming session at iteration {iteration}")
                # On resume, ask for follow-ups based on last result
                proposals = self._get_followups(catalog)

            # Main loop
            while self._within_budget(iteration):
                if not proposals:
                    logger.info("No more experiments proposed. Ending session.")
                    break

                # Run each proposed experiment
                for proposal in proposals:
                    if not self._within_budget(iteration):
                        break

                    exp_type = proposal.get("experiment_type", "")
                    params = proposal.get("parameters", {})
                    rationale = proposal.get("rationale", "")

                    logger.info(f"\n--- Iteration {iteration + 1} ---")
                    logger.info(f"Experiment: {exp_type}")
                    logger.info(f"Params: {params}")
                    logger.info(f"Rationale: {rationale}")

                    result = self._run_experiment(exp_type, params, iteration)
                    if result is None:
                        continue

                    self.results.append(result)
                    self.store.save_result(result, iteration)
                    iteration += 1

                    logger.info(f"Result: {'SIGNIFICANT' if result.is_significant else 'not significant'}")
                    logger.info(f"Summary: {result.summary}")

                # Ask Claude for follow-ups based on latest results
                if self._within_budget(iteration):
                    proposals = self._get_followups(catalog)
                else:
                    break

            # Session complete
            logger.info("\n" + "=" * 60)
            logger.info("SESSION COMPLETE")
            logger.info(f"Total experiments: {len(self.results)}")
            logger.info(f"Significant findings: {sum(1 for r in self.results if r.is_significant)}")
            logger.info(f"Runtime: {time.time() - self.start_time:.0f}s")
            logger.info(f"Claude calls: {self.researcher.call_count}")

            stats = self._session_stats()
            self.store.mark_complete(stats)

        except Exception as e:
            logger.error(f"Session failed: {e}")
            logger.error(traceback.format_exc())
            self.store.mark_failed(str(e))
            raise

        return self.results

    def get_synthesis(self) -> str:
        """Ask Claude for a final synthesis of all results."""
        if not self.results:
            return "No experiments completed."

        all_summaries = "\n\n".join(r.summary_for_claude() for r in self.results)
        stats = self._session_stats()
        return self.researcher.synthesize_findings(all_summaries, stats)

    def _run_experiment(self, exp_type: str, params: dict,
                        iteration: int) -> Optional[ExperimentResult]:
        """Run a single experiment with error handling."""
        try:
            experiment = self.registry.get(exp_type)
        except KeyError as e:
            logger.warning(f"Unknown experiment type '{exp_type}': {e}")
            return None

        if not experiment.validate_params(params):
            logger.warning(f"Invalid params for {exp_type}: {params}")
            return None

        try:
            result = experiment.run(params, self.config)

            # Link to parent if this is a follow-up
            if len(self.results) > 0:
                result.parent_id = self.results[-1].experiment_id

            return result

        except Exception as e:
            logger.error(f"Experiment {exp_type} failed: {e}")
            logger.error(traceback.format_exc())
            # Return a failure result so it's tracked
            return ExperimentResult(
                experiment_id=ExperimentResult.__dataclass_fields__["experiment_id"].default_factory()
                if hasattr(ExperimentResult.__dataclass_fields__["experiment_id"], "default_factory")
                else f"err-{iteration}",
                experiment_type=exp_type,
                strategy=params.get("strategy", "unknown"),
                parameters=params,
                metrics={},
                summary=f"FAILED: {str(e)[:200]}",
                statistical_tests={},
                is_significant=False,
                metadata={"error": str(e)},
            )

    def _get_followups(self, catalog: str) -> List[Dict]:
        """Ask Claude for follow-up experiments."""
        if not self.results:
            return []

        latest_summary = self.results[-1].summary_for_claude()
        all_summaries = "\n\n".join(r.summary_for_claude() for r in self.results)
        completed = [
            f"{r.experiment_type}:{r.strategy}:{json.dumps(r.parameters)}"
            for r in self.results
        ]

        proposals = self.researcher.propose_followups(
            latest_result_summary=latest_summary,
            all_results_summary=all_summaries,
            experiment_catalog=catalog,
            completed_experiments=completed,
        )
        return proposals

    def _within_budget(self, iteration: int) -> bool:
        """Check if we're still within time and iteration budgets."""
        elapsed = time.time() - self.start_time
        if iteration >= self.config.max_iterations:
            logger.info(f"Hit iteration limit ({self.config.max_iterations})")
            return False
        if elapsed >= self.config.max_runtime_seconds:
            logger.info(f"Hit time limit ({elapsed:.0f}s >= {self.config.max_runtime_seconds}s)")
            return False
        return True

    def _build_data_overview(self) -> str:
        """Build a compact overview of available data for Claude."""
        lines = ["DATA OVERVIEW:", ""]

        for strategy in self.config.strategies:
            if strategy == "reversal":
                csv_path = self.config.reversal_csv
                target_col = "reversal_open_close_pct"
                pnl_sign = -1
            elif strategy == "bounce":
                csv_path = self.config.bounce_csv
                target_col = "bounce_open_close_pct"
                pnl_sign = 1
            else:
                continue

            try:
                df = pd.read_csv(str(csv_path)).dropna(subset=["ticker", "date"])
                pnl = df[target_col] * pnl_sign * 100

                lines.append(f"## {strategy.upper()}")
                lines.append(f"Total trades: {len(df)}")
                lines.append(f"Date range: {df['date'].min()} to {df['date'].max()}")
                lines.append(f"Win rate: {(pnl > 0).mean() * 100:.1f}%")
                lines.append(f"Avg P&L (expectancy): {pnl.mean():+.2f}%")
                lines.append(f"Median P&L: {pnl.median():+.2f}%")

                if "trade_grade" in df.columns:
                    for grade in ["A", "B", "C"]:
                        g = df[df["trade_grade"] == grade]
                        if len(g) > 0:
                            g_pnl = g[target_col] * pnl_sign * 100
                            lines.append(
                                f"  Grade {grade}: N={len(g)}, "
                                f"WR={((g_pnl > 0).mean() * 100):.1f}%, "
                                f"Avg={g_pnl.mean():+.2f}%"
                            )

                if "cap" in df.columns:
                    for cap_val in df["cap"].dropna().unique():
                        c = df[df["cap"] == cap_val]
                        if len(c) >= 5:
                            c_pnl = c[target_col] * pnl_sign * 100
                            lines.append(
                                f"  {cap_val}: N={len(c)}, "
                                f"WR={((c_pnl > 0).mean() * 100):.1f}%, "
                                f"Avg={c_pnl.mean():+.2f}%"
                            )

                lines.append(f"Numeric columns: {len(df.select_dtypes(include='number').columns)}")
                lines.append("")

            except Exception as e:
                lines.append(f"## {strategy.upper()}: Error loading data: {e}")
                lines.append("")

        return "\n".join(lines)

    def _session_stats(self) -> Dict:
        return {
            "total_experiments": len(self.results),
            "significant_findings": sum(1 for r in self.results if r.is_significant),
            "runtime_seconds": round(time.time() - self.start_time, 1),
            "claude_calls": self.researcher.call_count,
            "strategies_tested": list(set(r.strategy for r in self.results)),
            "experiment_types_used": list(set(r.experiment_type for r in self.results)),
            "statistical_guard_summary": self.guard.summary(),
        }


# Need json import for _get_followups
import json
