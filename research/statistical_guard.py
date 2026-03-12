"""Statistical guard — Holm-Bonferroni correction, min-N, effect size, bootstrap gates."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from analyzers.bootstrap import bootstrap_mean_pnl, bootstrap_win_rate, BootstrapResult

logger = logging.getLogger(__name__)


class StatisticalGuard:
    """Tracks all hypothesis tests in a session and applies corrections."""

    def __init__(self, alpha: float = 0.05, min_sample: int = 10,
                 min_effect_size: float = 1.0, correction: str = "holm"):
        self.alpha = alpha
        self.min_sample = min_sample
        self.min_effect_size = min_effect_size  # absolute % improvement in expectancy
        self.correction = correction
        self._p_values: List[Tuple[str, float]] = []  # (experiment_id, p_value)

    def register_test(self, experiment_id: str, p_value: float):
        """Register a p-value from a hypothesis test."""
        self._p_values.append((experiment_id, p_value))

    def get_holm_threshold(self, experiment_id: str) -> float:
        """Get the Holm-Bonferroni corrected threshold for a specific test."""
        if not self._p_values:
            return self.alpha

        # Sort all p-values
        sorted_pvals = sorted(self._p_values, key=lambda x: x[1])
        m = len(sorted_pvals)

        for rank, (eid, pval) in enumerate(sorted_pvals):
            threshold = self.alpha / (m - rank)
            if eid == experiment_id:
                return threshold

        return self.alpha  # fallback

    def evaluate(
        self,
        experiment_id: str,
        n_treatment: int,
        n_control: int,
        p_value: Optional[float],
        effect_size: float,
        pnl_array_treatment: np.ndarray = None,
        baseline_expectancy: float = 0.0,
    ) -> Dict:
        """
        Full evaluation of a finding.

        Returns dict with:
          - passes_min_n: bool
          - passes_effect_size: bool
          - passes_significance: bool
          - passes_bootstrap_ci: bool
          - is_significant: bool (all gates pass)
          - holm_threshold: float
          - bootstrap_ci: dict or None
          - reasons: list of failure reasons
        """
        result = {
            "passes_min_n": True,
            "passes_effect_size": True,
            "passes_significance": True,
            "passes_bootstrap_ci": True,
            "is_significant": False,
            "holm_threshold": None,
            "bootstrap_ci": None,
            "reasons": [],
        }

        # Gate 1: Minimum sample size
        if n_treatment < self.min_sample:
            result["passes_min_n"] = False
            result["reasons"].append(f"N={n_treatment} < min_sample={self.min_sample}")

        if n_control < self.min_sample:
            result["passes_min_n"] = False
            result["reasons"].append(f"N_control={n_control} < min_sample={self.min_sample}")

        # Gate 2: Effect size (absolute improvement in expectancy %)
        if abs(effect_size) < self.min_effect_size:
            result["passes_effect_size"] = False
            result["reasons"].append(
                f"Effect size {effect_size:.2f}% < min {self.min_effect_size}%"
            )

        # Gate 3: Statistical significance with Holm-Bonferroni
        if p_value is not None:
            self.register_test(experiment_id, p_value)
            holm_threshold = self.get_holm_threshold(experiment_id)
            result["holm_threshold"] = holm_threshold

            if p_value >= holm_threshold:
                result["passes_significance"] = False
                result["reasons"].append(
                    f"p={p_value:.4f} >= Holm threshold={holm_threshold:.4f}"
                )
        else:
            # No p-value available — can't pass significance
            result["passes_significance"] = False
            result["reasons"].append("No p-value available")

        # Gate 4: Bootstrap CI lower bound must show improvement
        if pnl_array_treatment is not None and len(pnl_array_treatment) >= 5:
            ci = bootstrap_mean_pnl(pnl_array_treatment)
            if ci is not None:
                result["bootstrap_ci"] = {
                    "point": round(ci.point_estimate, 2),
                    "ci_lower": round(ci.ci_lower, 2),
                    "ci_upper": round(ci.ci_upper, 2),
                    "n": ci.n,
                    "method": ci.method,
                }
                # CI lower bound must still beat baseline
                if ci.ci_lower <= baseline_expectancy:
                    result["passes_bootstrap_ci"] = False
                    result["reasons"].append(
                        f"Bootstrap CI lower ({ci.ci_lower:.2f}) <= baseline ({baseline_expectancy:.2f})"
                    )

        # Final verdict
        result["is_significant"] = all([
            result["passes_min_n"],
            result["passes_effect_size"],
            result["passes_significance"],
            result["passes_bootstrap_ci"],
        ])

        return result

    @property
    def total_tests(self) -> int:
        return len(self._p_values)

    def summary(self) -> str:
        """Return a summary of all registered tests."""
        if not self._p_values:
            return "No tests registered yet."
        sorted_pvals = sorted(self._p_values, key=lambda x: x[1])
        m = len(sorted_pvals)
        lines = [f"Total tests: {m}, Base alpha: {self.alpha}"]
        for rank, (eid, pval) in enumerate(sorted_pvals):
            threshold = self.alpha / (m - rank)
            sig = "PASS" if pval < threshold else "fail"
            lines.append(f"  {eid}: p={pval:.4f} vs threshold={threshold:.4f} [{sig}]")
        return "\n".join(lines)
