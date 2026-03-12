"""Experiment registry — maps names to classes and provides a catalog for Claude."""

from typing import Dict, Type
from research.experiments.base import BaseExperiment


class ExperimentRegistry:
    """Central registry of all available experiment types."""

    def __init__(self):
        self._experiments: Dict[str, BaseExperiment] = {}

    def register(self, experiment: BaseExperiment):
        self._experiments[experiment.name] = experiment

    def get(self, name: str) -> BaseExperiment:
        if name not in self._experiments:
            raise KeyError(f"Unknown experiment: {name}. Available: {list(self._experiments.keys())}")
        return self._experiments[name]

    def list_names(self):
        return list(self._experiments.keys())

    def catalog(self) -> str:
        """Return a formatted catalog string for Claude's context window."""
        lines = ["AVAILABLE EXPERIMENTS:", ""]
        for name, exp in self._experiments.items():
            lines.append(f"## {name}")
            lines.append(f"Strategies: {', '.join(exp.supported_strategies)}")
            lines.append(exp.describe_capabilities())
            lines.append("")
        return "\n".join(lines)


def build_default_registry() -> ExperimentRegistry:
    """Build registry with all 8 experiment types."""
    registry = ExperimentRegistry()

    from research.experiments.filter_sweep import FilterSweepExperiment
    from research.experiments.feature_importance import FeatureImportanceExperiment
    from research.experiments.walk_forward_sensitivity import WalkForwardSensitivityExperiment
    from research.experiments.exit_optimization import ExitOptimizationExperiment
    from research.experiments.regime_analysis import RegimeAnalysisExperiment
    from research.experiments.entry_signal_comparison import EntrySignalComparisonExperiment
    from research.experiments.risk_metrics import RiskMetricsExperiment
    from research.experiments.backscanner_validation import BackscannerValidationExperiment

    for cls in [
        FilterSweepExperiment,
        FeatureImportanceExperiment,
        WalkForwardSensitivityExperiment,
        ExitOptimizationExperiment,
        RegimeAnalysisExperiment,
        EntrySignalComparisonExperiment,
        RiskMetricsExperiment,
        BackscannerValidationExperiment,
    ]:
        registry.register(cls())

    return registry
