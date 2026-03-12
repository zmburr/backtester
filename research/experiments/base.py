"""Base classes for all experiments."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import uuid
from datetime import datetime


@dataclass
class ExperimentResult:
    """Standardized output from any experiment."""

    experiment_id: str
    experiment_type: str  # registry name
    strategy: str  # "reversal" or "bounce"
    parameters: Dict
    metrics: Dict  # key output numbers
    summary: str  # 2-3 sentence plain-English finding
    statistical_tests: Dict  # p-values, CIs, effect size
    is_significant: bool  # passed Holm-corrected threshold
    metadata: Dict = field(default_factory=dict)  # runtime, sample sizes, warnings
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    parent_id: Optional[str] = None  # which prior result spawned this

    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "experiment_type": self.experiment_type,
            "strategy": self.strategy,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "summary": self.summary,
            "statistical_tests": self.statistical_tests,
            "is_significant": self.is_significant,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "parent_id": self.parent_id,
        }

    def summary_for_claude(self) -> str:
        """Compact summary for Claude context window."""
        sig = "SIGNIFICANT" if self.is_significant else "not significant"
        lines = [
            f"[{self.experiment_type}] {self.strategy} — {sig}",
            f"  Params: {self.parameters}",
            f"  Key metrics: {self.metrics}",
            f"  {self.summary}",
        ]
        if self.statistical_tests:
            lines.append(f"  Stats: {self.statistical_tests}")
        return "\n".join(lines)


class BaseExperiment(ABC):
    """Abstract base for all experiment types."""

    name: str = ""
    description: str = ""
    supported_strategies: List[str] = []

    @abstractmethod
    def run(self, params: dict, config) -> ExperimentResult:
        """Execute the experiment and return a result."""
        ...

    @abstractmethod
    def validate_params(self, params: dict) -> bool:
        """Check that params are valid before running."""
        ...

    @abstractmethod
    def describe_capabilities(self) -> str:
        """Return a description for Claude's context about what this experiment can do."""
        ...

    @staticmethod
    def make_id() -> str:
        return str(uuid.uuid4())[:8]
