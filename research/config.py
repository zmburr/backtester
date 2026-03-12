"""Configuration for the overnight research system."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ResearchConfig:
    """All tunables for a research session."""

    # Strategies to explore
    strategies: List[str] = field(default_factory=lambda: ["reversal", "bounce"])

    # Session budget
    max_iterations: int = 15
    max_runtime_seconds: int = 7200  # 2 hours

    # Statistical guard
    min_sample: int = 10
    alpha: float = 0.05
    correction: str = "holm"  # Holm-Bonferroni
    min_effect_size: float = 1.0  # minimum absolute improvement in expectancy (%)

    # Polygon API budget (calls per session)
    polygon_budget: int = 500

    # Claude CLI
    claude_model: str = "opus"
    claude_timeout: int = 120  # seconds per call

    # Paths
    project_root: Path = _PROJECT_ROOT
    data_dir: Path = field(default=None)
    results_dir: Path = field(default=None)
    reports_dir: Path = field(default=None)

    # Email
    email_to: str = "zmburr@gmail.com"

    # Memory
    memory_dir: Path = field(default=None)

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.results_dir is None:
            self.results_dir = self.project_root / "research" / "results" / "sessions"
        if self.reports_dir is None:
            self.reports_dir = self.project_root / "research" / "reports"
        if self.memory_dir is None:
            self.memory_dir = self.project_root / "research" / "memory"

    @property
    def reversal_csv(self) -> Path:
        return self.data_dir / "reversal_data.csv"

    @property
    def bounce_csv(self) -> Path:
        return self.data_dir / "bounce_data.csv"

    @property
    def exit_analysis_csv(self) -> Path:
        return self.data_dir / "exit_analysis_results.csv"
