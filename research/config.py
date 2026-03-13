"""Configuration for the overnight research system."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Column classifications: pre-trade (predictive) vs outcome (post-trade)
#
# OUTCOME columns are data you only know AFTER the trade plays out.
# They must NEVER be used as predictive features — only as target variables.
# ---------------------------------------------------------------------------

OUTCOME_COLUMNS: Set[str] = {
    # --- Trade result / P&L ---
    "trade_grade",                          # assigned after the trade
    "bp", "npl", "size",                    # actual trading P&L and position size

    # --- Reversal outcome ---
    "reversal_open_close_pct",              # target variable (reversal P&L)
    "reversal_open_low_pct",                # max adverse excursion
    "reversal_open_post_low_pct",           # recovery after low
    "reversal_open_to_day_after_open_pct",  # overnight hold result
    "reversal_duration",                    # how long reversal took
    "time_of_reversal",                     # when reversal occurred

    # --- Bounce outcome ---
    "bounce_open_close_pct",                # target variable (bounce P&L)
    "bounce_open_high_pct",                 # max favorable excursion
    "bounce_open_low_pct",                  # max adverse excursion
    "bounce_open_to_day_after_open_pct",    # overnight hold result
    "bounce_duration",                      # how long bounce took
    "time_of_bounce",                       # when bounce occurred

    # --- Same-day price action (only known at/after close) ---
    "day_of_range_pct",                     # intraday range on trade day
    "close_at_lows",                        # did it close at lows
    "close_at_highs",                       # did it close at highs
    "close_green_red",                      # close direction
    "close_above_prior_close",              # close vs prior close
    "hit_green_red",                        # hit green/red during session
    "hit_prior_day_hilo",                   # hit prior day high/low
    "move_together",                        # stock & SPY same direction (same-day)
    "time_of_high_price",                   # when HOD occurred
    "time_of_high_bucket",                  # HOD time bucket
    "time_of_low",                          # when LOD occurred (reversal)
    "time_of_low_price",                    # when LOD occurred (bounce)
    "time_of_low_bucket",                   # LOD time bucket
    "high_to_low_duration_min",             # HOD-to-LOD duration

    # --- Same-day volume (total day only known at close) ---
    "vol_on_breakout_day",                  # total volume on trade day
    "percent_of_vol_on_breakout_day",       # % of avg volume on trade day

    # --- Same-day SPY (only known at close) ---
    "spy_open_close_pct",                   # SPY return on trade day
}

# Early-session columns: available minutes into the session, NOT premarket.
# These are valid predictive features but only if entry is after that window.
EARLY_SESSION_COLUMNS: Set[str] = {
    "vol_in_first_5_min", "vol_in_first_10_min",
    "vol_in_first_15_min", "vol_in_first_30_min",
    "percent_of_vol_in_first_5_min", "percent_of_vol_in_first_10_min",
    "percent_of_vol_in_first_15_min", "percent_of_vol_in_first_30_min",
    "vol_ratio_5min_to_pm",
}


def is_predictive(column: str) -> bool:
    """Return True if a column is available at decision time (pre-trade)."""
    return column not in OUTCOME_COLUMNS


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
