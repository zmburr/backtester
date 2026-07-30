"""Standalone runner: generate the semis relative strength/weakness heatmaps
for a given date (default today) WITHOUT sending the daily email report.

Usage (from project root):
    python -m scripts.run_semis_rs_chart            # today
    python -m scripts.run_semis_rs_chart 2026-06-03 # specific date
"""
import sys
import datetime

from scripts.generate_report import (
    create_rs_momentum_heatmap,
    create_rs_absolute_heatmap,
    SEMI_NAMES,
)


def main() -> None:
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.datetime.now().strftime("%Y-%m-%d")
    print(f"Generating semis RS heatmaps for {date}")
    print(f"Universe ({len(SEMI_NAMES)} names): {', '.join(SEMI_NAMES)}\n")

    self_rel = create_rs_momentum_heatmap(date, output_dir="charts")
    print(f"Self-relative (strength/weakness) heatmap: {self_rel}")

    absolute = create_rs_absolute_heatmap(date, output_dir="charts")
    print(f"Absolute (shared-scale) heatmap:          {absolute}")

    if not self_rel and not absolute:
        print("\nNo charts produced — check benchmark/data availability above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
