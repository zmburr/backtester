"""CLI entry point for the overnight research system.

Usage:
    python scripts/run_overnight_research.py
    python scripts/run_overnight_research.py --max-iterations 5 --dry-run
    python scripts/run_overnight_research.py --resume 2026-03-12_22-00
"""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.config import ResearchConfig
from research.orchestrator import ResearchOrchestrator
from research.report_generator import save_and_send_report


def setup_logging(log_dir: str, verbose: bool = False):
    """Configure logging to both file and console."""
    os.makedirs(log_dir, exist_ok=True)
    from datetime import datetime
    log_file = os.path.join(log_dir, f"research_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log")

    handlers = [
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
    return log_file


def main():
    parser = argparse.ArgumentParser(
        description="Overnight Backtester Researcher — autonomous experiment loop"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=15,
        help="Maximum experiments per session (default: 15)"
    )
    parser.add_argument(
        "--max-runtime", type=int, default=7200,
        help="Maximum runtime in seconds (default: 7200 = 2 hours)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume a prior session by name (e.g., 2026-03-12_22-00)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run 3 iterations max, skip email"
    )
    parser.add_argument(
        "--no-email", action="store_true",
        help="Skip sending email"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--strategies", type=str, default="reversal,bounce",
        help="Comma-separated strategies (default: reversal,bounce)"
    )

    args = parser.parse_args()

    # Config
    config = ResearchConfig(
        strategies=args.strategies.split(","),
        max_iterations=3 if args.dry_run else args.max_iterations,
        max_runtime_seconds=300 if args.dry_run else args.max_runtime,
    )

    # Logging
    log_dir = str(config.project_root / "research" / "logs")
    log_file = setup_logging(log_dir, args.verbose)
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")

    # Run
    try:
        orchestrator = ResearchOrchestrator(config=config, session_name=args.resume)
        results = orchestrator.run()

        if not results:
            logger.info("No experiments completed. Exiting.")
            return

        # Synthesis
        logger.info("Generating synthesis...")
        synthesis = orchestrator.get_synthesis()

        # Report
        stats = orchestrator._session_stats()

        if args.dry_run or args.no_email:
            # Just save the report, don't email
            from research.report_generator import generate_markdown_report
            config.reports_dir.mkdir(parents=True, exist_ok=True)
            from datetime import datetime
            report_path = config.reports_dir / f"report_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.md"
            md = generate_markdown_report(results, synthesis, stats, config)
            report_path.write_text(md, encoding="utf-8")
            logger.info(f"Report saved to {report_path} (email skipped)")
        else:
            report_path = save_and_send_report(results, synthesis, stats, config)
            logger.info(f"Report saved and emailed: {report_path}")

        # Print summary
        n_sig = sum(1 for r in results if r.is_significant)
        print(f"\n{'='*60}")
        print(f"SESSION COMPLETE")
        print(f"  Experiments: {len(results)}")
        print(f"  Significant: {n_sig}")
        print(f"  Runtime: {stats['runtime_seconds']:.0f}s")
        print(f"  Report: {report_path}")
        print(f"{'='*60}")

    except KeyboardInterrupt:
        logger.info("Session interrupted by user. Results saved for resume.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
