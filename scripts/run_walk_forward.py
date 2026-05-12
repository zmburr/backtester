"""Walk-forward validation entry point.

Usage:
    python scripts/run_walk_forward.py                          # all strategies
    python scripts/run_walk_forward.py --strategy reversal       # single strategy
    python scripts/run_walk_forward.py --train-end 2022-12-31   # custom split
"""

import argparse
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.walk_forward_engine import run_walk_forward
from validation.report import print_full_report


def main():
    parser = argparse.ArgumentParser(description='Walk-forward validation for backtester scoring systems')
    parser.add_argument('--strategy', '-s', type=str, default=None,
                        choices=['reversal', 'bounce', 'breakout', 'all'],
                        help='Strategy to validate (default: all)')
    parser.add_argument('--train-end', type=str, default='2023-12-31',
                        help='Last date of training period (default: 2023-12-31)')
    parser.add_argument('--validate-end', type=str, default='2024-12-31',
                        help='Last date of validation period (default: 2024-12-31)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Override CSV path (for single strategy)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING)

    strategies = ['reversal', 'bounce', 'breakout']
    if args.strategy and args.strategy != 'all':
        strategies = [args.strategy]

    print(f"\nWalk-Forward Validation")
    print(f"Train:    <= {args.train_end}")
    print(f"Validate: {args.train_end} - {args.validate_end}")
    print(f"Test:     > {args.validate_end}")
    print(f"Strategies: {', '.join(strategies)}")

    results = {}

    for strategy in strategies:
        try:
            print(f"\n{'=' * 80}")
            print(f"Running walk-forward for: {strategy.upper()}")
            print(f"{'=' * 80}")

            result = run_walk_forward(
                strategy=strategy,
                train_end=args.train_end,
                validate_end=args.validate_end,
                csv_path=args.csv,
            )

            print_full_report(result)
            results[strategy] = result

        except FileNotFoundError as e:
            print(f"\n  SKIPPED {strategy}: CSV not found ({e})")
        except Exception as e:
            print(f"\n  ERROR running {strategy}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # Cross-strategy summary
    if len(results) > 1:
        print(f"\n{'#' * 80}")
        print(f"#  {'CROSS-STRATEGY SUMMARY':<74}  #")
        print(f"{'#' * 80}")

        print(f"\n  {'Strategy':<12} {'Train WR':>10} {'OOS WR':>10} {'Delta':>8} {'Verdict':>12}")
        print(f"  {'-' * 55}")
        for strategy, result in results.items():
            t = result.train_metrics
            v = result.validate_metrics
            deg = result.train_vs_validate
            if t and v:
                verdict = deg.verdict.upper() if deg else '?'
                delta = deg.win_rate_change_pp if deg else 0
                print(f"  {strategy:<12} {t.win_rate:>9.1f}% {v.win_rate:>9.1f}% {delta:>+7.1f}pp {verdict:>12}")

    print("\nDone.")


if __name__ == '__main__':
    main()
