"""Walk-forward validation entry point.

Runs the real out-of-sample pipeline: derive thresholds on the training window,
score the unconditioned candidate population out-of-sample with both the derived
and the production scorers, and print the outcome-conditional comparison report.

Usage:
    python -m scripts.run_walk_forward                          # all strategies
    python -m scripts.run_walk_forward --strategy reversal      # single strategy
    python -m scripts.run_walk_forward --train-end 2022-12-31   # custom split

Artifacts (written every run, untracked by git):
    data/derived_thresholds_{strategy}.json   # derived thresholds + logs + window
    validation/reports/{strategy}_walk_forward.txt   # full text report
"""

import argparse
import sys
import os
import io
import json
import logging
import contextlib
from dataclasses import asdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.walk_forward_engine import run_walk_forward
from validation.report import print_full_report

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_ROOT, 'data')
_REPORT_DIR = os.path.join(_ROOT, 'validation', 'reports')


def _thresholds_to_json(result) -> dict:
    """JSON-serializable view of the derived thresholds + provenance."""
    derived = result.derived
    out = {
        'strategy': result.strategy,
        'train_window': {
            'train_end': result.split.train_end if result.split else None,
            'validate_end': result.split.validate_end if result.split else None,
            'train_trades': result.diagnostics.train_n if result.diagnostics else None,
            'train_date_range': result.diagnostics.train_date_range if result.diagnostics else None,
        },
        'derivation_log': list(derived.derivation_log) if derived else [],
        'cap_pooling_log': list(derived.cap_pooling_log) if derived else [],
    }
    if derived is None:
        return out

    if derived.reversal_cap_thresholds:
        out['reversal_cap_thresholds'] = {
            cap: asdict(t) for cap, t in derived.reversal_cap_thresholds.items()
        }
    if derived.reversal_setup_profiles:
        out['reversal_setup_profiles'] = {
            name: asdict(p) for name, p in derived.reversal_setup_profiles.items()
        }
    if derived.reversal_readiness_thresholds:
        out['reversal_readiness_thresholds'] = derived.reversal_readiness_thresholds
    if derived.reversal_ref_by_cap_group is not None:
        # Reference frames are large — record shape/date-range provenance only.
        out['reversal_ref_by_cap_group'] = {
            group: {
                'n': int(len(sub)),
                'date_min': str(sub['date'].min()) if 'date' in sub.columns and len(sub) else None,
                'date_max': str(sub['date'].max()) if 'date' in sub.columns and len(sub) else None,
            }
            for group, sub in derived.reversal_ref_by_cap_group.items()
        }
    if derived.bounce_setup_profiles:
        out['bounce_setup_profiles'] = {
            name: asdict(p) for name, p in derived.bounce_setup_profiles.items()
        }
    return out


def _persist_artifacts(result):
    """Write derived thresholds JSON and the full text report to disk."""
    os.makedirs(_REPORT_DIR, exist_ok=True)

    thresh_path = os.path.join(_DATA_DIR, f'derived_thresholds_{result.strategy}.json')
    with open(thresh_path, 'w', encoding='utf-8') as f:
        json.dump(_thresholds_to_json(result), f, indent=2, default=str)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_full_report(result)
    report_text = buf.getvalue()

    report_path = os.path.join(_REPORT_DIR, f'{result.strategy}_walk_forward.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    # Echo to console too.
    print(report_text)
    print(f"  [artifact] thresholds -> {thresh_path}")
    print(f"  [artifact] report     -> {report_path}")


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
                        help='Override curated CSV path (single strategy only)')
    parser.add_argument('--population', type=str, default=None,
                        help='Override candidate population CSV path (single strategy only)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
                population_path=args.population,
            )

            _persist_artifacts(result)
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

        print(f"\n  {'Strategy':<12} {'Base WR':>10} {'OOS WR':>10} {'Delta':>8} {'Verdict':>12}")
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
