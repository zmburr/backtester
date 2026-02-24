"""Standalone bootstrap confidence interval analysis for both reversal and bounce datasets.

Loads CSVs directly and computes CIs on key subsets without the full scoring pipeline.
Usage: python scripts/bootstrap_analysis.py
"""
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzers.bootstrap import bootstrap_win_rate, bootstrap_mean_pnl, format_ci

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def print_ci_row(label: str, pnl_values, indent: int = 2):
    """Print a single CI row with label, n, WR CI, and avg P&L CI."""
    n = len(pnl_values)
    if n == 0:
        return
    wr = bootstrap_win_rate(pnl_values)
    avg = bootstrap_mean_pnl(pnl_values)
    pad = ' ' * indent
    print(f'{pad}{label:30s}: {n:3d} trades | WR: {format_ci(wr)} | Avg: {format_ci(avg, is_pnl=True)}')


def analyze_reversal():
    """Bootstrap CIs for the reversal dataset."""
    path = os.path.join(DATA_DIR, 'reversal_data.csv')
    if not os.path.exists(path):
        print('reversal_data.csv not found, skipping.')
        return

    df = pd.read_csv(path)
    df['pnl'] = -df['reversal_open_close_pct'] * 100
    print(f'\nReversal dataset: {len(df)} trades')

    # Overall
    print_ci_row('Overall', df['pnl'].values)

    # By setup type
    print('\n  By setup type:')
    for setup in df['setup'].value_counts().index:
        subset = df[df['setup'] == setup]
        print_ci_row(setup, subset['pnl'].values, indent=4)

    # By grade
    print('\n  By grade:')
    for grade in ['A', 'B', 'C']:
        subset = df[df['trade_grade'] == grade]
        if len(subset) > 0:
            print_ci_row(f'Grade {grade}', subset['pnl'].values, indent=4)

    # By cap
    print('\n  By cap:')
    for cap in ['ETF', 'Large', 'Medium', 'Small', 'Micro']:
        subset = df[df['cap'] == cap]
        if len(subset) > 0:
            print_ci_row(cap, subset['pnl'].values, indent=4)


def analyze_bounce():
    """Bootstrap CIs for the bounce dataset."""
    path = os.path.join(DATA_DIR, 'bounce_data.csv')
    if not os.path.exists(path):
        print('bounce_data.csv not found, skipping.')
        return

    df = pd.read_csv(path)
    df['pnl'] = df['bounce_open_close_pct'] * 100
    # Exclude IntradayCapitch
    gf = df[~df['Setup'].str.contains('IntradayCapitch', case=False, na=False)].copy()
    print(f'\nBounce dataset: {len(gf)} GapFade trades (of {len(df)} total)')

    # Overall
    print_ci_row('Overall (GapFade)', gf['pnl'].dropna().values)

    # By setup
    print('\n  By setup:')
    for setup in gf['Setup'].value_counts().index:
        subset = gf[gf['Setup'] == setup]
        pnl = subset['pnl'].dropna()
        if len(pnl) > 0:
            print_ci_row(setup, pnl.values, indent=4)

    # By grade
    print('\n  By grade:')
    for grade in ['A', 'B', 'C']:
        subset = gf[gf['trade_grade'] == grade]
        pnl = subset['pnl'].dropna()
        if len(pnl) > 0:
            print_ci_row(f'Grade {grade}', pnl.values, indent=4)

    # By cap
    print('\n  By cap:')
    for cap in ['ETF', 'Large', 'Medium', 'Small']:
        subset = gf[gf['cap'] == cap]
        pnl = subset['pnl'].dropna()
        if len(pnl) > 0:
            print_ci_row(cap, pnl.values, indent=4)


if __name__ == '__main__':
    print('=' * 70)
    print('BOOTSTRAP CONFIDENCE INTERVALS (95%, BCa method, 10K resamples)')
    print('=' * 70)

    analyze_reversal()
    print()
    analyze_bounce()

    print('\n' + '=' * 70)
    print('Note: \u2020 = n < 10, CI uses percentile method (less reliable)')
    print('=' * 70)
