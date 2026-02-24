"""Standalone bootstrap confidence interval analysis for both reversal and bounce datasets.

Loads CSVs directly and computes CIs on key subsets without the full scoring pipeline.
Usage: python scripts/bootstrap_analysis.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzers.bootstrap import bootstrap_win_rate, bootstrap_mean_pnl, format_ci, BootstrapResult

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
CHARTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'charts')


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


def _collect_ci_rows(df, pnl_col, groups):
    """Collect bootstrap results for a list of (label, subset_pnl_array) pairs."""
    rows = []
    for label, pnl_values in groups:
        if len(pnl_values) == 0:
            continue
        wr = bootstrap_win_rate(pnl_values)
        avg = bootstrap_mean_pnl(pnl_values)
        if wr and avg:
            rows.append((label, len(pnl_values), wr, avg))
    return rows


def generate_forest_plot():
    """Generate a two-panel forest plot (WR + Avg P&L) for key subsets."""
    rev_path = os.path.join(DATA_DIR, 'reversal_data.csv')
    bounce_path = os.path.join(DATA_DIR, 'bounce_data.csv')

    sections = []  # list of (section_title, rows) where rows = [(label, n, wr_ci, avg_ci)]

    # --- Reversal ---
    if os.path.exists(rev_path):
        df_rev = pd.read_csv(rev_path)
        df_rev['pnl'] = -df_rev['reversal_open_close_pct'] * 100

        # Overall
        sections.append(('REVERSAL - Overall', _collect_ci_rows(
            df_rev, 'pnl', [('All Reversals', df_rev['pnl'].values)])))

        # By setup (top 5)
        top_setups = df_rev['setup'].value_counts().head(5).index
        groups = [(s, df_rev.loc[df_rev['setup'] == s, 'pnl'].values) for s in top_setups]
        sections.append(('Reversal - By Setup', _collect_ci_rows(df_rev, 'pnl', groups)))

        # By grade
        groups = [(f'Grade {g}', df_rev.loc[df_rev['trade_grade'] == g, 'pnl'].values) for g in ['A', 'B', 'C']]
        sections.append(('Reversal - By Grade', _collect_ci_rows(df_rev, 'pnl', groups)))

        # By cap
        groups = [(c, df_rev.loc[df_rev['cap'] == c, 'pnl'].values) for c in ['ETF', 'Large', 'Medium', 'Small', 'Micro']]
        sections.append(('Reversal - By Cap', _collect_ci_rows(df_rev, 'pnl', groups)))

    # --- Bounce ---
    if os.path.exists(bounce_path):
        df_b = pd.read_csv(bounce_path)
        df_b['pnl'] = df_b['bounce_open_close_pct'] * 100
        gf = df_b[~df_b['Setup'].str.contains('IntradayCapitch', case=False, na=False)].copy()

        sections.append(('BOUNCE - Overall', _collect_ci_rows(
            gf, 'pnl', [('All Bounces', gf['pnl'].dropna().values)])))

        # By setup (top 4)
        top_setups = gf['Setup'].value_counts().head(4).index
        groups = [(s, gf.loc[gf['Setup'] == s, 'pnl'].dropna().values) for s in top_setups]
        sections.append(('Bounce - By Setup', _collect_ci_rows(gf, 'pnl', groups)))

        # By grade
        groups = [(f'Grade {g}', gf.loc[gf['trade_grade'] == g, 'pnl'].dropna().values) for g in ['A', 'B', 'C']]
        sections.append(('Bounce - By Grade', _collect_ci_rows(gf, 'pnl', groups)))

        # By cap
        groups = [(c, gf.loc[gf['cap'] == c, 'pnl'].dropna().values) for c in ['ETF', 'Large', 'Medium', 'Small']]
        sections.append(('Bounce - By Cap', _collect_ci_rows(gf, 'pnl', groups)))

    # --- Build the plot ---
    # Flatten into ordered labels with section headers
    labels = []
    wr_points, wr_lows, wr_highs = [], [], []
    avg_points, avg_lows, avg_highs = [], [], []
    ns = []
    is_header = []
    small_n = []

    for section_title, rows in reversed(sections):  # reversed so top section is at top of chart
        if not rows:
            continue
        for label, n, wr, avg in reversed(rows):
            labels.append(f'{label} (n={n})')
            wr_points.append(wr.point_estimate)
            wr_lows.append(wr.ci_lower)
            wr_highs.append(wr.ci_upper)
            avg_points.append(avg.point_estimate)
            avg_lows.append(avg.ci_lower)
            avg_highs.append(avg.ci_upper)
            ns.append(n)
            is_header.append(False)
            small_n.append(n < 10)
        # Section header spacer
        labels.append(f'  {section_title}')
        wr_points.append(np.nan)
        wr_lows.append(np.nan)
        wr_highs.append(np.nan)
        avg_points.append(np.nan)
        avg_lows.append(np.nan)
        avg_highs.append(np.nan)
        ns.append(0)
        is_header.append(True)
        small_n.append(False)

    n_rows = len(labels)
    y_pos = np.arange(n_rows)

    fig, (ax_wr, ax_pnl) = plt.subplots(1, 2, figsize=(18, max(10, n_rows * 0.38)),
                                          sharey=True, gridspec_kw={'wspace': 0.08})

    # Colors
    color_main = '#2563EB'
    color_small = '#F59E0B'
    bg_dark = '#0F172A'
    bg_card = '#1E293B'
    text_color = '#E2E8F0'
    grid_color = '#334155'

    fig.patch.set_facecolor(bg_dark)
    for ax in [ax_wr, ax_pnl]:
        ax.set_facecolor(bg_card)
        ax.tick_params(colors=text_color, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(grid_color)

    # Plot each row
    for i in range(n_rows):
        if is_header[i]:
            continue
        c = color_small if small_n[i] else color_main
        marker = 'D' if small_n[i] else 'o'
        ms = 5 if small_n[i] else 7

        # Win Rate
        ax_wr.errorbar(wr_points[i], y_pos[i],
                       xerr=[[wr_points[i] - wr_lows[i]], [wr_highs[i] - wr_points[i]]],
                       fmt=marker, color=c, ecolor=c, elinewidth=1.8, capsize=4,
                       markersize=ms, markeredgecolor='white', markeredgewidth=0.5, alpha=0.9)

        # Avg P&L
        ax_pnl.errorbar(avg_points[i], y_pos[i],
                        xerr=[[avg_points[i] - avg_lows[i]], [avg_highs[i] - avg_points[i]]],
                        fmt=marker, color=c, ecolor=c, elinewidth=1.8, capsize=4,
                        markersize=ms, markeredgecolor='white', markeredgewidth=0.5, alpha=0.9)

    # Y-axis labels
    ax_wr.set_yticks(y_pos)
    label_colors = []
    for i, lbl in enumerate(labels):
        if is_header[i]:
            label_colors.append('#94A3B8')
        elif small_n[i]:
            label_colors.append(color_small)
        else:
            label_colors.append(text_color)

    ax_wr.set_yticklabels(labels, fontsize=9, fontfamily='monospace')
    for tick_label, color in zip(ax_wr.get_yticklabels(), label_colors):
        tick_label.set_color(color)
        if color == '#94A3B8':
            tick_label.set_fontweight('bold')
            tick_label.set_fontsize(10)

    # Section header background bands
    for i in range(n_rows):
        if is_header[i]:
            for ax in [ax_wr, ax_pnl]:
                ax.axhspan(y_pos[i] - 0.5, y_pos[i] + 0.5, color='#1A2332', zorder=0)

    # Axes formatting
    ax_wr.set_xlabel('Win Rate (%)', color=text_color, fontsize=11, fontweight='bold')
    ax_wr.set_xlim(0, 105)
    ax_wr.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax_wr.axvline(50, color='#EF4444', linewidth=0.8, linestyle='--', alpha=0.5, label='50% (coin flip)')
    ax_wr.grid(axis='x', color=grid_color, linewidth=0.5, alpha=0.5)
    ax_wr.legend(loc='lower right', fontsize=8, facecolor=bg_card, edgecolor=grid_color, labelcolor=text_color)

    ax_pnl.set_xlabel('Avg P&L (%)', color=text_color, fontsize=11, fontweight='bold')
    ax_pnl.axvline(0, color='#EF4444', linewidth=0.8, linestyle='--', alpha=0.5, label='0% (breakeven)')
    ax_pnl.grid(axis='x', color=grid_color, linewidth=0.5, alpha=0.5)
    ax_pnl.xaxis.set_major_formatter(mticker.FormatStrFormatter('%+.0f%%'))
    ax_pnl.legend(loc='lower right', fontsize=8, facecolor=bg_card, edgecolor=grid_color, labelcolor=text_color)

    fig.suptitle('Bootstrap 95% Confidence Intervals — Win Rate & Avg P&L',
                 color='white', fontsize=14, fontweight='bold', y=0.98)
    fig.text(0.5, 0.01,
             'Blue = n\u226510 (BCa method)    Amber = n<10 (percentile, less reliable)',
             ha='center', color='#94A3B8', fontsize=9)

    os.makedirs(CHARTS_DIR, exist_ok=True)
    out_path = os.path.join(CHARTS_DIR, 'bootstrap_forest_plot.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=bg_dark, edgecolor='none')
    plt.close(fig)
    print(f'\nForest plot saved to {out_path}')
    return out_path


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

    chart_path = generate_forest_plot()
    if chart_path:
        # Open the chart
        import subprocess
        subprocess.Popen(['cmd', '/c', 'start', '', chart_path], shell=False)
