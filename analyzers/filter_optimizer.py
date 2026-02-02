"""
Filter Optimizer for Grade A Parabolic Short Reversals

This script tests various threshold combinations on technical indicators
to find filters that improve signal quality and P&L for reversal setups.
"""

import pandas as pd
import numpy as np
from itertools import product
from tabulate import tabulate
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_reversal_data(filepath='C:\\Users\\zmbur\\PycharmProjects\\backtester\\data\\reversal_data.csv'):
    """Load and prepare reversal data."""
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['ticker', 'date'])
    return df


def calculate_metrics(df, target_col='reversal_open_close_pct'):
    """Calculate performance metrics for a filtered dataset."""
    if len(df) == 0:
        return {
            'count': 0,
            'win_rate': 0,
            'avg_pnl': 0,
            'median_pnl': 0,
            'total_pnl': 0,
            'max_win': 0,
            'max_loss': 0,
            'profit_factor': 0,
            'avg_winner': 0,
            'avg_loser': 0
        }

    # For shorts, negative return = profit
    pnl = -df[target_col]  # Flip sign for short trades
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]

    win_rate = len(winners) / len(df) * 100 if len(df) > 0 else 0
    avg_pnl = pnl.mean() * 100  # Convert to percentage
    median_pnl = pnl.median() * 100
    total_pnl = pnl.sum() * 100

    gross_profit = winners.sum() if len(winners) > 0 else 0
    gross_loss = abs(losers.sum()) if len(losers) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        'count': len(df),
        'win_rate': round(win_rate, 1),
        'avg_pnl': round(avg_pnl, 2),
        'median_pnl': round(median_pnl, 2),
        'total_pnl': round(total_pnl, 1),
        'max_win': round(-df[target_col].min() * 100, 2),  # Best trade (most negative original)
        'max_loss': round(-df[target_col].max() * 100, 2),  # Worst trade (most positive original)
        'profit_factor': round(profit_factor, 2),
        'avg_winner': round(winners.mean() * 100, 2) if len(winners) > 0 else 0,
        'avg_loser': round(losers.mean() * 100, 2) if len(losers) > 0 else 0
    }


def test_single_filter(df, column, operator, threshold, target_col='reversal_open_close_pct'):
    """Test a single filter condition."""
    if column not in df.columns:
        return None

    if operator == '>':
        filtered = df[df[column] > threshold]
    elif operator == '>=':
        filtered = df[df[column] >= threshold]
    elif operator == '<':
        filtered = df[df[column] < threshold]
    elif operator == '<=':
        filtered = df[df[column] <= threshold]
    elif operator == '==':
        filtered = df[df[column] == threshold]
    else:
        return None

    return calculate_metrics(filtered, target_col)


def test_filter_thresholds(df, column, thresholds, operator='>=', target_col='reversal_open_close_pct'):
    """Test multiple thresholds for a single filter."""
    results = []
    baseline = calculate_metrics(df, target_col)

    for thresh in thresholds:
        metrics = test_single_filter(df, column, operator, thresh, target_col)
        if metrics and metrics['count'] >= 5:  # Minimum sample size
            metrics['filter'] = f"{column} {operator} {thresh}"
            metrics['threshold'] = thresh
            metrics['improvement'] = round(metrics['avg_pnl'] - baseline['avg_pnl'], 2)
            results.append(metrics)

    return results


def analyze_indicator_impact(df, target_col='reversal_open_close_pct'):
    """Analyze correlation and impact of each indicator on performance."""
    # For shorts, we want negative target values (profitable)
    pnl = -df[target_col]

    indicators = [
        # Existing indicators
        'pct_change_15', 'pct_change_30', 'pct_change_3',
        'pct_from_10mav', 'pct_from_20mav', 'pct_from_50mav', 'pct_from_9ema',
        'atr_distance_from_50mav', 'day_of_range_pct',
        'percent_of_premarket_vol', 'percent_of_vol_in_first_5_min',
        'gap_pct',
        # New indicators
        'upper_band_distance', 'bollinger_width',
        'vol_ratio_5min_to_pm', 'rvol_score',
        'prior_day_close_vs_high_pct', 'consecutive_up_days', 'prior_day_range_atr',
        'time_of_high_bucket', 'gap_from_pm_high',
        'spy_5day_return', 'uvxy_close'
    ]

    correlations = []
    for ind in indicators:
        if ind in df.columns:
            valid_data = df[[ind, target_col]].dropna()
            if len(valid_data) > 10:
                corr = valid_data[ind].corr(-valid_data[target_col])  # Flip for short P&L
                correlations.append({
                    'indicator': ind,
                    'correlation': round(corr, 3),
                    'abs_corr': abs(round(corr, 3)),
                    'direction': 'higher=better' if corr > 0 else 'lower=better'
                })

    return sorted(correlations, key=lambda x: x['abs_corr'], reverse=True)


def find_optimal_filters(df, grade='A', min_sample=10, target_col='reversal_open_close_pct'):
    """Find optimal filter thresholds for each indicator."""

    # Filter to grade if specified
    if grade:
        df = df[df['trade_grade'] == grade].copy()

    baseline = calculate_metrics(df, target_col)
    logging.info(f"Baseline ({len(df)} trades): Win Rate={baseline['win_rate']}%, Avg P&L={baseline['avg_pnl']}%")

    # Define filter configurations: (column, operator, thresholds)
    filter_configs = [
        # Price momentum filters
        ('pct_change_15', '>=', [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]),
        ('pct_change_30', '>=', [0.5, 0.75, 1.0, 1.5, 2.0]),
        ('pct_change_3', '>=', [0.1, 0.2, 0.3, 0.5]),

        # Distance from MA filters
        ('pct_from_50mav', '>=', [0.5, 1.0, 1.5, 2.0, 2.5]),
        ('pct_from_20mav', '>=', [0.3, 0.5, 0.75, 1.0, 1.5]),
        ('pct_from_9ema', '>=', [0.2, 0.3, 0.5, 0.75, 1.0]),
        ('atr_distance_from_50mav', '>=', [4, 5, 6, 7, 8]),

        # Volume filters
        ('percent_of_premarket_vol', '>=', [0.1, 0.15, 0.2, 0.3, 0.4]),
        ('percent_of_vol_in_first_5_min', '>=', [0.1, 0.15, 0.2, 0.25]),
        ('rvol_score', '>=', [1.5, 2.0, 2.5, 3.0, 4.0]),
        ('vol_ratio_5min_to_pm', '>=', [0.3, 0.5, 0.75, 1.0]),
        ('vol_ratio_5min_to_pm', '<=', [0.5, 0.75, 1.0, 1.5]),

        # Gap filters
        ('gap_pct', '>=', [0.05, 0.1, 0.15, 0.2, 0.3]),
        ('gap_from_pm_high', '<=', [0, -0.01, -0.02, -0.03]),  # Opened below PM high

        # Bollinger Band filters
        ('upper_band_distance', '>=', [0, 0.01, 0.02, 0.03, 0.05]),
        ('bollinger_width', '>=', [0.1, 0.15, 0.2, 0.25, 0.3]),
        ('closed_outside_upper_band', '==', [True]),

        # Prior day context filters
        ('prior_day_close_vs_high_pct', '>=', [0.7, 0.8, 0.9, 0.95]),
        ('consecutive_up_days', '>=', [2, 3, 4, 5]),
        ('prior_day_range_atr', '>=', [1.5, 2.0, 2.5, 3.0]),

        # Timing filters
        ('time_of_high_bucket', '==', [1]),  # High in first 30 min
        ('time_of_high_bucket', '<=', [2]),  # High before 11 AM

        # Market context filters
        ('spy_5day_return', '>=', [0, 0.01, 0.02]),  # Market up
        ('spy_5day_return', '<=', [0, -0.01, -0.02]),  # Market down
        ('uvxy_close', '>=', [15, 20, 25, 30]),

        # Range expansion
        ('day_of_range_pct', '>=', [2.0, 2.5, 3.0, 3.5, 4.0]),
    ]

    all_results = []

    for column, operator, thresholds in filter_configs:
        results = test_filter_thresholds(df, column, thresholds, operator, target_col)
        all_results.extend(results)

    # Sort by improvement in avg P&L
    all_results = sorted(all_results, key=lambda x: x['improvement'], reverse=True)

    return baseline, all_results


def test_combined_filters(df, filters, target_col='reversal_open_close_pct'):
    """Test a combination of filters."""
    filtered = df.copy()

    for column, operator, threshold in filters:
        if column not in filtered.columns:
            continue
        if operator == '>':
            filtered = filtered[filtered[column] > threshold]
        elif operator == '>=':
            filtered = filtered[filtered[column] >= threshold]
        elif operator == '<':
            filtered = filtered[filtered[column] < threshold]
        elif operator == '<=':
            filtered = filtered[filtered[column] <= threshold]
        elif operator == '==':
            filtered = filtered[filtered[column] == threshold]

    return calculate_metrics(filtered, target_col), filtered


def find_best_filter_combinations(df, top_filters, max_combo_size=3, min_sample=8, target_col='reversal_open_close_pct'):
    """Find the best combinations of top-performing filters."""

    # Parse top filters into tuples
    filter_tuples = []
    for f in top_filters[:15]:  # Use top 15 filters
        parts = f['filter'].split(' ')
        col = parts[0]
        op = parts[1]
        thresh = float(parts[2]) if parts[2] not in ['True', 'False'] else (parts[2] == 'True')
        filter_tuples.append((col, op, thresh, f['filter']))

    best_combos = []
    baseline = calculate_metrics(df, target_col)

    # Test pairs
    for i in range(len(filter_tuples)):
        for j in range(i+1, len(filter_tuples)):
            combo = [filter_tuples[i][:3], filter_tuples[j][:3]]
            metrics, filtered_df = test_combined_filters(df, combo, target_col)

            if metrics['count'] >= min_sample:
                best_combos.append({
                    'filters': f"{filter_tuples[i][3]} + {filter_tuples[j][3]}",
                    'count': metrics['count'],
                    'win_rate': metrics['win_rate'],
                    'avg_pnl': metrics['avg_pnl'],
                    'improvement': round(metrics['avg_pnl'] - baseline['avg_pnl'], 2),
                    'profit_factor': metrics['profit_factor']
                })

    # Test triples (optional, can be slow)
    if max_combo_size >= 3:
        for i in range(len(filter_tuples)):
            for j in range(i+1, len(filter_tuples)):
                for k in range(j+1, len(filter_tuples)):
                    combo = [filter_tuples[i][:3], filter_tuples[j][:3], filter_tuples[k][:3]]
                    metrics, filtered_df = test_combined_filters(df, combo, target_col)

                    if metrics['count'] >= min_sample:
                        best_combos.append({
                            'filters': f"{filter_tuples[i][3]} + {filter_tuples[j][3]} + {filter_tuples[k][3]}",
                            'count': metrics['count'],
                            'win_rate': metrics['win_rate'],
                            'avg_pnl': metrics['avg_pnl'],
                            'improvement': round(metrics['avg_pnl'] - baseline['avg_pnl'], 2),
                            'profit_factor': metrics['profit_factor']
                        })

    return sorted(best_combos, key=lambda x: x['improvement'], reverse=True)


def analyze_conditional_rates(df, target_col='reversal_open_close_pct'):
    """Analyze performance by boolean conditional columns."""
    conditionals = [
        'hit_green_red', 'close_green_red', 'close_at_lows',
        'move_together', 'breaks_fifty_two_wk', 'breaks_ath',
        'hit_prior_day_hilo', 'closed_outside_upper_band'
    ]

    results = []
    baseline = calculate_metrics(df, target_col)

    for col in conditionals:
        if col not in df.columns:
            continue

        # True condition
        true_df = df[df[col] == True]
        if len(true_df) >= 5:
            metrics = calculate_metrics(true_df, target_col)
            results.append({
                'condition': f'{col} = True',
                'count': metrics['count'],
                'win_rate': metrics['win_rate'],
                'avg_pnl': metrics['avg_pnl'],
                'improvement': round(metrics['avg_pnl'] - baseline['avg_pnl'], 2)
            })

        # False condition
        false_df = df[df[col] == False]
        if len(false_df) >= 5:
            metrics = calculate_metrics(false_df, target_col)
            results.append({
                'condition': f'{col} = False',
                'count': metrics['count'],
                'win_rate': metrics['win_rate'],
                'avg_pnl': metrics['avg_pnl'],
                'improvement': round(metrics['avg_pnl'] - baseline['avg_pnl'], 2)
            })

    return sorted(results, key=lambda x: x['improvement'], reverse=True)


def generate_report(df, grade='A'):
    """Generate a comprehensive filter optimization report."""

    if grade:
        df = df[df['trade_grade'] == grade].copy()

    print("=" * 80)
    print(f"FILTER OPTIMIZATION REPORT - Grade {grade} Reversals")
    print("=" * 80)
    print()

    # Baseline metrics
    baseline = calculate_metrics(df)
    print("BASELINE PERFORMANCE")
    print("-" * 40)
    print(f"Total Trades: {baseline['count']}")
    print(f"Win Rate: {baseline['win_rate']}%")
    print(f"Avg P&L: {baseline['avg_pnl']}%")
    print(f"Median P&L: {baseline['median_pnl']}%")
    print(f"Profit Factor: {baseline['profit_factor']}")
    print(f"Best Trade: +{baseline['max_win']}%")
    print(f"Worst Trade: {baseline['max_loss']}%")
    print()

    # Indicator correlations
    print("INDICATOR CORRELATIONS WITH P&L")
    print("-" * 40)
    correlations = analyze_indicator_impact(df)
    corr_df = pd.DataFrame(correlations[:15])
    print(tabulate(corr_df, headers='keys', tablefmt='simple', showindex=False))
    print()

    # Best single filters
    print("TOP SINGLE FILTERS (by P&L improvement)")
    print("-" * 40)
    _, single_filters = find_optimal_filters(df, grade=None)  # Already filtered
    top_single = single_filters[:20]
    single_df = pd.DataFrame(top_single)[['filter', 'count', 'win_rate', 'avg_pnl', 'improvement', 'profit_factor']]
    print(tabulate(single_df, headers='keys', tablefmt='simple', showindex=False))
    print()

    # Conditional analysis
    print("CONDITIONAL PERFORMANCE")
    print("-" * 40)
    cond_results = analyze_conditional_rates(df)
    cond_df = pd.DataFrame(cond_results)
    print(tabulate(cond_df, headers='keys', tablefmt='simple', showindex=False))
    print()

    # Best filter combinations
    print("TOP FILTER COMBINATIONS")
    print("-" * 40)
    combos = find_best_filter_combinations(df, single_filters, max_combo_size=3, min_sample=8)
    combo_df = pd.DataFrame(combos[:15])
    print(tabulate(combo_df, headers='keys', tablefmt='simple', showindex=False))
    print()

    return {
        'baseline': baseline,
        'correlations': correlations,
        'single_filters': single_filters,
        'conditionals': cond_results,
        'combinations': combos
    }


if __name__ == '__main__':
    # Load data
    df = load_reversal_data()

    # Generate report for Grade A
    results = generate_report(df, grade='A')

    # Save results to CSV
    pd.DataFrame(results['single_filters']).to_csv(
        'C:\\Users\\zmbur\\PycharmProjects\\backtester\\data\\filter_optimization_results.csv',
        index=False
    )
    print("\nResults saved to data/filter_optimization_results.csv")
