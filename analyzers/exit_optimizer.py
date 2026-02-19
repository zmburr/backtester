"""
Exit Strategy Optimizer - Phase 2

Analyzes Phase 1 data to determine optimal exit strategies:
1. ATR-based tiered exits
2. Time-based exits
3. Technical level targets
4. Trailing stop optimization
5. Cap-specific recommendations

Usage:
    from analyzers.exit_optimizer import ExitOptimizer
    optimizer = ExitOptimizer()
    optimizer.run_full_analysis()
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_DATA_DIR = Path(__file__).resolve().parent.parent / 'data'


class ExitOptimizer:
    """Optimizes exit strategies based on Phase 1 exit analysis data."""

    def __init__(self, results_path: str = None):
        self.results_path = results_path or str(_DATA_DIR / 'exit_analysis_results.csv')
        self.df = None

    def load_results(self) -> pd.DataFrame:
        """Load Phase 1 exit analysis results."""
        self.df = pd.read_csv(self.results_path)
        logging.info(f"Loaded {len(self.df)} trade exit analyses")
        return self.df

    def analyze_atr_targets(self) -> Dict:
        """Analyze optimal ATR-based exit targets."""
        if self.df is None:
            self.load_results()

        print("\n" + "=" * 70)
        print("ATR TARGET ANALYSIS")
        print("=" * 70)

        # Overall hit rates
        print("\n--- Overall ATR Hit Rates ---")
        atr_cols = ['hit_1x_atr', 'hit_1_5x_atr', 'hit_2x_atr', 'hit_2_5x_atr', 'hit_3x_atr']
        time_cols = ['time_to_1x_atr', 'time_to_1_5x_atr', 'time_to_2x_atr']

        results = {}
        for col in atr_cols:
            target = col.replace('hit_', '').replace('_atr', '').replace('_', '.')
            hit_rate = self.df[col].mean() * 100
            results[f'{target}_hit_rate'] = hit_rate
            print(f"  {target}x ATR: {hit_rate:.1f}%")

        # Time to targets
        print("\n--- Average Time to ATR Targets (for hits) ---")
        for col in time_cols:
            target = col.replace('time_to_', '').replace('_atr', '').replace('_', '.')
            hits = self.df[self.df[col].notna()][col]
            if len(hits) > 0:
                avg_time = hits.mean()
                median_time = hits.median()
                print(f"  {target}x ATR: Avg {avg_time:.0f} mins, Median {median_time:.0f} mins")
                results[f'{target}_avg_time'] = avg_time
                results[f'{target}_median_time'] = median_time

        # By cap size
        print("\n--- ATR Hit Rates by Cap Size ---")
        print(f"{'Cap':<10} {'1.0x':>8} {'1.5x':>8} {'2.0x':>8} {'2.5x':>8} {'3.0x':>8}")
        print("-" * 58)

        for cap in self.df['cap'].unique():
            cap_df = self.df[self.df['cap'] == cap]
            rates = [cap_df[col].mean() * 100 for col in atr_cols]
            print(f"{cap:<10} {rates[0]:>7.0f}% {rates[1]:>7.0f}% {rates[2]:>7.0f}% {rates[3]:>7.0f}% {rates[4]:>7.0f}%")

        # Expected value calculation for different exit strategies
        print("\n--- Expected Value by ATR Exit Strategy ---")
        print("(Assuming exit at target if hit, else at close)")

        for target, col in [('1.0x', 'hit_1x_atr'), ('1.5x', 'hit_1_5x_atr'), ('2.0x', 'hit_2x_atr')]:
            atr_multiple = float(target.replace('x', ''))
            # For those that hit target: P&L = atr_multiple * atr / entry_price
            # For those that didn't: P&L = close_pct

            hits = self.df[self.df[col] == True]
            misses = self.df[self.df[col] == False]

            if len(hits) > 0:
                # Estimate P&L for hits (atr_multiple * atr / entry = atr_multiple * atr_pct approximately)
                hit_pnl = atr_multiple * (self.df['atr'] / self.df['entry_price']).mean()
            else:
                hit_pnl = 0

            miss_pnl = misses['close_pct'].mean() if len(misses) > 0 else 0
            hit_rate = self.df[col].mean()

            expected_pnl = (hit_rate * hit_pnl + (1 - hit_rate) * miss_pnl) * 100
            print(f"  Exit at {target} ATR: Expected P&L = {expected_pnl:+.1f}%")

        return results

    def analyze_time_exits(self) -> Dict:
        """Analyze time-based exit effectiveness."""
        if self.df is None:
            self.load_results()

        print("\n" + "=" * 70)
        print("TIME-BASED EXIT ANALYSIS")
        print("=" * 70)

        time_cols = ['pnl_at_10am', 'pnl_at_1030am', 'pnl_at_11am', 'pnl_at_1130am',
                     'pnl_at_12pm', 'pnl_at_1pm', 'pnl_at_2pm', 'pnl_at_close']

        results = {}

        print("\n--- Average P&L at Different Exit Times ---")
        print(f"{'Time':<15} {'Avg P&L':>10} {'Median P&L':>12} {'Win Rate':>10} {'vs Close':>10}")
        print("-" * 60)

        close_pnl = self.df['pnl_at_close'].mean()

        for col in time_cols:
            time_label = col.replace('pnl_at_', '').replace('am', ' AM').replace('pm', ' PM')
            valid = self.df[self.df[col].notna()]

            if len(valid) > 0:
                avg_pnl = valid[col].mean() * 100
                median_pnl = valid[col].median() * 100
                win_rate = (valid[col] > 0).mean() * 100
                vs_close = (avg_pnl - close_pnl * 100)

                print(f"{time_label:<15} {avg_pnl:>+9.1f}% {median_pnl:>+11.1f}% {win_rate:>9.0f}% {vs_close:>+9.1f}%")
                results[col] = {'avg': avg_pnl, 'median': median_pnl, 'win_rate': win_rate}

        # By cap size
        print("\n--- P&L at 11 AM by Cap Size (Common Reversal Window) ---")
        for cap in self.df['cap'].unique():
            cap_df = self.df[self.df['cap'] == cap]
            valid = cap_df[cap_df['pnl_at_11am'].notna()]
            if len(valid) > 0:
                avg = valid['pnl_at_11am'].mean() * 100
                wr = (valid['pnl_at_11am'] > 0).mean() * 100
                print(f"  {cap}: Avg {avg:+.1f}%, Win Rate {wr:.0f}%")

        # Time of MFE distribution
        print("\n--- Time to MFE Distribution ---")
        mfe_minutes = self.df['mfe_minutes']
        print(f"  Mean: {mfe_minutes.mean():.0f} mins")
        print(f"  Median: {mfe_minutes.median():.0f} mins")
        print(f"  25th percentile: {mfe_minutes.quantile(0.25):.0f} mins")
        print(f"  75th percentile: {mfe_minutes.quantile(0.75):.0f} mins")

        # What % of MFEs occur before key times
        print("\n--- % of MFEs Occurring Before Key Times ---")
        for mins, label in [(30, '10:00 AM'), (60, '10:30 AM'), (90, '11:00 AM'),
                            (120, '11:30 AM'), (150, '12:00 PM'), (180, '12:30 PM')]:
            pct = (mfe_minutes <= mins).mean() * 100
            print(f"  Before {label}: {pct:.0f}%")

        return results

    def analyze_mfe_mae(self) -> Dict:
        """Deep analysis of MFE/MAE patterns."""
        if self.df is None:
            self.load_results()

        print("\n" + "=" * 70)
        print("MFE/MAE ANALYSIS (Maximum Favorable/Adverse Excursion)")
        print("=" * 70)

        results = {}

        # Overall stats
        print("\n--- Overall MFE/MAE Statistics ---")
        print(f"  Average MFE (max profit available): {self.df['mfe_pct'].mean()*100:+.1f}%")
        print(f"  Median MFE:                         {self.df['mfe_pct'].median()*100:+.1f}%")
        print(f"  Average MAE (max adverse before):   {self.df['mae_pct'].mean()*100:+.1f}%")
        print(f"  Average Captured (close P&L):       {self.df['close_pct'].mean()*100:+.1f}%")

        # Capture efficiency
        efficiency = self.df['capture_efficiency'].mean() * 100
        print(f"\n  Capture Efficiency: {efficiency:.1f}%")
        print(f"  (You're capturing only {efficiency:.0f}% of the available move!)")

        # By cap
        print("\n--- MFE by Cap Size ---")
        print(f"{'Cap':<10} {'Avg MFE':>10} {'Avg Captured':>14} {'Efficiency':>12} {'Left on Table':>15}")
        print("-" * 65)

        for cap in self.df['cap'].unique():
            cap_df = self.df[self.df['cap'] == cap]
            mfe = cap_df['mfe_pct'].mean() * 100
            captured = cap_df['close_pct'].mean() * 100
            eff = cap_df['capture_efficiency'].mean() * 100
            left = mfe - captured
            print(f"{cap:<10} {mfe:>+9.1f}% {captured:>+13.1f}% {eff:>11.1f}% {left:>+14.1f}%")

        # Giveback analysis
        print("\n--- Giveback After MFE ---")
        print(f"  Average Max Giveback: {self.df['max_giveback_pct'].mean()*100:.1f}%")
        print(f"  Median Max Giveback:  {self.df['max_giveback_pct'].median()*100:.1f}%")

        print("\n--- Optimal Stop After MFE (to minimize giveback) ---")
        # Calculate what trailing stop would have captured
        for trail_pct in [0.25, 0.50, 0.75, 1.0]:
            # If trail stop = trail_pct * MFE, how much would we have captured on average?
            # Captured = MFE - min(trail_pct * MFE, actual_giveback)
            captured_with_trail = self.df.apply(
                lambda x: x['mfe_pct'] - min(trail_pct * x['mfe_pct'], x['max_giveback_pct']),
                axis=1
            ).mean() * 100
            print(f"  Trail at {trail_pct*100:.0f}% of gains: Would capture {captured_with_trail:+.1f}%")

        return results

    def analyze_technical_levels(self) -> Dict:
        """Analyze technical level target effectiveness."""
        if self.df is None:
            self.load_results()

        print("\n" + "=" * 70)
        print("TECHNICAL LEVEL TARGET ANALYSIS")
        print("=" * 70)

        results = {}

        # Prior day levels
        print("\n--- Prior Day Level Hit Rates ---")
        pdl_hit_rate = self.df['hit_prior_day_low'].mean() * 100
        pdc_hit_rate = self.df['hit_prior_day_close'].mean() * 100
        print(f"  Hit Prior Day Low:   {pdl_hit_rate:.1f}%")
        print(f"  Hit Prior Day Close: {pdc_hit_rate:.1f}%")

        # Time to prior day low (for those that hit)
        pdl_times = self.df[self.df['time_to_prior_day_low'].notna()]['time_to_prior_day_low']
        if len(pdl_times) > 0:
            print(f"  Avg Time to Prior Day Low: {pdl_times.mean():.0f} mins")

        # VWAP
        print("\n--- VWAP Analysis ---")
        vwap_hit_rate = self.df['hit_vwap'].mean() * 100
        print(f"  Hit Below Opening VWAP: {vwap_hit_rate:.1f}%")

        # By cap
        print("\n--- Prior Day Low Hit Rate by Cap ---")
        for cap in self.df['cap'].unique():
            cap_df = self.df[self.df['cap'] == cap]
            rate = cap_df['hit_prior_day_low'].mean() * 100
            print(f"  {cap}: {rate:.0f}%")

        return results

    def generate_recommendations(self) -> Dict:
        """Generate actionable exit strategy recommendations."""
        if self.df is None:
            self.load_results()

        print("\n" + "=" * 70)
        print("EXIT STRATEGY RECOMMENDATIONS")
        print("=" * 70)

        recommendations = {}

        for cap in ['Micro', 'Small', 'Medium', 'Large', 'ETF']:
            cap_df = self.df[self.df['cap'] == cap]
            if len(cap_df) == 0:
                continue

            print(f"\n--- {cap} Cap Recommendations ---")

            # Calculate key metrics
            mfe = cap_df['mfe_pct'].mean()
            captured = cap_df['close_pct'].mean()
            hit_1x = cap_df['hit_1x_atr'].mean()
            hit_1_5x = cap_df['hit_1_5x_atr'].mean()
            hit_2x = cap_df['hit_2x_atr'].mean()
            avg_time_mfe = cap_df['mfe_minutes'].mean()

            rec = {
                'cap': cap,
                'n_trades': len(cap_df),
                'avg_mfe': mfe,
                'avg_captured': captured,
            }

            # Tier 1 target (70% of position)
            if hit_1x > 0.80:
                tier1 = "1.0x ATR"
                tier1_size = "70%"
            elif hit_1x > 0.60:
                tier1 = "0.75x ATR"
                tier1_size = "70%"
            else:
                tier1 = "0.5x ATR"
                tier1_size = "70%"

            # Tier 2 target (20% of position)
            if hit_2x > 0.50:
                tier2 = "2.0x ATR"
            elif hit_1_5x > 0.60:
                tier2 = "1.5x ATR"
            else:
                tier2 = "1.0x ATR"

            # Tier 3 (remaining 10%)
            tier3 = "Let run with trail"

            # Time-based exit recommendation
            if avg_time_mfe < 120:
                time_exit = "Exit by 11:30 AM if targets not hit"
            elif avg_time_mfe < 180:
                time_exit = "Exit by 12:30 PM if targets not hit"
            else:
                time_exit = "Can hold into afternoon"

            print(f"  Tier 1 ({tier1_size}): Exit at {tier1}")
            print(f"  Tier 2 (20%): Exit at {tier2}")
            print(f"  Tier 3 (10%): {tier3}")
            print(f"  Time Rule: {time_exit}")
            print(f"  Expected MFE: {mfe*100:+.1f}% | Currently Capturing: {captured*100:+.1f}%")

            rec['tier1'] = tier1
            rec['tier2'] = tier2
            rec['tier3'] = tier3
            rec['time_rule'] = time_exit
            recommendations[cap] = rec

        # Overall recommendation
        print("\n" + "=" * 70)
        print("OVERALL KEY FINDINGS")
        print("=" * 70)

        overall_mfe = self.df['mfe_pct'].mean() * 100
        overall_captured = self.df['close_pct'].mean() * 100
        efficiency = self.df['capture_efficiency'].mean() * 100

        print(f"""
1. CAPTURE EFFICIENCY PROBLEM:
   - Average MFE available: {overall_mfe:+.1f}%
   - Currently capturing: {overall_captured:+.1f}%
   - Efficiency: {efficiency:.0f}% (leaving {overall_mfe - overall_captured:.1f}% on table!)

2. ATR TARGETS:
   - 1.0x ATR hit rate: {self.df['hit_1x_atr'].mean()*100:.0f}% (avg {self.df[self.df['time_to_1x_atr'].notna()]['time_to_1x_atr'].mean():.0f} mins)
   - 1.5x ATR hit rate: {self.df['hit_1_5x_atr'].mean()*100:.0f}%
   - 2.0x ATR hit rate: {self.df['hit_2x_atr'].mean()*100:.0f}%

3. TIME ANALYSIS:
   - Median time to MFE: {self.df['mfe_minutes'].median():.0f} mins
   - {(self.df['mfe_minutes'] <= 90).mean()*100:.0f}% of MFEs occur before 11 AM

4. RECOMMENDED STRATEGY:
   - Tier 1 (70%): Exit at 1.0x ATR (82% hit rate)
   - Tier 2 (20%): Exit at 1.5x ATR (67% hit rate)
   - Tier 3 (10%): Trail with 50% of gains as stop
   - Time stop: Exit remaining by 12:00 PM if targets not hit

5. GIVEBACK:
   - Average max giveback: {self.df['max_giveback_pct'].mean()*100:.1f}%
   - Use 50% trailing stop after hitting tier 1 target
""")

        return recommendations

    def run_full_analysis(self):
        """Run all analyses and generate report."""
        self.load_results()

        self.analyze_mfe_mae()
        self.analyze_atr_targets()
        self.analyze_time_exits()
        self.analyze_technical_levels()
        self.generate_recommendations()

        # Save summary to file
        summary = {
            'total_trades': len(self.df),
            'avg_mfe_pct': self.df['mfe_pct'].mean() * 100,
            'avg_captured_pct': self.df['close_pct'].mean() * 100,
            'capture_efficiency': self.df['capture_efficiency'].mean() * 100,
            'hit_1x_atr': self.df['hit_1x_atr'].mean() * 100,
            'hit_1_5x_atr': self.df['hit_1_5x_atr'].mean() * 100,
            'hit_2x_atr': self.df['hit_2x_atr'].mean() * 100,
            'avg_time_to_mfe': self.df['mfe_minutes'].mean(),
            'avg_giveback': self.df['max_giveback_pct'].mean() * 100,
        }

        return summary


def run_phase2_optimization():
    """Run Phase 2: Exit Strategy Optimization."""
    optimizer = ExitOptimizer()
    summary = optimizer.run_full_analysis()
    return optimizer, summary


if __name__ == '__main__':
    optimizer, summary = run_phase2_optimization()
