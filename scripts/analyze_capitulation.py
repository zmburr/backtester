"""
Capitulation Analysis Script
Analyzes reversal_data.csv to identify capitulation day patterns and next-day moves.

Usage:
    python scripts/analyze_capitulation.py           # Full analysis
    python scripts/analyze_capitulation.py --stats   # Summary only
    python scripts/analyze_capitulation.py --export  # Export tagged data
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
REVERSAL_CSV = DATA_DIR / "reversal_data.csv"
OUTPUT_CSV = DATA_DIR / "capitulation_analysis.csv"


@dataclass
class CapitulationMetrics:
    """Metrics for a capitulation day."""
    date: str
    ticker: str
    is_capitulation: bool
    capitulation_score: float  # 0-100 composite score
    
    # Individual signals
    large_down_move: bool
    volume_spike: bool
    vix_spike: bool
    close_at_lows: bool
    range_expansion: bool
    
    # Next day data (if available)
    next_day_gap_pct: float = np.nan
    next_day_max_bounce_pct: float = np.nan
    next_day_close_pct: float = np.nan
    next_day_win: bool = False


def load_reversal_data() -> pd.DataFrame:
    """Load and clean reversal data."""
    df = pd.read_csv(REVERSAL_CSV)
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    
    # Ensure numeric columns
    numeric_cols = ['pct_change_3', 'pct_change_30', 'gap_pct', 'close_green_red',
                   'vol_on_breakout_day', 'vol_one_day_before', 'atr_pct',
                   'percent_of_vol_in_first_30_min', 'reversal_open_low_pct']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def calculate_capitulation_signals(row: pd.Series) -> Dict[str, bool]:
    """Calculate individual capitulation signals for a row."""
    signals = {}
    
    # 1. Large down move (3-day decline > 10% or single day > 5%)
    pct_change_3 = row.get('pct_change_3', 0)
    reversal_open_low = abs(row.get('reversal_open_low_pct', 0))
    signals['large_down_move'] = pct_change_3 < -0.10 or reversal_open_low > 0.05
    
    # 2. Volume spike (RVOL > 2x)
    vol_today = row.get('vol_on_breakout_day', 0)
    vol_prior = row.get('vol_one_day_before', 0)
    if vol_prior and vol_prior > 0:
        rvol = vol_today / vol_prior
        signals['volume_spike'] = rvol > 2.0
    else:
        signals['volume_spike'] = False
    
    # 3. Close at/near lows (down > 3% and close in bottom 20% of range)
    # Using reversal_open_close_pct as proxy for close performance
    open_close = row.get('reversal_open_close_pct', 0)
    signals['close_at_lows'] = open_close < -0.03
    
    # 4. Range expansion (today's range > 2x ATR)
    atr_pct = row.get('atr_pct', 0.02)
    # Estimate range from reversal data
    range_estimate = abs(row.get('reversal_open_low_pct', 0)) + abs(open_close)
    signals['range_expansion'] = range_estimate > (2 * atr_pct)
    
    # 5. VIX spike context (not in data, infer from market conditions)
    # For now, use gap down as proxy for fear
    gap_pct = row.get('gap_pct', 0)
    signals['vix_spike'] = gap_pct < -0.02  # Gap down suggests fear
    
    return signals


def calculate_capitulation_score(signals: Dict[str, bool]) -> float:
    """Calculate composite capitulation score (0-100)."""
    weights = {
        'large_down_move': 30,
        'volume_spike': 25,
        'close_at_lows': 20,
        'range_expansion': 15,
        'vix_spike': 10
    }
    
    score = sum(weights[k] for k, v in signals.items() if v)
    return score


def tag_capitulation_days(df: pd.DataFrame, threshold: float = 60.0) -> pd.DataFrame:
    """Tag capitulation days in the dataset."""
    print(f"Analyzing {len(df)} reversal records...")
    
    # Calculate signals for each row
    results = []
    for idx, row in df.iterrows():
        signals = calculate_capitulation_signals(row)
        score = calculate_capitulation_score(signals)
        
        results.append({
            'date': row['date'],
            'ticker': row['ticker'],
            'is_capitulation': score >= threshold,
            'capitulation_score': score,
            **signals,
            'trade_grade': row.get('trade_grade', ''),
            'cap': row.get('cap', ''),
            'setup': row.get('setup', ''),
            'pct_change_3': row.get('pct_change_3', np.nan),
            'gap_pct': row.get('gap_pct', np.nan),
            'reversal_open_close_pct': row.get('reversal_open_close_pct', np.nan),
            'reversal_open_low_pct': row.get('reversal_open_low_pct', np.nan),
        })
    
    result_df = pd.DataFrame(results)
    return result_df


def analyze_capitulation_stats(df: pd.DataFrame) -> Dict:
    """Calculate statistics on capitulation vs non-capitulation days."""
    cap_days = df[df['is_capitulation'] == True]
    non_cap_days = df[df['is_capitulation'] == False]
    
    stats = {
        'total_records': len(df),
        'capitulation_days': len(cap_days),
        'non_capitulation_days': len(non_cap_days),
        'capitulation_rate': len(cap_days) / len(df) * 100 if len(df) > 0 else 0,
        
        'capitulation_by_grade': cap_days['trade_grade'].value_counts().to_dict() if len(cap_days) > 0 else {},
        'capitulation_by_cap': cap_days['cap'].value_counts().to_dict() if len(cap_days) > 0 else {},
        
        'avg_score_all': df['capitulation_score'].mean(),
        'avg_score_capitulation': cap_days['capitulation_score'].mean() if len(cap_days) > 0 else 0,
        'avg_score_non_capitulation': non_cap_days['capitulation_score'].mean() if len(non_cap_days) > 0 else 0,
    }
    
    return stats


def print_analysis(stats: Dict, df: pd.DataFrame):
    """Print formatted analysis results."""
    print("\n" + "=" * 60)
    print("CAPITULATION ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Total reversal records: {stats['total_records']}")
    print(f"   Capitulation days: {stats['capitulation_days']} ({stats['capitulation_rate']:.1f}%)")
    print(f"   Non-capitulation days: {stats['non_capitulation_days']}")
    
    print(f"\n📈 Capitulation Scores:")
    print(f"   Average (all): {stats['avg_score_all']:.1f}/100")
    print(f"   Average (capitulation): {stats['avg_score_capitulation']:.1f}/100")
    print(f"   Average (non-capitulation): {stats['avg_score_non_capitulation']:.1f}/100")
    
    if stats['capitulation_by_grade']:
        print(f"\n🎯 Capitulation Days by Trade Grade:")
        for grade, count in sorted(stats['capitulation_by_grade'].items()):
            print(f"   {grade}: {count}")
    
    if stats['capitulation_by_cap']:
        print(f"\n💰 Capitulation Days by Market Cap:")
        for cap, count in sorted(stats['capitulation_by_cap'].items()):
            print(f"   {cap}: {count}")
    
    # Show examples
    cap_days = df[df['is_capitulation'] == True].sort_values('capitulation_score', ascending=False)
    if len(cap_days) > 0:
        print(f"\n🔥 Top Capitulation Examples (highest scores):")
        for idx, row in cap_days.head(10).iterrows():
            date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
            print(f"   {date_str} {row['ticker']:6s} | Score: {row['capitulation_score']:5.1f} | "
                  f"3-day: {row['pct_change_3']*100:6.1f}% | Setup: {row['setup']}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze capitulation patterns in reversal data")
    parser.add_argument('--stats', action='store_true', help='Show summary statistics only')
    parser.add_argument('--export', action='store_true', help='Export tagged data to CSV')
    parser.add_argument('--threshold', type=float, default=60.0, help='Capitulation score threshold (default: 60)')
    
    args = parser.parse_args()
    
    # Check if data exists
    if not REVERSAL_CSV.exists():
        print(f"Error: {REVERSAL_CSV} not found")
        return
    
    # Load data
    print(f"Loading reversal data from {REVERSAL_CSV}...")
    df = load_reversal_data()
    print(f"Loaded {len(df)} records")
    
    # Tag capitulation days
    print(f"\nTagging capitulation days (threshold: {args.threshold})...")
    tagged_df = tag_capitulation_days(df, threshold=args.threshold)
    
    # Calculate stats
    stats = analyze_capitulation_stats(tagged_df)
    
    # Print results
    print_analysis(stats, tagged_df)
    
    # Export if requested
    if args.export:
        print(f"\n💾 Exporting tagged data to {OUTPUT_CSV}...")
        tagged_df.to_csv(OUTPUT_CSV, index=False)
        print(f"   Exported {len(tagged_df)} records")
    
    # Summary for progress log
    summary = {
        'total_records': stats['total_records'],
        'capitulation_days': stats['capitulation_days'],
        'capitulation_rate': round(stats['capitulation_rate'], 1),
        'avg_score_capitulation': round(stats['avg_score_capitulation'], 1),
        'threshold_used': args.threshold,
        'export_path': str(OUTPUT_CSV) if args.export else None
    }
    
    print("\n📋 JSON Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
