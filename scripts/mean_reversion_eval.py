"""
Mean Reversion GO/NOGO Evaluation Script

Evaluates a ticker on a given date BEFORE market open to determine if it's a
candidate for mean reversion short based on the playbook criteria.

Usage:
    python scripts/mean_reversion_eval.py GLD 2026-01-29
    python scripts/mean_reversion_eval.py NVDA 2026-01-15 --verbose
"""

import sys
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, List

sys.path.insert(0, r'C:\Users\zmbur\PycharmProjects\backtester')

from data_queries.polygon_queries import (
    get_levels_data,
    get_atr,
    fetch_and_calculate_volumes,
    get_ticker_pct_move,
    get_ticker_mavs_open,
    get_daily,
    adjust_date_to_market,
    get_intraday,
)
from analyzers.exit_targets import (
    get_exit_framework,
    calculate_exit_targets,
    format_exit_targets_text,
    CAP_STATISTICS,
)


# Known ETFs for classification
KNOWN_ETFS = {
    'SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'SLV', 'TLT', 'XLF', 'XLE', 'XLK',
    'XLV', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU', 'XLRE', 'VXX', 'UVXY', 'SQQQ',
    'TQQQ', 'SPXU', 'SPXS', 'ARKK', 'ARKG', 'ARKW', 'ARKF', 'ARKQ', 'SMH',
    'SOXX', 'XBI', 'IBB', 'KRE', 'XOP', 'OIH', 'GDX', 'GDXJ', 'SLV', 'USO',
    'UNG', 'EEM', 'EFA', 'FXI', 'EWZ', 'EWJ', 'KWEB', 'MCHI', 'HYG', 'LQD',
    'JNK', 'TIP', 'BND', 'AGG', 'VTI', 'VOO', 'VEA', 'VWO', 'VNQ', 'VNQI',
}


@dataclass
class ReversalScore:
    """Holds the GO/NOGO evaluation results"""
    ticker: str
    date: str
    cap: str = "Unknown"

    # Individual criteria scores (True = GO, False = NOGO, None = N/A)
    consecutive_gains: Optional[bool] = None
    range_expansion: Optional[bool] = None
    volume_expansion: Optional[bool] = None
    premarket_volume: Optional[bool] = None
    extended_from_9ema: Optional[bool] = None
    extended_from_50ma: Optional[bool] = None
    atr_distance_extreme: Optional[bool] = None

    # Raw values for display
    days_of_gains: int = 0
    range_pct_1d: float = 0.0
    range_pct_2d: float = 0.0
    vol_pct_1d: float = 0.0
    vol_pct_2d: float = 0.0
    vol_pct_3d: float = 0.0
    premarket_vol_pct: float = 0.0
    pct_from_9ema: float = 0.0
    pct_from_50ma: float = 0.0
    atr_distance: float = 0.0
    pct_change_3d: float = 0.0
    pct_change_15d: float = 0.0

    # Reference prices
    prior_close: float = 0.0
    prior_low: float = 0.0
    premarket_high: float = 0.0
    open_price: float = 0.0
    atr: float = 0.0
    gap_pct: float = 0.0

    # Exit targets (calculated from framework)
    exit_targets: Dict = field(default_factory=dict)

    def total_go_count(self) -> int:
        """Count how many criteria are GO"""
        criteria = [
            self.consecutive_gains,
            self.range_expansion,
            self.volume_expansion,
            self.premarket_volume,
            self.extended_from_9ema,
            self.extended_from_50ma,
            self.atr_distance_extreme,
        ]
        return sum(1 for c in criteria if c is not None and bool(c))

    def total_criteria(self) -> int:
        """Count total evaluable criteria"""
        criteria = [
            self.consecutive_gains,
            self.range_expansion,
            self.volume_expansion,
            self.premarket_volume,
            self.extended_from_9ema,
            self.extended_from_50ma,
            self.atr_distance_extreme,
        ]
        return sum(1 for c in criteria if c is not None)

    def verdict(self) -> str:
        """Return overall GO/NOGO verdict"""
        go_count = self.total_go_count()
        total = self.total_criteria()

        if total == 0:
            return "INSUFFICIENT DATA"

        pct = go_count / total
        if pct >= 0.7:
            return "STRONG GO"
        elif pct >= 0.5:
            return "GO"
        elif pct >= 0.3:
            return "MARGINAL"
        else:
            return "NO GO"


def get_cap_category(ticker: str, avg_daily_vol: float = 0) -> str:
    """
    Determine market cap category for a ticker.

    Categories: ETF, Large, Medium, Small, Micro

    Note: This is a simplified heuristic. For production, you'd want to
    pull actual market cap data from an API.
    """
    # Check if it's an ETF
    if ticker.upper() in KNOWN_ETFS:
        return 'ETF'

    # Heuristic based on average daily volume as proxy for size
    # This is imperfect but works for quick classification
    if avg_daily_vol >= 10_000_000:
        return 'Large'
    elif avg_daily_vol >= 2_000_000:
        return 'Medium'
    elif avg_daily_vol >= 500_000:
        return 'Small'
    else:
        return 'Micro'


def get_premarket_high(ticker: str, date: str) -> Optional[float]:
    """Get the premarket high for a given date"""
    try:
        data = get_intraday(ticker, date, multiplier=1, timespan='minute')
        if data is None or data.empty:
            return None
        premarket = data.between_time('04:00:00', '09:29:00')
        if premarket.empty:
            return None
        return premarket['high'].max()
    except Exception as e:
        print(f"Error getting premarket high: {e}")
        return None


def get_open_price(ticker: str, date: str) -> Optional[float]:
    """Get the open price for a given date"""
    try:
        daily = get_daily(ticker, date)
        if daily:
            return daily.open
        return None
    except Exception:
        return None


def count_consecutive_green_days(df) -> int:
    """Count consecutive green days going backwards from most recent"""
    if df is None or len(df) < 2:
        return 0

    count = 0
    # Start from second-to-last (prior day) and go backwards
    for i in range(len(df) - 2, -1, -1):
        if df['close'].iloc[i] > df['open'].iloc[i]:
            count += 1
        else:
            break
    return count


def evaluate_reversal(ticker: str, date: str, verbose: bool = False) -> ReversalScore:
    """
    Evaluate a ticker for mean reversion potential on the morning of a given date.

    This simulates what you would see BEFORE the market opens - using only
    prior day data and premarket activity.
    """
    score = ReversalScore(ticker=ticker, date=date)

    # Get prior trading day
    prior_date = adjust_date_to_market(date, 1)

    if verbose:
        print(f"\nEvaluating {ticker} for {date}")
        print(f"Prior day: {prior_date}")

    # === Get historical daily data ===
    df = get_levels_data(ticker, date, 60, 1, 'day')
    if df is None or df.empty:
        print(f"No historical data for {ticker}")
        return score

    # Calculate range metrics
    df['high-low'] = df['high'] - df['low']
    df['high-previous_close'] = abs(df['high'] - df['close'].shift())
    df['low-previous_close'] = abs(df['low'] - df['close'].shift())
    df['TR'] = df[['high-low', 'high-previous_close', 'low-previous_close']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14, min_periods=1).mean()
    df['PCT_ATR'] = df['TR'] / df['ATR']

    # ADV calculation
    df['ADV'] = df['volume'].rolling(window=20, min_periods=1).mean()
    df['VOL_PCT'] = df['volume'] / df['ADV']

    # Get ADV for cap classification
    avg_daily_vol = df['ADV'].iloc[-2] if len(df) >= 2 else 0
    score.cap = get_cap_category(ticker, avg_daily_vol)

    # === 1. Consecutive Gains ===
    consecutive_days = count_consecutive_green_days(df)
    score.days_of_gains = consecutive_days
    score.consecutive_gains = consecutive_days >= 2  # GO if 2+ consecutive green days

    # === 2. Range Expansion ===
    if len(df) >= 4:
        score.range_pct_1d = float(df['PCT_ATR'].iloc[-2]) if len(df) >= 2 else 0
        score.range_pct_2d = float(df['PCT_ATR'].iloc[-3]) if len(df) >= 3 else 0
        # GO if either of last 2 days had range > 150% of ATR
        score.range_expansion = score.range_pct_1d > 1.5 or score.range_pct_2d > 1.5

    # === 3. Volume Expansion ===
    if len(df) >= 4:
        score.vol_pct_1d = float(df['VOL_PCT'].iloc[-2]) if len(df) >= 2 else 0
        score.vol_pct_2d = float(df['VOL_PCT'].iloc[-3]) if len(df) >= 3 else 0
        score.vol_pct_3d = float(df['VOL_PCT'].iloc[-4]) if len(df) >= 4 else 0
        # GO if volume building (each day higher than prior, or any day > 150% ADV)
        score.volume_expansion = (
            score.vol_pct_1d > 1.5 or
            (score.vol_pct_1d > score.vol_pct_2d > score.vol_pct_3d and score.vol_pct_1d > 1.2)
        )

    # === 4. Premarket Volume ===
    vol_data = fetch_and_calculate_volumes(ticker, date)
    if vol_data:
        adv = vol_data.get('avg_daily_vol', 1)
        premarket_vol = vol_data.get('premarket_vol', 0)
        if adv > 0:
            score.premarket_vol_pct = premarket_vol / adv
            score.premarket_volume = score.premarket_vol_pct > 0.10  # GO if premarket > 10% ADV

    # === Get prior day data for targets ===
    prior_daily = get_daily(ticker, prior_date)
    if prior_daily:
        score.prior_close = prior_daily.close
        score.prior_low = prior_daily.low

    # === Get ATR ===
    atr = get_atr(ticker, date)
    if atr:
        score.atr = atr

    # === 5 & 6. Extension from Moving Averages ===
    mav_data = get_ticker_mavs_open(ticker, prior_date)
    if mav_data:
        score.pct_from_9ema = mav_data.get('pct_from_9ema', 0)
        score.pct_from_50ma = mav_data.get('pct_from_50mav', 0)
        score.atr_distance = mav_data.get('atr_distance_from_50mav', 0)

        score.extended_from_9ema = score.pct_from_9ema > 0.05  # GO if > 5% above 9 EMA
        score.extended_from_50ma = score.pct_from_50ma > 0.15  # GO if > 15% above 50 MA
        score.atr_distance_extreme = score.atr_distance > 3.0  # GO if > 3 ATRs from 50 MA

    # === Price Changes ===
    if prior_daily:
        pct_data = get_ticker_pct_move(ticker, prior_date, prior_daily.high)
        if pct_data:
            score.pct_change_3d = pct_data.get('pct_change_3', 0)
            score.pct_change_15d = pct_data.get('pct_change_15', 0)

    # === Get premarket high and open for target date ===
    pm_high = get_premarket_high(ticker, date)
    if pm_high:
        score.premarket_high = pm_high

    open_price = get_open_price(ticker, date)
    if open_price:
        score.open_price = open_price
        if score.prior_close > 0:
            score.gap_pct = (open_price - score.prior_close) / score.prior_close

    # === Calculate Exit Targets using framework ===
    if score.open_price > 0 and score.atr > 0:
        score.exit_targets = calculate_exit_targets(
            cap=score.cap,
            entry_price=score.open_price,  # Targets from OPEN
            atr=score.atr,
            prior_close=score.prior_close,
            prior_low=score.prior_low,
        )

    return score


def print_evaluation(score: ReversalScore, verbose: bool = False):
    """Print formatted evaluation results"""

    def status(val: Optional[bool]) -> str:
        if val is None:
            return "N/A"
        return "[GO]" if bool(val) else "[NO]"

    print(f"\n{'='*70}")
    print(f"MEAN REVERSION GO/NOGO EVALUATION")
    print(f"Ticker: {score.ticker}  |  Date: {score.date}  |  Cap: {score.cap}")
    print(f"{'='*70}")

    print(f"\n>>> VERDICT: {score.verdict()} ({score.total_go_count()}/{score.total_criteria()} criteria met)")

    print(f"\n--- GO/NOGO CHECKLIST ---")
    print(f"{'Criteria':<35} {'Status':<10} {'Value':<20}")
    print("-" * 70)

    print(f"{'Consecutive Green Days (>=2)':<35} {status(score.consecutive_gains):<10} {score.days_of_gains} days")
    print(f"{'Range Expansion (>150% ATR)':<35} {status(score.range_expansion):<10} 1d: {score.range_pct_1d*100:.0f}%, 2d: {score.range_pct_2d*100:.0f}%")
    print(f"{'Volume Expansion (>150% ADV)':<35} {status(score.volume_expansion):<10} 1d: {score.vol_pct_1d*100:.0f}%, 2d: {score.vol_pct_2d*100:.0f}%")
    print(f"{'Premarket Volume (>10% ADV)':<35} {status(score.premarket_volume):<10} {score.premarket_vol_pct*100:.1f}%")
    print(f"{'Extended from 9 EMA (>5%)':<35} {status(score.extended_from_9ema):<10} {score.pct_from_9ema*100:.1f}%")
    print(f"{'Extended from 50 MA (>15%)':<35} {status(score.extended_from_50ma):<10} {score.pct_from_50ma*100:.1f}%")
    print(f"{'ATR Distance Extreme (>3 ATRs)':<35} {status(score.atr_distance_extreme):<10} {score.atr_distance:.1f} ATRs")

    print(f"\n--- PRICE CONTEXT ---")
    print(f"Prior Close: ${score.prior_close:.2f}  |  Prior Low: ${score.prior_low:.2f}")
    if score.open_price > 0:
        print(f"Open: ${score.open_price:.2f}  |  Gap: {score.gap_pct*100:+.1f}%")
    if score.premarket_high > 0:
        print(f"Premarket High: ${score.premarket_high:.2f}")
    print(f"ATR: ${score.atr:.2f}")
    print(f"3-Day Move: {score.pct_change_3d*100:+.1f}%  |  15-Day Move: {score.pct_change_15d*100:+.1f}%")

    # === EXIT TARGETS (from framework) ===
    if score.exit_targets and score.exit_targets.get('tiers'):
        print(f"\n--- EXIT TARGETS ({score.cap} Cap - from Open ${score.open_price:.2f}) ---")
        print(f"{'Tier':<8} {'Target':<25} {'Level':<15} {'Hit Rate':<12} {'Size':<8}")
        print("-" * 70)

        for tier in score.exit_targets['tiers']:
            price_str = f"${tier['target_price']:.2f}" if tier.get('target_price') else "N/A"
            pct_str = f"(-{tier['target_pct']*100:.1f}%)" if tier.get('target_pct') else ""
            hit_str = f"{tier['hit_rate']*100:.0f}%"
            size_str = f"{tier['position_pct']*100:.0f}%"

            print(f"T{tier['tier']:<7} {tier['name']:<25} {price_str:<15} {hit_str:<12} {size_str:<8}")

        print(f"\nTime Stop: {score.exit_targets.get('time_stop', 'N/A')}")

        # Add squeeze risk info
        stats = CAP_STATISTICS.get(score.cap, {})
        if stats.get('avg_squeeze_pct'):
            avg_sq = stats['avg_squeeze_pct']
            max_sq = stats['max_squeeze_pct']
            potential_hod = score.open_price * (1 + avg_sq/100)
            print(f"\n[!] Squeeze Risk: Avg +{avg_sq:.1f}% above open (max +{max_sq:.0f}%)")
            print(f"    Potential HOD (stop area): ${potential_hod:.2f}")

    print(f"\n{'='*70}")

    if verbose:
        print("\n--- INTERPRETATION ---")
        if score.verdict() in ["STRONG GO", "GO"]:
            print("Setup meets criteria for mean reversion short candidate.")
            print("Key factors to confirm:")
            print("  - Wait for price action confirmation (weak open, VWAP fail)")
            print("  - Watch for 2-min bar break below prior bar low")
            print(f"  - Stop above premarket high (${score.premarket_high:.2f})" if score.premarket_high > 0 else "  - Stop above morning high")
            print("\nTarget execution:")
            print("  - Exit 1/3 at Tier 1")
            print("  - Exit 1/3 at Tier 2")
            print("  - Trail remaining 1/3 to Tier 3")
        elif score.verdict() == "MARGINAL":
            print("Setup has some but not all ideal characteristics.")
            print("Consider smaller size or wait for better setup.")
        else:
            print("Setup does not meet minimum criteria for mean reversion.")
            print("Missing key elements - pass on this trade.")

        if score.exit_targets:
            print(f"\nNote: {score.exit_targets.get('notes', '')}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate mean reversion setup')
    parser.add_argument('ticker', help='Stock ticker symbol')
    parser.add_argument('date', help='Date to evaluate (YYYY-MM-DD)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')

    args = parser.parse_args()

    # Normalize date format
    date = args.date
    if '/' in date:
        # Convert M/D/YY to YYYY-MM-DD
        parts = date.split('/')
        if len(parts[2]) == 2:
            parts[2] = '20' + parts[2]
        date = f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"

    score = evaluate_reversal(args.ticker.upper(), date, args.verbose)
    print_evaluation(score, args.verbose)


if __name__ == '__main__':
    main()
