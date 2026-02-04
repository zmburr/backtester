"""
Data-Informed Exit Target Framework for Parabolic Short Reversals

Based on analysis of 52 Grade A trades:
- MFE/MAE analysis
- ATR target hit rates
- EMA target hit rates
- Technical level hit rates
- Time-based analysis

Each cap size has different optimal targets based on historical hit rates.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)


@dataclass
class ExitTarget:
    """Single exit target with metadata."""
    name: str
    target_type: str  # 'atr', 'ema', 'level', 'trail'
    value: float      # ATR multiple, EMA period, or price level
    hit_rate: float   # Historical hit rate (0-1)
    avg_time_mins: Optional[int]  # Average time to hit
    position_pct: float  # % of position to exit here


@dataclass
class ExitFramework:
    """Complete exit framework for a cap size."""
    cap: str
    tier1: ExitTarget
    tier2: ExitTarget
    tier3: ExitTarget
    time_stop: str
    notes: str


# =============================================================================
# DATA-INFORMED EXIT FRAMEWORKS BY CAP SIZE
# =============================================================================

EXIT_FRAMEWORKS = {

    'Large': ExitFramework(
        cap='Large',
        tier1=ExitTarget(
            name='Gap Fill / Prior Close',
            target_type='level',
            value=0,  # Prior close level
            hit_rate=1.00,  # 100% hit rate
            avg_time_mins=60,
            position_pct=0.33
        ),
        tier2=ExitTarget(
            name='4-Day EMA',
            target_type='ema',
            value=4,
            hit_rate=0.71,  # 71% hit rate
            avg_time_mins=120,
            position_pct=0.33
        ),
        tier3=ExitTarget(
            name='Prior Day Low',
            target_type='level',
            value=0,  # Prior day low
            hit_rate=0.86,  # 86% hit rate
            avg_time_mins=180,
            position_pct=0.34
        ),
        time_stop='Exit remaining by 2:00 PM if targets not hit',
        notes='Large caps have small moves but high hit rates. 4-day EMA works here.'
    ),

    'ETF': ExitFramework(
        cap='ETF',
        tier1=ExitTarget(
            name='1.0x ATR',
            target_type='atr',
            value=1.0,
            hit_rate=0.80,  # 80% hit rate
            avg_time_mins=90,
            position_pct=0.33
        ),
        tier2=ExitTarget(
            name='Gap Fill / Prior Close',
            target_type='level',
            value=0,
            hit_rate=0.60,  # 60% hit rate
            avg_time_mins=120,
            position_pct=0.33
        ),
        tier3=ExitTarget(
            name='1.5x ATR with Trail',
            target_type='atr',
            value=1.5,
            hit_rate=0.80,  # 80% hit rate
            avg_time_mins=150,
            position_pct=0.34
        ),
        time_stop='Exit remaining by 1:00 PM if targets not hit',
        notes='ETFs have smaller moves. Use ATR targets, not EMAs.'
    ),

    'Medium': ExitFramework(
        cap='Medium',
        tier1=ExitTarget(
            name='Gap Fill / Prior Close',
            target_type='level',
            value=0,
            hit_rate=0.81,  # 81% hit rate
            avg_time_mins=90,
            position_pct=0.33
        ),
        tier2=ExitTarget(
            name='1.5x ATR',
            target_type='atr',
            value=1.5,
            hit_rate=0.65,  # 65% hit rate
            avg_time_mins=107,
            position_pct=0.33
        ),
        tier3=ExitTarget(
            name='2.0x ATR with Trail',
            target_type='atr',
            value=2.0,
            hit_rate=0.45,  # 45% hit rate
            avg_time_mins=150,
            position_pct=0.34
        ),
        time_stop='Can hold into afternoon - most MFEs occur late',
        notes='EMAs DO NOT WORK for Medium cap (too extended). Use gap fill + ATR.'
    ),

    'Small': ExitFramework(
        cap='Small',
        tier1=ExitTarget(
            name='1.0x ATR',
            target_type='atr',
            value=1.0,
            hit_rate=0.80,  # 80% hit rate
            avg_time_mins=60,
            position_pct=0.33
        ),
        tier2=ExitTarget(
            name='1.5x ATR',
            target_type='atr',
            value=1.5,
            hit_rate=0.80,  # 80% hit rate
            avg_time_mins=90,
            position_pct=0.33
        ),
        tier3=ExitTarget(
            name='2.0x ATR with Trail',
            target_type='atr',
            value=2.0,
            hit_rate=0.80,  # 80% hit rate
            avg_time_mins=120,
            position_pct=0.34
        ),
        time_stop='Can hold into afternoon',
        notes='EMAs DO NOT WORK for Small cap. ATR targets are reliable.'
    ),

    'Micro': ExitFramework(
        cap='Micro',
        tier1=ExitTarget(
            name='1.5x ATR',
            target_type='atr',
            value=1.5,
            hit_rate=1.00,  # 100% hit rate!
            avg_time_mins=90,
            position_pct=0.33
        ),
        tier2=ExitTarget(
            name='2.0x ATR',
            target_type='atr',
            value=2.0,
            hit_rate=1.00,  # 100% hit rate!
            avg_time_mins=120,
            position_pct=0.33
        ),
        tier3=ExitTarget(
            name='2.5x ATR with Trail',
            target_type='atr',
            value=2.5,
            hit_rate=1.00,  # 100% hit rate!
            avg_time_mins=180,
            position_pct=0.33
        ),
        time_stop='Let them run - biggest moves, can hold all day',
        notes='Micro caps have HUGE moves. All ATR targets hit 100%. Be patient!'
    ),
}


# =============================================================================
# SUMMARY STATISTICS FROM ANALYSIS
# =============================================================================

CAP_STATISTICS = {
    'Large': {
        'avg_mfe': 9.0,
        'avg_captured': 8.2,
        'efficiency': 87.9,
        'hit_1x_atr': 86,
        'hit_1_5x_atr': 43,
        'hit_2x_atr': 14,
        'hit_gap_fill': 100,
        'hit_prior_day_low': 86,
        'hit_4ema': 71,
        'avg_time_to_mfe': 309,
        # Squeeze risk: How much stock typically runs ABOVE open before reversing
        'avg_squeeze_pct': 2.2,   # Average MAE (adverse move before MFE)
        'max_squeeze_pct': 11.1,  # Max observed squeeze above open
    },
    'ETF': {
        'avg_mfe': 8.1,
        'avg_captured': 3.3,
        'efficiency': 59.9,
        'hit_1x_atr': 80,
        'hit_1_5x_atr': 80,
        'hit_2x_atr': 40,
        'hit_gap_fill': 60,
        'hit_prior_day_low': 40,
        'hit_4ema': 40,
        'avg_time_to_mfe': 251,
        'avg_squeeze_pct': 0.4,
        'max_squeeze_pct': 0.7,
    },
    'Medium': {
        'avg_mfe': 22.9,
        'avg_captured': 12.5,
        'efficiency': 11.1,
        'hit_1x_atr': 81,
        'hit_1_5x_atr': 65,
        'hit_2x_atr': 45,
        'hit_gap_fill': 81,
        'hit_prior_day_low': 29,
        'hit_4ema': 20,
        'avg_time_to_mfe': 205,
        'avg_squeeze_pct': 9.8,   # Medium caps can squeeze significantly
        'max_squeeze_pct': 49.4,  # GME-style squeezes possible
    },
    'Small': {
        'avg_mfe': 21.6,
        'avg_captured': 9.4,
        'efficiency': 17.2,
        'hit_1x_atr': 80,
        'hit_1_5x_atr': 80,
        'hit_2x_atr': 80,
        'hit_gap_fill': 60,
        'hit_prior_day_low': 0,
        'hit_4ema': 0,
        'avg_time_to_mfe': 201,
        'avg_squeeze_pct': 9.5,
        'max_squeeze_pct': 28.2,
    },
    'Micro': {
        'avg_mfe': 55.2,
        'avg_captured': 48.1,
        'efficiency': 86.5,
        'hit_1x_atr': 100,
        'hit_1_5x_atr': 100,
        'hit_2x_atr': 100,
        'hit_gap_fill': 100,
        'hit_prior_day_low': 75,
        'hit_4ema': 0,
        'avg_time_to_mfe': 254,
        'avg_squeeze_pct': 13.8,  # Micro caps very volatile
        'max_squeeze_pct': 27.2,
    },
}


def get_exit_framework(cap: str) -> ExitFramework:
    """Get the exit framework for a given cap size."""
    if cap not in EXIT_FRAMEWORKS:
        logging.warning(f"Unknown cap '{cap}', defaulting to Medium")
        cap = 'Medium'
    return EXIT_FRAMEWORKS[cap]


def calculate_exit_targets(cap: str, entry_price: float, atr: float,
                           prior_close: float = None, prior_low: float = None,
                           ema_4: float = None) -> Dict:
    """
    Calculate target PRICE LEVELS for a short trade.

    IMPORTANT: These are fixed price levels measured from the OPEN, not entry-relative targets.
    The analysis showed these levels are historically reached X% of the time.
    Your actual entry can be anywhere (open, after a squeeze, etc.) - the target LEVELS stay fixed.

    Args:
        cap: Market cap category
        entry_price: OPEN price (reference point for calculating levels)
        atr: Average True Range
        prior_close: Prior day's close (for gap fill)
        prior_low: Prior day's low
        ema_4: 4-day EMA value

    Returns:
        Dictionary with target price levels and historical hit rates
    """
    framework = get_exit_framework(cap)

    targets = {
        'cap': cap,
        'entry_price': entry_price,
        'atr': atr,
        'framework': framework,
        'tiers': []
    }

    for tier_num, tier in enumerate([framework.tier1, framework.tier2, framework.tier3], 1):
        target_info = {
            'tier': tier_num,
            'name': tier.name,
            'type': tier.target_type,
            'hit_rate': tier.hit_rate,
            'position_pct': tier.position_pct,
            'avg_time': tier.avg_time_mins,
        }

        # Calculate actual target price
        if tier.target_type == 'atr':
            target_price = entry_price - (tier.value * atr)
            target_info['target_price'] = target_price
            target_info['target_pct'] = (entry_price - target_price) / entry_price

        elif tier.target_type == 'ema':
            if tier.value == 4 and ema_4:
                target_price = ema_4
                target_info['target_price'] = target_price
                target_info['target_pct'] = (entry_price - target_price) / entry_price
            else:
                target_info['target_price'] = None
                target_info['note'] = f'Need {int(tier.value)}-day EMA value'

        elif tier.target_type == 'level':
            if 'Prior Close' in tier.name and prior_close:
                target_price = prior_close
                target_info['target_price'] = target_price
                target_info['target_pct'] = (entry_price - target_price) / entry_price
            elif 'Prior Day Low' in tier.name and prior_low:
                target_price = prior_low
                target_info['target_price'] = target_price
                target_info['target_pct'] = (entry_price - target_price) / entry_price
            else:
                target_info['target_price'] = None

        targets['tiers'].append(target_info)

    targets['time_stop'] = framework.time_stop
    targets['notes'] = framework.notes

    return targets


def format_exit_targets_text(targets: Dict) -> str:
    """Format target price levels as readable text."""
    lines = []
    lines.append(f"TARGET LEVELS - {targets['cap']} Cap")
    lines.append("=" * 50)
    lines.append(f"Open: ${targets['entry_price']:.2f} | ATR: ${targets['atr']:.2f}")
    lines.append("(Fixed levels from open - mark on chart)")
    lines.append("")

    for tier in targets['tiers']:
        pct_str = f"({tier['target_pct']*100:.1f}%)" if tier.get('target_pct') else ""
        price_str = f"${tier['target_price']:.2f}" if tier.get('target_price') else "N/A"
        lines.append(f"Tier {tier['tier']} ({tier['position_pct']*100:.0f}%): {tier['name']}")
        lines.append(f"  Target: {price_str} {pct_str}")
        lines.append(f"  Hit Rate: {tier['hit_rate']*100:.0f}% | Avg Time: {tier['avg_time']} mins")
        lines.append("")

    lines.append(f"Time Stop: {targets['time_stop']}")
    lines.append(f"Note: {targets['notes']}")

    return "\n".join(lines)


def format_exit_targets_html(targets: Dict) -> str:
    """Format exit targets as HTML for reports."""
    cap = targets['cap']

    # Color coding by cap
    cap_colors = {
        'Large': '#2196F3',
        'ETF': '#9C27B0',
        'Medium': '#FF9800',
        'Small': '#4CAF50',
        'Micro': '#F44336',
    }
    color = cap_colors.get(cap, '#666')

    open_price = targets['entry_price']
    atr = targets['atr']
    html = f'''
    <div style="border: 2px solid {color}; padding: 12px; margin: 10px 0; border-radius: 8px; background: #f9f9f9;">
        <h4 style="color: {color}; margin: 0 0 10px 0;">TARGET LEVELS - {cap} Cap</h4>
        <p style="margin: 0 0 8px 0; font-size: 0.85em;">Ref: ${open_price:.2f} (live) | ATR: ${atr:.2f}</p>
        <p style="margin: 0 0 8px 0; font-size: 0.8em; color: #666;"><em>Price levels from live price - mark these on your chart</em></p>
        <table style="width: 100%; font-size: 0.9em; border-collapse: collapse;">
            <tr style="background: #eee;">
                <th style="padding: 5px; text-align: left;">Tier</th>
                <th style="padding: 5px; text-align: left;">Target</th>
                <th style="padding: 5px; text-align: right;">Level</th>
                <th style="padding: 5px; text-align: right;">Hit Rate</th>
                <th style="padding: 5px; text-align: right;">Size</th>
            </tr>
    '''

    for tier in targets['tiers']:
        price_str = f"${tier['target_price']:.2f}" if tier.get('target_price') else "—"
        pct_str = f"({tier['target_pct']*100:.1f}%)" if tier.get('target_pct') else ""

        html += f'''
            <tr>
                <td style="padding: 5px;"><strong>T{tier['tier']}</strong></td>
                <td style="padding: 5px;">{tier['name']}</td>
                <td style="padding: 5px; text-align: right;">{price_str} {pct_str}</td>
                <td style="padding: 5px; text-align: right;">{tier['hit_rate']*100:.0f}%</td>
                <td style="padding: 5px; text-align: right;">{tier['position_pct']*100:.0f}%</td>
            </tr>
        '''

    # Add squeeze risk context
    stats = CAP_STATISTICS.get(cap, {})
    avg_squeeze = stats.get('avg_squeeze_pct', 0)
    max_squeeze = stats.get('max_squeeze_pct', 0)
    potential_hod = open_price * (1 + avg_squeeze/100) if avg_squeeze else None

    squeeze_html = ""
    if avg_squeeze:
        squeeze_html = f'''
        <p style="margin: 8px 0; font-size: 0.85em; background: #fff3cd; padding: 6px; border-radius: 4px;">
            <strong>⚠️ Squeeze Risk:</strong> Avg +{avg_squeeze:.1f}% above open before reversal (max +{max_squeeze:.0f}%)<br>
            <strong>Potential HOD:</strong> ${potential_hod:.2f} (based on avg squeeze)
        </p>
        '''

    html += f'''
        </table>
        {squeeze_html}
        <p style="margin: 10px 0 0 0; font-size: 0.85em; color: #666;">
            <strong>Time:</strong> {targets['time_stop']}<br>
            <strong>Note:</strong> {targets['notes']}
        </p>
    </div>
    '''

    return html


def print_all_frameworks():
    """Print all exit frameworks for reference."""
    print("\n" + "=" * 70)
    print("DATA-INFORMED EXIT TARGET FRAMEWORK")
    print("Based on 52 Grade A Parabolic Short Reversals")
    print("=" * 70)

    for cap in ['Large', 'ETF', 'Medium', 'Small', 'Micro']:
        framework = EXIT_FRAMEWORKS[cap]
        stats = CAP_STATISTICS[cap]

        print(f"\n{'='*70}")
        print(f"{cap.upper()} CAP")
        print(f"{'='*70}")
        print(f"Historical: MFE {stats['avg_mfe']:+.1f}% | Captured {stats['avg_captured']:+.1f}% | Efficiency {stats['efficiency']:.0f}%")
        print()

        print(f"TIER 1 ({framework.tier1.position_pct*100:.0f}% of position):")
        print(f"  Target: {framework.tier1.name}")
        print(f"  Hit Rate: {framework.tier1.hit_rate*100:.0f}%")
        print(f"  Avg Time: {framework.tier1.avg_time_mins} mins")
        print()

        print(f"TIER 2 ({framework.tier2.position_pct*100:.0f}% of position):")
        print(f"  Target: {framework.tier2.name}")
        print(f"  Hit Rate: {framework.tier2.hit_rate*100:.0f}%")
        print(f"  Avg Time: {framework.tier2.avg_time_mins} mins")
        print()

        print(f"TIER 3 ({framework.tier3.position_pct*100:.0f}% of position):")
        print(f"  Target: {framework.tier3.name}")
        print(f"  Hit Rate: {framework.tier3.hit_rate*100:.0f}%")
        print(f"  Avg Time: {framework.tier3.avg_time_mins} mins")
        print()

        print(f"TIME STOP: {framework.time_stop}")
        print(f"NOTES: {framework.notes}")


if __name__ == '__main__':
    print_all_frameworks()
