"""
Data-Informed Exit Target Framework for Bounce Trades (Long)

Based on analysis of 123 bounce trades:
- ATR-based hit rates (targets ABOVE open)
- Gap fill (red-to-green) hit rates
- Prior day hi/lo hit rates
- Dip risk (MAE below open before bounce)

Each cap size has different optimal targets based on historical hit rates.
Targets are ABOVE open since bounces are long trades.
"""

from typing import Dict
import logging

from analyzers.exit_targets import ExitTarget, ExitFramework

logging.basicConfig(level=logging.INFO)


# =============================================================================
# DATA-INFORMED BOUNCE EXIT FRAMEWORKS BY CAP SIZE
# Targets are ABOVE open (long trade direction)
# =============================================================================

BOUNCE_EXIT_FRAMEWORKS = {

    'ETF': ExitFramework(
        cap='ETF',
        tier1=ExitTarget(
            name='0.5x ATR',
            target_type='atr',
            value=0.5,
            hit_rate=0.81,
            avg_time_mins=None,
            position_pct=0.33
        ),
        tier2=ExitTarget(
            name='1.0x ATR',
            target_type='atr',
            value=1.0,
            hit_rate=0.81,
            avg_time_mins=None,
            position_pct=0.33
        ),
        tier3=ExitTarget(
            name='Gap Fill (Red-to-Green)',
            target_type='level',
            value=0,  # Prior close level
            hit_rate=0.50,
            avg_time_mins=None,
            position_pct=0.34
        ),
        time_stop='Exit remaining by 2:00 PM if targets not hit',
        notes='n=16. ETFs have smaller moves. ATR targets more reliable than gap fill.'
    ),

    'Medium': ExitFramework(
        cap='Medium',
        tier1=ExitTarget(
            name='0.5x ATR',
            target_type='atr',
            value=0.5,
            hit_rate=0.80,
            avg_time_mins=None,
            position_pct=0.33
        ),
        tier2=ExitTarget(
            name='1.0x ATR',
            target_type='atr',
            value=1.0,
            hit_rate=0.64,
            avg_time_mins=None,
            position_pct=0.33
        ),
        tier3=ExitTarget(
            name='Gap Fill (Red-to-Green)',
            target_type='level',
            value=0,
            hit_rate=0.67,
            avg_time_mins=None,
            position_pct=0.34
        ),
        time_stop='Can hold into afternoon - bounces can develop slowly',
        notes='n=75. Largest sample. Gap fill works well as middle target.'
    ),

    'Small': ExitFramework(
        cap='Small',
        tier1=ExitTarget(
            name='0.5x ATR',
            target_type='atr',
            value=0.5,
            hit_rate=0.75,
            avg_time_mins=None,
            position_pct=0.33
        ),
        tier2=ExitTarget(
            name='1.0x ATR',
            target_type='atr',
            value=1.0,
            hit_rate=0.75,
            avg_time_mins=None,
            position_pct=0.33
        ),
        tier3=ExitTarget(
            name='Gap Fill (Red-to-Green)',
            target_type='level',
            value=0,
            hit_rate=0.75,
            avg_time_mins=None,
            position_pct=0.34
        ),
        time_stop='Can hold into afternoon',
        notes='n=8. Small sample — use with caution. Gap fill strongest target.'
    ),

    'Large': ExitFramework(
        cap='Large',
        tier1=ExitTarget(
            name='0.5x ATR',
            target_type='atr',
            value=0.5,
            hit_rate=0.96,
            avg_time_mins=None,
            position_pct=0.33
        ),
        tier2=ExitTarget(
            name='1.0x ATR',
            target_type='atr',
            value=1.0,
            hit_rate=0.83,
            avg_time_mins=None,
            position_pct=0.33
        ),
        tier3=ExitTarget(
            name='Gap Fill (Red-to-Green)',
            target_type='level',
            value=0,
            hit_rate=0.67,
            avg_time_mins=None,
            position_pct=0.34
        ),
        time_stop='Can hold into afternoon',
        notes='n=24. Large caps have highest 0.5x ATR hit rate (96%).'
    ),

    # Micro: use Small defaults (n=0 in dataset)
    'Micro': ExitFramework(
        cap='Micro',
        tier1=ExitTarget(
            name='0.5x ATR',
            target_type='atr',
            value=0.5,
            hit_rate=0.75,
            avg_time_mins=None,
            position_pct=0.33
        ),
        tier2=ExitTarget(
            name='1.0x ATR',
            target_type='atr',
            value=1.0,
            hit_rate=0.75,
            avg_time_mins=None,
            position_pct=0.33
        ),
        tier3=ExitTarget(
            name='Gap Fill (Red-to-Green)',
            target_type='level',
            value=0,
            hit_rate=0.75,
            avg_time_mins=None,
            position_pct=0.34
        ),
        time_stop='Can hold into afternoon',
        notes='n=0. Using Small cap defaults — no Micro cap bounce data.'
    ),
}


# =============================================================================
# SUMMARY STATISTICS FROM BOUNCE ANALYSIS (123 trades)
# =============================================================================

BOUNCE_CAP_STATISTICS = {
    'ETF': {
        'n': 16,
        'hit_0_5x_atr': 81,
        'hit_1_0x_atr': 81,
        'hit_1_5x_atr': 69,
        'hit_2_0x_atr': 69,
        'hit_gap_fill': 50,
        'hit_prior_day_hilo': 97,
        # Dip risk: how much stock drops BELOW open before bouncing
        'avg_dip_pct': 2.5,
        'avg_dip_atrs': 0.71,
        'max_dip_pct': 7.7,
    },
    'Medium': {
        'n': 75,
        'hit_0_5x_atr': 80,
        'hit_1_0x_atr': 64,
        'hit_1_5x_atr': 43,
        'hit_2_0x_atr': 31,
        'hit_gap_fill': 67,
        'hit_prior_day_hilo': 97,
        'avg_dip_pct': 9.7,
        'avg_dip_atrs': 1.81,
        'max_dip_pct': 60.4,
    },
    'Small': {
        'n': 8,
        'hit_0_5x_atr': 75,
        'hit_1_0x_atr': 75,
        'hit_1_5x_atr': 50,
        'hit_2_0x_atr': 25,
        'hit_gap_fill': 75,
        'hit_prior_day_hilo': 97,
        'avg_dip_pct': 18.1,
        'avg_dip_atrs': 1.59,
        'max_dip_pct': 53.6,
    },
    'Large': {
        'n': 24,
        'hit_0_5x_atr': 96,
        'hit_1_0x_atr': 83,
        'hit_1_5x_atr': 54,
        'hit_2_0x_atr': 42,
        'hit_gap_fill': 67,
        'hit_prior_day_hilo': 97,
        'avg_dip_pct': 2.5,
        'avg_dip_atrs': 0.86,
        'max_dip_pct': 10.6,
    },
    'Micro': {
        'n': 0,
        'hit_0_5x_atr': 75,
        'hit_1_0x_atr': 75,
        'hit_1_5x_atr': 50,
        'hit_2_0x_atr': 25,
        'hit_gap_fill': 75,
        'hit_prior_day_hilo': 97,
        'avg_dip_pct': 18.1,
        'avg_dip_atrs': 1.59,
        'max_dip_pct': 53.6,
    },
}


def get_bounce_exit_framework(cap: str) -> ExitFramework:
    """Get the bounce exit framework for a given cap size."""
    if cap not in BOUNCE_EXIT_FRAMEWORKS:
        logging.warning(f"Unknown cap '{cap}', defaulting to Medium for bounce exits")
        cap = 'Medium'
    return BOUNCE_EXIT_FRAMEWORKS[cap]


def calculate_bounce_exit_targets(cap: str, entry_price: float, atr: float,
                                  prior_close: float = None,
                                  prior_high: float = None) -> Dict:
    """
    Calculate target PRICE LEVELS for a bounce (long) trade.

    Targets are ABOVE open (long direction): target = entry + (mult * atr).
    Gap fill is only valid when prior_close > entry_price (stock gapped down).

    Args:
        cap: Market cap category
        entry_price: OPEN price (reference point for calculating levels)
        atr: Average True Range
        prior_close: Prior day's close (for gap fill / red-to-green)
        prior_high: Prior day's high

    Returns:
        Dictionary with target price levels and historical hit rates
    """
    framework = get_bounce_exit_framework(cap)

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

        if tier.target_type == 'atr':
            # ABOVE open for long bounce trades
            target_price = entry_price + (tier.value * atr)
            target_info['target_price'] = target_price
            target_info['target_pct'] = (target_price - entry_price) / entry_price

        elif tier.target_type == 'level':
            if 'Gap Fill' in tier.name or 'Prior Close' in tier.name:
                # Gap fill only valid when prior_close > entry (stock gapped down)
                if prior_close and prior_close > entry_price:
                    target_info['target_price'] = prior_close
                    target_info['target_pct'] = (prior_close - entry_price) / entry_price
                else:
                    target_info['target_price'] = None
                    target_info['note'] = 'No gap down — gap fill N/A'
            elif 'Prior' in tier.name and 'High' in tier.name and prior_high:
                target_info['target_price'] = prior_high
                target_info['target_pct'] = (prior_high - entry_price) / entry_price
            else:
                target_info['target_price'] = None

        targets['tiers'].append(target_info)

    # Sort tiers by target price ascending so T1 < T2 < T3 (scale out higher).
    # Tiers with no target price (None) go last.
    targets['tiers'].sort(key=lambda t: (t.get('target_price') is None, t.get('target_price') or 0))
    for i, tier in enumerate(targets['tiers'], 1):
        tier['tier'] = i

    targets['time_stop'] = framework.time_stop
    targets['notes'] = framework.notes

    return targets


def format_bounce_exit_targets_html(targets: Dict) -> str:
    """Format bounce exit targets as HTML for reports.

    Green-accented theme, dip risk warning, sample size warnings.
    """
    cap = targets['cap']
    stats = BOUNCE_CAP_STATISTICS.get(cap, {})
    n = stats.get('n', 0)

    # Green theme for bounce (long) trades
    color = '#2e7d32'  # dark green

    open_price = targets['entry_price']
    atr = targets['atr']
    src = (targets.get('entry_price_source') or targets.get('open_price_source') or '').strip().lower()
    if src == 'open':
        ref_label = 'OPEN'
    elif src == 'live':
        ref_label = 'LIVE'
    elif src == 'prior_close':
        ref_label = 'PRIOR CLOSE'
    else:
        ref_label = 'REF'

    # Sample size warning
    sample_warning = ''
    if n < 10:
        sample_warning = (
            f'<p style="margin: 4px 0; font-size: 0.8em; background: #fff3cd; '
            f'padding: 4px 6px; border-radius: 4px; color: #856404;">'
            f'&#9888; Small sample size (n={n}) — use targets as guidelines, not guarantees</p>'
        )

    html = f'''
    <div style="border: 2px solid {color}; padding: 12px; margin: 10px 0; border-radius: 8px; background: #f1f8e9;">
        <h4 style="color: {color}; margin: 0 0 10px 0;">BOUNCE TARGET LEVELS - {cap} Cap (n={n})</h4>
        <p style="margin: 0 0 8px 0; font-size: 0.85em;">Ref: ${open_price:.2f} ({ref_label}) | ATR: ${atr:.2f}</p>
        <p style="margin: 0 0 8px 0; font-size: 0.8em; color: #666;"><em>Price levels ABOVE the reference price — mark these on your chart. Gap Fill = Red-to-Green move.</em></p>
        {sample_warning}
        <table style="width: 100%; font-size: 0.9em; border-collapse: collapse;">
            <tr style="background: #c8e6c9;">
                <th style="padding: 5px; text-align: left;">Tier</th>
                <th style="padding: 5px; text-align: left;">Target</th>
                <th style="padding: 5px; text-align: right;">Level</th>
                <th style="padding: 5px; text-align: right;">Hit Rate</th>
                <th style="padding: 5px; text-align: right;">Size</th>
            </tr>
    '''

    for tier in targets['tiers']:
        price_str = f"${tier['target_price']:.2f}" if tier.get('target_price') else "—"
        pct_str = f"(+{tier['target_pct']*100:.1f}%)" if tier.get('target_pct') else ""
        note_str = f' <span style="color:#999;font-size:0.8em;">({tier["note"]})</span>' if tier.get('note') else ""

        html += f'''
            <tr>
                <td style="padding: 5px;"><strong>T{tier['tier']}</strong></td>
                <td style="padding: 5px;">{tier['name']}{note_str}</td>
                <td style="padding: 5px; text-align: right;">{price_str} {pct_str}</td>
                <td style="padding: 5px; text-align: right;">{tier['hit_rate']*100:.0f}%</td>
                <td style="padding: 5px; text-align: right;">{tier['position_pct']*100:.0f}%</td>
            </tr>
        '''

    # Dip risk context (analogous to squeeze risk for reversals)
    avg_dip = stats.get('avg_dip_pct', 0)
    avg_dip_atrs = stats.get('avg_dip_atrs', 0)
    max_dip = stats.get('max_dip_pct', 0)
    potential_lod = open_price * (1 - avg_dip / 100) if avg_dip else None

    dip_html = ""
    if avg_dip:
        dip_html = f'''
        <p style="margin: 8px 0; font-size: 0.85em; background: #ffebee; padding: 6px; border-radius: 4px;">
            <strong>&#9888; Dip Risk:</strong> Avg -{avg_dip:.0f}% below open before bounce ({avg_dip_atrs:.1f} ATRs) | Max -{max_dip:.0f}%<br>
            <strong>Potential LOD:</strong> ${potential_lod:.2f} (based on avg dip)
        </p>
        '''

    html += f'''
        </table>
        {dip_html}
        <p style="margin: 10px 0 0 0; font-size: 0.85em; color: #666;">
            <strong>Time:</strong> {targets['time_stop']}<br>
            <strong>Note:</strong> {targets['notes']}
        </p>
    </div>
    '''

    return html
