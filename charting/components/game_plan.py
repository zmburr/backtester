"""
Bounce + Reversal game plan renderers.
Reuses existing scoring/validation functions — no reimplementation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from charting.theme import rec_badge_html, criteria_table_html, intensity_meter_html


# ---------------------------------------------------------------------------
# Bounce Game Plan
# ---------------------------------------------------------------------------

def render_bounce_game_plan(ticker: str, date: str, cap: str):
    """
    Render bounce game plan: pre-trade checklist + exit targets.

    Args:
        ticker: Stock symbol
        date: Date in YYYY-MM-DD format
        cap: Market cap category
    """
    if st.button("Generate Bounce Game Plan", key="bounce_gp_btn"):
        with st.spinner("Fetching bounce metrics..."):
            try:
                from analyzers.bounce_scorer import (
                    fetch_bounce_metrics, BouncePretrade, classify_stock,
                )
                from analyzers.bounce_exit_targets import (
                    calculate_bounce_exit_targets, format_bounce_exit_targets_html,
                )
                from data_queries.polygon_queries import get_atr

                metrics = fetch_bounce_metrics(ticker, date)
                checker = BouncePretrade()
                result = checker.validate(ticker, metrics, cap=cap)

                # Get ATR and entry price for exit targets
                atr = get_atr(ticker, date)
                entry_price = metrics.get('current_price')
                prior_close = metrics.get('prior_close')

                st.session_state.bounce_game_plan = {
                    'result': result,
                    'metrics': metrics,
                    'atr': atr,
                    'entry_price': entry_price,
                    'prior_close': prior_close,
                }
            except Exception as e:
                st.error(f"Failed to generate bounce game plan: {e}")
                return

    # Render cached game plan
    gp = st.session_state.get('bounce_game_plan')
    if gp is None:
        st.info("Click 'Generate Bounce Game Plan' to run the pre-trade checklist.")
        return

    result = gp['result']
    atr = gp['atr']
    entry_price = gp['entry_price']
    prior_close = gp['prior_close']

    # Recommendation badge + score header
    badge = rec_badge_html(result.recommendation)
    st.markdown(
        f"{badge} &nbsp; "
        f'<span style="font-family:JetBrains Mono,monospace; font-size:0.85rem; color:#c0c8d8;">'
        f"Score: {result.score}/{result.max_score}</span> &nbsp; "
        f'<span style="font-family:JetBrains Mono,monospace; font-size:0.75rem; color:#5a6578;">'
        f"{result.setup_type}</span>",
        unsafe_allow_html=True,
    )
    st.caption(result.summary)

    # Criteria table (styled HTML)
    criteria_rows = []
    for item in result.items:
        criteria_rows.append({
            'Status': "PASS" if item.passed else "FAIL",
            'Criterion': item.description,
            'Required': item.threshold_display,
            'Actual': item.actual_display,
            'Reference': item.reference,
        })
    st.markdown(
        criteria_table_html(criteria_rows, ['Status', 'Criterion', 'Required', 'Actual', 'Reference']),
        unsafe_allow_html=True,
    )

    # Bonuses and warnings (inline, compact)
    col_b, col_w = st.columns(2)
    with col_b:
        if result.bonuses:
            for bonus in result.bonuses:
                st.markdown(
                    f'<span style="font-family:JetBrains Mono,monospace; font-size:0.7rem; color:#6ee7b7;">'
                    f'+ {bonus}</span>',
                    unsafe_allow_html=True,
                )
    with col_w:
        if result.warnings:
            for warning in result.warnings:
                st.markdown(
                    f'<span style="font-family:JetBrains Mono,monospace; font-size:0.7rem; color:#fbbf24;">'
                    f'! {warning}</span>',
                    unsafe_allow_html=True,
                )

    # Exit targets
    if atr and entry_price:
        st.markdown(
            '<div style="border-top:1px solid #141924; margin:8px 0;"></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<span style="font-family:JetBrains Mono,monospace; font-size:0.65rem; '
            'text-transform:uppercase; letter-spacing:0.1em; color:#3d4a5c;">Exit Targets</span>',
            unsafe_allow_html=True,
        )
        try:
            from analyzers.bounce_exit_targets import (
                calculate_bounce_exit_targets, format_bounce_exit_targets_html,
            )
            targets = calculate_bounce_exit_targets(
                cap=result.cap,
                entry_price=entry_price,
                atr=atr,
                prior_close=prior_close,
            )
            html = format_bounce_exit_targets_html(targets)
            st.markdown(html, unsafe_allow_html=True)

            # Store target levels for chart overlay
            chart_targets = []
            for tier in targets.get('tiers', []):
                tp = tier.get('target_price')
                if tp is not None:
                    label = f"B-T{tier['tier']}: {tier['name']} (${tp:.2f})"
                    chart_targets.append((tp, '#6ee7b7', label))
            st.session_state.chart_targets = chart_targets
        except Exception as e:
            st.warning(f"Exit target calculation failed: {e}")
    else:
        if not atr:
            st.warning("ATR unavailable — cannot compute exit targets")
        if not entry_price:
            st.warning("Entry price unavailable — cannot compute exit targets")


def get_bounce_chart_targets() -> List[Tuple[float, str, str]]:
    """Get bounce target levels for chart overlay."""
    gp = st.session_state.get('bounce_game_plan')
    if gp is None:
        return []
    return st.session_state.get('chart_targets', [])


# ---------------------------------------------------------------------------
# Reversal Game Plan
# ---------------------------------------------------------------------------

def render_reversal_game_plan(ticker: str, date: str, cap: str):
    """
    Render reversal game plan: typed classification, checklist + exit targets.

    Args:
        ticker: Stock symbol
        date: Date in YYYY-MM-DD format
        cap: Market cap category
    """
    if st.button("Generate Reversal Game Plan", key="reversal_gp_btn"):
        with st.spinner("Fetching reversal metrics..."):
            try:
                from analyzers.reversal_pretrade import (
                    ReversalPretrade, classify_reversal_setup,
                )
                from analyzers.reversal_scorer import (
                    ReversalScorer, compute_reversal_intensity,
                )
                from analyzers.exit_targets import (
                    calculate_exit_targets, format_exit_targets_html,
                )
                from data_queries.polygon_queries import (
                    get_ticker_mavs_open, get_range_vol_expansion_data,
                    get_daily, get_atr, get_levels_data, adjust_date_to_market,
                )

                # Assemble metrics from multiple Polygon calls
                metrics = {}

                # MA distances
                mavs = get_ticker_mavs_open(ticker, date)
                if mavs:
                    metrics.update(mavs)

                # Range / RVOL
                range_data = get_range_vol_expansion_data(ticker, date)
                if range_data:
                    metrics.update(range_data)
                    metrics['prior_day_range_atr'] = range_data.get('one_day_before_range_pct')
                    metrics['rvol_score'] = range_data.get('percent_of_vol_one_day_before')

                # Gap calculation
                prior_date = adjust_date_to_market(date, 1)
                daily = get_daily(ticker, date)
                prior_daily = get_daily(ticker, prior_date)
                if daily and prior_daily:
                    today_open = getattr(daily, 'open', None)
                    prior_close = getattr(prior_daily, 'close', None)
                    if today_open and prior_close and prior_close != 0:
                        metrics['gap_pct'] = (today_open - prior_close) / prior_close
                    metrics['entry_price'] = today_open
                    metrics['prior_close'] = prior_close
                elif daily:
                    metrics['entry_price'] = getattr(daily, 'open', None)

                # Consecutive up days and momentum
                levels = get_levels_data(ticker, date, 30, 1, 'day')
                if levels is not None and not levels.empty:
                    consecutive_up = 0
                    for i in range(len(levels) - 2, 0, -1):
                        if levels.iloc[i]['close'] > levels.iloc[i - 1]['close']:
                            consecutive_up += 1
                        else:
                            break
                    metrics['consecutive_up_days'] = consecutive_up

                    if len(levels) >= 4:
                        close_3ago = levels.iloc[-4]['close']
                        today_close = levels.iloc[-1]['close']
                        if close_3ago and close_3ago != 0:
                            metrics['pct_change_3'] = (today_close - close_3ago) / close_3ago

                # ATR
                atr = get_atr(ticker, date)
                if atr:
                    metrics['atr'] = atr
                    entry = metrics.get('entry_price')
                    if entry and entry != 0:
                        metrics['atr_pct'] = atr / entry

                # Try typed classification first
                setup_type = classify_reversal_setup(metrics)
                typed_result = None
                generic_result = None
                intensity = None

                if setup_type:
                    pretrade = ReversalPretrade()
                    typed_result = pretrade.validate(
                        ticker=ticker,
                        metrics=metrics,
                        setup_type=setup_type,
                        cap=cap,
                    )

                # Always run generic scorer for comparison / fallback
                scorer = ReversalScorer()
                generic_result = scorer.score_setup(
                    ticker=ticker, date=date, cap=cap, metrics=metrics,
                )

                # Intensity score
                if metrics.get('atr_pct'):
                    intensity_result = compute_reversal_intensity(metrics, cap=cap)
                    intensity = intensity_result.get('composite')

                st.session_state.reversal_game_plan = {
                    'setup_type': setup_type,
                    'typed_result': typed_result,
                    'generic_result': generic_result,
                    'intensity': intensity,
                    'metrics': metrics,
                    'atr': atr,
                    'cap': cap,
                }
            except Exception as e:
                st.error(f"Failed to generate reversal game plan: {e}")
                return

    # Render cached game plan
    gp = st.session_state.get('reversal_game_plan')
    if gp is None:
        st.info("Click 'Generate Reversal Game Plan' to run the pre-trade checklist.")
        return

    setup_type = gp['setup_type']
    typed_result = gp['typed_result']
    generic = gp['generic_result']
    intensity = gp['intensity']
    metrics = gp['metrics']
    atr = gp['atr']
    cap = gp['cap']

    # Intensity meter (prominent)
    if intensity is not None:
        st.markdown(
            '<span style="font-family:JetBrains Mono,monospace; font-size:0.65rem; '
            'text-transform:uppercase; letter-spacing:0.1em; color:#3d4a5c;">Intensity</span>',
            unsafe_allow_html=True,
        )
        st.markdown(intensity_meter_html(intensity), unsafe_allow_html=True)

    # Typed setup result (if classified)
    if typed_result:
        badge = rec_badge_html(typed_result.recommendation)
        st.markdown(
            f'{badge} &nbsp; '
            f'<span style="font-family:JetBrains Mono,monospace; font-size:0.85rem; color:#c0c8d8;">'
            f'Score: {typed_result.score}/{typed_result.max_score}</span> &nbsp; '
            f'<span style="font-family:JetBrains Mono,monospace; font-size:0.75rem; color:#c084fc;">'
            f'{setup_type}</span>',
            unsafe_allow_html=True,
        )
        st.caption(typed_result.summary)

        # Criteria table (styled HTML)
        criteria_rows = []
        for item in typed_result.items:
            criteria_rows.append({
                'Status': "PASS" if item.passed else "FAIL",
                'Criterion': item.description,
                'Required': item.threshold_display,
                'Actual': item.actual_display,
            })
        st.markdown(
            criteria_table_html(criteria_rows, ['Status', 'Criterion', 'Required', 'Actual']),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span style="font-family:JetBrains Mono,monospace; font-size:0.72rem; color:#5a6578;">'
            'No typed setup detected — using generic reversal scorer</span>',
            unsafe_allow_html=True,
        )

    # Generic scorer result
    with st.expander("Generic Reversal Score" if typed_result else "Reversal Score", expanded=not bool(typed_result)):
        pretrade_rec = generic.get('pretrade_recommendation', 'N/A')
        badge = rec_badge_html(pretrade_rec)
        st.markdown(
            f"{badge} &nbsp; "
            f'<span style="font-family:JetBrains Mono,monospace; font-size:0.8rem; color:#c0c8d8;">'
            f"Pre-Trade: {generic['pretrade_score']}/{generic['pretrade_max']} "
            f"({generic['pretrade_grade']})</span>",
            unsafe_allow_html=True,
        )

        criteria_rows = []
        for criterion, details in generic.get('criteria_details', {}).items():
            is_pretrade = criterion != 'reversal_pct'
            tag = "PRE" if is_pretrade else "OUT"
            actual = details['actual']
            threshold = details['threshold']

            if criterion in ('pct_from_9ema', 'gap_pct', 'reversal_pct', 'pct_change_3'):
                actual_str = f"{actual * 100:.1f}%" if actual is not None else "N/A"
                threshold_str = f"{threshold * 100:.1f}%"
            else:
                actual_str = f"{actual:.1f}x" if actual is not None else "N/A"
                threshold_str = f"{threshold:.1f}x"

            criteria_rows.append({
                'Status': "PASS" if details['passed'] else "FAIL",
                'Tag': tag,
                'Criterion': details['name'],
                'Required': f"≥ {threshold_str}",
                'Actual': actual_str,
            })
        st.markdown(
            criteria_table_html(criteria_rows, ['Status', 'Tag', 'Criterion', 'Required', 'Actual']),
            unsafe_allow_html=True,
        )

    # Exit targets (SHORT direction — below entry)
    entry_price = metrics.get('entry_price')
    prior_close = metrics.get('prior_close')

    if atr and entry_price:
        st.markdown(
            '<div style="border-top:1px solid #141924; margin:8px 0;"></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<span style="font-family:JetBrains Mono,monospace; font-size:0.65rem; '
            'text-transform:uppercase; letter-spacing:0.1em; color:#3d4a5c;">Exit Targets (Short)</span>',
            unsafe_allow_html=True,
        )
        try:
            from analyzers.exit_targets import (
                calculate_exit_targets, format_exit_targets_html,
            )
            targets = calculate_exit_targets(
                cap=cap,
                entry_price=entry_price,
                atr=atr,
                prior_close=prior_close,
            )
            html = format_exit_targets_html(targets)
            st.markdown(html, unsafe_allow_html=True)

            # Store target levels for chart overlay (red for short)
            chart_targets = []
            for tier in targets.get('tiers', []):
                tp = tier.get('target_price')
                if tp is not None:
                    label = f"R-T{tier['tier']}: {tier['name']} (${tp:.2f})"
                    chart_targets.append((tp, '#f87171', label))
            st.session_state.chart_targets = chart_targets
        except Exception as e:
            st.warning(f"Exit target calculation failed: {e}")


def get_reversal_chart_targets() -> List[Tuple[float, str, str]]:
    """Get reversal target levels for chart overlay."""
    gp = st.session_state.get('reversal_game_plan')
    if gp is None:
        return []
    return st.session_state.get('chart_targets', [])
