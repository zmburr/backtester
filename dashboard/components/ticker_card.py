"""
Per-ticker report card — renders one TickerReportData object.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from charting.theme import (
    criteria_table_html,
    intensity_meter_html,
    rec_badge_html,
)


def render_ticker_card(report):
    """
    Render a complete card for one TickerReportData object.

    Args:
        report: TickerReportData instance from report_engine
    """
    # Header: ticker + bucket badge + setup type
    bucket_color = "#6ee7b7" if report.bucket == "bounce" else "#4fc3f7"
    bucket_label = report.bucket.upper()
    cap_str = f" | {report.cap}" if report.cap else ""
    setup_str = ""
    if report.bucket == "bounce" and report.bounce_setup_type:
        setup_str = f" | {report.bounce_setup_type.replace('GapFade_', '')}"

    st.markdown(
        f'<div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">'
        f'<span style="font-family:JetBrains Mono,monospace; font-size:1.3rem; font-weight:700; color:#c0c8d8;">{report.ticker}</span>'
        f'<span style="background:{bucket_color}22; color:{bucket_color}; padding:2px 8px; border-radius:4px; '
        f'font-family:JetBrains Mono,monospace; font-size:0.7rem; font-weight:600; '
        f'border:1px solid {bucket_color}44;">{bucket_label}</span>'
        f'<span style="color:#6b7a90; font-size:0.78rem;">{cap_str}{setup_str}</span>'
        f'</div>'
        f'<div style="color:#6b7a90; font-size:0.72rem; margin-bottom:8px;">{report.bucket_reason}</div>',
        unsafe_allow_html=True,
    )

    if report.bucket == "bounce":
        _render_bounce_scoring(report)
    else:
        _render_reversal_scoring(report)

    # Exit targets
    if report.exit_targets and report.exit_targets.targets:
        _render_exit_targets(report)

    # Percentiles and MA distances in expanders
    col_pct, col_ma = st.columns(2)
    with col_pct:
        if report.percentiles:
            with st.expander(f"Percentiles ({report.percentile_ref_label or 'ref'})", expanded=False):
                _render_percentiles(report.percentiles)
    with col_ma:
        if report.mav_data:
            with st.expander("MA Distances", expanded=False):
                _render_ma_distances(report.mav_data)

    # Chart
    _render_chart(report)

    st.divider()


def _render_bounce_scoring(report):
    """Render bounce-specific scoring: checklist + intensity meter."""
    if report.bounce_result is not None:
        result = report.bounce_result
        rec = getattr(result, 'recommendation', 'NO-GO')

        # Recommendation badge + score
        score = getattr(result, 'score', 0)
        max_score = getattr(result, 'max_score', 6)
        st.markdown(
            f'{rec_badge_html(rec)} &nbsp; Score: **{score}/{max_score}**',
            unsafe_allow_html=True,
        )

        # Checklist items
        items = getattr(result, 'items', [])
        if items:
            rows = []
            for item in items:
                name = getattr(item, 'name', '')
                passed = getattr(item, 'passed', False)
                status = "PASS" if passed else "FAIL"
                actual = getattr(item, 'actual', None)
                threshold = getattr(item, 'threshold', None)

                actual_str = _fmt_val(actual)
                thresh_str = _fmt_val(threshold)

                rows.append({
                    "Criteria": name,
                    "Status": status,
                    "Actual": actual_str,
                    "Threshold": thresh_str,
                })
            st.markdown(
                criteria_table_html(rows, ["Criteria", "Status", "Actual", "Threshold"]),
                unsafe_allow_html=True,
            )

    # Intensity meter
    if report.bounce_intensity and report.bounce_intensity.get('composite') is not None:
        composite = report.bounce_intensity['composite']
        st.markdown(
            f'<div style="color:#6b7a90; font-size:0.65rem; text-transform:uppercase; '
            f'letter-spacing:0.06em; margin-top:8px;">Bounce Intensity</div>',
            unsafe_allow_html=True,
        )
        st.markdown(intensity_meter_html(composite), unsafe_allow_html=True)

        # Historical context from BOUNCE_SCORE_STATISTICS
        from dashboard.data.report_engine import BOUNCE_SCORE_STATISTICS
        if report.bounce_result:
            rec = getattr(report.bounce_result, 'recommendation', '')
            stats = BOUNCE_SCORE_STATISTICS.get(rec)
            if stats:
                st.caption(
                    f"Historical {rec}: {stats['trades']} trades, "
                    f"{stats['win_rate']}% WR, +{stats['avg_pnl']}% avg P&L"
                )


def _render_reversal_scoring(report):
    """Render reversal-specific scoring: 5 criteria with pass/fail."""
    if report.score_result is None:
        st.caption("No pre-trade metrics available")
        return

    sr = report.score_result
    rec = sr.get('recommendation', 'NO-GO')
    score = sr.get('score', 0)
    max_score = sr.get('max_score', 5)

    st.markdown(
        f'{rec_badge_html(rec)} &nbsp; Score: **{score}/{max_score}**',
        unsafe_allow_html=True,
    )

    criteria = sr.get('criteria', [])
    if criteria:
        rows = []
        for c in criteria:
            name = c.get('name', '')
            passed = c.get('passed', False)
            status = "PASS" if passed else "FAIL"

            actual = c.get('display_actual') or _fmt_val(c.get('actual'))
            threshold = c.get('display_threshold') or _fmt_val(c.get('threshold'))

            rows.append({
                "Criteria": name,
                "Status": status,
                "Actual": actual,
                "Threshold": threshold,
            })
        st.markdown(
            criteria_table_html(rows, ["Criteria", "Status", "Actual", "Threshold"]),
            unsafe_allow_html=True,
        )

    # Historical context from SCORE_STATISTICS
    from dashboard.data.report_engine import SCORE_STATISTICS
    stats = SCORE_STATISTICS.get(score)
    if stats and stats.get('trades'):
        st.caption(
            f"Historical score={score}: {stats['trades']} trades, "
            f"{stats['win_rate']}% WR, +{stats['avg_pnl']}% avg P&L"
        )


def _render_exit_targets(report):
    """Render exit target table."""
    et = report.exit_targets
    targets = et.targets or {}

    entry_price = et.entry_price
    entry_src = et.entry_source or "unknown"
    atr = et.atr

    st.markdown(
        f'<div style="color:#6b7a90; font-size:0.65rem; text-transform:uppercase; '
        f'letter-spacing:0.06em; margin-top:8px;">Exit Targets</div>',
        unsafe_allow_html=True,
    )

    if entry_price:
        st.caption(f"Entry: ${entry_price:.2f} ({entry_src}) | ATR: ${atr:.2f}" if atr else f"Entry: ${entry_price:.2f} ({entry_src})")

    # Build table from targets dict
    target_rows = []
    for key, val in targets.items():
        if key.startswith('entry_price') or key == 'atr':
            continue
        if isinstance(val, dict):
            price = val.get('price')
            hit_rate = val.get('hit_rate')
            label = val.get('label', key)
            if price is not None:
                hr_str = f"{hit_rate:.0f}%" if hit_rate is not None else "N/A"
                target_rows.append({
                    "Target": label,
                    "Price": f"${price:.2f}",
                    "Hit Rate": hr_str,
                })
        elif isinstance(val, (int, float)) and not pd.isna(val):
            target_rows.append({
                "Target": key,
                "Price": f"${val:.2f}",
                "Hit Rate": "N/A",
            })

    if target_rows:
        st.markdown(
            criteria_table_html(target_rows, ["Target", "Price", "Hit Rate"]),
            unsafe_allow_html=True,
        )


def _render_percentiles(percentiles: Dict[str, float]):
    """Render color-coded percentile table."""
    from dashboard.data.report_engine import PERCENTILE_ORDER

    rows = []
    for key in PERCENTILE_ORDER:
        if key in percentiles:
            val = percentiles[key]
            rows.append({"Metric": _clean_metric_name(key), "Percentile": val})

    # Add remaining keys not in PERCENTILE_ORDER
    for key, val in percentiles.items():
        if key not in PERCENTILE_ORDER:
            rows.append({"Metric": _clean_metric_name(key), "Percentile": val})

    if not rows:
        st.caption("No percentile data")
        return

    # Render as HTML with color coding
    html = '<table class="criteria-table"><thead><tr><th>Metric</th><th>Percentile</th></tr></thead><tbody>'
    for row in rows:
        pct = row["Percentile"]
        if pct is not None and not pd.isna(pct):
            if pct >= 75:
                color = "#6ee7b7"
                bg = "rgba(110,231,183,0.08)"
            elif pct <= 25:
                color = "#f87171"
                bg = "rgba(248,113,113,0.08)"
            else:
                color = "#c0c8d8"
                bg = "transparent"
            html += f'<tr style="background:{bg}"><td>{row["Metric"]}</td><td style="color:{color}; font-weight:600;">{pct:.0f}th</td></tr>'
        else:
            html += f'<tr><td>{row["Metric"]}</td><td style="color:#6b7a90;">N/A</td></tr>'
    html += '</tbody></table>'
    st.markdown(html, unsafe_allow_html=True)


def _render_ma_distances(mav_data: Dict[str, float]):
    """Render MA distances table."""
    ma_keys = [
        ("pct_from_10mav", "10 SMA"),
        ("pct_from_20mav", "20 SMA"),
        ("pct_from_50mav", "50 SMA"),
        ("pct_from_200mav", "200 SMA"),
    ]

    rows = []
    for key, label in ma_keys:
        val = mav_data.get(key)
        if val is not None:
            try:
                val_f = float(val)
                if not pd.isna(val_f):
                    color = "#6ee7b7" if val_f > 0 else "#f87171"
                    rows.append({"MA": label, "Distance": f"{val_f:+.1%}", "color": color})
            except (ValueError, TypeError):
                pass

    if not rows:
        st.caption("No MA data")
        return

    html = '<table class="criteria-table"><thead><tr><th>MA</th><th>Distance</th></tr></thead><tbody>'
    for row in rows:
        html += f'<tr><td>{row["MA"]}</td><td style="color:{row["color"]}; font-weight:600;">{row["Distance"]}</td></tr>'
    html += '</tbody></table>'
    st.markdown(html, unsafe_allow_html=True)


def _render_chart(report):
    """Render interactive candlestick chart for this ticker."""
    try:
        from charting.components.candlestick import fetch_daily_data, build_candlestick_figure

        end_date_str = date.today().strftime('%Y-%m-%d')
        df = fetch_daily_data(report.ticker, end_date_str, window=120)

        if df is not None and not df.empty:
            target_levels = report.chart_hlines if report.chart_hlines else []
            fig = build_candlestick_figure(
                df=df,
                ticker=report.ticker,
                target_levels=target_levels,
                show_volume=True,
                show_mas=True,
            )
            fig.update_layout(height=400, margin=dict(l=40, r=10, t=10, b=30))
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{report.ticker}")
    except Exception as e:
        st.caption(f"Chart unavailable: {e}")


def _fmt_val(val) -> str:
    """Format a numeric value for display."""
    if val is None:
        return "N/A"
    try:
        v = float(val)
        if pd.isna(v):
            return "N/A"
        if abs(v) < 1:
            return f"{v:.2%}"
        return f"{v:.2f}"
    except (ValueError, TypeError):
        return str(val)


def _clean_metric_name(key: str) -> str:
    """Convert column name to readable label."""
    return (
        key.replace("pct_change_", "")
        .replace("pct_from_", "from ")
        .replace("mav", "MA")
        .replace("_", " ")
        .title()
    )
