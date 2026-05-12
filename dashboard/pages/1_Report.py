"""
Daily Watchlist Report — interactive replacement for the email report.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from charting.theme import inject_css
from dashboard.components.summary_bar import render_summary_bar
from dashboard.components.ticker_card import render_ticker_card
from dashboard.components.report_header import render_trading_rules
from dashboard.data.report_engine import (
    compute_report,
    TickerReportData,
    SCORE_STATISTICS,
    BOUNCE_SCORE_STATISTICS,
)

inject_css()

# ---------------------------------------------------------------------------
# Cached report computation
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Computing report...")
def _compute_report_cached(watchlist_tuple: tuple, report_date: str):
    """Cached wrapper around compute_report(). Uses tuple for hashability."""
    return compute_report(list(watchlist_tuple), date=report_date)


# ---------------------------------------------------------------------------
# Watchlist management (sidebar)
# ---------------------------------------------------------------------------

def _load_watchlist():
    """Load watchlist from the charting watchlist JSON."""
    from charting.components.watchlist import _load_watchlist as _load
    return _load()


def _save_watchlist(tickers):
    """Save watchlist to the charting watchlist JSON."""
    from charting.components.watchlist import _save_watchlist as _save
    _save(tickers)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("#### Report")

    report_date = date.today().strftime('%Y-%m-%d')
    st.caption(f"Date: {report_date}")

    if st.button("Refresh Report", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    # Watchlist editor
    st.markdown("#### Watchlist")

    if "report_watchlist" not in st.session_state:
        st.session_state.report_watchlist = _load_watchlist()

    watchlist = st.session_state.report_watchlist

    # Add ticker
    col1, col2 = st.columns([3, 1])
    with col1:
        new_ticker = st.text_input(
            "Add ticker",
            value="",
            key="report_add_input",
            label_visibility="collapsed",
            placeholder="Add ticker...",
        )
    with col2:
        add_clicked = st.button("+", key="report_add_btn")

    if add_clicked and new_ticker.strip():
        t = new_ticker.strip().upper()
        if t not in watchlist:
            watchlist.append(t)
            _save_watchlist(watchlist)
            st.cache_data.clear()
            st.rerun()

    # Show current watchlist count
    st.caption(f"{len(watchlist)} tickers")

    # Manage watchlist
    with st.expander("Manage Watchlist", expanded=False):
        remove_ticker = st.selectbox(
            "Remove",
            options=[""] + list(watchlist),
            key="report_remove_select",
            label_visibility="collapsed",
        )
        if remove_ticker and st.button("Remove", key="report_remove_btn"):
            watchlist.remove(remove_ticker)
            _save_watchlist(watchlist)
            st.cache_data.clear()
            st.rerun()

    st.divider()

    # Filters
    st.markdown("#### Filters")
    filter_bucket = st.selectbox(
        "Setup Type",
        ["All", "Bounce", "Reversal"],
        index=0,
        key="report_filter_bucket",
    )
    min_score = st.slider(
        "Min Score",
        min_value=0,
        max_value=6,
        value=0,
        key="report_min_score",
    )

    st.divider()

    # Trading rules
    render_trading_rules()


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.markdown(
    '<h2 style="font-family:Outfit,sans-serif; font-weight:700; margin-bottom:0;">Daily Report</h2>',
    unsafe_allow_html=True,
)

if not watchlist:
    st.info("Add tickers to your watchlist in the sidebar to generate the report.")
    st.stop()

# Compute report
watchlist_tuple = tuple(watchlist)

try:
    reports = _compute_report_cached(watchlist_tuple, report_date)
except Exception as e:
    st.error(f"Error computing report: {e}")
    reports = None

if not reports:
    st.warning("No report data generated. Check your watchlist and API keys.")
    st.stop()

# Store for Chat page
st.session_state.report_data = reports

# Apply filters
filtered = reports
if filter_bucket != "All":
    filtered = [r for r in filtered if r.bucket == filter_bucket.lower()]

if min_score > 0:
    def _get_score(r):
        if r.bucket == "bounce" and r.bounce_result:
            return getattr(r.bounce_result, 'score', 0)
        elif r.bucket == "reversal" and r.score_result:
            return r.score_result.get('score', 0)
        return 0
    filtered = [r for r in filtered if _get_score(r) >= min_score]

# Summary bar
render_summary_bar(filtered, report_date)

st.markdown(
    f'<div style="color:#6b7a90; font-size:0.72rem; margin:4px 0 16px 0;">'
    f'Last computed: {datetime.now().strftime("%I:%M %p")} | '
    f'Showing {len(filtered)} of {len(reports)} tickers</div>',
    unsafe_allow_html=True,
)

st.divider()

# Render ticker cards
for report in filtered:
    render_ticker_card(report)
