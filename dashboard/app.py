"""
Trading Dashboard — Multi-page Streamlit app.

Entry point. Streamlit's multi-page system automatically picks up files
in the pages/ directory. This file serves as the home/default page and
redirects users to the Report view.

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Trading Dashboard", layout="wide", page_icon="\U0001f4ca")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from charting.theme import inject_css

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
inject_css()

# ---------------------------------------------------------------------------
# Sidebar (shared across all pages)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("TRADING DASHBOARD")
    st.caption(f"{date.today().strftime('%A, %B %d, %Y')}")

    if st.button("Refresh All", key="home_refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("Navigate using the sidebar pages above.")

# ---------------------------------------------------------------------------
# Home page — show the Report content directly
# ---------------------------------------------------------------------------

# Import and render the Report page content inline so users land on the report
from dashboard.components.summary_bar import render_summary_bar
from dashboard.components.ticker_card import render_ticker_card
from dashboard.components.report_header import render_trading_rules
from dashboard.data.report_engine import compute_report

from charting.components.watchlist import _load_watchlist, _save_watchlist


st.markdown(
    '<h2 style="font-family:Outfit,sans-serif; font-weight:700; margin-bottom:0;">Daily Report</h2>',
    unsafe_allow_html=True,
)

report_date = date.today().strftime('%Y-%m-%d')


@st.cache_data(ttl=3600, show_spinner="Computing report...")
def _compute_home_report(watchlist_tuple: tuple, rdate: str):
    return compute_report(list(watchlist_tuple), date=rdate)


# Load watchlist
if "home_watchlist" not in st.session_state:
    st.session_state.home_watchlist = _load_watchlist()

watchlist = st.session_state.home_watchlist

if not watchlist:
    st.info("Add tickers to your watchlist to generate the report. Go to the Report page for full controls.")
    st.stop()

# Sidebar: quick watchlist + filters for home page
with st.sidebar:
    st.markdown("#### Quick Filters")
    home_filter = st.selectbox("Setup Type", ["All", "Bounce", "Reversal"], key="home_filter")

    st.divider()
    with st.expander("Trading Rules", expanded=False):
        render_trading_rules()

# Compute
watchlist_tuple = tuple(watchlist)

try:
    reports = _compute_home_report(watchlist_tuple, report_date)
except Exception as e:
    st.error(f"Error computing report: {e}")
    reports = None

if not reports:
    st.warning("No report data generated. Check your watchlist and API keys.")
    st.stop()

# Store for Chat page
st.session_state.report_data = reports

# Filter
filtered = reports
if home_filter != "All":
    filtered = [r for r in filtered if r.bucket == home_filter.lower()]

# Render
render_summary_bar(filtered, report_date)

st.markdown(
    f'<div style="color:#6b7a90; font-size:0.72rem; margin:4px 0 16px 0;">'
    f'Last computed: {datetime.now().strftime("%I:%M %p")} | '
    f'Showing {len(filtered)} of {len(reports)} tickers</div>',
    unsafe_allow_html=True,
)

st.divider()

for report in filtered:
    render_ticker_card(report)
