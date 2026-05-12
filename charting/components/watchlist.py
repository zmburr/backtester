"""
Persistent watchlist backed by a JSON file.
Compact chip layout with bounce_data dedup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WATCHLIST_PATH = REPO_ROOT / "data" / "charting_watchlist.json"
BOUNCE_CSV = REPO_ROOT / "data" / "bounce_data.csv"


# Default watchlist — seeded from scanners/stock_screener.py
_DEFAULT_WATCHLIST = [
    'BIDU', 'AMD', 'AAPL', 'GOOGL', 'NVDA', 'AVGO', 'PLTR', 'ORCL', 'LITE',
    'MSFT', 'MU', 'IONQ', 'WDC', 'STX', 'BITF', 'IREN', 'HYMC', 'HL', 'PAAS',
    'SLV', 'GLD', 'MP', 'GDXJ', 'BE', 'OKLO', 'SMR', 'QS', 'RKLB', 'GWRE',
    'APP', 'OPEN', 'CRML', 'FIGR', 'SNDK', 'PL', 'BETR', 'RGTI', 'CRWV',
    'NBIS', 'CRDO', 'USAR', 'TSLA', 'HUBS', 'DOCU', 'DUOL', 'FIG', 'IBIT',
    'ETHE', 'TEAM', 'MSTR',
]


def _load_watchlist() -> List[str]:
    """Load watchlist from JSON file, seeding defaults on first run."""
    if WATCHLIST_PATH.exists():
        try:
            with open(WATCHLIST_PATH, 'r') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data
        except (json.JSONDecodeError, IOError):
            pass
    # First run or corrupted — seed from stock_screener watchlist
    _save_watchlist(_DEFAULT_WATCHLIST)
    return list(_DEFAULT_WATCHLIST)


def _save_watchlist(tickers: List[str]):
    """Save watchlist to JSON file."""
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WATCHLIST_PATH, 'w') as f:
        json.dump(tickers, f, indent=2)


def _get_bounce_tickers() -> Set[str]:
    """Load the set of tickers already in bounce_data.csv."""
    if not BOUNCE_CSV.exists():
        return set()
    try:
        df = pd.read_csv(BOUNCE_CSV, usecols=['ticker'])
        return set(df['ticker'].str.upper().str.strip().dropna().unique())
    except Exception:
        return set()


def render_watchlist() -> Optional[str]:
    """
    Render the watchlist in the sidebar as a compact chip grid.

    Returns:
        Ticker that was clicked (to load its chart), or None
    """
    # Initialize from file if not in session state
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = _load_watchlist()

    watchlist = st.session_state.watchlist
    st.markdown("#### Watchlist")

    # Add ticker — inline row
    col1, col2 = st.columns([3, 1])
    with col1:
        new_ticker = st.text_input(
            "Add ticker",
            value="",
            key="watchlist_add_input",
            label_visibility="collapsed",
            placeholder="Add ticker...",
        )
    with col2:
        add_clicked = st.button("+", key="watchlist_add_btn")

    if add_clicked and new_ticker.strip():
        t = new_ticker.strip().upper()
        if t not in watchlist:
            watchlist.append(t)
            _save_watchlist(watchlist)
            st.rerun()

    # Filter out tickers already in bounce_data.csv
    already_traded = _get_bounce_tickers()
    tickers = [t for t in watchlist if t.upper() not in already_traded]
    hidden = len(watchlist) - len(tickers)

    if hidden:
        st.caption(f"{hidden} hidden (in bounce_data)")

    if not tickers:
        st.caption("Watchlist empty after filtering.")
        return None

    # Render compact ticker grid (3 per row)
    clicked_ticker = None
    cols_per_row = 3

    for row_start in range(0, len(tickers), cols_per_row):
        row_tickers = tickers[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for j, t in enumerate(row_tickers):
            with cols[j]:
                if st.button(t, key=f"wl_{row_start + j}", use_container_width=True):
                    clicked_ticker = t

    # Remove ticker (compact, hidden in expander)
    with st.expander("Manage", expanded=False):
        remove_ticker = st.selectbox(
            "Remove",
            options=[""] + list(watchlist),
            key="wl_remove_select",
            label_visibility="collapsed",
        )
        if remove_ticker and st.button("Remove", key="wl_remove_btn"):
            watchlist.remove(remove_ticker)
            _save_watchlist(watchlist)
            st.rerun()

    return clicked_ticker
