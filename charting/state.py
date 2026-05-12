"""
Centralized session state schema for the charting app.
"""

import streamlit as st


# Default session state keys and their initial values
_DEFAULTS = {
    "ticker": "NVDA",
    "cap": "Medium",
    "end_date": None,           # datetime.date — defaults to today
    "history_days": 250,
    "show_mas": True,
    "show_volume": True,
    "show_historical_trades": False,

    # News cache: {(ticker, date_str): {"summary": str, "full": str|None}}
    "news_cache": {},

    # Game plan results (last generated)
    "bounce_game_plan": None,
    "reversal_game_plan": None,

    # Target levels for chart overlay: list of (price, color, label)
    "chart_targets": [],

    # Watchlist: loaded from JSON file
    "watchlist": [],
}


def init_state():
    """Initialize session state with defaults (only sets keys that don't exist)."""
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default
