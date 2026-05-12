"""
Trading Chat — Streamlit page.

Chat with Claude about your trades, get live setup scoring,
query historical data, and search trading journals.
"""

import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Theme injection
# ---------------------------------------------------------------------------
try:
    from charting.theme import inject_css
    inject_css()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Trading Chat")
    st.caption("AI-powered trading assistant with live data tools")
    st.divider()

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
from dashboard.components.chat_interface import render_chat

st.title("Trading Chat")
st.caption(
    "Ask about setups, score tickers, query historical trades, "
    "search journals, or get exit targets."
)

# Load report data if available in session state
report_data = st.session_state.get("report_data", None)

render_chat(report_data=report_data)
