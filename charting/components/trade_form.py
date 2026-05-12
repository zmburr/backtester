"""
Quick trade entry form — appends a row to bounce_data.csv or reversal_data.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"


def render_trade_form(default_ticker: str = "", default_cap: str = "Medium"):
    """
    Render a quick trade entry form that appends to the appropriate CSV.

    Args:
        default_ticker: Pre-fill ticker from sidebar
        default_cap: Pre-fill cap from sidebar
    """
    st.markdown(
        '<span style="font-family:JetBrains Mono,monospace; font-size:0.65rem; '
        'text-transform:uppercase; letter-spacing:0.1em; color:#3d4a5c;">Quick Trade Entry</span>',
        unsafe_allow_html=True,
    )
    st.caption("Adds a row to the CSV. Missing columns filled later by data collector.")

    with st.form("trade_entry_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            trade_type = st.selectbox("Type", ["Bounce", "Reversal"], key="trade_type")
            ticker = st.text_input("Ticker", value=default_ticker.upper(), key="trade_ticker")

        with col2:
            grade = st.selectbox("Grade", ["A", "B", "C"], key="trade_grade")
            trade_date = st.date_input("Date", value=datetime.now(), key="trade_date")

        with col3:
            cap = st.selectbox("Cap", ["ETF", "Large", "Medium", "Small", "Micro"],
                               index=["ETF", "Large", "Medium", "Small", "Micro"].index(default_cap),
                               key="trade_cap")
            setup_type = st.text_input("Setup Type", value="", key="trade_setup",
                                       placeholder="e.g., GapFade_weakstock")

        notes = st.text_area("Notes", value="", key="trade_notes", height=60,
                             placeholder="Optional notes...")

        submitted = st.form_submit_button("Add Trade")

    if submitted:
        if not ticker.strip():
            st.error("Ticker is required.")
            return

        ticker = ticker.strip().upper()
        date_str = trade_date.strftime("%m/%d/%Y")

        if trade_type == "Bounce":
            csv_path = DATA_DIR / "bounce_data.csv"
        else:
            csv_path = DATA_DIR / "reversal_data.csv"

        try:
            # Read existing CSV to get column order
            existing = pd.read_csv(csv_path)
            columns = existing.columns.tolist()

            # Build new row with NaN for unfilled columns
            new_row = {col: pd.NA for col in columns}
            new_row['date'] = date_str
            new_row['ticker'] = ticker
            new_row['trade_grade'] = grade
            new_row['cap'] = cap

            if trade_type == "Bounce":
                if setup_type.strip():
                    new_row['Setup'] = setup_type.strip()
            else:
                if setup_type.strip():
                    new_row['setup'] = setup_type.strip()

            # Append to CSV
            new_df = pd.DataFrame([new_row])
            new_df.to_csv(csv_path, mode='a', header=False, index=False)

            st.success(f"Added {ticker} ({date_str}) to {csv_path.name}")

        except FileNotFoundError:
            st.error(f"CSV file not found: {csv_path}")
        except Exception as e:
            st.error(f"Failed to add trade: {e}")
