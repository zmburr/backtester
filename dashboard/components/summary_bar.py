"""
Summary bar — top-level stats for the report page.
"""

from __future__ import annotations

from typing import List

import streamlit as st


def render_summary_bar(reports: list, report_date: str):
    """
    Show metrics: date, bounce count, reversal count, total tickers, timestamp.

    Args:
        reports: List of TickerReportData objects
        report_date: Date string (YYYY-MM-DD)
    """
    bounce_count = sum(1 for r in reports if r.bucket == "bounce")
    reversal_count = sum(1 for r in reports if r.bucket == "reversal")
    total = len(reports)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Report Date", report_date)
    c2.metric("Bounce Setups", bounce_count)
    c3.metric("Reversal Setups", reversal_count)
    c4.metric("Total Tickers", total)
