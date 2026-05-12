"""
News lookup panel using OpenAI gpt-4.1-mini with web search.
Caches results in session state and provides annotations for the chart.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from charting.theme import news_summary_html


def _query_openai_websearch(query: str, system_prompt: str) -> str:
    """Call OpenAI gpt-4.1-mini with web_search_preview tool."""
    from openai import OpenAI
    from support.config import OPENAI_API_KEY

    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": query}]},
        ],
        tools=[{
            "type": "web_search_preview",
            "user_location": {
                "type": "approximate",
                "country": "US", "region": "NY", "city": "New York",
            },
            "search_context_size": "medium",
        }],
        temperature=0.5,
        max_output_tokens=1024,
    )
    return resp.output_text.strip()


# ---------------------------------------------------------------------------
# News query functions (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Querying news...")
def query_news_summary(ticker: str, date: str) -> str:
    """Query for a one-sentence financial news summary."""
    system_prompt = (
        "You are a financial news analyst. Given a stock ticker and date, "
        "provide a single concise sentence summarizing the most significant "
        "financial news or market-moving event for that stock on that date. "
        "If there was no significant news, say 'No significant news for [TICKER] on [DATE].' "
        "Do not include disclaimers or caveats."
    )
    query = f"{ticker} stock news {date}"
    try:
        return _query_openai_websearch(query, system_prompt)
    except Exception as e:
        return f"News query failed: {e}"


@st.cache_data(ttl=3600, show_spinner="Fetching full report...")
def query_news_full_report(ticker: str, date: str) -> str:
    """Query for a detailed 3-5 paragraph news report."""
    system_prompt = (
        "You are a financial news analyst providing pre-market research. "
        "Given a stock ticker and date, provide a detailed 3-5 paragraph report covering: "
        "1) The main news event or catalyst for the stock on that date "
        "2) Market reaction and price action context "
        "3) Any relevant sector/macro factors "
        "4) Key levels or analyst targets if applicable. "
        "Be specific with numbers and facts. No disclaimers."
    )
    query = f"{ticker} stock news analysis {date}"
    try:
        return _query_openai_websearch(query, system_prompt)
    except Exception as e:
        return f"Full report query failed: {e}"


# ---------------------------------------------------------------------------
# UI renderer
# ---------------------------------------------------------------------------

def render_news_panel(ticker: str, chart_dates: list):
    """
    Render the news panel with date selector, summary, and full report.

    Args:
        ticker: Current ticker
        chart_dates: List of date strings from chart DataFrame (most recent first)
    """
    if not chart_dates:
        st.info("Load a chart to query news.")
        return

    # Use chart-clicked date if available, otherwise default to most recent
    default_idx = 0
    clicked_date = st.session_state.get("chart_clicked_date")
    if clicked_date and clicked_date in chart_dates:
        default_idx = chart_dates.index(clicked_date)

    selected_date = st.selectbox(
        "Select date for news",
        options=chart_dates,
        index=default_idx,
        key="news_date_selector",
    )

    if selected_date:
        col1, col2 = st.columns([4, 1])

        with col1:
            summary = query_news_summary(ticker, selected_date)
            st.markdown(news_summary_html(selected_date, summary), unsafe_allow_html=True)

            # Cache to session state for chart annotations
            cache_key = (ticker, selected_date)
            if "news_cache" not in st.session_state:
                st.session_state.news_cache = {}

            if "no significant news" not in summary.lower() and "no news found" not in summary.lower():
                st.session_state.news_cache[cache_key] = summary

        with col2:
            if st.button("Full Report", key="news_full_report_btn"):
                st.session_state["show_full_report"] = True

        if st.session_state.get("show_full_report"):
            with st.expander("Detailed News Report", expanded=True):
                report = query_news_full_report(ticker, selected_date)
                st.markdown(report)

    # News history
    if st.session_state.get("news_cache"):
        ticker_news = {
            k: v for k, v in st.session_state.news_cache.items()
            if k[0] == ticker
        }
        if ticker_news:
            with st.expander(f"News history ({len(ticker_news)} dates)", expanded=False):
                for (t, d), s in sorted(ticker_news.items(), key=lambda x: x[0][1], reverse=True):
                    st.markdown(news_summary_html(d, s), unsafe_allow_html=True)


def get_news_annotations(ticker: str) -> Dict[str, str]:
    """Get cached news annotations for chart overlay."""
    if "news_cache" not in st.session_state:
        return {}
    return {
        date: summary
        for (t, date), summary in st.session_state.news_cache.items()
        if t == ticker
    }
