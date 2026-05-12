"""
Trade Prep & Analysis Tool — Charting App

Standalone Streamlit app for pre-market prep combining interactive daily
candlestick charts, Perplexity news, game plan generation, and trade tracking.

Run:
    streamlit run charting/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Trade Prep", layout="wide", page_icon="📊")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Local imports (after sys.path)
from charting.theme import inject_css, news_summary_html
from charting.state import init_state
from charting.components.candlestick import fetch_daily_data, build_candlestick_figure
from charting.components.news_panel import (
    render_news_panel, get_news_annotations, query_news_summary,
)
from charting.components.annotations import (
    load_annotations, add_annotation, render_annotation_manager,
)
from charting.components.game_plan import (
    render_bounce_game_plan, render_reversal_game_plan,
    get_bounce_chart_targets, get_reversal_chart_targets,
)
from charting.components.trade_form import render_trade_form
from charting.components.watchlist import render_watchlist

# ---------------------------------------------------------------------------
# Theme + state initialization
# ---------------------------------------------------------------------------
inject_css()
init_state()

DATA_DIR = REPO_ROOT / "data"


# ---------------------------------------------------------------------------
# Compress news to short annotation label
# ---------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def _compress_for_annotation(text: str) -> str:
    """Compress a news summary to 5-10 words for a chart label."""
    try:
        from openai import OpenAI
        from support.config import OPENAI_API_KEY
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Compress the following news into 5-10 words. Return ONLY the short label, nothing else."},
                {"role": "user", "content": text},
            ],
            max_tokens=25,
            temperature=0,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # Fallback: take first 8 words
        words = text.split()
        return " ".join(words[:8]) + ("..." if len(words) > 8 else "")


# ---------------------------------------------------------------------------
# Historical trade loading (for overlay)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def load_historical_trades(ticker: str):
    """Load historical trades for a ticker from both CSVs."""
    trades = []

    # Bounce trades
    bounce_path = DATA_DIR / "bounce_data.csv"
    if bounce_path.exists():
        try:
            df = pd.read_csv(bounce_path)
            df = df[df['ticker'].str.upper().str.strip() == ticker.upper()]
            if not df.empty:
                df['pnl'] = pd.to_numeric(df.get('bounce_open_close_pct'), errors='coerce') * 100
                df['date_dt'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
                df['date_str'] = df['date_dt'].dt.strftime('%Y-%m-%d')
                df['trade_type'] = 'Bounce'
                trades.append(df[['date_str', 'ticker', 'pnl', 'trade_type']].dropna(subset=['date_str']))
        except Exception:
            pass

    # Reversal trades
    reversal_path = DATA_DIR / "reversal_data.csv"
    if reversal_path.exists():
        try:
            df = pd.read_csv(reversal_path)
            df = df[df['ticker'].str.upper().str.strip() == ticker.upper()]
            if not df.empty:
                df['pnl'] = -pd.to_numeric(df.get('reversal_open_close_pct'), errors='coerce') * 100
                df['date_dt'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
                df['date_str'] = df['date_dt'].dt.strftime('%Y-%m-%d')
                df['trade_type'] = 'Reversal'
                trades.append(df[['date_str', 'ticker', 'pnl', 'trade_type']].dropna(subset=['date_str']))
        except Exception:
            pass

    if trades:
        return pd.concat(trades, ignore_index=True)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("TRADE PREP")

    # Watchlist
    st.divider()
    clicked_ticker = render_watchlist()
    if clicked_ticker:
        st.session_state.ticker = clicked_ticker

    # Settings
    st.divider()
    st.markdown("#### Settings")

    ticker = st.text_input(
        "Ticker",
        value=st.session_state.ticker,
        key="ticker_input",
    ).strip().upper()
    if ticker:
        st.session_state.ticker = ticker

    cap = st.selectbox(
        "Cap",
        ["ETF", "Large", "Medium", "Small", "Micro"],
        index=["ETF", "Large", "Medium", "Small", "Micro"].index(st.session_state.cap),
        key="cap_input",
    )
    st.session_state.cap = cap

    end_date = st.date_input(
        "End Date",
        value=st.session_state.end_date or date.today(),
        key="end_date_input",
    )
    st.session_state.end_date = end_date

    history = st.slider(
        "History (days)",
        min_value=30,
        max_value=500,
        value=st.session_state.history_days,
        step=10,
        key="history_input",
    )
    st.session_state.history_days = history

    st.divider()
    st.markdown("#### Overlays")
    show_mas = st.checkbox("Moving Averages", value=st.session_state.show_mas, key="show_mas_input")
    st.session_state.show_mas = show_mas

    show_vol = st.checkbox("Volume", value=st.session_state.show_volume, key="show_vol_input")
    st.session_state.show_volume = show_vol

    show_trades = st.checkbox("Historical Trades", value=st.session_state.show_historical_trades, key="show_trades_input")
    st.session_state.show_historical_trades = show_trades


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
ticker = st.session_state.ticker
cap = st.session_state.cap
end_date_str = st.session_state.end_date.strftime('%Y-%m-%d') if st.session_state.end_date else date.today().strftime('%Y-%m-%d')

if not ticker:
    st.info("Enter a ticker in the sidebar to get started.")
    st.stop()

# Fetch data
df = fetch_daily_data(ticker, end_date_str, window=st.session_state.history_days)

if df is None or df.empty:
    st.warning(f"No data available for {ticker}. Check the ticker symbol and date range.")
    st.stop()

# Chart header — ticker, last price, change
last_close = df['close'].iloc[-1]
prev_close = df['close'].iloc[-2] if len(df) > 1 else last_close
change_pct = ((last_close - prev_close) / prev_close * 100) if prev_close else 0
change_cls = "change-positive" if change_pct >= 0 else "change-negative"
change_sign = "+" if change_pct >= 0 else ""

st.markdown(
    f'<div class="chart-header">'
    f'<span class="ticker-label">{ticker}</span>'
    f'<span class="price-label">${last_close:.2f}</span> '
    f'<span class="{change_cls}">{change_sign}{change_pct:.2f}%</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Date selector + auto news query (above chart, always visible)
# ---------------------------------------------------------------------------
chart_dates = df['date_str'].tolist()[::-1]  # Most recent first

# Pick up chart-clicked date if available
default_idx = 0
clicked_date = st.session_state.get("chart_clicked_date")
if clicked_date and clicked_date in chart_dates:
    default_idx = chart_dates.index(clicked_date)

col_sel, col_query, col_pin = st.columns([3, 1, 1])
with col_sel:
    selected_date = st.selectbox(
        "Date",
        options=chart_dates,
        index=default_idx,
        key="quick_date_selector",
        label_visibility="collapsed",
    )
with col_query:
    query_clicked = st.button("Query News", key="quick_news_btn")
with col_pin:
    pin_clicked = st.button("Pin to Chart", key="quick_pin_btn")

# Auto-query news when date changes or button pressed
if selected_date:
    # Track which date we last queried to auto-fire on date change
    prev_queried = st.session_state.get("_last_queried_date")
    date_changed = (selected_date != prev_queried)

    if query_clicked or date_changed:
        summary = query_news_summary(ticker, selected_date)
        st.session_state["_last_news_summary"] = summary
        st.session_state["_last_queried_date"] = selected_date

        # Cache for chart annotations (news arrows)
        if "news_cache" not in st.session_state:
            st.session_state.news_cache = {}
        if "no significant news" not in summary.lower() and "no news found" not in summary.lower():
            st.session_state.news_cache[(ticker, selected_date)] = summary

    # Show the last summary
    last_summary = st.session_state.get("_last_news_summary", "")
    if last_summary:
        st.markdown(news_summary_html(selected_date, last_summary), unsafe_allow_html=True)

    # Pin as annotation — compress to 5-10 words first
    if pin_clicked and last_summary:
        with st.spinner("Compressing..."):
            short = _compress_for_annotation(last_summary)
        add_annotation(ticker, selected_date, short)
        st.rerun()

# ---------------------------------------------------------------------------
# Build chart with annotations
# ---------------------------------------------------------------------------
text_annotations = load_annotations(ticker)
target_levels = st.session_state.get('chart_targets', [])

historical_trades_df = None
if st.session_state.show_historical_trades:
    historical_trades_df = load_historical_trades(ticker)
    if historical_trades_df.empty:
        historical_trades_df = None

fig = build_candlestick_figure(
    df=df,
    ticker=ticker,
    text_annotations=text_annotations,
    target_levels=target_levels,
    historical_trades=historical_trades_df,
    show_volume=st.session_state.show_volume,
    show_mas=st.session_state.show_mas,
)

# Render chart with selection support
event = st.plotly_chart(
    fig, use_container_width=True,
    on_select="rerun", selection_mode=["points"],
    key="main_chart",
)

# Capture clicked date from chart point selection
if event and hasattr(event, 'selection') and event.selection:
    points = event.selection.get("points", [])
    if points:
        clicked_x = points[0].get("x")
        if clicked_x and clicked_x != st.session_state.get("chart_clicked_date"):
            st.session_state["chart_clicked_date"] = clicked_x
            st.rerun()


# ---------------------------------------------------------------------------
# Tabs below the chart
# ---------------------------------------------------------------------------
tab_annotations, tab_news, tab_bounce, tab_reversal, tab_trade = st.tabs(
    ["Annotations", "News", "Bounce GP", "Reversal GP", "Add Trade"]
)

with tab_annotations:
    render_annotation_manager(ticker, chart_dates)

with tab_news:
    render_news_panel(ticker, chart_dates)

with tab_bounce:
    render_bounce_game_plan(ticker, end_date_str, cap)

with tab_reversal:
    render_reversal_game_plan(ticker, end_date_str, cap)

with tab_trade:
    render_trade_form(default_ticker=ticker, default_cap=cap)
