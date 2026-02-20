"""
Bounce Trade Chart Review Tool (Streamlit)

Visual confirmation tool for bounce_data.csv trades.
Shows long-term daily chart + intraday chart side by side
so you can visually verify each trade setup.

Run:
    streamlit run scripts/bounce_chart_review.py
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data_queries.polygon_queries import get_levels_data, get_intraday

st.set_page_config(page_title="Bounce Chart Review", layout="wide")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, .stApp, .stApp * { font-family: 'Outfit', sans-serif !important; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080c12 0%, #0d1117 100%) !important;
    border-right: 1px solid #1c2333 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
DATA_PATH = ROOT / "data" / "bounce_data.csv"
REMOVALS_PATH = ROOT / "data" / "bounce_removals.json"
ORIGINAL_ROW_COUNT = 54  # rows before the recent expansion


@st.cache_data
def load_bounce_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["_row_idx"] = range(len(df))
    return df


def load_removals() -> list[dict]:
    """Load the removal list from JSON. Returns list of {date, ticker} dicts."""
    if REMOVALS_PATH.exists():
        return json.loads(REMOVALS_PATH.read_text())
    return []


def save_removals(removals: list[dict]):
    """Persist the removal list to JSON."""
    REMOVALS_PATH.write_text(json.dumps(removals, indent=2))


def is_flagged(removals: list[dict], date: str, ticker: str) -> bool:
    return any(r["date"] == date and r["ticker"] == ticker for r in removals)


@st.cache_data(ttl=3600)
def fetch_daily(ticker: str, date_str: str) -> pd.DataFrame | None:
    """Fetch ~200 trading days of daily data ending on *date_str* (YYYY-MM-DD)."""
    try:
        return get_levels_data(ticker, date_str, window=200, multiplier=1, timespan="day")
    except Exception as e:
        st.warning(f"Daily data error for {ticker}: {e}")
        return None


@st.cache_data(ttl=3600)
def fetch_intraday(ticker: str, date_str: str) -> pd.DataFrame | None:
    """Fetch 5-min intraday bars for *ticker* on *date_str* (YYYY-MM-DD)."""
    try:
        return get_intraday(ticker, date_str, multiplier=5, timespan="minute")
    except Exception as e:
        st.warning(f"Intraday data error for {ticker}: {e}")
        return None


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def build_daily_chart(df: pd.DataFrame, ticker: str, bounce_date: str) -> go.Figure:
    """Long-term daily candlestick with MAs and bounce date highlighted."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Moving averages
    ma_configs = [
        (200, "#e41a1c", "SMA200"), (50, "#4daf4a", "SMA50"),
        (20, "#377eb8", "SMA20"), (10, "#984ea3", "SMA10"),
    ]
    for window, color, label in ma_configs:
        if len(df) >= window:
            ma = df["close"].rolling(window).mean()
            fig.add_trace(go.Scatter(
                x=df.index, y=ma, mode="lines", name=label,
                line=dict(color=color, width=1.2),
            ), row=1, col=1)

    # 9 EMA
    ema9 = df["close"].ewm(span=9, adjust=False).mean()
    fig.add_trace(go.Scatter(
        x=df.index, y=ema9, mode="lines", name="EMA9",
        line=dict(color="orange", width=1, dash="dash"),
    ), row=1, col=1)

    # Volume bars
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], name="Volume",
        marker_color=colors, opacity=0.5,
    ), row=2, col=1)

    # Highlight bounce date
    try:
        bd = pd.Timestamp(bounce_date)
        fig.add_vline(x=bd, line_color="cyan", line_width=2, line_dash="dot",
                      annotation_text="BOUNCE", annotation_position="top right",
                      annotation_font_color="cyan")
    except Exception:
        pass

    fig.update_layout(
        title=f"{ticker} — Daily (1Y context)",
        template="plotly_dark", height=600,
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        margin=dict(l=50, r=20, t=60, b=30),
    )
    fig.update_xaxes(type="category", nticks=30, row=1, col=1)
    fig.update_xaxes(type="category", nticks=30, row=2, col=1)
    return fig


def build_intraday_chart(df: pd.DataFrame, ticker: str, date_str: str,
                         gap_pct: float | None = None,
                         bounce_open_high_pct: float | None = None) -> go.Figure:
    """Intraday 5-min candlestick chart for bounce day."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # VWAP
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    vwap = cum_tp_vol / cum_vol
    fig.add_trace(go.Scatter(
        x=df.index, y=vwap, mode="lines", name="VWAP",
        line=dict(color="#ff9800", width=1.5, dash="dot"),
    ), row=1, col=1)

    # Prior close line (open adjusted by gap)
    if gap_pct is not None and gap_pct != 0:
        open_price = df["open"].iloc[0]
        prior_close = open_price / (1 + gap_pct)
        fig.add_hline(y=prior_close, line_color="yellow", line_dash="dash",
                      annotation_text="Prior Close", annotation_font_color="yellow")

    # Volume bars
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], name="Volume",
        marker_color=colors, opacity=0.5,
    ), row=2, col=1)

    # Annotations
    annotations = []
    if gap_pct is not None:
        annotations.append(f"Gap: {gap_pct*100:+.1f}%")
    if bounce_open_high_pct is not None:
        annotations.append(f"Open-High: {bounce_open_high_pct*100:+.1f}%")
    subtitle = " | ".join(annotations) if annotations else ""

    fig.update_layout(
        title=f"{ticker} — Intraday 5min ({date_str})<br><sup>{subtitle}</sup>",
        template="plotly_dark", height=600,
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        margin=dict(l=50, r=20, t=80, b=30),
    )
    return fig


# ---------------------------------------------------------------------------
# Navigation callbacks
# ---------------------------------------------------------------------------

def _safe_idx() -> int:
    val = st.session_state.get("trade_idx", 0)
    return int(val) if isinstance(val, (int, float)) else 0


def go_prev():
    st.session_state.trade_idx = max(0, _safe_idx() - 1)


def go_next(max_idx: int):
    st.session_state.trade_idx = min(max_idx, _safe_idx() + 1)


def on_select():
    val = st.session_state.trade_select
    st.session_state.trade_idx = val if isinstance(val, int) else 0


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    df = load_bounce_data()
    removals = load_removals()

    # Sidebar controls
    st.sidebar.title("Bounce Review")

    show_new_only = st.sidebar.checkbox(
        "Show only recently added trades",
        value=True,
        help=f"Filter to rows {ORIGINAL_ROW_COUNT}+ (added in the recent expansion)",
    )

    # Reset index when filter changes to avoid stale state
    prev_filter = st.session_state.get("_prev_filter", True)
    if show_new_only != prev_filter:
        st.session_state.trade_idx = 0
        # Clear the selectbox key so it doesn't hold a stale string
        st.session_state.pop("trade_select", None)
    st.session_state._prev_filter = show_new_only

    if show_new_only:
        view_df = df[df["_row_idx"] >= ORIGINAL_ROW_COUNT].reset_index(drop=True)
    else:
        view_df = df.copy()

    if view_df.empty:
        st.error("No trades to review with current filter.")
        return

    # Build display labels
    labels = []
    for i, (_, r) in enumerate(view_df.iterrows()):
        flag = " [REMOVE]" if is_flagged(removals, r["date"], r["ticker"]) else ""
        labels.append(
            f"{i+1}/{len(view_df)}  {r['ticker']}  {r['date']}  [{r['trade_grade']}] {r['Setup']}{flag}"
        )

    # Init session state
    if "trade_idx" not in st.session_state:
        st.session_state.trade_idx = 0
    # Clamp to valid range
    st.session_state.trade_idx = min(st.session_state.trade_idx, len(view_df) - 1)

    # Navigation buttons
    col_prev, _, col_next = st.sidebar.columns([1, 3, 1])
    with col_prev:
        st.button("Prev", on_click=go_prev)
    with col_next:
        st.button("Next", on_click=go_next, args=(len(view_df) - 1,))

    # Trade selector (synced via callback)
    st.sidebar.selectbox(
        "Trade", range(len(labels)), format_func=lambda i: labels[i],
        index=st.session_state.trade_idx,
        key="trade_select", on_change=on_select,
    )

    selected_idx = st.session_state.trade_idx
    trade = view_df.iloc[selected_idx]

    # Parse date
    raw_date = trade["date"]
    try:
        dt = pd.to_datetime(raw_date, format="%m/%d/%Y")
    except Exception:
        dt = pd.to_datetime(raw_date)
    api_date = dt.strftime("%Y-%m-%d")

    # --- Flag for removal checkbox ---
    st.sidebar.markdown("---")
    currently_flagged = is_flagged(removals, raw_date, trade["ticker"])
    flag_it = st.sidebar.checkbox(
        "Flag for removal",
        value=currently_flagged,
        key=f"flag_{raw_date}_{trade['ticker']}",
        help="Check to add this trade to the removal list. "
             "Uncheck to keep it. Changes save immediately to bounce_removals.json.",
    )

    # Sync checkbox state to JSON
    if flag_it and not currently_flagged:
        removals.append({"date": raw_date, "ticker": trade["ticker"]})
        save_removals(removals)
        st.sidebar.success(f"Flagged {trade['ticker']} {raw_date} for removal")
    elif not flag_it and currently_flagged:
        removals = [r for r in removals if not (r["date"] == raw_date and r["ticker"] == trade["ticker"])]
        save_removals(removals)
        st.sidebar.info(f"Un-flagged {trade['ticker']} {raw_date}")

    # Show removal summary
    if removals:
        st.sidebar.markdown(f"**Flagged for removal: {len(removals)}**")
        with st.sidebar.expander("View flagged trades"):
            for r in removals:
                st.write(f"- {r['ticker']} {r['date']}")

        if st.sidebar.button("Apply removals to bounce_data.csv"):
            original_df = pd.read_csv(DATA_PATH)
            before_count = len(original_df)
            for r in removals:
                mask = (original_df["date"] == r["date"]) & (original_df["ticker"] == r["ticker"])
                original_df = original_df[~mask]
            after_count = len(original_df)
            removed = before_count - after_count
            original_df.to_csv(DATA_PATH, index=False)
            # Clear the removals file
            save_removals([])
            # Bust the cache so we reload
            load_bounce_data.clear()
            st.sidebar.success(f"Removed {removed} trades from bounce_data.csv")
            st.rerun()

    # Trade info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Ticker:** {trade['ticker']}")
    st.sidebar.markdown(f"**Date:** {raw_date}")
    st.sidebar.markdown(f"**Grade:** {trade['trade_grade']}")
    st.sidebar.markdown(f"**Cap:** {trade['cap']}")
    st.sidebar.markdown(f"**Setup:** {trade['Setup']}")

    # Key metrics
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Key Metrics")
    metric_map = {
        "Gap %": ("gap_pct", True),
        "Open-High %": ("bounce_open_high_pct", True),
        "Open-Close %": ("bounce_open_close_pct", True),
        "Open-Low %": ("bounce_open_low_pct", True),
        "Consec Down Days": ("consecutive_down_days", False),
        "% Off 30d High": ("pct_off_30d_high", True),
        "% Off 52wk High": ("pct_off_52wk_high", True),
        "Selloff Total %": ("selloff_total_pct", True),
        "ATR %": ("atr_pct", True),
        "RVOL": ("rvol_score", False),
        "Closed Outside Lower BB": ("closed_outside_lower_band", False),
    }
    for label, (col, is_pct) in metric_map.items():
        val = trade.get(col)
        if pd.notna(val):
            if is_pct and isinstance(val, (int, float)):
                st.sidebar.markdown(f"**{label}:** {val*100:.1f}%" if abs(val) < 10 else f"**{label}:** {val:.2f}")
            else:
                st.sidebar.markdown(f"**{label}:** {val}")

    # Fetch and display charts
    header_color = "red" if currently_flagged else "inherit"
    flag_label = " [FLAGGED FOR REMOVAL]" if currently_flagged else ""
    st.markdown(
        f"## {trade['ticker']} — {raw_date}  |  Grade: {trade['trade_grade']}  |  {trade['Setup']}"
        f"<span style='color:red'>{flag_label}</span>",
        unsafe_allow_html=True,
    )

    with st.spinner(f"Fetching data for {trade['ticker']}..."):
        daily_df = fetch_daily(trade["ticker"], api_date)
        intraday_df = fetch_intraday(trade["ticker"], api_date)

    col_daily, col_intra = st.columns(2)

    with col_daily:
        if daily_df is not None and not daily_df.empty:
            fig_daily = build_daily_chart(daily_df, trade["ticker"], api_date)
            st.plotly_chart(fig_daily, use_container_width=True)
        else:
            st.error("Could not fetch daily data")

    with col_intra:
        if intraday_df is not None and not intraday_df.empty:
            gap_pct = trade.get("gap_pct") if pd.notna(trade.get("gap_pct")) else None
            open_high = trade.get("bounce_open_high_pct") if pd.notna(trade.get("bounce_open_high_pct")) else None
            fig_intra = build_intraday_chart(
                intraday_df, trade["ticker"], api_date,
                gap_pct=gap_pct, bounce_open_high_pct=open_high,
            )
            st.plotly_chart(fig_intra, use_container_width=True)
        else:
            st.error("Could not fetch intraday data")

    # P&L row
    pnl_cols = st.columns(6)
    pnl_metrics = [
        ("Run %", "run_pct"), ("Reversal %", "reversal_pct"),
        ("Run Duration", "run_duration"), ("SPY O-C %", "spy_open_close_pct"),
        ("Time of Low", "time_of_low_bucket"), ("Day After Open %", "bounce_open_to_day_after_open_pct"),
    ]
    for col_widget, (label, col_name) in zip(pnl_cols, pnl_metrics):
        val = trade.get(col_name)
        if pd.notna(val):
            if isinstance(val, float) and "pct" in col_name.lower():
                col_widget.metric(label, f"{val*100:.1f}%" if abs(val) < 10 else f"{val:.2f}")
            else:
                col_widget.metric(label, val)
        else:
            col_widget.metric(label, "—")


if __name__ == "__main__":
    main()
