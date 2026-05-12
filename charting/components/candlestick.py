"""
Candlestick chart builder with MA overlays, volume subplot,
text annotations, target level lines, and historical trade markers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_queries.polygon_queries import get_levels_data


# ---------------------------------------------------------------------------
# Data fetching (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner="Fetching daily data...")
def fetch_daily_data(ticker: str, end_date: str, window: int = 250) -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLCV data and compute moving averages locally.

    Args:
        ticker: Stock symbol
        end_date: End date in YYYY-MM-DD format
        window: Number of calendar days to look back

    Returns:
        DataFrame with OHLCV + computed MAs, indexed by date
    """
    df = get_levels_data(ticker, end_date, window, 1, 'day')
    if df is None or df.empty:
        return None

    # Compute moving averages locally
    df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['SMA10'] = df['close'].rolling(10).mean()
    df['SMA20'] = df['close'].rolling(20).mean()
    df['SMA50'] = df['close'].rolling(50).mean()
    df['SMA200'] = df['close'].rolling(200).mean()

    # Convert timezone-aware index to date for cleaner display
    df['date'] = df.index.date
    df['date_str'] = df.index.strftime('%Y-%m-%d')

    return df


# ---------------------------------------------------------------------------
# Chart builder
# ---------------------------------------------------------------------------

def build_candlestick_figure(
    df: pd.DataFrame,
    ticker: str,
    text_annotations: Optional[List[dict]] = None,
    target_levels: Optional[List[Tuple[float, str, str]]] = None,
    historical_trades: Optional[pd.DataFrame] = None,
    show_volume: bool = True,
    show_mas: bool = True,
) -> go.Figure:
    """
    Build a Plotly candlestick chart with optional overlays.

    Args:
        df: DataFrame from fetch_daily_data()
        ticker: Symbol for title
        text_annotations: list of {"date", "text"} dicts for text labels on chart
        target_levels: [(price, color, label), ...] for horizontal lines
        historical_trades: DataFrame with 'date_str' and 'pnl' columns
        show_volume: Whether to show volume subplot
        show_mas: Whether to show moving average overlays

    Returns:
        Plotly Figure
    """
    if text_annotations is None:
        text_annotations = []
    if target_levels is None:
        target_levels = []

    row_heights = [0.75, 0.25] if show_volume else [1.0]
    rows = 2 if show_volume else 1

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    dates = df['date_str'].tolist()

    # Candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            increasing_line_color='#6ee7b7',
            decreasing_line_color='#f87171',
            increasing_fillcolor='#6ee7b7',
            decreasing_fillcolor='#f87171',
            name='Price',
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Invisible scatter overlay for click-to-select-date
    # (Candlestick traces don't support Plotly point selection,
    #  but scatter traces do — this makes candles "clickable")
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df['close'],
            mode='markers',
            marker=dict(size=12, color='rgba(0,0,0,0)'),
            hoverinfo='skip',
            showlegend=False,
            name='_click_target',
        ),
        row=1, col=1,
    )

    # Moving averages
    if show_mas:
        ma_traces = [
            ('EMA9', '#fbbf24', 1.0),      # amber
            ('SMA10', '#4fc3f7', 0.8),      # cyan
            ('SMA20', '#c084fc', 0.8),      # lavender
            ('SMA50', '#6ee7b7', 1.0),      # green
            ('SMA200', '#f87171', 1.0),     # red
        ]
        for col_name, color, width in ma_traces:
            if col_name in df.columns:
                vals = df[col_name]
                if vals.notna().sum() > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=vals,
                            mode='lines',
                            name=col_name,
                            line=dict(color=color, width=width),
                            opacity=0.7,
                        ),
                        row=1, col=1,
                    )

    # Volume bars
    if show_volume:
        colors = [
            '#6ee7b7' if c >= o else '#f87171'
            for c, o in zip(df['close'], df['open'])
        ]
        fig.add_trace(
            go.Bar(
                x=dates,
                y=df['volume'],
                marker_color=colors,
                opacity=0.5,
                name='Volume',
                showlegend=False,
            ),
            row=2, col=1,
        )

    # Text annotations (TC2000-style labels with connecting arrow)
    for ann in text_annotations:
        ann_date = ann.get("date", "")
        ann_text = ann.get("text", "")
        if ann_date in dates and ann_text:
            idx = dates.index(ann_date)
            price_high = df['high'].iloc[idx]
            fig.add_annotation(
                x=ann_date,
                y=price_high,
                text=ann_text,
                showarrow=True,
                arrowhead=0,
                arrowwidth=1,
                arrowcolor='#6b7a90',
                ax=0,
                ay=-45,
                font=dict(
                    family="Outfit, sans-serif",
                    size=11,
                    color="#c0c8d8",
                ),
                bgcolor="rgba(13,17,23,0.85)",
                bordercolor="#2a3548",
                borderwidth=1,
                borderpad=4,
                xref="x",
                yref="y",
            )

    # Target level lines
    for price, color, label in target_levels:
        if price is not None:
            fig.add_hline(
                y=price,
                line_dash="dot",
                line_color=color,
                line_width=1,
                annotation_text=label,
                annotation_position="right",
                annotation_font=dict(size=10, color=color),
                row=1, col=1,
            )

    # Historical trade markers
    if historical_trades is not None and not historical_trades.empty:
        for _, trade in historical_trades.iterrows():
            trade_date = trade.get('date_str', '')
            pnl = trade.get('pnl', 0)
            if trade_date in dates:
                idx = dates.index(trade_date)
                is_win = pnl > 0 if pd.notna(pnl) else True
                price = df['low'].iloc[idx] if is_win else df['high'].iloc[idx]
                marker_color = '#6ee7b7' if is_win else '#f87171'
                symbol = 'triangle-up' if is_win else 'triangle-down'
                pnl_str = f"{pnl:+.1f}%" if pd.notna(pnl) else "N/A"

                fig.add_trace(
                    go.Scatter(
                        x=[trade_date],
                        y=[price],
                        mode='markers',
                        marker=dict(
                            symbol=symbol,
                            size=12,
                            color=marker_color,
                            line=dict(width=1, color='white'),
                        ),
                        name=f"Trade {pnl_str}",
                        hovertext=f"{trade.get('ticker', '')} {trade_date}<br>P&L: {pnl_str}",
                        hoverinfo='text',
                        showlegend=False,
                    ),
                    row=1, col=1,
                )

    # Layout — no title (handled by custom header above chart)
    fig.update_layout(
        title="",
        height=600,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=9, color="#5a6578"),
        ),
        margin=dict(l=48, r=12, t=20, b=32),
    )

    # Hide weekends / gaps
    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])],
        type="category",
        tickangle=-45,
        nticks=20,
    )

    # Y-axis labels
    fig.update_yaxes(title_text="", row=1, col=1)
    if show_volume:
        fig.update_yaxes(title_text="", row=2, col=1)

    return fig
