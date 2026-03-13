"""
Plotly figure builders for the Options Replay dashboard.
All figures use the Copper & Ink dark theme.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from options_replay.theme import C
from options_replay.chain_analyzer import contract_label, contract_key


def _apply_dark_theme(fig):
    """Apply Copper & Ink dark theme to a figure."""
    fig.update_layout(
        paper_bgcolor=C["surface"],
        plot_bgcolor=C["bg"],
        font=dict(family="Outfit, sans-serif", color=C["text2"], size=11),
        title_font=dict(family="DM Serif Display, serif", color=C["text"], size=15),
        legend_font_color=C["text2"],
        margin=dict(l=50, r=20, t=50, b=40),
    )
    fig.update_xaxes(gridcolor=C["border"], zerolinecolor=C["border"])
    fig.update_yaxes(gridcolor=C["border"], zerolinecolor=C["border"])
    return fig


def fig_underlying_candlestick(ohlc_df: pd.DataFrame, entry_time: datetime,
                                exit_time: datetime = None, entry_price: float = 0,
                                side: int = 1, hold_minutes: int = 30) -> go.Figure:
    """1-min candlestick chart of the underlying stock.

    Shows 30 min before to hold_minutes after headline.
    Entry/exit arrows and shaded analysis window.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)

    if ohlc_df.empty:
        fig.add_annotation(text="No underlying data available",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                          font=dict(color=C["text3"], size=14))
        return _apply_dark_theme(fig)

    df = ohlc_df.copy()

    # Window: 30 min before to hold_minutes after
    window_start = entry_time - timedelta(minutes=30)
    window_end = entry_time + timedelta(minutes=hold_minutes)

    # Handle timezone matching
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        if window_start.tzinfo is None:
            from pytz import timezone
            et = timezone("US/Eastern")
            window_start = et.localize(window_start)
            window_end = et.localize(window_end)

    df = df[(df.index >= window_start) & (df.index <= window_end)]

    if df.empty:
        fig.add_annotation(text="No data in time window",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                          font=dict(color=C["text3"], size=14))
        return _apply_dark_theme(fig)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color=C["profit"], decreasing_line_color=C["loss"],
        increasing_fillcolor=C["profit"], decreasing_fillcolor=C["loss"],
        name="Price",
        showlegend=False,
    ), row=1, col=1)

    # Volume bars
    if "volume" in df.columns:
        colors = [C["profit"] if c >= o else C["loss"]
                  for o, c in zip(df["open"], df["close"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["volume"],
            marker_color=colors, opacity=0.4,
            name="Volume", showlegend=False,
        ), row=2, col=1)

    # Shaded analysis window
    analysis_start = entry_time
    analysis_end = entry_time + timedelta(minutes=hold_minutes)
    if hasattr(df.index, 'tz') and df.index.tz is not None and analysis_start.tzinfo is None:
        from pytz import timezone
        et = timezone("US/Eastern")
        analysis_start = et.localize(analysis_start)
        analysis_end = et.localize(analysis_end)

    fig.add_vrect(
        x0=analysis_start, x1=analysis_end,
        fillcolor=C["gold"], opacity=0.06,
        line_width=0, row=1, col=1,
    )

    # Entry marker
    entry_marker_time = entry_time
    if hasattr(df.index, 'tz') and df.index.tz is not None and entry_marker_time.tzinfo is None:
        from pytz import timezone
        entry_marker_time = timezone("US/Eastern").localize(entry_marker_time)

    if entry_price > 0:
        marker_color = C["profit"] if side == 1 else C["loss"]
        marker_symbol = "triangle-up" if side == 1 else "triangle-down"
        fig.add_trace(go.Scatter(
            x=[entry_marker_time], y=[entry_price],
            mode="markers",
            marker=dict(size=14, color=marker_color, symbol=marker_symbol,
                       line=dict(width=1, color=C["text"])),
            name="Entry",
            showlegend=True,
        ), row=1, col=1)

        # Entry price line
        fig.add_hline(y=entry_price, line_dash="dash", line_color=C["text3"],
                      line_width=1, row=1, col=1)

    fig.update_layout(
        title="Underlying Price Action",
        xaxis_rangeslider_visible=False,
        xaxis2_title="Time",
        yaxis_title="Price",
        yaxis2_title="Volume",
        height=450,
    )

    return _apply_dark_theme(fig)


def fig_chain_heatmap(chain_df: pd.DataFrame, underlying_price: float) -> go.Figure:
    """Heatmap of the chain at headline time.

    X = strike, Y = expiration, Color = spread_pct (lower = better = greener).
    Annotations show volume/OI.
    """
    fig = go.Figure()

    if chain_df.empty:
        fig.add_annotation(text="No chain data available",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                          font=dict(color=C["text3"], size=14))
        return _apply_dark_theme(fig)

    df = chain_df.copy()

    # Only show primary right
    if "is_primary" in df.columns:
        df = df[df["is_primary"]].copy()

    if df.empty or "strike" not in df.columns or "expiration" not in df.columns:
        return _apply_dark_theme(fig)

    # Pivot for heatmap
    strikes = sorted(df["strike"].unique())
    expirations = sorted(df["expiration"].unique())

    z_data = []
    text_data = []
    for exp in expirations:
        z_row = []
        text_row = []
        for strike in strikes:
            mask = (df["expiration"] == exp) & (df["strike"] == strike)
            match = df[mask]
            if not match.empty:
                row = match.iloc[0]
                spread_pct = row.get("spread_pct", 0)
                mid = row.get("mid", 0)
                vol = int(row.get("volume", 0))
                oi = int(row.get("open_interest", 0))
                z_row.append(spread_pct * 100 if pd.notna(spread_pct) else None)
                text_row.append(f"Mid: ${mid:.2f}<br>Spread: {spread_pct:.1%}<br>Vol: {vol}<br>OI: {oi}")
            else:
                z_row.append(None)
                text_row.append("")
        z_data.append(z_row)
        text_data.append(text_row)

    # Format labels
    strike_labels = [f"${s:.0f}" for s in strikes]
    exp_labels = []
    for e in expirations:
        try:
            exp_labels.append(pd.Timestamp(e).strftime("%b-%d"))
        except Exception:
            exp_labels.append(str(e))

    fig.add_trace(go.Heatmap(
        z=z_data,
        x=strike_labels,
        y=exp_labels,
        text=text_data,
        hovertemplate="%{text}<extra></extra>",
        colorscale=[[0, C["profit"]], [0.5, C["gold"]], [1, C["loss"]]],
        colorbar=dict(title=dict(text="Spread %", font=dict(color=C["text2"])),
                     ticksuffix="%", tickfont=dict(color=C["text2"])),
        zmin=0, zmax=20,
    ))

    # ATM marker
    atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - underlying_price))
    fig.add_vline(x=atm_idx, line_dash="dash", line_color=C["gold"],
                  line_width=1, opacity=0.5)

    fig.update_layout(
        title="Chain Snapshot — Spread % by Strike & Expiry",
        xaxis_title="Strike",
        yaxis_title="Expiration",
        height=300,
    )

    return _apply_dark_theme(fig)


def fig_top_options_table(scored_df: pd.DataFrame) -> go.Figure:
    """Styled table showing top ranked options with all metrics."""
    fig = go.Figure()

    if scored_df.empty:
        fig.add_annotation(text="No scored options",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                          font=dict(color=C["text3"], size=14))
        return _apply_dark_theme(fig)

    df = scored_df.copy()

    # Build labels
    labels = [contract_label(row) for _, row in df.iterrows()]

    # Helper to format greeks safely (may be None)
    def _fmt_greek(series, decimals=2):
        return series.apply(lambda x: f"{x:.{decimals}f}" if pd.notna(x) else "—")

    # Format columns
    headers = ["Rank", "Contract", "DTE", "Entry (Ask)", "Max (Bid)", "Raw %", "Real %", "Spread Cost",
               "Delta", "IV",
               "Spread %", "Volume", "OI", "Score"]

    cell_values = [
        df["rank"],
        labels,
        df.get("dte", pd.Series(dtype=int)),
        df.get("entry_ask", df["entry_mid"]).apply(lambda x: f"${x:.2f}"),
        df.get("max_bid", df["max_mid"]).apply(lambda x: f"${x:.2f}"),
        df["raw_return_pct"].apply(lambda x: f"{x:.0%}"),
        df.get("realistic_return_pct", df["raw_return_pct"]).apply(lambda x: f"{x:.0%}"),
        df.get("spread_cost_pct", pd.Series(0, index=df.index)).apply(lambda x: f"{x:.0%}"),
        _fmt_greek(df.get("delta", pd.Series(dtype=float)), 2),
        df.get("implied_vol", pd.Series(dtype=float)).apply(
            lambda x: f"{x:.0%}" if pd.notna(x) else "—"),
        df["avg_spread_pct_window"].apply(lambda x: f"{x:.1%}"),
        df["volume_during_window"].apply(lambda x: f"{x:,}"),
        df.get("open_interest", pd.Series(0, index=df.index)).apply(lambda x: f"{int(x):,}"),
        df["composite_score"].apply(lambda x: f"{x:.1f}"),
    ]

    fig.add_trace(go.Table(
        header=dict(
            values=headers,
            fill_color=C["elevated"],
            font=dict(color=C["gold"], size=11, family="Outfit, sans-serif"),
            align="center",
            line_color=C["border"],
        ),
        cells=dict(
            values=cell_values,
            fill_color=[
                [C["surface"] if i % 2 == 0 else C["elevated"] for i in range(len(df))]
            ] * len(headers),
            font=dict(color=C["text"], size=11, family="JetBrains Mono, monospace"),
            align="center",
            line_color=C["border"],
            height=30,
        ),
    ))

    fig.update_layout(
        title="Top Options Ranked by Liquidity-Adjusted Return",
        height=max(200, 50 + len(df) * 32),
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return _apply_dark_theme(fig)


def fig_option_price_chart(quotes_df: pd.DataFrame, ohlc_df: pd.DataFrame,
                           label: str, entry_mid: float,
                           snapshot_time: datetime, hold_minutes: int = 30) -> go.Figure:
    """Individual 1-min price chart for a single option contract.

    Shows bid/ask/mid lines, shaded bid-ask spread area, and volume bars.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.05)

    has_data = False

    # Window
    window_start = snapshot_time
    window_end = snapshot_time + timedelta(minutes=hold_minutes)

    # Quote data: bid/ask/mid
    if not quotes_df.empty:
        qdf = quotes_df.copy()

        if hasattr(qdf.index, 'tz') and qdf.index.tz is not None and window_start.tzinfo is None:
            from pytz import timezone
            et = timezone("US/Eastern")
            window_start = et.localize(window_start)
            window_end = et.localize(window_end)

        qdf = qdf[(qdf.index >= window_start) & (qdf.index <= window_end)]

        if not qdf.empty and "bid" in qdf.columns and "ask" in qdf.columns:
            has_data = True
            mid = (qdf["bid"] + qdf["ask"]) / 2 if "mid" not in qdf.columns else qdf["mid"]

            # Shaded bid-ask area
            fig.add_trace(go.Scatter(
                x=list(qdf.index) + list(qdf.index[::-1]),
                y=list(qdf["ask"]) + list(qdf["bid"][::-1]),
                fill="toself",
                fillcolor=f"rgba(200, 164, 110, 0.1)",
                line=dict(width=0),
                name="Bid-Ask Spread",
                showlegend=False,
                hoverinfo="skip",
            ), row=1, col=1)

            # Mid line (gold)
            fig.add_trace(go.Scatter(
                x=qdf.index, y=mid,
                mode="lines",
                line=dict(color=C["gold"], width=2),
                name="Mid",
            ), row=1, col=1)

            # Bid line
            fig.add_trace(go.Scatter(
                x=qdf.index, y=qdf["bid"],
                mode="lines",
                line=dict(color=C["loss"], width=1, dash="dot"),
                name="Bid",
                opacity=0.6,
            ), row=1, col=1)

            # Ask line
            fig.add_trace(go.Scatter(
                x=qdf.index, y=qdf["ask"],
                mode="lines",
                line=dict(color=C["steel"], width=1, dash="dot"),
                name="Ask",
                opacity=0.6,
            ), row=1, col=1)

    # Volume from OHLC data
    if not ohlc_df.empty and "volume" in ohlc_df.columns:
        vdf = ohlc_df.copy()
        # Re-compute window (may need TZ)
        ws = snapshot_time
        we = snapshot_time + timedelta(minutes=hold_minutes)
        if hasattr(vdf.index, 'tz') and vdf.index.tz is not None and ws.tzinfo is None:
            from pytz import timezone
            et = timezone("US/Eastern")
            ws = et.localize(ws)
            we = et.localize(we)
        vdf = vdf[(vdf.index >= ws) & (vdf.index <= we)]

        if not vdf.empty:
            has_data = True
            fig.add_trace(go.Bar(
                x=vdf.index, y=vdf["volume"],
                marker_color=C["steel"], opacity=0.4,
                name="Volume", showlegend=False,
            ), row=2, col=1)

    if not has_data:
        fig.add_annotation(text="No data",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                          font=dict(color=C["text3"], size=12))

    # Entry line
    if entry_mid > 0:
        fig.add_hline(y=entry_mid, line_dash="dash", line_color=C["text3"],
                      line_width=1, row=1, col=1,
                      annotation_text=f"Entry ${entry_mid:.2f}",
                      annotation_font_color=C["text3"],
                      annotation_font_size=9)

    fig.update_layout(
        title=label,
        title_font_size=12,
        height=280,
        showlegend=False,
        margin=dict(l=40, r=10, t=35, b=20),
    )
    fig.update_yaxes(title_text="Premium", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)

    return _apply_dark_theme(fig)


def fig_return_comparison(scored_df: pd.DataFrame) -> go.Figure:
    """Grouped horizontal bar chart: raw return vs realistic return."""
    fig = go.Figure()

    if scored_df.empty:
        return _apply_dark_theme(fig)

    df = scored_df.sort_values("composite_score", ascending=True)
    labels = [contract_label(row) for _, row in df.iterrows()]

    has_realistic = "realistic_return_pct" in df.columns

    # Raw return (muted)
    fig.add_trace(go.Bar(
        y=labels,
        x=df["raw_return_pct"] * 100,
        orientation="h",
        marker_color=C["steel"],
        opacity=0.4,
        name="Raw (mid→high)",
        text=df["raw_return_pct"].apply(lambda x: f"{x:.0%}"),
        textposition="auto",
        textfont=dict(color=C["text3"], size=9),
    ))

    # Realistic return (prominent)
    if has_realistic:
        fig.add_trace(go.Bar(
            y=labels,
            x=df["realistic_return_pct"] * 100,
            orientation="h",
            marker_color=C["gold"],
            name="Realistic (ask→bid)",
            text=df["realistic_return_pct"].apply(lambda x: f"{x:.0%}"),
            textposition="auto",
            textfont=dict(color=C["text"], size=10),
        ))

    fig.update_layout(
        title="Return Comparison — Raw vs Realistic",
        xaxis_title="Return %",
        barmode="group",
        height=max(280, len(df) * 45 + 80),
        margin=dict(l=120, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return _apply_dark_theme(fig)


def fig_liquidity_comparison(scored_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: spread %, volume, OI across top options."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=["Spread %", "Volume", "Open Interest"],
                        horizontal_spacing=0.08)

    if scored_df.empty:
        return _apply_dark_theme(fig)

    df = scored_df.sort_values("rank")
    labels = [contract_label(row) for _, row in df.iterrows()]

    # Spread %
    fig.add_trace(go.Bar(
        x=labels, y=df["avg_spread_pct_window"] * 100,
        marker_color=C["loss"], name="Spread %",
        showlegend=False,
    ), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(
        x=labels, y=df["volume_during_window"],
        marker_color=C["steel"], name="Volume",
        showlegend=False,
    ), row=1, col=2)

    # OI
    oi = df.get("open_interest", pd.Series(0, index=df.index))
    fig.add_trace(go.Bar(
        x=labels, y=oi,
        marker_color=C["gold"], name="OI",
        showlegend=False,
    ), row=1, col=3)

    fig.update_layout(
        title="Liquidity Comparison",
        height=300,
    )
    fig.update_xaxes(tickangle=45, tickfont_size=9)

    return _apply_dark_theme(fig)


def fig_option_price_grid(scored_df: pd.DataFrame, quotes_dict: dict,
                          ohlc_dict: dict, snapshot_time: datetime,
                          hold_minutes: int = 30) -> list:
    """Build a list of individual option price chart figures for the top options."""
    figures = []

    if scored_df.empty:
        return figures

    for _, row in scored_df.iterrows():
        key = contract_key(row)
        label = f"#{int(row['rank'])} {contract_label(row)}"
        entry_mid = row.get("entry_mid", 0)

        quotes = quotes_dict.get(key, pd.DataFrame())
        ohlc = ohlc_dict.get(key, pd.DataFrame())

        fig = fig_option_price_chart(
            quotes_df=quotes,
            ohlc_df=ohlc,
            label=label,
            entry_mid=entry_mid,
            snapshot_time=snapshot_time,
            hold_minutes=hold_minutes,
        )
        figures.append(fig)

    return figures
