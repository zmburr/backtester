"""
Plotly chart builders for the Cap Deep Dive analyzer.
All figures use the Copper & Ink dark theme.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from options_replay.theme import C


def _apply_dark_theme(fig):
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


# ---------------------------------------------------------------------------
# 1. Delta Return Curve (the hero chart)
# ---------------------------------------------------------------------------

def fig_delta_return_curve(delta_curve_df: pd.DataFrame) -> go.Figure:
    """Line chart: x=delta, y=realistic return, one line per hold window.

    Area fill shows spread cost. Annotations at the peak per window.
    """
    fig = go.Figure()
    if delta_curve_df.empty:
        return _apply_dark_theme(fig)

    colors = {
        30: C["steel"],
        60: C["gold"],
        120: C["profit"],
    }
    dashes = {
        30: "dot",
        60: "solid",
        120: "solid",
    }
    widths = {
        30: 1.5,
        60: 2.5,
        120: 2,
    }

    for hw in sorted(delta_curve_df["hold_window"].unique()):
        sub = delta_curve_df[delta_curve_df["hold_window"] == hw].sort_values("delta_mid")
        if sub.empty:
            continue

        color = colors.get(hw, C["text2"])
        dash = dashes.get(hw, "solid")
        width = widths.get(hw, 2)

        # Realistic return line
        fig.add_trace(go.Scatter(
            x=sub["delta_mid"],
            y=sub["realistic_return"],
            mode="lines+markers",
            name=f"{hw}min hold",
            line=dict(color=color, dash=dash, width=width),
            marker=dict(size=6),
            text=[f"n={int(n)}" for n in sub["count"]],
            hovertemplate=(
                "Delta: %{x:.2f}<br>"
                "Return: %{y:.0%}<br>"
                "%{text}<br>"
                f"Hold: {hw}min"
                "<extra></extra>"
            ),
        ))

        # Spread cost as area (below the return line)
        net_of_spread = sub["realistic_return"] + sub["spread_cost"]
        fig.add_trace(go.Scatter(
            x=sub["delta_mid"],
            y=net_of_spread,
            mode="lines",
            name=f"{hw}min raw (before spread)",
            line=dict(color=color, width=0.5, dash="dot"),
            showlegend=False,
        ))

        # Annotate peak
        peak = sub.sort_values("realistic_return", ascending=False).iloc[0]
        fig.add_annotation(
            x=peak["delta_mid"],
            y=peak["realistic_return"],
            text=f"<b>{peak['realistic_return']:.0%}</b><br>δ={peak['delta_mid']:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor=color,
            font=dict(size=10, color=color),
            bgcolor=C["surface"],
            bordercolor=color,
            borderwidth=1,
        )

    fig.update_layout(
        title="Delta Return Curve — Realistic Return by Delta",
        xaxis_title="Delta (absolute)",
        yaxis_title="Realistic Return %",
        yaxis_tickformat=".0%",
        hovermode="x unified",
    )

    # Sweet spot shading (0.15-0.30)
    fig.add_vrect(
        x0=0.15, x1=0.30,
        fillcolor=C["gold"], opacity=0.06,
        line_width=0,
        annotation_text="sweet spot",
        annotation_position="top left",
        annotation_font_size=9,
        annotation_font_color=C["text3"],
    )

    return _apply_dark_theme(fig)


# ---------------------------------------------------------------------------
# 2. DTE Comparison
# ---------------------------------------------------------------------------

def fig_dte_comparison(dte_df: pd.DataFrame) -> go.Figure:
    """Grouped bars: x=strike, groups=DTE, y=realistic return."""
    fig = go.Figure()
    if dte_df.empty:
        return _apply_dark_theme(fig)

    dte_colors = {}
    palette = [C["gold"], C["steel"], C["profit"], C["loss"]]
    for i, dte in enumerate(sorted(dte_df["dte"].unique())):
        dte_colors[dte] = palette[i % len(palette)]

    for dte in sorted(dte_df["dte"].unique()):
        sub = dte_df[dte_df["dte"] == dte].sort_values("strike")
        if sub.empty:
            continue

        fig.add_trace(go.Bar(
            name=f"{int(dte)} DTE",
            x=[f"${s:.0f}" for s in sub["strike"]],
            y=sub["realistic_return_pct"],
            marker_color=dte_colors.get(dte, C["text2"]),
            text=[f"{r:.0%}<br>δ={d:.2f}" for r, d in
                  zip(sub["realistic_return_pct"], sub["delta"].fillna(0))],
            textposition="outside",
            textfont_size=9,
        ))

    fig.update_layout(
        title="DTE Comparison — Same Strikes Across Expirations",
        xaxis_title="Strike",
        yaxis_title="Realistic Return %",
        yaxis_tickformat=".0%",
        barmode="group",
    )
    return _apply_dark_theme(fig)


# ---------------------------------------------------------------------------
# 3. IV Decomposition
# ---------------------------------------------------------------------------

def fig_iv_decomposition(iv_decomp_df: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """Stacked bar: delta / vega / theta / residual P&L for top contracts."""
    fig = go.Figure()
    if iv_decomp_df.empty:
        return _apply_dark_theme(fig)

    # Top N by absolute actual P&L
    df = iv_decomp_df.copy()
    df["abs_pnl"] = df["actual_pnl"].abs()
    df = df[df["abs_pnl"] > 0].sort_values("abs_pnl", ascending=False).head(top_n)
    df = df.sort_values("delta_entry", key=lambda x: x.abs())

    if df.empty:
        return _apply_dark_theme(fig)

    labels = df["contract"].tolist()

    fig.add_trace(go.Bar(
        name="Delta P&L",
        x=labels, y=df["delta_pnl"],
        marker_color=C["profit"],
        hovertemplate="Delta: $%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Vega P&L",
        x=labels, y=df["vega_pnl"],
        marker_color=C["steel"],
        hovertemplate="Vega: $%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Theta P&L",
        x=labels, y=df["theta_pnl"],
        marker_color=C["loss"],
        hovertemplate="Theta: $%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Residual (gamma + spread)",
        x=labels, y=df["residual_pnl"],
        marker_color=C["text3"],
        hovertemplate="Residual: $%{y:.3f}<extra></extra>",
    ))

    # Reference line: actual P&L
    fig.add_trace(go.Scatter(
        x=labels, y=df["actual_pnl"],
        mode="markers",
        name="Actual P&L",
        marker=dict(color=C["gold"], size=10, symbol="diamond"),
        hovertemplate="Actual: $%{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title="P&L Decomposition — Delta vs Vega vs Theta",
        xaxis_title="Contract",
        yaxis_title="P&L per Share ($)",
        barmode="relative",
    )
    return _apply_dark_theme(fig)


# ---------------------------------------------------------------------------
# 4. Liquidity Matrix
# ---------------------------------------------------------------------------

def fig_liquidity_matrix(liquidity_df: pd.DataFrame) -> go.Figure:
    """Scatter: x=delta, y=return, bubble size=volume+OI, color=spread."""
    fig = go.Figure()
    if liquidity_df.empty:
        return _apply_dark_theme(fig)

    df = liquidity_df.copy()
    df["abs_delta"] = df["delta"].abs()

    vol = df.get("volume_during_window", pd.Series(0, index=df.index)).fillna(0)
    oi = df.get("open_interest", pd.Series(0, index=df.index)).fillna(0)
    df["liq_size"] = vol + oi
    df["liq_size_scaled"] = np.clip(df["liq_size"], 10, 10000)

    spread = df.get("avg_spread_pct_window", pd.Series(0, index=df.index)).fillna(0)

    # Color by liquidity grade
    grade_colors = {"A": C["profit"], "B": C["gold"], "C": C["loss"]}
    colors = [grade_colors.get(g, C["text3"]) for g in df.get("liquidity_grade", ["C"] * len(df))]

    # Build hover text
    hover = []
    for _, r in df.iterrows():
        text = (f"${r.get('strike', 0):.0f} {int(r.get('dte', 0))}DTE<br>"
                f"Return: {r.get('realistic_return_pct', 0):.0%}<br>"
                f"Delta: {r.get('delta', 0):.3f}<br>"
                f"Spread: {spread.loc[r.name]:.1%}<br>"
                f"Volume: {vol.loc[r.name]:.0f} | OI: {oi.loc[r.name]:.0f}<br>"
                f"Grade: {r.get('liquidity_grade', 'C')}")
        hover.append(text)

    fig.add_trace(go.Scatter(
        x=df["abs_delta"],
        y=df["realistic_return_pct"],
        mode="markers",
        marker=dict(
            size=np.log1p(df["liq_size_scaled"]) * 4,
            color=colors,
            line=dict(width=1, color=C["border"]),
            opacity=0.85,
        ),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
    ))

    # Label each point with strike
    for _, r in df.iterrows():
        fig.add_annotation(
            x=abs(r.get("delta", 0)),
            y=r.get("realistic_return_pct", 0),
            text=f"${r.get('strike', 0):.0f}",
            showarrow=False,
            font=dict(size=8, color=C["text3"]),
            yshift=12,
        )

    fig.update_layout(
        title="Liquidity Matrix — Return vs Delta (bubble = volume+OI)",
        xaxis_title="Delta (absolute)",
        yaxis_title="Realistic Return %",
        yaxis_tickformat=".0%",
        showlegend=False,
    )

    # Sweet spot shading
    fig.add_vrect(
        x0=0.15, x1=0.30,
        fillcolor=C["gold"], opacity=0.06,
        line_width=0,
    )

    return _apply_dark_theme(fig)


# ---------------------------------------------------------------------------
# 5. Underlying with targets
# ---------------------------------------------------------------------------

def fig_underlying_with_targets(underlying_bars: pd.DataFrame,
                                 entry_time, target_levels: dict,
                                 side: int, hold_minutes: int = 120) -> go.Figure:
    """Candlestick of underlying with ATR target levels and hold window."""
    fig = go.Figure()
    if underlying_bars.empty:
        return _apply_dark_theme(fig)

    df = underlying_bars.copy()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color=C["profit"],
        decreasing_line_color=C["loss"],
        name="Price",
    ))

    # Entry line + hold window end (use add_shape to avoid Plotly datetime annotation bug)
    from datetime import timedelta
    exit_time = entry_time + timedelta(minutes=hold_minutes)

    fig.add_shape(type="line", x0=entry_time, x1=entry_time, y0=0, y1=1,
                  yref="paper", line=dict(color=C["gold"], dash="dash", width=1))
    fig.add_annotation(x=entry_time, y=1.02, yref="paper", text="Entry",
                       showarrow=False, font=dict(size=10, color=C["gold"]))

    fig.add_shape(type="line", x0=exit_time, x1=exit_time, y0=0, y1=1,
                  yref="paper", line=dict(color=C["text3"], dash="dot", width=1))

    # Shade hold window
    fig.add_shape(type="rect", x0=entry_time, x1=exit_time, y0=0, y1=1,
                  yref="paper", fillcolor=C["gold"], opacity=0.04, line_width=0)

    # Target levels
    target_colors = [C["profit"], C["steel"], C["gold"], C["loss"]]
    for i, (name, level) in enumerate(target_levels.items()):
        color = target_colors[i % len(target_colors)]
        fig.add_shape(type="line", x0=0, x1=1, xref="paper",
                      y0=level, y1=level,
                      line=dict(color=color, dash="dash", width=1))
        fig.add_annotation(x=1.01, xref="paper", y=level,
                           text=f"{name} ATR (${level:.2f})",
                           showarrow=False, font=dict(size=9, color=color),
                           xanchor="left")

    fig.update_layout(
        title="Underlying Price Action with ATR Targets",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        showlegend=False,
    )
    return _apply_dark_theme(fig)


# ---------------------------------------------------------------------------
# 6. Dollar return by contract (supplementary)
# ---------------------------------------------------------------------------

def fig_dollar_return_scatter(liquidity_df: pd.DataFrame) -> go.Figure:
    """Scatter: x=entry cost, y=dollar return per contract."""
    fig = go.Figure()
    if liquidity_df.empty or "dollar_return" not in liquidity_df.columns:
        return _apply_dark_theme(fig)

    df = liquidity_df.copy()
    grade_colors = {"A": C["profit"], "B": C["gold"], "C": C["loss"]}
    colors = [grade_colors.get(g, C["text3"]) for g in df.get("liquidity_grade", ["C"] * len(df))]

    fig.add_trace(go.Scatter(
        x=df["entry_ask"],
        y=df["dollar_return"],
        mode="markers+text",
        marker=dict(size=10, color=colors, line=dict(width=1, color=C["border"])),
        text=[f"${s:.0f}" for s in df["strike"]],
        textposition="top center",
        textfont=dict(size=8, color=C["text3"]),
        hovertemplate=(
            "Strike: $%{text}<br>"
            "Entry: $%{x:.2f}<br>"
            "Dollar Return: $%{y:.0f}/contract<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title="Dollar Return per Contract vs Entry Cost",
        xaxis_title="Entry Ask ($)",
        yaxis_title="Dollar Return per Contract ($)",
    )
    return _apply_dark_theme(fig)
