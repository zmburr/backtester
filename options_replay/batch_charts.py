"""
Plotly chart builders for batch analysis results.
All figures use the Copper & Ink dark theme.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from options_replay.theme import C
from options_replay.batch_aggregator import MONEYNESS_ORDER


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


def fig_return_by_delta_bucket(agg_df: pd.DataFrame) -> go.Figure:
    """Bar chart: avg realistic return by delta bucket with error bars."""
    fig = go.Figure()
    if agg_df.empty:
        return _apply_dark_theme(fig)

    fig.add_trace(go.Bar(
        x=agg_df["delta_bucket"],
        y=agg_df["mean_return"] * 100,
        error_y=dict(type="data", array=agg_df["std_return"] * 100, visible=True,
                     color=C["text3"]),
        marker_color=C["gold"],
        text=agg_df.apply(lambda r: f"{r['mean_return']:.0%}<br>n={int(r['count'])}", axis=1),
        textposition="outside",
        textfont=dict(color=C["text2"], size=9),
    ))

    fig.update_layout(
        title="Avg Return by Delta",
        xaxis_title="Absolute Delta",
        yaxis_title="Avg Realistic Return %",
        height=350,
    )
    return _apply_dark_theme(fig)


def fig_return_by_dte_bucket(agg_df: pd.DataFrame) -> go.Figure:
    """Bar chart: avg return by DTE bucket."""
    fig = go.Figure()
    if agg_df.empty:
        return _apply_dark_theme(fig)

    fig.add_trace(go.Bar(
        x=agg_df["dte_bucket"],
        y=agg_df["mean_return"] * 100,
        marker_color=C["steel"],
        text=agg_df.apply(lambda r: f"{r['mean_return']:.0%}<br>n={int(r['count'])}", axis=1),
        textposition="outside",
        textfont=dict(color=C["text2"], size=9),
    ))

    fig.update_layout(
        title="Avg Return by DTE",
        xaxis_title="Days to Expiration",
        yaxis_title="Avg Realistic Return %",
        height=350,
    )
    return _apply_dark_theme(fig)


def fig_return_by_moneyness(agg_df: pd.DataFrame) -> go.Figure:
    """Bar chart: avg return by moneyness with win rate annotation."""
    fig = go.Figure()
    if agg_df.empty:
        return _apply_dark_theme(fig)

    # Sort by moneyness order
    order = {v: i for i, v in enumerate(MONEYNESS_ORDER)}
    df = agg_df.copy()
    df["_sort"] = df["moneyness_5"].map(order).fillna(99)
    df = df.sort_values("_sort")

    fig.add_trace(go.Bar(
        x=df["moneyness_5"],
        y=df["mean_return"] * 100,
        marker_color=C["gold"],
        text=df.apply(
            lambda r: f"{r['mean_return']:.0%}<br>WR: {r['win_rate']:.0%}<br>n={int(r['count'])}",
            axis=1
        ),
        textposition="outside",
        textfont=dict(color=C["text2"], size=9),
    ))

    fig.update_layout(
        title="Avg Return by Moneyness",
        xaxis_title="Moneyness",
        yaxis_title="Avg Realistic Return %",
        height=350,
    )
    return _apply_dark_theme(fig)


def fig_delta_vs_return_scatter(results_df: pd.DataFrame) -> go.Figure:
    """Scatter: delta (x) vs realistic return (y), color=composite score."""
    fig = go.Figure()
    if results_df.empty:
        return _apply_dark_theme(fig)

    df = results_df.dropna(subset=["delta"])
    if df.empty:
        return _apply_dark_theme(fig)

    return_col = "realistic_return_pct" if "realistic_return_pct" in df.columns else "raw_return_pct"

    fig.add_trace(go.Scatter(
        x=df["delta"].abs(),
        y=df[return_col] * 100,
        mode="markers",
        marker=dict(
            size=8,
            color=df.get("composite_score", pd.Series(50, index=df.index)),
            colorscale=[[0, C["steel"]], [1, C["gold"]]],
            colorbar=dict(title=dict(text="Score", font=dict(color=C["text2"])),
                         tickfont=dict(color=C["text2"])),
            opacity=0.7,
            line=dict(width=0.5, color=C["border"]),
        ),
        text=df.apply(
            lambda r: f"{r.get('symbol','')}<br>{r.get('trade_date','')}<br>Delta: {abs(r['delta']):.2f}",
            axis=1
        ),
        hovertemplate="%{text}<br>Return: %{y:.0f}%<extra></extra>",
    ))

    fig.update_layout(
        title="Delta vs Return",
        xaxis_title="Absolute Delta",
        yaxis_title="Realistic Return %",
        height=400,
    )
    return _apply_dark_theme(fig)


def fig_iv_vs_return_scatter(results_df: pd.DataFrame) -> go.Figure:
    """Scatter: implied vol (x) vs realistic return (y), color=moneyness."""
    fig = go.Figure()
    if results_df.empty:
        return _apply_dark_theme(fig)

    df = results_df.dropna(subset=["implied_vol"])
    if df.empty:
        return _apply_dark_theme(fig)

    return_col = "realistic_return_pct" if "realistic_return_pct" in df.columns else "raw_return_pct"
    moneyness_col = "moneyness_5" if "moneyness_5" in df.columns else "moneyness_label"

    color_map = {
        "Deep ITM": C["profit"], "ITM": C["steel"],
        "ATM": C["gold"], "OTM": C["loss"], "Deep OTM": "#8B0000",
    }

    for cat in df[moneyness_col].unique():
        sub = df[df[moneyness_col] == cat]
        fig.add_trace(go.Scatter(
            x=sub["implied_vol"] * 100,
            y=sub[return_col] * 100,
            mode="markers",
            name=cat,
            marker=dict(size=7, color=color_map.get(cat, C["text3"]),
                       opacity=0.7, line=dict(width=0.5, color=C["border"])),
        ))

    fig.update_layout(
        title="IV vs Return",
        xaxis_title="Implied Volatility %",
        yaxis_title="Realistic Return %",
        height=400,
    )
    return _apply_dark_theme(fig)


def fig_moneyness_dte_heatmap(values_pivot: pd.DataFrame,
                               count_pivot: pd.DataFrame) -> go.Figure:
    """Heatmap: moneyness × DTE, color=avg return, annotated with counts."""
    fig = go.Figure()
    if values_pivot.empty:
        return _apply_dark_theme(fig)

    # Build annotation text
    text_data = []
    for i in range(len(values_pivot.index)):
        row_text = []
        for j in range(len(values_pivot.columns)):
            val = values_pivot.iloc[i, j]
            cnt = count_pivot.iloc[i, j] if not count_pivot.empty else 0
            if pd.notna(val) and pd.notna(cnt):
                row_text.append(f"{val:.0%}\nn={int(cnt)}")
            else:
                row_text.append("")
        text_data.append(row_text)

    fig.add_trace(go.Heatmap(
        z=values_pivot.values * 100,
        x=[str(c) for c in values_pivot.columns],
        y=[str(i) for i in values_pivot.index],
        text=text_data,
        texttemplate="%{text}",
        textfont=dict(size=10, color=C["text"]),
        colorscale=[[0, C["loss"]], [0.5, C["bg"]], [1, C["profit"]]],
        colorbar=dict(title=dict(text="Avg Return %", font=dict(color=C["text2"])),
                     ticksuffix="%", tickfont=dict(color=C["text2"])),
        zmid=0,
    ))

    fig.update_layout(
        title="Moneyness x DTE — Avg Realistic Return",
        xaxis_title="DTE Bucket",
        yaxis_title="Moneyness",
        height=350,
    )
    return _apply_dark_theme(fig)


def fig_hold_window_comparison(window_stats: pd.DataFrame) -> go.Figure:
    """Line chart: hold window vs avg return, one line per category."""
    fig = go.Figure()
    if window_stats.empty:
        return _apply_dark_theme(fig)

    color_cycle = [C["gold"], C["steel"], C["profit"], C["loss"], C["text2"]]

    for i, cat in enumerate(window_stats["category"].unique()):
        sub = window_stats[window_stats["category"] == cat].sort_values("hold_window")
        fig.add_trace(go.Scatter(
            x=sub["hold_window"],
            y=sub["mean_return"] * 100,
            mode="lines+markers",
            name=cat,
            line=dict(color=color_cycle[i % len(color_cycle)], width=2),
            marker=dict(size=8),
        ))

    fig.update_layout(
        title="Return by Hold Window",
        xaxis_title="Hold Window (minutes)",
        yaxis_title="Avg Realistic Return %",
        height=350,
    )
    return _apply_dark_theme(fig)


def fig_spread_cost_by_category(delta_agg: pd.DataFrame,
                                 moneyness_agg: pd.DataFrame) -> go.Figure:
    """Side-by-side bars: avg spread cost by delta bucket and moneyness."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["By Delta", "By Moneyness"],
                        horizontal_spacing=0.12)

    if not delta_agg.empty and "avg_spread_cost" in delta_agg.columns:
        fig.add_trace(go.Bar(
            x=delta_agg["delta_bucket"],
            y=delta_agg["avg_spread_cost"] * 100,
            marker_color=C["loss"],
            name="Delta",
            showlegend=False,
        ), row=1, col=1)

    if not moneyness_agg.empty and "avg_spread_cost" in moneyness_agg.columns:
        order = {v: i for i, v in enumerate(MONEYNESS_ORDER)}
        m = moneyness_agg.copy()
        m["_sort"] = m["moneyness_5"].map(order).fillna(99)
        m = m.sort_values("_sort")
        fig.add_trace(go.Bar(
            x=m["moneyness_5"],
            y=m["avg_spread_cost"] * 100,
            marker_color=C["loss"],
            name="Moneyness",
            showlegend=False,
        ), row=1, col=2)

    fig.update_layout(title="Spread Cost by Category", height=320)
    fig.update_yaxes(title_text="Avg Spread Cost %")
    fig.update_xaxes(tickangle=45, tickfont_size=9)
    return _apply_dark_theme(fig)


def fig_return_distribution(results_df: pd.DataFrame) -> go.Figure:
    """Histogram of realistic returns with breakeven line."""
    fig = go.Figure()
    if results_df.empty:
        return _apply_dark_theme(fig)

    return_col = "realistic_return_pct" if "realistic_return_pct" in results_df.columns else "raw_return_pct"
    returns = results_df[return_col].dropna() * 100

    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=40,
        marker_color=C["gold"],
        opacity=0.8,
    ))

    # Breakeven line
    fig.add_vline(x=0, line_dash="dash", line_color=C["loss"], line_width=2)

    # Stats annotation
    win_pct = (returns > 0).mean()
    fig.add_annotation(
        text=f"Win Rate: {win_pct:.0%} | Mean: {returns.mean():.0f}% | Median: {returns.median():.0f}%",
        xref="paper", yref="paper", x=0.98, y=0.95,
        showarrow=False,
        font=dict(color=C["gold"], size=11),
        align="right",
    )

    fig.update_layout(
        title="Return Distribution (All Contracts)",
        xaxis_title="Realistic Return %",
        yaxis_title="Count",
        height=300,
    )
    return _apply_dark_theme(fig)


def fig_win_rate_by_category(delta_agg: pd.DataFrame,
                              moneyness_agg: pd.DataFrame) -> go.Figure:
    """Side-by-side win rate bars by delta and moneyness."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["By Delta", "By Moneyness"],
                        horizontal_spacing=0.12)

    if not delta_agg.empty and "win_rate" in delta_agg.columns:
        fig.add_trace(go.Bar(
            x=delta_agg["delta_bucket"],
            y=delta_agg["win_rate"] * 100,
            marker_color=C["profit"],
            text=delta_agg["win_rate"].apply(lambda x: f"{x:.0%}"),
            textposition="outside",
            textfont=dict(color=C["text2"], size=9),
            showlegend=False,
        ), row=1, col=1)

    if not moneyness_agg.empty and "win_rate" in moneyness_agg.columns:
        order = {v: i for i, v in enumerate(MONEYNESS_ORDER)}
        m = moneyness_agg.copy()
        m["_sort"] = m["moneyness_5"].map(order).fillna(99)
        m = m.sort_values("_sort")
        fig.add_trace(go.Bar(
            x=m["moneyness_5"],
            y=m["win_rate"] * 100,
            marker_color=C["profit"],
            text=m["win_rate"].apply(lambda x: f"{x:.0%}"),
            textposition="outside",
            textfont=dict(color=C["text2"], size=9),
            showlegend=False,
        ), row=1, col=2)

    fig.update_layout(title="Win Rate by Category", height=320)
    fig.update_yaxes(title_text="Win Rate %", range=[0, 105])
    fig.update_xaxes(tickangle=45, tickfont_size=9)
    return _apply_dark_theme(fig)


def fig_summary_stats_table(summary: dict) -> go.Figure:
    """Summary overview table."""
    fig = go.Figure()

    fig.add_trace(go.Table(
        header=dict(
            values=["Metric", "Value"],
            fill_color=C["elevated"],
            font=dict(color=C["gold"], size=12, family="Outfit, sans-serif"),
            align="left",
            line_color=C["border"],
        ),
        cells=dict(
            values=[
                ["Trades Analyzed", "Contracts Scored", "Win Rate",
                 "Avg Realistic Return", "Median Return",
                 "Avg Spread Cost", "Best Delta Bucket", "Worst Delta Bucket"],
                [
                    f"{summary['total_trades']:,}",
                    f"{summary['total_contracts']:,}",
                    f"{summary['win_rate']:.0%}",
                    f"{summary['avg_return']:.0%}",
                    f"{summary['median_return']:.0%}",
                    f"{summary['avg_spread_cost']:.0%}",
                    summary["best_delta_bucket"],
                    summary["worst_delta_bucket"],
                ],
            ],
            fill_color=C["surface"],
            font=dict(color=C["text"], size=12, family="JetBrains Mono, monospace"),
            align="left",
            line_color=C["border"],
            height=28,
        ),
    ))

    fig.update_layout(
        title="Batch Summary",
        height=310,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return _apply_dark_theme(fig)
