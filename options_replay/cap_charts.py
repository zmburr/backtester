"""
Plotly chart builders for the Capitulation tab.
All figures use the Copper & Ink dark theme.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go

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


def fig_target_hit_heatmap(target_df: pd.DataFrame) -> go.Figure:
    """Heatmap: delta bucket (y) × ATR target (x), color = hit rate."""
    fig = go.Figure()
    if target_df.empty:
        return _apply_dark_theme(fig)

    pivot = target_df.pivot(index="delta_bucket", columns="target", values="hit_rate")

    text_data = []
    for i in range(len(pivot.index)):
        row_text = []
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                row_text.append(f"{val:.0%}")
            else:
                row_text.append("")
        text_data.append(row_text)

    fig.add_trace(go.Heatmap(
        z=pivot.values,
        x=[str(c) + " ATR" for c in pivot.columns],
        y=pivot.index.tolist(),
        text=text_data,
        texttemplate="%{text}",
        colorscale=[[0, C["bg"]], [0.5, C["steel"]], [1, C["profit"]]],
        colorbar=dict(title="Hit Rate"),
    ))

    fig.update_layout(
        title="ATR Target Hit Rate by Delta Bucket",
        xaxis_title="Target",
        yaxis_title="Delta Bucket",
    )
    return _apply_dark_theme(fig)


def fig_entry_offset_comparison(offset_df: pd.DataFrame) -> go.Figure:
    """Grouped bar: return, win rate, edge by entry offset."""
    fig = go.Figure()
    if offset_df.empty:
        return _apply_dark_theme(fig)

    offsets = offset_df["entry_offset"].tolist()

    fig.add_trace(go.Bar(
        name="Avg Return",
        x=offsets,
        y=offset_df["avg_return"],
        marker_color=C["gold"],
        text=[f"{v:.0%}" for v in offset_df["avg_return"]],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Win Rate",
        x=offsets,
        y=offset_df["win_rate"],
        marker_color=C["profit"],
        text=[f"{v:.0%}" for v in offset_df["win_rate"]],
        textposition="outside",
    ))

    fig.update_layout(
        title="Entry Offset Comparison",
        xaxis_title="Entry Offset",
        yaxis_title="Value",
        barmode="group",
        yaxis_tickformat=".0%",
    )
    return _apply_dark_theme(fig)


def fig_time_to_target(target_df: pd.DataFrame) -> go.Figure:
    """Bar chart: average time to hit each ATR target by delta bucket."""
    fig = go.Figure()
    if target_df.empty:
        return _apply_dark_theme(fig)

    # Filter to rows that actually hit targets
    hit = target_df[target_df["hit_rate"] > 0].copy()
    if hit.empty:
        return _apply_dark_theme(fig)

    for target_name in sorted(hit["target"].unique()):
        sub = hit[hit["target"] == target_name]
        fig.add_trace(go.Bar(
            name=f"{target_name} ATR",
            x=sub["delta_bucket"].tolist(),
            y=sub["avg_time_to_target_min"],
            text=[f"{v:.0f}m" if pd.notna(v) else "" for v in sub["avg_time_to_target_min"]],
            textposition="outside",
        ))

    fig.update_layout(
        title="Avg Time to Hit ATR Target by Delta",
        xaxis_title="Delta Bucket",
        yaxis_title="Minutes",
        barmode="group",
    )
    return _apply_dark_theme(fig)


def fig_cap_hotkey_heatmap(grid_df: pd.DataFrame, metric: str = "edge") -> go.Figure:
    """Heatmap: max_price (y) × max_otm (x) — same as news but for cap trades."""
    fig = go.Figure()
    if grid_df.empty:
        return _apply_dark_theme(fig)

    pivot = grid_df.pivot(index="max_price", columns="max_otm", values=metric)
    count_pivot = grid_df.pivot(index="max_price", columns="max_otm", values="count")

    text_data = []
    for i in range(len(pivot.index)):
        row_text = []
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            cnt = count_pivot.iloc[i, j]
            if pd.notna(val) and cnt > 0:
                if metric == "edge":
                    row_text.append(f"{val:.2f}\nn={int(cnt)}")
                else:
                    row_text.append(f"{val:.0%}\nn={int(cnt)}")
            else:
                row_text.append("")
        text_data.append(row_text)

    fig.add_trace(go.Heatmap(
        z=pivot.values,
        x=[f"<{c}% OTM" for c in pivot.columns],
        y=[f"<${p:.2f}" for p in pivot.index],
        text=text_data,
        texttemplate="%{text}",
        colorscale=[[0, C["bg"]], [0.5, C["steel"]], [1, C["gold"]]],
        colorbar=dict(title=metric.title()),
    ))

    fig.update_layout(
        title=f"Cap Hotkey Grid — {metric.title()}",
        xaxis_title="Max % OTM",
        yaxis_title="Max Price",
    )
    return _apply_dark_theme(fig)


def fig_iv_analysis(iv_df: pd.DataFrame) -> go.Figure:
    """Bar chart: average IV at entry by delta bucket — key for cap trades."""
    fig = go.Figure()
    if iv_df.empty:
        return _apply_dark_theme(fig)

    fig.add_trace(go.Bar(
        name="Avg IV",
        x=iv_df["delta_bucket"].tolist(),
        y=iv_df["avg_iv"],
        marker_color=C["loss"],
        text=[f"{v:.0%}" if pd.notna(v) else "" for v in iv_df["avg_iv"]],
        textposition="outside",
    ))

    fig.add_trace(go.Bar(
        name="Avg Return",
        x=iv_df["delta_bucket"].tolist(),
        y=iv_df["avg_return"],
        marker_color=C["profit"],
        text=[f"{v:.0%}" for v in iv_df["avg_return"]],
        textposition="outside",
    ))

    fig.update_layout(
        title="IV at Entry vs Return by Delta",
        xaxis_title="Delta Bucket",
        yaxis_title="Value",
        barmode="group",
    )
    return _apply_dark_theme(fig)


def fig_bounce_vs_reversal(df: pd.DataFrame) -> go.Figure:
    """Side-by-side comparison of bounce (long/calls) vs reversal (short/puts)."""
    fig = go.Figure()
    if df.empty or "source" not in df.columns:
        return _apply_dark_theme(fig)

    metrics = []
    for source in ["bounce", "reversal"]:
        sub = df[df["source"] == source]
        if sub.empty:
            continue
        metrics.append({
            "source": source.title(),
            "count": len(sub),
            "avg_return": sub["realistic_return_pct"].mean(),
            "win_rate": (sub["realistic_return_pct"] > 0).mean(),
            "avg_spread": sub["spread_cost_pct"].mean() if "spread_cost_pct" in sub.columns else 0,
        })

    if not metrics:
        return _apply_dark_theme(fig)

    sources = [m["source"] for m in metrics]

    fig.add_trace(go.Bar(name="Avg Return", x=sources,
                         y=[m["avg_return"] for m in metrics],
                         marker_color=C["gold"],
                         text=[f"{m['avg_return']:.0%}" for m in metrics],
                         textposition="outside"))

    fig.add_trace(go.Bar(name="Win Rate", x=sources,
                         y=[m["win_rate"] for m in metrics],
                         marker_color=C["profit"],
                         text=[f"{m['win_rate']:.0%}" for m in metrics],
                         textposition="outside"))

    fig.update_layout(
        title="Bounce vs Reversal Performance",
        barmode="group",
        yaxis_tickformat=".0%",
    )
    return _apply_dark_theme(fig)


def fig_batch_delta_curve(delta_curve_df: pd.DataFrame) -> go.Figure:
    """Batch delta return curve — same style as deep dive but aggregated."""
    fig = go.Figure()
    if delta_curve_df.empty:
        return _apply_dark_theme(fig)

    colors = {30: C["steel"], 60: C["gold"], 120: C["profit"]}
    dashes = {30: "dot", 60: "solid", 120: "solid"}
    widths = {30: 1.5, 60: 2.5, 120: 2}

    for hw in sorted(delta_curve_df["hold_window"].unique()):
        sub = delta_curve_df[delta_curve_df["hold_window"] == hw].sort_values("delta_mid")
        if sub.empty:
            continue

        color = colors.get(int(hw), C["text2"])
        fig.add_trace(go.Scatter(
            x=sub["delta_mid"], y=sub["realistic_return"],
            mode="lines+markers",
            name=f"{int(hw)}min hold",
            line=dict(color=color, dash=dashes.get(int(hw), "solid"),
                      width=widths.get(int(hw), 2)),
            marker=dict(size=6),
            text=[f"n={int(n)}" for n in sub["count"]],
            hovertemplate="Delta: %{x:.2f}<br>Return: %{y:.0%}<br>%{text}<extra></extra>",
        ))

        # Annotate peak
        peak = sub.sort_values("realistic_return", ascending=False).iloc[0]
        fig.add_annotation(
            x=peak["delta_mid"], y=peak["realistic_return"],
            text=f"<b>{peak['realistic_return']:.0%}</b>",
            showarrow=True, arrowhead=2, arrowcolor=color,
            font=dict(size=10, color=color),
            bgcolor=C["surface"], bordercolor=color, borderwidth=1,
        )

    # Sweet spot shading
    fig.add_vrect(x0=0.15, x1=0.30, fillcolor=C["gold"], opacity=0.06, line_width=0,
                  annotation_text="sweet spot", annotation_position="top left",
                  annotation_font_size=9, annotation_font_color=C["text3"])

    fig.update_layout(
        title="Batch Delta Return Curve - Avg Return by Delta Across All Trades",
        xaxis_title="Delta (absolute)", yaxis_title="Avg Realistic Return %",
        yaxis_tickformat=".0%", hovermode="x unified",
    )
    return _apply_dark_theme(fig)


def fig_batch_iv_summary(iv_summary_df: pd.DataFrame) -> go.Figure:
    """Stacked bar showing P&L attribution by delta bucket."""
    fig = go.Figure()
    if iv_summary_df.empty:
        return _apply_dark_theme(fig)

    labels = iv_summary_df["delta_bucket"].tolist()

    fig.add_trace(go.Bar(name="Delta %", x=labels, y=iv_summary_df["avg_delta_pct"],
                         marker_color=C["profit"],
                         hovertemplate="Delta: %{y:.0%}<extra></extra>"))
    fig.add_trace(go.Bar(name="Vega %", x=labels, y=iv_summary_df["avg_vega_pct"],
                         marker_color=C["steel"],
                         hovertemplate="Vega: %{y:.0%}<extra></extra>"))
    fig.add_trace(go.Bar(name="Theta %", x=labels, y=iv_summary_df["avg_theta_pct"],
                         marker_color=C["loss"],
                         hovertemplate="Theta: %{y:.0%}<extra></extra>"))
    fig.add_trace(go.Bar(name="Residual %", x=labels, y=iv_summary_df["avg_residual_pct"],
                         marker_color=C["text3"],
                         hovertemplate="Residual: %{y:.0%}<extra></extra>"))

    fig.update_layout(
        title="P&L Attribution by Delta Bucket (Batch Average)",
        xaxis_title="Delta Bucket", yaxis_title="Avg % of P&L",
        yaxis_tickformat=".0%", barmode="relative",
    )
    return _apply_dark_theme(fig)


def fig_liquidity_grade_comparison(liq_summary_df: pd.DataFrame) -> go.Figure:
    """Grouped bar: grade A/B/C performance comparison."""
    fig = go.Figure()
    if liq_summary_df.empty:
        return _apply_dark_theme(fig)

    grades = liq_summary_df["grade"].tolist()
    grade_colors = {"A": C["profit"], "B": C["gold"], "C": C["loss"]}
    colors = [grade_colors.get(g, C["text3"]) for g in grades]

    fig.add_trace(go.Bar(
        name="Avg Return", x=grades, y=liq_summary_df["avg_return"],
        marker_color=[grade_colors.get(g, C["text3"]) for g in grades],
        text=[f"{r:.0%}\nn={int(n)}" for r, n in
              zip(liq_summary_df["avg_return"], liq_summary_df["count"])],
        textposition="outside", textfont_size=10,
    ))
    fig.add_trace(go.Bar(
        name="Avg Spread Cost", x=grades, y=liq_summary_df["avg_spread_cost"],
        marker_color=[C["text3"]] * len(grades),
        text=[f"{s:.1%}" for s in liq_summary_df["avg_spread_cost"]],
        textposition="outside", textfont_size=10,
    ))

    fig.update_layout(
        title="Return & Spread by Liquidity Grade",
        xaxis_title="Liquidity Grade", yaxis_title="Value",
        yaxis_tickformat=".0%", barmode="group",
    )
    return _apply_dark_theme(fig)
