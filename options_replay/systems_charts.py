"""
Plotly chart builders for the Systems tab (hotkey optimization).
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


def fig_hotkey_heatmap(grid_df: pd.DataFrame, metric: str = "edge") -> go.Figure:
    """Heatmap: max_price (y) × max_otm (x), color = edge/return/win_rate.

    Cells with count < 10 are dimmed.
    """
    fig = go.Figure()
    if grid_df.empty:
        return _apply_dark_theme(fig)

    # Pivot for heatmap
    pivot = grid_df.pivot(index="max_price", columns="max_otm", values=metric)
    count_pivot = grid_df.pivot(index="max_price", columns="max_otm", values="count")

    # Build annotation text
    text_data = []
    for i in range(len(pivot.index)):
        row_text = []
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            cnt = count_pivot.iloc[i, j]
            if pd.notna(val) and cnt > 0:
                if metric == "win_rate":
                    row_text.append(f"{val:.0%}\nn={int(cnt)}")
                elif metric == "edge":
                    row_text.append(f"{val:.1f}\nn={int(cnt)}")
                else:
                    row_text.append(f"{val:.1f}%\nn={int(cnt)}")
            else:
                row_text.append("")
        text_data.append(row_text)

    # Labels
    metric_labels = {"edge": "Edge", "avg_return": "Avg Return %", "win_rate": "Win Rate"}
    color_label = metric_labels.get(metric, metric)

    z_values = pivot.values
    if metric == "win_rate":
        z_values = z_values * 100

    fig.add_trace(go.Heatmap(
        z=z_values,
        x=[f"{c}% OTM" for c in pivot.columns],
        y=[f"<${r:.2f}" for r in pivot.index],
        text=text_data,
        texttemplate="%{text}",
        textfont=dict(size=10, color=C["text"]),
        colorscale=[[0, C["loss"]], [0.5, C["bg"]], [1, C["profit"]]],
        colorbar=dict(
            title=dict(text=color_label, font=dict(color=C["text2"])),
            tickfont=dict(color=C["text2"]),
        ),
        zmid=0 if metric != "win_rate" else 50,
    ))

    fig.update_layout(
        title=f"Hotkey Combos — {color_label}",
        xaxis_title="Max % OTM",
        yaxis_title="Max Contract Price",
        height=380,
    )
    return _apply_dark_theme(fig)


def fig_price_otm_grid(grid_stats: pd.DataFrame) -> go.Figure:
    """Heatmap: price_band × otm_band, color = avg return."""
    fig = go.Figure()
    if grid_stats.empty:
        return _apply_dark_theme(fig)

    pivot = grid_stats.pivot(index="price_band", columns="otm_band", values="avg_return")
    count_pivot = grid_stats.pivot(index="price_band", columns="otm_band", values="count")

    text_data = []
    for i in range(len(pivot.index)):
        row_text = []
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            cnt = count_pivot.iloc[i, j]
            if pd.notna(val) and pd.notna(cnt) and cnt > 0:
                row_text.append(f"{val:.0%}\nn={int(cnt)}")
            else:
                row_text.append("")
        text_data.append(row_text)

    fig.add_trace(go.Heatmap(
        z=pivot.values * 100,
        x=[str(c) for c in pivot.columns],
        y=[str(i) for i in pivot.index],
        text=text_data,
        texttemplate="%{text}",
        textfont=dict(size=10, color=C["text"]),
        colorscale=[[0, C["loss"]], [0.5, C["bg"]], [1, C["profit"]]],
        colorbar=dict(
            title=dict(text="Avg Return %", font=dict(color=C["text2"])),
            ticksuffix="%",
            tickfont=dict(color=C["text2"]),
        ),
        zmid=0,
    ))

    fig.update_layout(
        title="Price Band × OTM Band — Avg Return",
        xaxis_title="% Out of the Money",
        yaxis_title="Entry Price Band",
        height=380,
    )
    return _apply_dark_theme(fig)


def fig_hotkey_comparison_table(recommendations: list) -> go.Figure:
    """Table showing recommended hotkeys side by side."""
    fig = go.Figure()
    if not recommendations:
        return _apply_dark_theme(fig)

    metrics = [
        "Filter", "Count", "Avg Return", "Win Rate", "Edge",
        "Spread Cost", "Avg Delta", "Med Entry", "Rationale",
    ]
    values = [[] for _ in metrics]

    for rec in recommendations:
        values[0].append(f"<${rec['max_price']:.2f} & <{rec['max_otm']:.0f}% OTM")
        values[1].append(f"{int(rec['count'])}")
        values[2].append(f"{rec['avg_return']:.1%}")
        values[3].append(f"{rec['win_rate']:.0%}")
        values[4].append(f"{rec['edge']:.1f}")
        values[5].append(f"{rec['avg_spread_cost']:.1%}")
        values[6].append(f"{rec['avg_delta']:.3f}")
        values[7].append(f"${rec['median_entry']:.2f}")
        values[8].append(rec.get("rationale", ""))

    fig.add_trace(go.Table(
        header=dict(
            values=[f"<b>{rec.get('label', '')}</b>" for rec in recommendations],
            fill_color=C["elevated"],
            font=dict(color=C["gold"], size=12, family="Outfit, sans-serif"),
            align="center",
            line_color=C["border"],
        ),
        cells=dict(
            values=[[m] for m in metrics],  # Row headers
            fill_color=C["surface"],
            font=dict(color=C["text2"], size=11, family="Outfit, sans-serif"),
            align="left",
            line_color=C["border"],
            height=26,
        ),
    ))

    # Transpose: rows = metrics, columns = hotkeys
    header_vals = ["Metric"] + [rec.get("label", f"#{i+1}") for i, rec in enumerate(recommendations)]
    cell_values = [metrics]
    for rec in recommendations:
        col = [
            f"<${rec['max_price']:.2f} & <{rec['max_otm']:.0f}% OTM",
            str(int(rec["count"])),
            f"{rec['avg_return']:.1%}",
            f"{rec['win_rate']:.0%}",
            f"{rec['edge']:.1f}",
            f"{rec['avg_spread_cost']:.1%}",
            f"{rec['avg_delta']:.3f}",
            f"${rec['median_entry']:.2f}",
            rec.get("rationale", ""),
        ]
        cell_values.append(col)

    fig.data = []  # Clear the initial attempt
    fig.add_trace(go.Table(
        header=dict(
            values=header_vals,
            fill_color=C["elevated"],
            font=dict(color=C["gold"], size=12, family="Outfit, sans-serif"),
            align="left",
            line_color=C["border"],
        ),
        cells=dict(
            values=cell_values,
            fill_color=C["surface"],
            font=dict(color=C["text"], size=11, family="JetBrains Mono, monospace"),
            align="left",
            line_color=C["border"],
            height=26,
        ),
    ))

    fig.update_layout(
        title="Hotkey Comparison",
        height=340,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return _apply_dark_theme(fig)


def fig_delta_distribution(df: pd.DataFrame, hotkeys: list) -> go.Figure:
    """Overlaid histograms of delta distribution for each hotkey band."""
    fig = go.Figure()
    if df.empty or not hotkeys:
        return _apply_dark_theme(fig)

    colors = [C["gold"], C["steel"], C["profit"]]

    for i, hk in enumerate(hotkeys):
        mask = (df["entry_ask"] < hk["max_price"]) & (df["pct_otm"] < hk["max_otm"])
        subset = df[mask]
        if subset.empty:
            continue
        deltas = subset["delta"].dropna().abs()
        label = f"<${hk['max_price']:.2f}, <{hk['max_otm']:.0f}%"

        fig.add_trace(go.Histogram(
            x=deltas,
            name=f"{hk.get('label', label)} ({label})",
            marker_color=colors[i % len(colors)],
            opacity=0.6,
            nbinsx=20,
        ))

    fig.update_layout(
        title="Delta Distribution by Hotkey",
        xaxis_title="Absolute Delta",
        yaxis_title="Count",
        barmode="overlay",
        height=350,
    )
    return _apply_dark_theme(fig)


def fig_return_distribution_by_hotkey(df: pd.DataFrame, hotkeys: list) -> go.Figure:
    """Overlaid histograms of return distribution for each hotkey band."""
    fig = go.Figure()
    if df.empty or not hotkeys:
        return _apply_dark_theme(fig)

    colors = [C["gold"], C["steel"], C["profit"]]

    for i, hk in enumerate(hotkeys):
        mask = (df["entry_ask"] < hk["max_price"]) & (df["pct_otm"] < hk["max_otm"])
        subset = df[mask]
        if subset.empty:
            continue
        rets = subset["realistic_return_pct"] * 100
        label = f"<${hk['max_price']:.2f}, <{hk['max_otm']:.0f}%"

        fig.add_trace(go.Histogram(
            x=rets,
            name=f"{hk.get('label', label)} ({label})",
            marker_color=colors[i % len(colors)],
            opacity=0.6,
            nbinsx=25,
        ))

    # Breakeven line
    fig.add_vline(x=0, line_dash="dash", line_color=C["loss"], line_width=2)

    fig.update_layout(
        title="Return Distribution by Hotkey",
        xaxis_title="Realistic Return %",
        yaxis_title="Count",
        barmode="overlay",
        height=350,
    )
    return _apply_dark_theme(fig)


def fig_price_vs_return_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter: entry_ask (x) vs realistic return (y), color by OTM band."""
    fig = go.Figure()
    if df.empty:
        return _apply_dark_theme(fig)

    color_map = {
        "0-1%": C["profit"],
        "1-2%": C["steel"],
        "2-3%": C["gold"],
        "3-5%": C["loss"],
        "5%+": "#8B0000",
    }

    for band in df["otm_band"].unique():
        if pd.isna(band):
            continue
        sub = df[df["otm_band"] == band]
        fig.add_trace(go.Scatter(
            x=sub["entry_ask"],
            y=sub["realistic_return_pct"] * 100,
            mode="markers",
            name=str(band),
            marker=dict(
                size=9,
                color=color_map.get(str(band), C["text3"]),
                opacity=0.7,
                line=dict(width=0.5, color=C["border"]),
            ),
            text=sub.apply(
                lambda r: f"{r.get('symbol','')}<br>${r['entry_ask']:.2f}<br>{r['pct_otm']:.1f}% OTM",
                axis=1,
            ),
            hovertemplate="%{text}<br>Return: %{y:.0f}%<extra></extra>",
        ))

    # Breakeven line
    fig.add_hline(y=0, line_dash="dash", line_color=C["loss"], line_width=1)

    fig.update_layout(
        title="Entry Price vs Return (by OTM Band)",
        xaxis_title="Entry Ask ($)",
        yaxis_title="Realistic Return %",
        height=400,
    )
    return _apply_dark_theme(fig)
