"""De-SPAC study dashboard - review the flip-trade edge and its patterns.

Run:  streamlit run despac_study/dashboard.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from despac_study.config import DATA_DIR, MASTER_CSV  # noqa: E402

THEMED_CSV = DATA_DIR / "despac_themed.csv"

# validated categorical palette (dataviz reference), dark-surface steps -
# streamlit runs dark here, charts sit on the app surface
CAT = ["#3987e5", "#199e70", "#c98500", "#008300", "#9085e9", "#e66767", "#d55181", "#d95926"]
SEQ = ["#0d366b", "#104281", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#6da7ec", "#9ec5f4"]
ORD = ["#184f95", "#1c5cab", "#256abf", "#3987e5", "#6da7ec"]  # ordinal, 5 steps
INK, INK2 = "#ffffff", "#c3c2b7"
SURFACE, GRID = "#0e1117", "#2b2b33"

st.set_page_config(page_title="De-SPAC Study", layout="wide")


def base_layout(fig, title, ytitle="", xtitle=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK)),
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        font=dict(color=INK2, size=12),
        margin=dict(l=50, r=20, t=48, b=40),
        yaxis=dict(title=ytitle, gridcolor=GRID, zerolinecolor="#4a4a52"),
        xaxis=dict(title=xtitle, gridcolor=SURFACE),
        legend=dict(orientation="h", y=1.08, x=0),
        hovermode="x unified",
    )
    return fig


@st.cache_data
def load():
    themed = pd.read_csv(THEMED_CSV)
    master = pd.read_csv(MASTER_CSV)
    master = master[master["is_spac"] == True].copy()
    for df in (themed, master):
        df["year"] = df["close_date"].astype(str).str[:4]
    return themed, master


themed, master = load()

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Filters")
years = sorted(themed["year"].dropna().unique())
yr = st.sidebar.select_slider("Close year", options=years, value=(years[0], years[-1]))
themes_all = sorted(themed["theme"].fillna("other").unique())
sel_themes = st.sidebar.multiselect("Themes", themes_all, default=themes_all)
red_min, red_max = st.sidebar.slider("Redemption % (missing kept)", 0, 100, (0, 100))
near_trust_only = st.sidebar.checkbox("Near-trust entries only (<= $10.50)")

f = themed[(themed["year"] >= yr[0]) & (themed["year"] <= yr[1])
           & themed["theme"].fillna("other").isin(sel_themes)]
red = f["redemption_pct_best"]
f = f[red.isna() | ((red >= red_min) & (red <= red_max))]
if near_trust_only:
    f = f[f["last_old_close"] <= 10.5]

st.title("De-SPAC flip study")
st.caption(f"{len(f)} tradeable 1:1 flips in view (of {len(themed)} total; {len(master)} confirmed de-SPACs 2020-present). "
           "Entry = last close under the old SPAC ticker.")

tab1, tab2, tab3, tab4 = st.tabs(["The edge", "Redemptions & float", "Themes & eras", "Deal browser"])

# ---------------------------------------------------------------- tab 1
with tab1:
    c = st.columns(5)
    c[0].metric("Flip-day high (med)", f"+{f['flip_high_ret_pct'].median():.1f}%",
                f"{(f['flip_high_ret_pct'] > 0).mean() * 100:.0f}% trade above entry")
    c[1].metric("Flip-day close (med)", f"{f['flip_day_ret_pct'].median():+.1f}%")
    c[2].metric("Day 5 (med)", f"{f['post_flip_ret_5d_pct'].median():+.1f}%")
    c[3].metric("10d max runup (med)", f"+{f['max_runup_10d_pct'].median():.1f}%",
                f"{(f['max_runup_10d_pct'] > 25).mean() * 100:.0f}% run >25%")
    c[4].metric("10d max drawdown (med)", f"{f['max_drawdown_10d_pct'].median():.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        # median outcome path: hold-decay curve
        pts = [("entry", 0.0),
               ("flip open", f["flip_gap_pct"].median()),
               ("flip high", f["flip_high_ret_pct"].median()),
               ("flip close", f["flip_day_ret_pct"].median()),
               ("+1d", f["post_flip_ret_1d_pct"].median()),
               ("+3d", f["post_flip_ret_3d_pct"].median()),
               ("+5d", f["post_flip_ret_5d_pct"].median()),
               ("+10d", f["post_flip_ret_10d_pct"].median())]
        fig = go.Figure(go.Scatter(
            x=[p[0] for p in pts], y=[p[1] for p in pts],
            mode="lines+markers+text", line=dict(color=CAT[0], width=2),
            marker=dict(size=8), text=[f"{p[1]:+.1f}%" for p in pts],
            textposition="top center", textfont=dict(size=11, color=INK),
            name="median return vs entry"))
        fig.add_hline(y=0, line_color="#4a4a52", line_width=1)
        st.plotly_chart(base_layout(fig, "Median outcome path - sell the pop, don't hold", "% vs entry"),
                        use_container_width=True, theme=None)
    with col2:
        d = f["flip_high_ret_pct"].dropna().clip(-40, 100)
        fig = go.Figure(go.Histogram(x=d, nbinsx=40, marker=dict(color=CAT[0])))
        fig.add_vline(x=0, line_color=INK2, line_width=1)
        st.plotly_chart(base_layout(fig, "Flip-day high vs entry (clipped +/-)", "deals", "%"),
                        use_container_width=True, theme=None)

    g = f.groupby("year")
    fig = go.Figure()
    fig.add_bar(x=list(g.groups), y=g["flip_high_ret_pct"].median(), name="flip-day high",
                marker=dict(color=CAT[0]))
    fig.add_bar(x=list(g.groups), y=g["flip_day_ret_pct"].median(), name="flip-day close",
                marker=dict(color=CAT[2]))
    fig.add_bar(x=list(g.groups), y=g["post_flip_ret_5d_pct"].median(), name="+5d",
                marker=dict(color=CAT[5]))
    fig.update_layout(barmode="group")
    st.plotly_chart(base_layout(fig, "Regime by close year (medians) - 2022-24 bled, 2025-26 pays", "%"),
                    use_container_width=True, theme=None)

# ---------------------------------------------------------------- tab 2
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        d = f.dropna(subset=["redemption_pct_best", "max_runup_10d_pct"])
        fig = go.Figure(go.Scatter(
            x=d["redemption_pct_best"], y=d["max_runup_10d_pct"].clip(-60, 300),
            mode="markers", marker=dict(size=7, color=CAT[0], opacity=0.5),
            text=d["flip_ticker"], hovertemplate="%{text}: %{x:.0f}% red, %{y:.0f}% runup<extra></extra>",
            name="deal"))
        fig.add_vline(x=90, line_dash="dot", line_color=CAT[5])
        st.plotly_chart(base_layout(fig, "Redemption % vs 10d max runup (>=90% = microfloat zone)",
                                    "10d max runup % (clipped 300)", "redemption % of public shares"),
                        use_container_width=True, theme=None)
    with col2:
        buckets = ["0-25%", "25-50%", "50-75%", "75-90%", "90-100%"]
        b = pd.cut(f["redemption_pct_best"], [-0.1, 25, 50, 75, 90, 100.1], labels=buckets)
        med = f.groupby(b, observed=True)["max_runup_10d_pct"].median().reindex(buckets)
        n = f.groupby(b, observed=True)["max_runup_10d_pct"].count().reindex(buckets)
        fig = go.Figure(go.Bar(
            x=buckets, y=med, marker=dict(color=ORD),
            text=[f"+{v:.0f}%<br>n={int(c)}" if pd.notna(v) else "" for v, c in zip(med, n)],
            textposition="outside", textfont=dict(color=INK)))
        fig.update_yaxes(range=[0, float(med.max()) * 1.35])
        st.plotly_chart(base_layout(fig, "Median 10d max runup by redemption bucket", "%"),
                        use_container_width=True, theme=None)

    col3, col4 = st.columns(2)
    with col3:
        grp = f.assign(hi=(f["redemption_pct_best"] >= 90)).dropna(subset=["redemption_pct_best"])
        g2 = grp.groupby("hi")
        fig = go.Figure(go.Bar(
            x=["< 90% redeemed", ">= 90% redeemed"],
            y=[(g2["max_runup_10d_pct"].apply(lambda s: (s > 50).mean() * 100)).get(k, 0) for k in [False, True]],
            marker=dict(color=[CAT[0], CAT[5]]),
            text=None))
        fig.update_traces(text=[f"{v:.0f}%" for v in fig.data[0].y], textposition="outside")
        st.plotly_chart(base_layout(fig, "Share of deals running >50% within 10d", "% of deals"),
                        use_container_width=True, theme=None)
    with col4:
        grp = f.assign(nt=f["last_old_close"] <= 10.5)
        g3 = grp.groupby("nt")
        fig = go.Figure()
        fig.add_bar(x=["premium entry (>$10.50)", "near-trust entry (<=$10.50)"],
                    y=[g3["flip_high_ret_pct"].median().get(k) for k in [False, True]],
                    name="flip-day high", marker=dict(color=CAT[0]))
        fig.add_bar(x=["premium entry (>$10.50)", "near-trust entry (<=$10.50)"],
                    y=[g3["max_runup_10d_pct"].median().get(k) for k in [False, True]],
                    name="10d max runup", marker=dict(color=CAT[1]))
        fig.update_layout(barmode="group")
        st.plotly_chart(base_layout(fig, "Entry price vs trust (medians)", "%"),
                        use_container_width=True, theme=None)

# ---------------------------------------------------------------- tab 3
with tab3:
    col1, col2 = st.columns(2)
    with col1:
        tg = f.groupby(f["theme"].fillna("other"))
        med = tg["max_runup_10d_pct"].median()
        n = tg.size()
        keep = n[n >= 3].index
        med = med.loc[keep].sort_values()
        fig = go.Figure(go.Bar(
            x=med.values, y=med.index, orientation="h", marker=dict(color=CAT[0]),
            text=[f"+{v:.0f}%  (n={n[i]})" for i, v in med.items()], textposition="outside",
            textfont=dict(color=INK)))
        fig.update_layout(xaxis_range=[0, max(med.max() * 1.35, 10)])
        base_layout(fig, "Median 10d max runup by theme (n>=3)", "", "%")
        fig.update_layout(margin=dict(l=100))
        st.plotly_chart(fig, use_container_width=True, theme=None)
    with col2:
        pv = f.pivot_table(index="theme", columns="era", values="max_runup_10d_pct", aggfunc="median")
        pvn = f.pivot_table(index="theme", columns="era", values="max_runup_10d_pct", aggfunc="count")
        fig = go.Figure(go.Heatmap(
            z=pv.values, x=pv.columns, y=pv.index, colorscale=[[i / 7, c] for i, c in enumerate(SEQ)],
            text=[[f"{v:.0f}% (n={int(c)})" if pd.notna(v) else "" for v, c in zip(rv, rc)]
                  for rv, rc in zip(pv.values, pvn.fillna(0).values)],
            texttemplate="%{text}", textfont=dict(size=10),
            colorbar=dict(title="med runup %")))
        base_layout(fig, "Theme x era: median 10d max runup - heat moves")
        fig.update_layout(margin=dict(l=100))
        st.plotly_chart(fig, use_container_width=True, theme=None)

    st.markdown("**Announcement validation** - deals whose announcement 2-day high was >= +5% "
                "hold up after the flip; the rest bleed. This is the hold/day-trade switch.")
    ap = f.assign(v=f["ann_2d_high_ret_pct"] >= 5).dropna(subset=["ann_2d_high_ret_pct"])
    g4 = ap.groupby("v")
    fig = go.Figure()
    for k, label, color in [(False, "no announcement pop", CAT[5]), (True, "announcement popped >=5%", CAT[1])]:
        if k in g4.groups:
            fig.add_bar(x=["flip high", "+1d", "+5d", "+10d"],
                        y=[g4["flip_high_ret_pct"].median().get(k), g4["post_flip_ret_1d_pct"].median().get(k),
                           g4["post_flip_ret_5d_pct"].median().get(k), g4["post_flip_ret_10d_pct"].median().get(k)],
                        name=label, marker=dict(color=color))
    fig.update_layout(barmode="group")
    st.plotly_chart(base_layout(fig, "Post-flip path by announcement validation (medians)", "%"),
                    use_container_width=True, theme=None)

# ---------------------------------------------------------------- tab 4
with tab4:
    cols = ["flip_ticker", "old_ticker", "company_name", "theme", "sector", "close_date", "flip_date",
            "redemption_pct_best", "last_old_close", "flip_gap_pct", "flip_high_ret_pct",
            "flip_day_ret_pct", "post_flip_ret_5d_pct", "max_runup_10d_pct", "ann_2d_high_ret_pct"]
    st.dataframe(f[cols].sort_values("max_runup_10d_pct", ascending=False),
                 use_container_width=True, height=420)
    st.download_button("Download filtered CSV", f[cols].to_csv(index=False), "despac_filtered.csv")

    st.markdown("---")
    pick = st.selectbox("Replay a deal (price path around the flip, from cached bars)",
                        [""] + sorted(f["flip_ticker"].dropna().unique()))
    if pick:
        row = f[f["flip_ticker"] == pick].iloc[0]
        from despac_study.fetch_prices import get_daily_bars
        flip_d = str(row["flip_date"])
        start = (pd.Timestamp(flip_d) - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
        end = (pd.Timestamp(flip_d) + pd.Timedelta(days=25)).strftime("%Y-%m-%d")
        fig = go.Figure()
        if isinstance(row.get("old_ticker"), str):
            old = get_daily_bars(row["old_ticker"], start, end)
            if not old.empty:
                fig.add_scatter(x=old.index, y=old["c"], mode="lines", name=row["old_ticker"],
                                line=dict(color=CAT[0], width=2))
        new = get_daily_bars(pick, start, end)
        if not new.empty:
            fig.add_scatter(x=new.index, y=new["c"], mode="lines", name=pick,
                            line=dict(color=CAT[5], width=2))
        if pd.notna(row.get("last_old_close")):
            fig.add_hline(y=row["last_old_close"], line_dash="dot", line_color=INK2,
                          annotation_text=f"entry {row['last_old_close']:.2f}")
        fig.add_vline(x=flip_d, line_dash="dot", line_color=CAT[2],
                      annotation_text=f"flip {flip_d}")
        meta = (f"{row['company_name']}  |  red {row['redemption_pct_best']}%  |  "
                f"gap {row['flip_gap_pct']}%  high {row['flip_high_ret_pct']}%  "
                f"+5d {row['post_flip_ret_5d_pct']}%")
        st.plotly_chart(base_layout(fig, meta, "close ($)"), use_container_width=True, theme=None)
