"""
Backtester Data Dashboard (Streamlit)

Goal: explore `bounce_data.csv` + `reversal_data.csv` and simulate the *same* checklist
logic used in `scripts/generate_report.py`, so you can refine thresholds with real stats.

Run (from repo root):
  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Must be the first Streamlit command in the script.
st.set_page_config(page_title="Trade Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# Custom CSS — financial terminal aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Global typography ─────────────────────────────────────────────────── */
html, body, .stApp, .stApp *, [data-testid="stAppViewContainer"] {
    font-family: 'Outfit', sans-serif !important;
}

/* ── Sidebar ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080c12 0%, #0a0e14 40%, #0d1117 100%) !important;
    border-right: 1px solid #1c2333 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    background: linear-gradient(135deg, #4fc3f7 0%, #81d4fa 50%, #b3e5fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.4rem !important;
}

/* ── Metric cards ──────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #111827 0%, #0d1117 100%);
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 14px 18px;
    transition: border-color 0.2s ease;
}
[data-testid="stMetric"]:hover {
    border-color: #4fc3f7;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 1.6rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Outfit', sans-serif !important;
    text-transform: uppercase !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    opacity: 0.55;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Tabs ──────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    border-bottom: 2px solid #1e293b;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500;
    letter-spacing: 0.03em;
    padding: 10px 24px;
}
.stTabs [aria-selected="true"] {
    border-bottom-color: #4fc3f7 !important;
}

/* ── Headers ───────────────────────────────────────────────────────────── */
h1, h2, h3, h4 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
}
h2, .stSubheader {
    letter-spacing: -0.01em;
}

/* ── DataFrames ────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ── Recommendation badges (injected via HTML) ─────────────────────────── */
.rec-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.78rem;
    letter-spacing: 0.04em;
}
.rec-go       { background: #064e3b; color: #6ee7b7; border: 1px solid #10b981; }
.rec-caution  { background: #78350f; color: #fcd34d; border: 1px solid #f59e0b; }
.rec-nogo     { background: #7f1d1d; color: #fca5a5; border: 1px solid #ef4444; }

/* ── Sidebar divider ───────────────────────────────────────────────────── */
hr {
    border-color: #1e293b !important;
}

/* ── Expander ──────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border-color: #1e293b !important;
    border-radius: 8px;
}

/* ── Page top spacer ───────────────────────────────────────────────────── */
.block-container {
    padding-top: 2rem !important;
}

/* ── Plotly charts transparent bg ──────────────────────────────────────── */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Plotly template — matches the dark terminal theme
# ---------------------------------------------------------------------------
_CHART_COLORS = [
    "#4fc3f7",  # electric blue
    "#6ee7b7",  # mint green
    "#fbbf24",  # amber
    "#f87171",  # coral red
    "#c084fc",  # lavender
    "#fb923c",  # tangerine
    "#67e8f9",  # cyan
    "#a78bfa",  # violet
]

_TRADING_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, Outfit, monospace", color="#c0c8d8", size=12),
        title=dict(font=dict(family="Outfit, sans-serif", size=16, color="#e0e0e8")),
        xaxis=dict(
            gridcolor="#1e293b",
            zerolinecolor="#1e293b",
            linecolor="#1e293b",
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            gridcolor="#1e293b",
            zerolinecolor="#1e293b",
            linecolor="#1e293b",
            tickfont=dict(size=10),
        ),
        colorway=_CHART_COLORS,
        hoverlabel=dict(
            bgcolor="#1e293b",
            bordercolor="#4fc3f7",
            font=dict(family="JetBrains Mono, monospace", size=12, color="#e0e0e8"),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#1e293b",
            font=dict(size=11),
        ),
        margin=dict(l=40, r=20, t=50, b=40),
    )
)
pio.templates["trading_dark"] = _TRADING_TEMPLATE
pio.templates.default = "trading_dark"


# -----------------------------------------------------------------------------
# Path setup (Streamlit runs with script dir on sys.path; add repo root)
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Local imports (after sys.path adjustment)
from analyzers.bounce_scorer import (  # noqa: E402
    SETUP_PROFILES,
    BouncePretrade,
    classify_from_setup_column,
)


# -----------------------------------------------------------------------------
# Reversal pre-trade thresholds (mirrors `scripts/generate_report.py`)
# NOTE: This is intentionally duplicated so the dashboard can run without
# importing the full report script (which has heavier dependencies/side-effects).
# -----------------------------------------------------------------------------
REVERSAL_PRETRADE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "Micro": {
        "pct_from_9ema": 0.80,
        "prior_day_range_atr": 3.0,
        "prior_day_rvol": 2.0,
        "premarket_rvol": 0.05,
        "consecutive_up_days": 3,
        "gap_pct": 0.15,
    },
    "Small": {
        "pct_from_9ema": 0.40,
        "prior_day_range_atr": 2.0,
        "prior_day_rvol": 2.0,
        "premarket_rvol": 0.05,
        "consecutive_up_days": 2,
        "gap_pct": 0.10,
    },
    "Medium": {
        "pct_from_9ema": 0.15,
        "prior_day_range_atr": 1.0,
        "prior_day_rvol": 1.5,
        "premarket_rvol": 0.05,
        "consecutive_up_days": 2,
        "gap_pct": 0.05,
    },
    "Large": {
        "pct_from_9ema": 0.08,
        "prior_day_range_atr": 0.8,
        "prior_day_rvol": 1.0,
        "premarket_rvol": 0.05,
        "consecutive_up_days": 1,
        "gap_pct": 0.00,
    },
    "ETF": {
        "pct_from_9ema": 0.04,
        "prior_day_range_atr": 1.0,
        "prior_day_rvol": 1.5,
        "premarket_rvol": 0.05,
        "consecutive_up_days": 1,
        "gap_pct": 0.00,
    },
}


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _safe_to_numeric(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _ensure_rvol_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns so checklist code can rely on:
      - prior_day_rvol: prior day volume / ADV (x)
      - premarket_rvol: premarket volume / ADV (x)

    Both CSVs already contain `percent_of_vol_one_day_before` and
    `percent_of_premarket_vol` most of the time; we use those as primary source.
    """
    out = df.copy()

    # Prior day RVOL
    if "prior_day_rvol" not in out.columns:
        out["prior_day_rvol"] = np.nan
    out["prior_day_rvol"] = _safe_to_numeric(out["prior_day_rvol"])
    if "percent_of_vol_one_day_before" in out.columns:
        out["prior_day_rvol"] = out["prior_day_rvol"].fillna(
            _safe_to_numeric(out["percent_of_vol_one_day_before"])
        )
    if {"vol_one_day_before", "avg_daily_vol"}.issubset(out.columns):
        out["prior_day_rvol"] = out["prior_day_rvol"].fillna(
            _safe_to_numeric(out["vol_one_day_before"]) / _safe_to_numeric(out["avg_daily_vol"])
        )

    # Premarket RVOL
    if "premarket_rvol" not in out.columns:
        out["premarket_rvol"] = np.nan
    out["premarket_rvol"] = _safe_to_numeric(out["premarket_rvol"])
    if "percent_of_premarket_vol" in out.columns:
        out["premarket_rvol"] = out["premarket_rvol"].fillna(
            _safe_to_numeric(out["percent_of_premarket_vol"])
        )
    if {"premarket_vol", "avg_daily_vol"}.issubset(out.columns):
        out["premarket_rvol"] = out["premarket_rvol"].fillna(
            _safe_to_numeric(out["premarket_vol"]) / _safe_to_numeric(out["avg_daily_vol"])
        )

    return out


def _percentile_rank_rankkind(ref_vals: np.ndarray, x: float) -> Optional[float]:
    """
    SciPy-free equivalent of `scipy.stats.percentileofscore(..., kind="rank")`.
    Returns 0-100 percentile or None.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    ref = np.asarray(ref_vals, dtype=float)
    ref = ref[~np.isnan(ref)]
    if ref.size == 0:
        return None
    less = float(np.sum(ref < float(x)))
    equal = float(np.sum(ref == float(x)))
    return 100.0 * (less + 0.5 * equal) / float(ref.size)


def _compute_bounce_intensity_row(
    row: Mapping[str, Any],
    ref_df: pd.DataFrame,
    weights: Dict[str, float],
) -> float:
    """
    Implements the same Bounce Intensity idea as `scripts/generate_report.py`.

    Higher = more extreme (better bounce candidate).
    """
    spec: List[Tuple[str, bool]] = [
        ("selloff_total_pct", False),  # deeper selloff = better => invert
        ("consecutive_down_days", True),
        ("prior_day_rvol", True),
        ("pct_off_30d_high", False),  # further off high (more negative) = better => invert
        ("gap_pct", False),  # larger gap down (more negative) = better => invert
    ]

    weighted_sum = 0.0
    total_weight = 0.0

    for col, higher_is_better in spec:
        w = float(weights.get(col, 0.0))
        if w <= 0:
            continue

        x = row.get(col)
        ref_vals = ref_df[col].to_numpy(dtype=float, copy=False) if col in ref_df.columns else np.array([])
        raw = _percentile_rank_rankkind(ref_vals, float(x)) if x is not None else None
        if raw is None:
            continue

        pctile = raw if higher_is_better else (100.0 - raw)
        weighted_sum += pctile * w
        total_weight += w

    return float(round(weighted_sum / total_weight, 1)) if total_weight > 0 else 0.0


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_bounce_df() -> pd.DataFrame:
    path = REPO_ROOT / "data" / "bounce_data.csv"
    df = pd.read_csv(path).dropna(subset=["ticker", "date"]).copy()

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["cap"] = df.get("cap", "Medium")
    df["cap"] = df["cap"].fillna("Medium").astype(str).str.strip()

    df["date_dt"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")

    if "Setup" in df.columns:
        df["setup_profile"] = df["Setup"].astype(str).apply(classify_from_setup_column)
    else:
        df["setup_profile"] = "Unknown"

    df = _ensure_rvol_columns(df)

    # Match bounce_scorer.py behavior: pnl = bounce_open_close_pct * 100
    df["pnl"] = _safe_to_numeric(df.get("bounce_open_close_pct")) * 100.0

    return df


@st.cache_data(show_spinner=False)
def load_reversal_df() -> pd.DataFrame:
    path = REPO_ROOT / "data" / "reversal_data.csv"
    df = pd.read_csv(path).dropna(subset=["ticker", "date"]).copy()

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["cap"] = df.get("cap", "Medium")
    df["cap"] = df["cap"].fillna("Medium").astype(str).str.strip()

    df["date_dt"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")

    df = _ensure_rvol_columns(df)

    # Simple "short P&L proxy": positive when open->close is negative (down day)
    df["pnl"] = -_safe_to_numeric(df.get("reversal_open_close_pct")) * 100.0

    return df


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------
def score_reversal_pretrade_row(
    row: Mapping[str, Any],
    thresholds: Dict[str, float],
) -> Tuple[int, str, List[str]]:
    failed: List[str] = []
    score = 0

    def _get_float(key: str) -> Optional[float]:
        v = row.get(key)
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            return float(v)
        except Exception:
            return None

    # 4 direct criteria
    checks: List[Tuple[str, str, str]] = [
        ("pct_from_9ema", "pct_from_9ema", ">="),
        ("prior_day_range_atr", "prior_day_range_atr", ">="),
        ("consecutive_up_days", "consecutive_up_days", ">="),
        ("gap_pct", "gap_pct", ">="),
    ]

    for name, key, op in checks:
        actual = _get_float(key)
        thresh = float(thresholds[name])
        passed = actual is not None and (actual >= thresh if op == ">=" else actual <= thresh)
        if passed:
            score += 1
        else:
            failed.append(name)

    # Vol signal: prior OR premarket meets threshold
    prior_rvol = _get_float("prior_day_rvol")
    pm_rvol = _get_float("premarket_rvol")
    prior_pass = prior_rvol is not None and prior_rvol >= float(thresholds["prior_day_rvol"])
    pm_pass = pm_rvol is not None and pm_rvol >= float(thresholds["premarket_rvol"])
    vol_pass = prior_pass or pm_pass
    if vol_pass:
        score += 1
    else:
        failed.append("vol_signal")

    if score >= 4:
        rec = "GO"
    elif score == 3:
        rec = "CAUTION"
    else:
        rec = "NO-GO"

    return score, rec, failed


def score_bounce_pretrade_override_row(
    row: Mapping[str, Any],
    required_selloff_pct: int,
    required_down_days: int,
    required_off_30d_high_pct: int,
    required_gap_down_pct: int,
    required_prior_day_rvol: float,
    required_premarket_rvol: float,
) -> Tuple[int, str, List[str]]:
    """
    What-if version of the BouncePretrade checklist (5 criteria).

    Notes:
      - selloff_total_pct, pct_off_30d_high, gap_pct are negative for drawdowns.
      - We accept *required* inputs as positive percentages and convert to negatives.
    """
    failed: List[str] = []
    score = 0

    def _f(key: str) -> Optional[float]:
        v = row.get(key)
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            return float(v)
        except Exception:
            return None

    selloff_thr = -abs(required_selloff_pct) / 100.0
    off30_thr = -abs(required_off_30d_high_pct) / 100.0
    gap_thr = -abs(required_gap_down_pct) / 100.0

    # 4 direct criteria
    criteria: List[Tuple[str, bool]] = [
        ("selloff_total_pct", (_f("selloff_total_pct") is not None and _f("selloff_total_pct") <= selloff_thr)),
        ("consecutive_down_days", (_f("consecutive_down_days") is not None and _f("consecutive_down_days") >= required_down_days)),
        ("pct_off_30d_high", (_f("pct_off_30d_high") is not None and _f("pct_off_30d_high") <= off30_thr)),
        ("gap_pct", (_f("gap_pct") is not None and _f("gap_pct") <= gap_thr)),
    ]

    for name, passed in criteria:
        if passed:
            score += 1
        else:
            failed.append(name)

    # Vol signal
    prior_rvol = _f("prior_day_rvol")
    pm_rvol = _f("premarket_rvol")
    vol_pass = (prior_rvol is not None and prior_rvol >= required_prior_day_rvol) or (
        pm_rvol is not None and pm_rvol >= required_premarket_rvol
    )
    if vol_pass:
        score += 1
    else:
        failed.append("vol_expansion")

    if score >= 4:
        rec = "GO"
    elif score == 3:
        rec = "CAUTION"
    else:
        rec = "NO-GO"

    return score, rec, failed


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------
def _filter_by_multiselect(df: pd.DataFrame, col: str, selected: List[str]) -> pd.DataFrame:
    if not selected or col not in df.columns:
        return df
    return df[df[col].isin(selected)]


def _date_range_filter(df: pd.DataFrame, date_col: str, start, end) -> pd.DataFrame:
    if date_col not in df.columns:
        return df
    out = df.copy()
    out = out[out[date_col].notna()]
    if start is not None:
        out = out[out[date_col].dt.date >= start]
    if end is not None:
        out = out[out[date_col].dt.date <= end]
    return out


def _summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    pnl = _safe_to_numeric(df.get("pnl"))
    pnl = pnl.dropna()
    if len(pnl) == 0:
        return {"trades": len(df), "win_rate": None, "avg_pnl": None, "median_pnl": None}
    return {
        "trades": int(len(df)),
        "win_rate": float(round((pnl > 0).mean() * 100.0, 1)),
        "avg_pnl": float(round(pnl.mean(), 2)),
        "median_pnl": float(round(pnl.median(), 2)),
    }


def _group_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()
    tmp = df.copy()
    tmp["pnl_num"] = _safe_to_numeric(tmp.get("pnl"))
    g = (
        tmp.groupby(group_col, dropna=False)
        .agg(
            trades=("ticker", "count"),
            win_rate=("pnl_num", lambda s: float((s > 0).mean() * 100.0) if len(s.dropna()) else np.nan),
            avg_pnl=("pnl_num", "mean"),
            median_pnl=("pnl_num", "median"),
        )
        .reset_index()
    )
    g["win_rate"] = g["win_rate"].round(1)
    g["avg_pnl"] = g["avg_pnl"].round(2)
    g["median_pnl"] = g["median_pnl"].round(2)
    return g.sort_values(group_col)


def _key_metrics_selector(defaults: List[str], df: pd.DataFrame) -> List[str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    options = [c for c in defaults if c in df.columns] + [c for c in numeric_cols if c not in defaults]
    return options


def _rec_badge_html(rec: str) -> str:
    """Return a styled HTML badge for GO / CAUTION / NO-GO."""
    cls = {"GO": "rec-go", "CAUTION": "rec-caution", "NO-GO": "rec-nogo"}.get(
        str(rec).upper(), "rec-nogo"
    )
    return f'<span class="rec-badge {cls}">{rec}</span>'


def _render_rec_summary(scored_df: pd.DataFrame) -> None:
    """Render GO / CAUTION / NO-GO badges with trade counts and P&L."""
    if "checklist_rec" not in scored_df.columns:
        return
    for rec in ["GO", "CAUTION", "NO-GO"]:
        sub = scored_df[scored_df["checklist_rec"] == rec]
        if sub.empty:
            continue
        pnl = _safe_to_numeric(sub["pnl"]).dropna()
        wr = f"{(pnl > 0).mean() * 100:.0f}%" if len(pnl) > 0 else "—"
        avg = f"{pnl.mean():+.1f}%" if len(pnl) > 0 else "—"
        badge = _rec_badge_html(rec)
        st.markdown(
            f"{badge} &nbsp; **{len(sub)}** trades &nbsp;·&nbsp; WR {wr} &nbsp;·&nbsp; Avg {avg}",
            unsafe_allow_html=True,
        )


def _cumulative_pnl_chart(
    df: pd.DataFrame,
    date_col: str = "date_dt",
    pnl_col: str = "pnl",
    title: str = "Equity Curve (cumulative P&L %)",
    color_col: Optional[str] = None,
) -> Optional[go.Figure]:
    """Cumulative P&L line chart ordered by trade date."""
    tmp = df.dropna(subset=[date_col, pnl_col]).sort_values(date_col).copy()
    if tmp.empty:
        return None
    tmp["_cum_pnl"] = _safe_to_numeric(tmp[pnl_col]).cumsum()
    tmp["_trade_num"] = range(1, len(tmp) + 1)

    hover = ["ticker", "date", "cap"] if "cap" in tmp.columns else ["ticker", "date"]
    hover = [h for h in hover if h in tmp.columns]

    fig = px.area(
        tmp,
        x="_trade_num",
        y="_cum_pnl",
        color=color_col if color_col and color_col in tmp.columns else None,
        title=title,
        hover_data=hover + [pnl_col],
        labels={"_trade_num": "Trade #", "_cum_pnl": "Cumulative P&L %"},
    )
    fig.update_traces(line=dict(width=2))
    # Shade positive green, negative red
    if color_col is None:
        fig.update_traces(
            fillcolor="rgba(79,195,247,0.08)",
            line_color="#4fc3f7",
        )
    fig.update_layout(showlegend=bool(color_col))
    return fig


def _rolling_winrate_chart(
    df: pd.DataFrame,
    date_col: str = "date_dt",
    pnl_col: str = "pnl",
    window: int = 20,
    title: str = "Rolling Win Rate",
) -> Optional[go.Figure]:
    """Rolling win-rate line chart (trades ordered by date)."""
    tmp = df.dropna(subset=[date_col, pnl_col]).sort_values(date_col).copy()
    if len(tmp) < 5:
        return None
    tmp["_win"] = (_safe_to_numeric(tmp[pnl_col]) > 0).astype(float)
    tmp["_rolling_wr"] = tmp["_win"].rolling(window=window, min_periods=5).mean() * 100
    tmp["_trade_num"] = range(1, len(tmp) + 1)

    fig = px.line(
        tmp,
        x="_trade_num",
        y="_rolling_wr",
        title=f"{title} ({window}-trade window)",
        hover_data=[c for c in ["ticker", "date", "cap"] if c in tmp.columns],
        labels={"_trade_num": "Trade #", "_rolling_wr": "Win Rate %"},
    )
    fig.update_traces(line=dict(width=2, color="#6ee7b7"))
    fig.add_hline(y=50, line_dash="dot", line_color="#4b5563", annotation_text="50%")
    return fig


def _metric_threshold_sweep(
    df: pd.DataFrame,
    metric: str,
    direction: str,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    """
    Sweep a single threshold for a metric and compute win-rate / pnl stats for each cutoff.

    direction:
      - '>=' keeps rows where metric >= threshold
      - '<=' keeps rows where metric <= threshold
    """
    if metric not in df.columns:
        return pd.DataFrame()

    base_n = int(len(df))
    if base_n == 0:
        return pd.DataFrame()

    metric_num = _safe_to_numeric(df[metric])

    rows: List[Dict[str, Any]] = []
    for thr in thresholds:
        try:
            thr_f = float(thr)
        except Exception:
            continue

        if direction == "<=":
            subset = df[metric_num <= thr_f]
        else:
            subset = df[metric_num >= thr_f]

        stats = _summary_stats(subset)
        rows.append(
            {
                "threshold": thr_f,
                "coverage_pct": round((len(subset) / base_n) * 100.0, 1),
                **stats,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("threshold").reset_index(drop=True)
    return out


def render_metric_optimizer(
    df: pd.DataFrame,
    metric_defaults: List[str],
    default_directions: Optional[Dict[str, str]] = None,
    *,
    key_prefix: str,
) -> None:
    """
    Interactive optimizer for a *single* metric threshold.

    Renders directly (no wrapper); caller places it inside a tab or section.
    Respects the current filtered dataset so it stacks with sidebar filters.
    """
    default_directions = default_directions or {}

    if df.empty:
        st.warning("No rows to optimize (empty filter result).")
        return

    # Optional cap filter (optimizer-only)
    df_opt = df
    if "cap" in df.columns:
        caps = sorted([str(c) for c in df["cap"].dropna().unique()])
        cap_choice = st.selectbox(
            "Cap (optimizer)",
            options=["All"] + caps,
            index=0,
            key=f"{key_prefix}_cap",
            help="Narrows the sweep to a single cap. The sidebar Cap filter still applies too.",
        )
        if cap_choice != "All":
            df_opt = df[df["cap"].astype(str) == cap_choice]
            st.caption(f"Optimizer subset: cap={cap_choice} (n={len(df_opt)})")

    if df_opt.empty:
        st.warning("No rows in the optimizer subset.")
        return

    numeric_cols = [c for c in df_opt.columns if pd.api.types.is_numeric_dtype(df_opt[c])]
    metric_options = _key_metrics_selector(metric_defaults, df_opt)
    metric_options = [m for m in metric_options if m in numeric_cols]
    if not metric_options:
        st.warning("No numeric metrics available in this dataset.")
        return

    c_metric, c_dir = st.columns([3, 1])
    metric = c_metric.selectbox("Metric", metric_options, index=0, key=f"{key_prefix}_metric")
    default_dir = default_directions.get(metric, ">=")
    direction = c_dir.radio(
        "Keep rows where …",
        [">=", "<="],
        index=0 if default_dir == ">=" else 1,
        horizontal=True,
        key=f"{key_prefix}_dir",
    )

    # Threshold candidates (quantiles over the observed values)
    vals = _safe_to_numeric(df_opt[metric]).dropna()
    if len(vals) == 0:
        st.warning("Selected metric has no numeric values after filtering.")
        return

    c1, c2, c3 = st.columns(3)
    sweep_points = c1.slider("Sweep points", min_value=10, max_value=250, value=60, step=5, key=f"{key_prefix}_points")
    min_trades = c2.slider(
        "Min trades",
        min_value=1,
        max_value=max(1, min(500, len(df_opt))),
        value=min(20, len(df_opt)),
        step=1,
        key=f"{key_prefix}_min_trades",
    )
    objective = c3.selectbox(
        "Objective",
        ["win_rate", "avg_pnl", "median_pnl", "trades"],
        index=0,
        key=f"{key_prefix}_obj",
    )

    qs = np.linspace(0.0, 1.0, int(sweep_points))
    thresholds = np.unique(np.quantile(vals.to_numpy(dtype=float), qs))
    # If metric looks integer-like, keep integer thresholds (avoids confusing 2.5-day cutoffs).
    try:
        is_int_like = bool(np.all(np.isclose(vals.to_numpy(dtype=float), np.round(vals.to_numpy(dtype=float)))))
    except Exception:
        is_int_like = False
    if is_int_like:
        thresholds = np.unique(np.round(thresholds))

    sweep = _metric_threshold_sweep(df_opt, metric=metric, direction=direction, thresholds=thresholds)
    if sweep.empty:
        st.warning("No sweep results.")
        return

    feasible = sweep[sweep["trades"] >= int(min_trades)].copy()
    if feasible.empty:
        st.warning("No thresholds meet the minimum trade constraint.")
    else:
        best = feasible.sort_values(objective, ascending=False).iloc[0].to_dict()
        st.success(
            f"**Best (by {objective})**: threshold **{best['threshold']:.6g}** "
            f"→ {int(best['trades'])} trades, "
            f"win rate {best['win_rate'] if best['win_rate'] is not None else '—'}%, "
            f"avg P&L {best['avg_pnl'] if best['avg_pnl'] is not None else '—'}%"
        )

    # Charts side-by-side where possible
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(px.line(sweep, x="threshold", y="win_rate", title="Win rate (%) vs threshold"), width="stretch")
    with chart_right:
        st.plotly_chart(px.line(sweep, x="threshold", y="avg_pnl", title="Avg P&L (%) vs threshold"), width="stretch")
    st.plotly_chart(px.line(sweep, x="threshold", y="trades", title="Trades vs threshold"), width="stretch")

    # Table
    with st.expander("Threshold details", expanded=False):
        table = feasible.sort_values(objective, ascending=False).head(40) if not feasible.empty else sweep.head(40)
        st.dataframe(table, width="stretch", height=380)
        st.download_button(
            "Download sweep CSV",
            data=sweep.to_csv(index=False).encode("utf-8"),
            file_name=f"{key_prefix}_metric_sweep.csv",
        )

# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
def _compute_delta(df: pd.DataFrame, pnl_col: str = "pnl", recent_n: int = 20) -> Optional[str]:
    """Compare last N trades' win rate to overall win rate; return delta string."""
    pnl = _safe_to_numeric(df.get(pnl_col)).dropna()
    if len(pnl) < recent_n + 5:
        return None
    recent = pnl.tail(recent_n)
    overall_wr = (pnl > 0).mean() * 100
    recent_wr = (recent > 0).mean() * 100
    diff = recent_wr - overall_wr
    return f"{diff:+.0f}pp (last {recent_n})"


def render_overview(bounce_df: pd.DataFrame, reversal_df: pd.DataFrame) -> None:
    st.markdown(
        '<span style="color:#64748b;font-size:0.9rem">Select a strategy in the sidebar. '
        "Each page has **Threshold Optimizer**, **Metric Explorer**, and **Checklist Simulator**.</span>",
        unsafe_allow_html=True,
    )

    # ── Strategy summary cards ────────────────────────────────────────────
    c1, c2 = st.columns(2)

    for col, label, sdf in [
        (c1, "Bounce (Long)", bounce_df),
        (c2, "Reversal (Short)", reversal_df),
    ]:
        with col:
            st.markdown(f"#### {label}")
            stats = _summary_stats(sdf)
            delta_str = _compute_delta(sdf)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Trades", stats["trades"])
            m2.metric(
                "Win Rate",
                "—" if stats["win_rate"] is None else f"{stats['win_rate']:.1f}%",
                delta=delta_str,
            )
            m3.metric(
                "Avg P&L",
                "—" if stats["avg_pnl"] is None else f"{stats['avg_pnl']:+.2f}%",
            )
            m4.metric(
                "Median",
                "—" if stats["median_pnl"] is None else f"{stats['median_pnl']:+.2f}%",
            )
            st.dataframe(
                _group_stats(sdf, "cap"),
                width="stretch",
                hide_index=True,
                height=200,
            )

    st.divider()

    # ── Equity curves ─────────────────────────────────────────────────────
    st.markdown("#### Equity Curves")
    ec1, ec2 = st.columns(2)
    with ec1:
        fig = _cumulative_pnl_chart(bounce_df, title="Bounce — Cumulative P&L %")
        if fig:
            fig.update_traces(fillcolor="rgba(110,231,183,0.08)", line_color="#6ee7b7")
            st.plotly_chart(fig, width="stretch")
    with ec2:
        fig = _cumulative_pnl_chart(reversal_df, title="Reversal — Cumulative P&L %")
        if fig:
            st.plotly_chart(fig, width="stretch")

    # ── Rolling win-rate ──────────────────────────────────────────────────
    wr1, wr2 = st.columns(2)
    with wr1:
        fig = _rolling_winrate_chart(bounce_df, title="Bounce Win Rate")
        if fig:
            st.plotly_chart(fig, width="stretch")
    with wr2:
        fig = _rolling_winrate_chart(reversal_df, title="Reversal Win Rate")
        if fig:
            st.plotly_chart(fig, width="stretch")

    st.divider()

    # ── Recent trades ─────────────────────────────────────────────────────
    st.markdown("#### Recent Trades")
    rt1, rt2 = st.columns(2)
    with rt1:
        st.caption("Bounce (last 10)")
        _show_recent(bounce_df, extra_cols=["setup_profile"])
    with rt2:
        st.caption("Reversal (last 10)")
        _show_recent(reversal_df, extra_cols=["setup"])


def _show_recent(df: pd.DataFrame, n: int = 10, extra_cols: Optional[List[str]] = None) -> None:
    """Display a compact recent-trades table with colored P&L."""
    base_cols = ["date", "ticker", "cap"]
    if extra_cols:
        base_cols += [c for c in extra_cols if c in df.columns]
    base_cols.append("pnl")
    avail = [c for c in base_cols if c in df.columns]

    recent = df.dropna(subset=["date_dt"]).sort_values("date_dt", ascending=False).head(n)
    if recent.empty:
        st.info("No trades.")
        return

    display = recent[avail].copy()
    if "pnl" in display.columns:
        display["pnl"] = display["pnl"].apply(
            lambda v: f"{v:+.1f}%" if pd.notna(v) else "—"
        )
    st.dataframe(display, width="stretch", hide_index=True, height=360)


def render_bounce(bounce_df: pd.DataFrame) -> None:
    st.subheader("Bounce (Long)")

    # Sidebar filters
    with st.sidebar:
        st.markdown("### Filters")
        grades = sorted([g for g in bounce_df.get("trade_grade", pd.Series(dtype=str)).dropna().unique()])
        caps = sorted([c for c in bounce_df.get("cap", pd.Series(dtype=str)).dropna().unique()])
        profiles = sorted([p for p in bounce_df.get("setup_profile", pd.Series(dtype=str)).dropna().unique()])

        sel_caps = st.multiselect("Cap", caps, default=caps)
        sel_grades = st.multiselect("Grade", grades, default=grades)
        sel_profiles = st.multiselect("Setup profile", profiles, default=profiles)

        dmin = bounce_df["date_dt"].min()
        dmax = bounce_df["date_dt"].max()
        with st.expander("Date range", expanded=False):
            start_end = st.date_input(
                "Range",
                value=(dmin.date() if pd.notna(dmin) else None, dmax.date() if pd.notna(dmax) else None),
                label_visibility="collapsed",
            )
            start_date = start_end[0] if isinstance(start_end, (list, tuple)) and len(start_end) > 0 else None
            end_date = start_end[1] if isinstance(start_end, (list, tuple)) and len(start_end) > 1 else None

    df = bounce_df.copy()
    df = _filter_by_multiselect(df, "trade_grade", sel_grades)
    df = _filter_by_multiselect(df, "cap", sel_caps)
    df = _filter_by_multiselect(df, "setup_profile", sel_profiles)
    df = _date_range_filter(df, "date_dt", start_date, end_date)

    if df.empty:
        st.warning("No bounce trades match the current filters.")
        return

    # Summary stats
    stats = _summary_stats(df)
    delta_str = _compute_delta(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades", stats["trades"])
    c2.metric("Win Rate", "—" if stats["win_rate"] is None else f"{stats['win_rate']:.1f}%", delta=delta_str)
    c3.metric("Avg P&L", "—" if stats["avg_pnl"] is None else f"{stats['avg_pnl']:+.2f}%")
    c4.metric("Median P&L", "—" if stats["median_pnl"] is None else f"{stats['median_pnl']:+.2f}%")

    st.caption("P&L = open→close % on bounce day. Win = P&L > 0.")

    # ── Equity curve + rolling WR ─────────────────────────────────────────
    ch1, ch2 = st.columns(2)
    with ch1:
        fig = _cumulative_pnl_chart(df, title="Bounce Equity Curve")
        if fig:
            fig.update_traces(fillcolor="rgba(110,231,183,0.08)", line_color="#6ee7b7")
            st.plotly_chart(fig, width="stretch")
    with ch2:
        fig = _rolling_winrate_chart(df, title="Bounce Win Rate")
        if fig:
            st.plotly_chart(fig, width="stretch")

    # Breakdowns (expanded by default — this is core data)
    with st.expander("Stats by cap / setup profile", expanded=True):
        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown("**By cap**")
            st.dataframe(_group_stats(df, "cap"), width="stretch", hide_index=True, height=200)
        with bc2:
            st.markdown("**By setup profile**")
            st.dataframe(_group_stats(df, "setup_profile"), width="stretch", hide_index=True, height=200)

    metric_defaults = [
        "selloff_total_pct",
        "consecutive_down_days",
        "pct_off_30d_high",
        "gap_pct",
        "prior_day_rvol",
        "premarket_rvol",
        "bounce_open_high_pct",
        "bounce_open_close_pct",
    ]

    # Tabs for different analysis modes
    tab_optimizer, tab_explore, tab_checklist = st.tabs(["Threshold Optimizer", "Metric Explorer", "Checklist Simulator"])

    with tab_optimizer:
        st.markdown("**Find optimal threshold for any metric by cap.** Sweeps values and shows win rate / avg P&L at each cutoff.")
        render_metric_optimizer(
            df,
            metric_defaults=metric_defaults,
            default_directions={
                "selloff_total_pct": "<=",
                "pct_off_30d_high": "<=",
                "gap_pct": "<=",
                "consecutive_down_days": ">=",
                "prior_day_rvol": ">=",
                "premarket_rvol": ">=",
            },
            key_prefix="bounce_opt",
        )

    with tab_explore:
        st.markdown("**Explore metric distributions and their relationship to P&L.**")
        metric_options = _key_metrics_selector(metric_defaults, df)
        metric = st.selectbox("Metric", metric_options, index=0, key="bounce_explore_metric")

        if metric in df.columns:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(
                    df,
                    x=metric,
                    nbins=35,
                    hover_data=["ticker", "date", "cap", "setup_profile", "trade_grade", "pnl"],
                    title=f"Distribution: {metric}",
                )
                st.plotly_chart(fig, width="stretch")

            with c2:
                if "pnl" in df.columns and pd.api.types.is_numeric_dtype(df["pnl"]):
                    scatter = px.scatter(
                        df,
                        x=metric,
                        y="pnl",
                        color="setup_profile" if "setup_profile" in df.columns else None,
                        hover_data=["ticker", "date", "cap", "trade_grade"],
                        title=f"{metric} vs P&L",
                    )
                    st.plotly_chart(scatter, width="stretch")

    with tab_checklist:
        st.markdown("**Test checklist logic on historical trades.** See how GO/CAUTION/NO-GO performs.")
        mode = st.radio(
            "Mode",
            ["Current thresholds", "Custom thresholds"],
            horizontal=True,
            key="bounce_checklist_mode",
        )

        if mode == "Current thresholds":
            checker = BouncePretrade()
            rows: List[Dict[str, Any]] = []
            for _, r in df.iterrows():
                cap = str(r.get("cap") or "Medium")
                setup = str(r.get("setup_profile") or "GapFade_strongstock")
                res = checker.validate(
                    ticker=str(r.get("ticker", "")),
                    metrics=r.to_dict(),
                    force_setup=setup,
                    cap=cap,
                )
                failed = [i.name for i in res.items if not i.passed]
                rows.append(
                    {
                        "checklist_score": int(res.score),
                        "checklist_max": int(res.max_score),
                        "checklist_rec": str(res.recommendation),
                        "checklist_failed": ", ".join(failed) if failed else "PERFECT",
                    }
                )
            scored = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

            st.markdown("**Results by recommendation:**")
            _render_rec_summary(scored)
            st.dataframe(_group_stats(scored, "checklist_rec"), width="stretch")

            box = px.box(
                scored,
                x="checklist_rec",
                y="pnl",
                points="all",
                hover_data=["ticker", "date", "cap", "setup_profile", "trade_grade", "checklist_score"],
                title="P&L by recommendation",
                color="checklist_rec",
                color_discrete_map={"GO": "#6ee7b7", "CAUTION": "#fbbf24", "NO-GO": "#f87171"},
            )
            box.update_layout(showlegend=False)
            st.plotly_chart(box, width="stretch")

            with st.expander("All trades", expanded=False):
                st.dataframe(
                    scored.sort_values(["checklist_rec", "pnl"], ascending=[True, False]),
                    width="stretch",
                    height=350,
                )
                csv_bytes = scored.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", data=csv_bytes, file_name="bounce_scored.csv")

        else:
            c1, c2 = st.columns(2)
            tune_profile = c1.selectbox("Setup profile", sorted(SETUP_PROFILES.keys()), index=0)
            tune_cap = c2.selectbox("Cap", sorted(df["cap"].dropna().unique()), index=0)

            profile = SETUP_PROFILES[tune_profile]
            base = {
                "selloff_total_pct": float(profile.get_threshold("selloff_total_pct", tune_cap)),
                "consecutive_down_days": int(profile.get_threshold("consecutive_down_days", tune_cap)),
                "pct_off_30d_high": float(profile.get_threshold("pct_off_30d_high", tune_cap)),
                "gap_pct": float(profile.get_threshold("gap_pct", tune_cap)),
                "prior_day_rvol": float(profile.get_threshold("vol_expansion", tune_cap)),
                "premarket_rvol": float(profile.vol_premarket),
            }

            st.markdown("**Adjust thresholds:**")
            c1, c2, c3 = st.columns(3)
            required_selloff = c1.slider("Selloff depth (%)", 0, 90, int(round(abs(base["selloff_total_pct"]) * 100)), 1)
            required_off30 = c2.slider("Off 30d high (%)", 0, 90, int(round(abs(base["pct_off_30d_high"]) * 100)), 1)
            required_gap = c3.slider("Gap down (%)", 0, 50, int(round(abs(base["gap_pct"]) * 100)), 1)

            c4, c5, c6 = st.columns(3)
            required_down_days = c4.slider("Down days", 0, 10, int(base["consecutive_down_days"]), 1)
            required_prior_rvol = c5.slider("Prior RVOL (x)", 0.0, 10.0, float(base["prior_day_rvol"]), 0.1)
            required_pm_rvol = c6.slider("PM RVOL (x)", 0.0, 0.50, float(base["premarket_rvol"]), 0.01)

            only_matching = st.checkbox("Only this profile + cap", value=True)
            to_score = df.copy()
            if only_matching:
                to_score = to_score[(to_score["setup_profile"] == tune_profile) & (to_score["cap"] == tune_cap)]

            if to_score.empty:
                st.warning("No trades match.")
            else:
                rows: List[Dict[str, Any]] = []
                for _, r in to_score.iterrows():
                    s, rec, failed = score_bounce_pretrade_override_row(
                        r.to_dict(),
                        required_selloff_pct=required_selloff,
                        required_down_days=required_down_days,
                        required_off_30d_high_pct=required_off30,
                        required_gap_down_pct=required_gap,
                        required_prior_day_rvol=required_prior_rvol,
                        required_premarket_rvol=required_pm_rvol,
                    )
                    rows.append(
                        {
                            "checklist_score": int(s),
                            "checklist_max": 5,
                            "checklist_rec": rec,
                            "checklist_failed": ", ".join(failed) if failed else "PERFECT",
                        }
                    )
                scored = pd.concat([to_score.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

                st.dataframe(_group_stats(scored, "checklist_rec"), width="stretch")
                box = px.box(
                    scored,
                    x="checklist_rec",
                    y="pnl",
                    points="all",
                    hover_data=["ticker", "date", "trade_grade"],
                    title="P&L by recommendation",
                )
                st.plotly_chart(box, width="stretch")

            with st.expander("Export config snippet"):
                st.code(
                    f"""# {tune_profile} / {tune_cap}
selloff_total_pct <= {-required_selloff/100:.2f}
consecutive_down_days >= {required_down_days}
pct_off_30d_high <= {-required_off30/100:.2f}
gap_pct <= {-required_gap/100:.2f}
prior_day_rvol >= {required_prior_rvol:.2f}
premarket_rvol >= {required_pm_rvol:.2f}""",
                    language="python",
                )


def render_reversal(reversal_df: pd.DataFrame) -> None:
    st.subheader("Reversal (Short)")

    # Sidebar filters
    with st.sidebar:
        st.markdown("### Filters")
        grades = sorted([g for g in reversal_df.get("trade_grade", pd.Series(dtype=str)).dropna().unique()])
        caps = sorted([c for c in reversal_df.get("cap", pd.Series(dtype=str)).dropna().unique()])
        setups = sorted([s for s in reversal_df.get("setup", pd.Series(dtype=str)).dropna().unique()])

        sel_caps = st.multiselect("Cap", caps, default=caps, key="rev_cap")
        sel_grades = st.multiselect("Grade", grades, default=grades, key="rev_grade")
        sel_setups = st.multiselect("Setup", setups, default=setups, key="rev_setup")

        dmin = reversal_df["date_dt"].min()
        dmax = reversal_df["date_dt"].max()
        with st.expander("Date range", expanded=False):
            start_end = st.date_input(
                "Range",
                value=(dmin.date() if pd.notna(dmin) else None, dmax.date() if pd.notna(dmax) else None),
                label_visibility="collapsed",
                key="rev_date",
            )
            start_date = start_end[0] if isinstance(start_end, (list, tuple)) and len(start_end) > 0 else None
            end_date = start_end[1] if isinstance(start_end, (list, tuple)) and len(start_end) > 1 else None

    df = reversal_df.copy()
    df = _filter_by_multiselect(df, "trade_grade", sel_grades)
    df = _filter_by_multiselect(df, "cap", sel_caps)
    df = _filter_by_multiselect(df, "setup", sel_setups)
    df = _date_range_filter(df, "date_dt", start_date, end_date)

    if df.empty:
        st.warning("No reversal trades match the current filters.")
        return

    # Summary stats
    stats = _summary_stats(df)
    c1, c2, c3, c4 = st.columns(4)
    delta_str = _compute_delta(df)
    c1.metric("Trades", stats["trades"])
    c2.metric("Win Rate", "—" if stats["win_rate"] is None else f"{stats['win_rate']:.1f}%", delta=delta_str)
    c3.metric("Avg P&L", "—" if stats["avg_pnl"] is None else f"{stats['avg_pnl']:+.2f}%")
    c4.metric("Median P&L", "—" if stats["median_pnl"] is None else f"{stats['median_pnl']:+.2f}%")

    st.caption("P&L = -(open→close %) on reversal day (short). Win = P&L > 0.")

    # ── Equity curve + rolling WR ─────────────────────────────────────────
    ch1, ch2 = st.columns(2)
    with ch1:
        fig = _cumulative_pnl_chart(df, title="Reversal Equity Curve")
        if fig:
            st.plotly_chart(fig, width="stretch")
    with ch2:
        fig = _rolling_winrate_chart(df, title="Reversal Win Rate")
        if fig:
            st.plotly_chart(fig, width="stretch")

    # Breakdowns (expanded by default)
    with st.expander("Stats by cap / setup", expanded=True):
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("**By cap**")
            st.dataframe(_group_stats(df, "cap"), width="stretch", hide_index=True, height=200)
        with rc2:
            st.markdown("**By setup**")
            st.dataframe(_group_stats(df, "setup"), width="stretch", hide_index=True, height=200)

    metric_defaults = [
        "pct_from_9ema",
        "prior_day_range_atr",
        "consecutive_up_days",
        "gap_pct",
        "prior_day_rvol",
        "premarket_rvol",
        "reversal_open_close_pct",
        "upper_band_distance",
        "bollinger_width",
    ]

    # Tabs for different analysis modes
    tab_optimizer, tab_explore, tab_checklist = st.tabs(["Threshold Optimizer", "Metric Explorer", "Checklist Simulator"])

    with tab_optimizer:
        st.markdown("**Find optimal threshold for any metric by cap.** Sweeps values and shows win rate / avg P&L at each cutoff.")
        render_metric_optimizer(
            df,
            metric_defaults=metric_defaults,
            default_directions={
                "reversal_open_close_pct": "<=",
                "prior_day_close_vs_high_pct": "<=",
            },
            key_prefix="reversal_opt",
        )

    with tab_explore:
        st.markdown("**Explore metric distributions and their relationship to P&L.**")
        metric_options = _key_metrics_selector(metric_defaults, df)
        metric = st.selectbox("Metric", metric_options, index=0, key="rev_explore_metric")

        if metric in df.columns:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(
                    df,
                    x=metric,
                    nbins=35,
                    hover_data=["ticker", "date", "cap", "setup", "trade_grade", "pnl"],
                    title=f"Distribution: {metric}",
                )
                st.plotly_chart(fig, width="stretch")

            with c2:
                scatter = px.scatter(
                    df,
                    x=metric,
                    y="pnl",
                    color="cap" if "cap" in df.columns else None,
                    hover_data=["ticker", "date", "setup", "trade_grade"],
                    title=f"{metric} vs P&L",
                )
                st.plotly_chart(scatter, width="stretch")

    with tab_checklist:
        st.markdown("**Test checklist logic on historical trades.** See how GO/CAUTION/NO-GO performs.")
        mode = st.radio(
            "Mode",
            ["Current thresholds", "Custom thresholds"],
            horizontal=True,
            key="rev_checklist_mode",
        )

        if mode == "Current thresholds":
            rows: List[Dict[str, Any]] = []
            for _, r in df.iterrows():
                cap = str(r.get("cap") or "Medium")
                thr = REVERSAL_PRETRADE_THRESHOLDS.get(cap, REVERSAL_PRETRADE_THRESHOLDS["Medium"])
                s, rec, failed = score_reversal_pretrade_row(r.to_dict(), thr)
                rows.append(
                    {
                        "checklist_score": int(s),
                        "checklist_max": 5,
                        "checklist_rec": rec,
                        "checklist_failed": ", ".join(failed) if failed else "PERFECT",
                    }
                )
            scored = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

            st.markdown("**Results by recommendation:**")
            _render_rec_summary(scored)
            st.dataframe(_group_stats(scored, "checklist_rec"), width="stretch")
            box = px.box(
                scored,
                x="checklist_rec",
                y="pnl",
                points="all",
                hover_data=["ticker", "date", "cap", "setup", "trade_grade", "checklist_score"],
                title="P&L by recommendation",
                color="checklist_rec",
                color_discrete_map={"GO": "#6ee7b7", "CAUTION": "#fbbf24", "NO-GO": "#f87171"},
            )
            box.update_layout(showlegend=False)
            st.plotly_chart(box, width="stretch")

            with st.expander("All trades", expanded=False):
                st.dataframe(scored.sort_values(["checklist_rec", "pnl"], ascending=[True, False]), width="stretch", height=350)
                csv_bytes = scored.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", data=csv_bytes, file_name="reversal_scored.csv")

        else:
            tune_cap = st.selectbox("Cap", sorted(df["cap"].dropna().unique()), index=0, key="rev_tune_cap")
            base = REVERSAL_PRETRADE_THRESHOLDS.get(tune_cap, REVERSAL_PRETRADE_THRESHOLDS["Medium"]).copy()

            st.markdown("**Adjust thresholds:**")
            c1, c2, c3 = st.columns(3)
            thr_pct9 = c1.slider("Above 9EMA (%)", 0.0, 2.0, float(base["pct_from_9ema"]), 0.01)
            thr_range = c2.slider("Range (x ATR)", 0.0, 10.0, float(base["prior_day_range_atr"]), 0.1)
            thr_up = c3.slider("Up days", 0, 10, int(base["consecutive_up_days"]), 1)

            c4, c5, c6 = st.columns(3)
            thr_gap = c4.slider("Gap up (%)", 0.0, 20.0, float(base["gap_pct"]) * 100.0, 0.5)
            thr_prior_rvol = c5.slider("Prior RVOL (x)", 0.0, 10.0, float(base["prior_day_rvol"]), 0.1)
            thr_pm_rvol = c6.slider("PM RVOL (x)", 0.0, 0.50, float(base["premarket_rvol"]), 0.01)

            base_override = {
                "pct_from_9ema": thr_pct9,
                "prior_day_range_atr": thr_range,
                "consecutive_up_days": thr_up,
                "gap_pct": thr_gap / 100.0,
                "prior_day_rvol": thr_prior_rvol,
                "premarket_rvol": thr_pm_rvol,
            }

            only_matching = st.checkbox("Only this cap", value=True, key="rev_only_matching")
            to_score = df.copy()
            if only_matching:
                to_score = to_score[to_score["cap"] == tune_cap]

            if to_score.empty:
                st.warning("No trades match.")
            else:
                rows: List[Dict[str, Any]] = []
                for _, r in to_score.iterrows():
                    s, rec, failed = score_reversal_pretrade_row(r.to_dict(), base_override)
                    rows.append(
                        {
                            "checklist_score": int(s),
                            "checklist_max": 5,
                            "checklist_rec": rec,
                            "checklist_failed": ", ".join(failed) if failed else "PERFECT",
                        }
                    )
                scored = pd.concat([to_score.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

                st.dataframe(_group_stats(scored, "checklist_rec"), width="stretch")
                box = px.box(
                    scored,
                    x="checklist_rec",
                    y="pnl",
                    points="all",
                    hover_data=["ticker", "date", "setup", "trade_grade"],
                    title="P&L by recommendation",
                )
                st.plotly_chart(box, width="stretch")

            with st.expander("Export config snippet"):
                st.code(
                    f"""PRETRADE_THRESHOLDS['{tune_cap}'] = {{
    'pct_from_9ema': {thr_pct9:.2f},
    'prior_day_range_atr': {thr_range:.1f},
    'prior_day_rvol': {thr_prior_rvol:.1f},
    'premarket_rvol': {thr_pm_rvol:.2f},
    'consecutive_up_days': {thr_up},
    'gap_pct': {thr_gap/100.0:.2f},
}}""",
                    language="python",
                )


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------

with st.sidebar:
    st.title("Trade Dashboard")
    st.caption("Strategy analysis & threshold tuning")
    page = st.radio(
        "Page",
        ["Overview", "Bounce (long)", "Reversal (short)"],
        index=0,
        label_visibility="collapsed",
    )
    st.divider()

bounce_df = load_bounce_df()
reversal_df = load_reversal_df()

# Pre-compute bounce intensity (always — removes opt-in friction)
if "bounce_intensity" not in bounce_df.columns:
    _weights = {
        "selloff_total_pct": 0.30,
        "consecutive_down_days": 0.10,
        "prior_day_rvol": 0.15,
        "pct_off_30d_high": 0.20,
        "gap_pct": 0.25,
    }
    bounce_df["bounce_intensity"] = bounce_df.apply(
        lambda r: _compute_bounce_intensity_row(r, bounce_df, _weights), axis=1
    )

if page == "Overview":
    render_overview(bounce_df, reversal_df)
elif page == "Bounce (long)":
    render_bounce(bounce_df)
else:
    render_reversal(reversal_df)

