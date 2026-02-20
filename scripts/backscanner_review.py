"""
Backscanner Review App (Streamlit)

Unified review tool for bounce and reversal backscanner CSVs.
Load backscanner CSV -> review each candidate with daily + intraday charts
-> tag ADD/SKIP/REJECT -> auto-append ADDed trades to bounce_data.csv or reversal_data.csv.

Run:
    streamlit run scripts/backscanner_review.py
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from data_queries.polygon_queries import get_levels_data, get_intraday

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Backscanner Review", layout="wide")

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "data"
BOUNCE_DATA_PATH = DATA_DIR / "bounce_data.csv"
REVERSAL_DATA_PATH = DATA_DIR / "reversal_data.csv"

BOUNCE_DISCRIMINATORS = {"selloff_total_pct", "consecutive_down_days"}
REVERSAL_DISCRIMINATORS = {"pct_from_9ema", "consecutive_up_days"}

BOUNCE_METRICS = {
    "Selloff %": ("selloff_total_pct", True),
    "Gap %": ("gap_pct", True),
    "Down Days": ("consecutive_down_days", False),
    "Off 30d High": ("pct_off_30d_high", True),
    "Off 52wk High": ("pct_off_52wk_high", True),
    "Range/ATR": ("prior_day_range_atr", False),
    "Pr. Day RVOL": ("prior_day_rvol", False),
    "ATR %": ("atr_pct", True),
    "From 50MA": ("pct_from_50mav", True),
    "From 200MA": ("pct_from_200mav", True),
    "Bounce Ret": ("bounce_day_return", True),
    "Price": ("current_price", False),
}

REVERSAL_METRICS = {
    "From 9EMA": ("pct_from_9ema", True),
    "Gap %": ("gap_pct", True),
    "Up Days": ("consecutive_up_days", False),
    "Range/ATR": ("prior_day_range_atr", False),
    "RVOL": ("rvol_score", False),
    "ATR %": ("atr_pct", True),
    "From 50MA": ("pct_from_50mav", True),
    "Fade Ret": ("fade_day_return", True),
    "Price": ("current_price", False),
}

# Accent colors per mode
MODE_ACCENTS = {
    "Bounce": {"accent": "#00d4ff", "accent_dim": "#0a3d5c", "glow": "rgba(0,212,255,0.15)",
               "label": "BOUNCE", "vline": "cyan"},
    "Reversal": {"accent": "#ff2d6f", "accent_dim": "#5c0a2a", "glow": "rgba(255,45,111,0.15)",
                 "label": "FADE", "vline": "magenta"},
}


# ---------------------------------------------------------------------------
# CSS — financial terminal aesthetic (matches dashboard/app.py design system)
# ---------------------------------------------------------------------------

def inject_css(mode: str):
    accent = MODE_ACCENTS.get(mode, MODE_ACCENTS["Bounce"])
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Global ───────────────────────────────────────────────────────────── */
html, body, .stApp, .stApp *, [data-testid="stAppViewContainer"] {{
    font-family: 'Outfit', sans-serif !important;
}}
.block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 1rem !important;
}}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #080c12 0%, #0a0e14 40%, #0d1117 100%) !important;
    border-right: 1px solid #1c2333 !important;
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {{
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    background: linear-gradient(135deg, {accent['accent']} 0%, #81d4fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.3rem !important;
    margin-bottom: 0.5rem !important;
}}
[data-testid="stSidebar"] hr {{
    border-color: #1e293b !important;
    margin: 0.5rem 0 !important;
}}

/* ── Metric cards (sidebar progress) ──────────────────────────────────── */
[data-testid="stMetric"] {{
    background: linear-gradient(135deg, #111827 0%, #0d1117 100%);
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 10px 12px;
}}
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 1.3rem !important;
}}
[data-testid="stMetricLabel"] {{
    font-family: 'Outfit', sans-serif !important;
    text-transform: uppercase !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    opacity: 0.5;
}}

/* ── Headers ──────────────────────────────────────────────────────────── */
h1, h2, h3, h4 {{
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
}}

/* ── Expander ─────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {{
    border-color: #1e293b !important;
    border-radius: 8px;
}}

/* ── Progress bar ─────────────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {{
    background-color: {accent['accent']} !important;
    box-shadow: 0 0 12px {accent['glow']};
}}

/* ── Plotly transparent ───────────────────────────────────────────────── */
.js-plotly-plot .plotly .main-svg {{
    background: transparent !important;
}}

/* ── Header card ──────────────────────────────────────────────────────── */
.review-header {{
    background: linear-gradient(135deg, #111827 0%, #0f1520 100%);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}}
.review-header::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {accent['accent']}, transparent);
}}
.review-header .ticker {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #f0f4f8;
    letter-spacing: -0.02em;
    margin: 0;
    display: inline-block;
}}
.review-header .date {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    color: #6b7b8d;
    margin-left: 16px;
    vertical-align: baseline;
}}
.review-header .cap-badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    background: #1e293b;
    color: #94a3b8;
    border: 1px solid #334155;
    margin-left: 12px;
    vertical-align: baseline;
}}
.review-header .setup-label {{
    font-family: 'Outfit', sans-serif;
    font-size: 0.85rem;
    color: #6b7b8d;
    margin-top: 4px;
}}
.review-header .rec-go {{
    display: inline-block;
    padding: 3px 12px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.78rem;
    letter-spacing: 0.04em;
    background: #064e3b;
    color: #6ee7b7;
    border: 1px solid #10b981;
}}
.review-header .rec-nogo {{
    display: inline-block;
    padding: 3px 12px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.78rem;
    letter-spacing: 0.04em;
    background: #7f1d1d;
    color: #fca5a5;
    border: 1px solid #ef4444;
}}
.review-header .rec-caution {{
    display: inline-block;
    padding: 3px 12px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.78rem;
    letter-spacing: 0.04em;
    background: #78350f;
    color: #fcd34d;
    border: 1px solid #f59e0b;
}}
.review-header .score-badge {{
    display: inline-block;
    padding: 3px 12px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    background: {accent['accent_dim']};
    color: {accent['accent']};
    border: 1px solid {accent['accent']}44;
    margin-right: 8px;
}}
.review-header .mode-badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.68rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    background: {accent['accent_dim']};
    color: {accent['accent']};
    border: 1px solid {accent['accent']}33;
    margin-left: 8px;
    vertical-align: middle;
}}

/* ── Metrics grid ─────────────────────────────────────────────────────── */
.metrics-grid {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 14px 0 4px 0;
}}
.metric-cell {{
    background: #0d1117;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 6px 12px;
    min-width: 85px;
    flex: 0 0 auto;
}}
.metric-cell .m-label {{
    font-family: 'Outfit', sans-serif;
    font-size: 0.6rem;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    line-height: 1.2;
}}
.metric-cell .m-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #e0e0e8;
    font-weight: 500;
    line-height: 1.4;
}}
.metric-cell .m-value.negative {{
    color: #f87171;
}}
.metric-cell .m-value.positive {{
    color: #6ee7b7;
}}

/* ── Action bar ───────────────────────────────────────────────────────── */
.action-bar {{
    background: linear-gradient(135deg, #111827 0%, #0f1520 100%);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px 24px;
    margin-top: 12px;
}}
.kbd {{
    display: inline-block;
    padding: 1px 6px;
    border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    background: #1e293b;
    color: #94a3b8;
    border: 1px solid #334155;
    margin-left: 4px;
    vertical-align: middle;
}}
.shortcut-hint {{
    font-family: 'Outfit', sans-serif;
    font-size: 0.72rem;
    color: #4b5563;
    text-align: center;
    margin-top: 10px;
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Custom Plotly template (matches dashboard)
# ---------------------------------------------------------------------------

CHART_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, Outfit, monospace", color="#c0c8d8", size=11),
        title=dict(font=dict(family="Outfit, sans-serif", size=14, color="#e0e0e8")),
        xaxis=dict(gridcolor="#1e293b", zerolinecolor="#1e293b", linecolor="#1e293b",
                   tickfont=dict(size=9)),
        yaxis=dict(gridcolor="#1e293b", zerolinecolor="#1e293b", linecolor="#1e293b",
                   tickfont=dict(size=10)),
        hoverlabel=dict(bgcolor="#1e293b", bordercolor="#4fc3f7",
                        font=dict(family="JetBrains Mono, monospace", size=11, color="#e0e0e8")),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e293b", font=dict(size=10)),
        margin=dict(l=50, r=20, t=60, b=30),
    )
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def detect_csv_mode(columns: list[str]) -> str:
    cols = set(columns)
    if BOUNCE_DISCRIMINATORS.issubset(cols):
        return "Bounce"
    if REVERSAL_DISCRIMINATORS.issubset(cols):
        return "Reversal"
    return "Unknown"


def normalize_to_api_date(date_str: str) -> str:
    date_str = str(date_str).strip()
    try:
        dt = pd.to_datetime(date_str)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return date_str


def normalize_to_csv_date(date_str: str) -> str:
    date_str = str(date_str).strip()
    try:
        dt = pd.to_datetime(date_str)
        return f"{dt.month}/{dt.day}/{dt.year}"
    except Exception:
        return date_str


def discover_backscanner_csvs() -> dict[str, Path]:
    files = {}
    for p in sorted(DATA_DIR.glob("bounce_backscanner_*.csv")):
        files[p.name] = p
    for p in sorted(DATA_DIR.glob("backscanner_*.csv")):
        if not p.name.startswith("bounce_"):
            files[p.name] = p
    return files


def check_duplicate(target_df: pd.DataFrame, ticker: str, date_str: str) -> bool:
    csv_date = normalize_to_csv_date(date_str)
    normalized_dates = target_df["date"].apply(normalize_to_csv_date)
    mask = (target_df["ticker"] == ticker) & (normalized_dates == csv_date)
    return mask.any()


# ---------------------------------------------------------------------------
# CSV operations
# ---------------------------------------------------------------------------

def load_backscanner_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "review_status" not in df.columns:
        df["review_status"] = ""
    df["review_status"] = df["review_status"].fillna("").astype(str)

    # Dedupe within the CSV itself on ticker+date (keep first occurrence)
    df["_norm_date_dedup"] = pd.to_datetime(df["date"], format="mixed").dt.strftime("%Y-%m-%d")
    df = df.drop_duplicates(subset=["ticker", "_norm_date_dedup"], keep="first").reset_index(drop=True)
    df = df.drop(columns=["_norm_date_dedup"])

    # Dedupe: drop rows where ticker+date already exists in the target CSV
    mode = detect_csv_mode(df.columns.tolist())
    target_path = BOUNCE_DATA_PATH if mode == "Bounce" else REVERSAL_DATA_PATH
    if target_path.exists():
        target_df = pd.read_csv(target_path)
        target_df["_norm_date"] = pd.to_datetime(target_df["date"], format="mixed").dt.strftime("%Y-%m-%d")
        df["_norm_date"] = pd.to_datetime(df["date"], format="mixed").dt.strftime("%Y-%m-%d")
        existing = set(zip(target_df["ticker"], target_df["_norm_date"]))
        mask = df.apply(lambda r: (r["ticker"], r["_norm_date"]) not in existing, axis=1)
        removed = (~mask).sum()
        if removed > 0:
            df = df[mask].reset_index(drop=True)
        df = df.drop(columns=["_norm_date"])

    return df


def save_backscanner_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)


def append_to_bounce_data(ticker: str, date_str: str, trade_grade: str,
                          cap: str, setup_type: str) -> tuple[bool, str]:
    target_df = pd.read_csv(BOUNCE_DATA_PATH)
    csv_date = normalize_to_csv_date(date_str)
    if check_duplicate(target_df, ticker, date_str):
        return False, "Duplicate: already exists in bounce_data.csv"
    new_row = {col: "" for col in target_df.columns}
    new_row["date"] = csv_date
    new_row["ticker"] = ticker
    new_row["trade_grade"] = trade_grade
    new_row["cap"] = cap
    new_row["Setup"] = setup_type
    target_df = pd.concat([target_df, pd.DataFrame([new_row])], ignore_index=True)
    target_df.to_csv(BOUNCE_DATA_PATH, index=False)
    return True, f"Added {ticker} {csv_date} to bounce_data.csv"


def append_to_reversal_data(ticker: str, date_str: str, trade_grade: str,
                            cap: str, setup_type: str) -> tuple[bool, str]:
    target_df = pd.read_csv(REVERSAL_DATA_PATH)
    csv_date = normalize_to_csv_date(date_str)
    if check_duplicate(target_df, ticker, date_str):
        return False, "Duplicate: already exists in reversal_data.csv"
    new_row = {col: "" for col in target_df.columns}
    new_row["date"] = csv_date
    new_row["ticker"] = ticker
    new_row["trade_grade"] = trade_grade
    new_row["cap"] = cap
    new_row["setup"] = setup_type
    target_df = pd.concat([target_df, pd.DataFrame([new_row])], ignore_index=True)
    target_df.to_csv(REVERSAL_DATA_PATH, index=False)
    return True, f"Added {ticker} {csv_date} to reversal_data.csv"


# ---------------------------------------------------------------------------
# Data fetching (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_daily(ticker: str, date_str: str) -> pd.DataFrame | None:
    try:
        return get_levels_data(ticker, date_str, window=200, multiplier=1, timespan="day")
    except Exception as e:
        st.warning(f"Daily data error for {ticker}: {e}")
        return None


@st.cache_data(ttl=3600)
def fetch_intraday(ticker: str, date_str: str) -> pd.DataFrame | None:
    try:
        return get_intraday(ticker, date_str, multiplier=5, timespan="minute")
    except Exception as e:
        st.warning(f"Intraday data error for {ticker}: {e}")
        return None


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def build_daily_chart(df: pd.DataFrame, ticker: str, trade_date: str,
                      mode: str) -> go.Figure:
    accent = MODE_ACCENTS.get(mode, MODE_ACCENTS["Bounce"])

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.02,
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    ma_configs = [
        (200, "#e41a1c", "200"), (50, "#4daf4a", "50"),
        (20, "#377eb8", "20"), (10, "#984ea3", "10"),
    ]
    for window, color, label in ma_configs:
        if len(df) >= window:
            ma = df["close"].rolling(window).mean()
            fig.add_trace(go.Scatter(
                x=df.index, y=ma, mode="lines", name=label,
                line=dict(color=color, width=1.2),
            ), row=1, col=1)

    ema9 = df["close"].ewm(span=9, adjust=False).mean()
    fig.add_trace(go.Scatter(
        x=df.index, y=ema9, mode="lines", name="9e",
        line=dict(color="orange", width=1, dash="dash"),
    ), row=1, col=1)

    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], name="Vol",
        marker_color=colors, opacity=0.45, showlegend=False,
    ), row=2, col=1)

    try:
        bd = pd.Timestamp(trade_date)
        fig.add_vline(x=bd, line_color=accent["vline"], line_width=2, line_dash="dot",
                      annotation_text=accent["label"], annotation_position="top right",
                      annotation_font_color=accent["vline"],
                      annotation_font_size=10)
    except Exception:
        pass

    fig.update_layout(
        template=CHART_TEMPLATE,
        title=f"{ticker} — Daily",
        height=550,
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center",
                    font=dict(size=9)),
    )
    fig.update_xaxes(type="category", nticks=25, row=1, col=1)
    fig.update_xaxes(type="category", nticks=25, row=2, col=1)
    return fig


def build_intraday_chart(df: pd.DataFrame, ticker: str, date_str: str,
                         mode: str, gap_pct: float | None = None) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.02,
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    vwap = cum_tp_vol / cum_vol
    fig.add_trace(go.Scatter(
        x=df.index, y=vwap, mode="lines", name="VWAP",
        line=dict(color="#ff9800", width=1.5, dash="dot"),
    ), row=1, col=1)

    if gap_pct is not None and gap_pct != 0:
        open_price = df["open"].iloc[0]
        prior_close = open_price / (1 + gap_pct)
        fig.add_hline(y=prior_close, line_color="#fbbf24", line_dash="dash",
                      annotation_text="Prev Close", annotation_font_color="#fbbf24",
                      annotation_font_size=10)

    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], name="Vol",
        marker_color=colors, opacity=0.45, showlegend=False,
    ), row=2, col=1)

    annotations = []
    if gap_pct is not None:
        annotations.append(f"Gap {gap_pct*100:+.1f}%")
    subtitle = " | ".join(annotations) if annotations else ""

    fig.update_layout(
        template=CHART_TEMPLATE,
        title=f"{ticker} — 5min ({date_str})" + (f"  <sub>{subtitle}</sub>" if subtitle else ""),
        height=550,
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center",
                    font=dict(size=9)),
    )
    return fig


# ---------------------------------------------------------------------------
# UI helpers — custom HTML components
# ---------------------------------------------------------------------------

def render_header_card(trade: pd.Series, mode: str):
    """Render the main header card with ticker, badges, and inline metrics."""
    accent = MODE_ACCENTS.get(mode, MODE_ACCENTS["Bounce"])

    ticker = trade["ticker"]
    date = str(trade["date"])
    cap = trade.get("cap", "—")
    setup = trade.get("setup_type", "—")
    score = trade.get("score", "?")
    grade = trade.get("grade", "?")
    rec = trade.get("recommendation", "—")

    rec_class = "rec-go" if rec == "GO" else ("rec-nogo" if rec == "NO-GO" else "rec-caution")

    # Build metrics HTML
    metrics = BOUNCE_METRICS if mode == "Bounce" else REVERSAL_METRICS
    cells = ""
    for label, (col, is_pct) in metrics.items():
        val = trade.get(col)
        if pd.notna(val):
            if is_pct and isinstance(val, (int, float)):
                formatted = f"{val*100:.1f}%" if abs(val) < 10 else f"{val:.2f}"
                val_class = "negative" if val < 0 else "positive"
            elif isinstance(val, float):
                formatted = f"{val:.2f}"
                val_class = "negative" if val < 0 else ("positive" if val > 0 else "")
            else:
                formatted = str(val)
                val_class = ""
            cells += f'<div class="metric-cell"><div class="m-label">{label}</div><div class="m-value {val_class}">{formatted}</div></div>'

    st.markdown(f"""
    <div class="review-header">
        <div>
            <span class="ticker">{ticker}</span>
            <span class="date">{date}</span>
            <span class="cap-badge">{cap}</span>
            <span class="mode-badge">{mode}</span>
        </div>
        <div style="margin-top: 8px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
            <span class="score-badge">{grade} ({score})</span>
            <span class="{rec_class}">{rec}</span>
            <span class="setup-label" style="margin-left: 4px;">{setup}</span>
        </div>
        <div class="metrics-grid">{cells}</div>
    </div>
    """, unsafe_allow_html=True)


def render_keyboard_shortcuts():
    """Inject keyboard shortcut handler via JS."""
    components.html("""
    <script>
    const doc = window.parent.document;
    if (!doc._reviewKeysAttached) {
        doc._reviewKeysAttached = true;
        doc.addEventListener('keydown', function(e) {
            const tag = e.target.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
            if (e.target.contentEditable === 'true') return;

            function clickBtn(text) {
                const buttons = doc.querySelectorAll('button');
                for (const b of buttons) {
                    if (b.innerText.trim() === text) { b.click(); return true; }
                }
                return false;
            }

            if (e.key === '1') { clickBtn('ADD'); e.preventDefault(); }
            else if (e.key === '2') { clickBtn('SKIP'); e.preventDefault(); }
            else if (e.key === '3') { clickBtn('REJECT'); e.preventDefault(); }
            else if (e.key === 'ArrowLeft') { clickBtn('Prev'); e.preventDefault(); }
            else if (e.key === 'ArrowRight') { clickBtn('Next'); e.preventDefault(); }
        });
    }
    </script>
    """, height=0)


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
    csv_files = discover_backscanner_csvs()
    if not csv_files:
        st.error("No backscanner CSVs found in data/ directory.")
        return

    # ── Sidebar ───────────────────────────────────────────────────────────
    st.sidebar.title("Backscanner Review")

    selected_name = st.sidebar.selectbox("CSV File", list(csv_files.keys()))
    csv_path = csv_files[selected_name]

    if st.session_state.get("_current_csv") != selected_name:
        st.session_state._current_csv = selected_name
        st.session_state.trade_idx = 0
        st.session_state.pop("trade_select", None)

    df = load_backscanner_csv(csv_path)
    mode = detect_csv_mode(df.columns.tolist())

    # Inject mode-aware CSS
    inject_css(mode)

    st.sidebar.markdown(
        f"**Mode:** {mode} &nbsp;&middot;&nbsp; {len(df)} rows")

    # Progress
    st.sidebar.markdown("---")
    added = (df["review_status"] == "ADD").sum()
    skipped = (df["review_status"] == "SKIP").sum()
    rejected = (df["review_status"] == "REJECT").sum()
    reviewed = added + skipped + rejected
    remaining = len(df) - reviewed

    row1 = st.sidebar.columns(2)
    row1[0].metric("Added", int(added))
    row1[1].metric("Skipped", int(skipped))
    row2 = st.sidebar.columns(2)
    row2[0].metric("Rejected", int(rejected))
    row2[1].metric("Remaining", int(remaining))

    if len(df) > 0:
        pct = reviewed / len(df)
        st.sidebar.progress(pct, text=f"{pct:.0%}")

    # Quick filters (always visible)
    st.sidebar.markdown("---")
    show_unreviewed = st.sidebar.checkbox("Show unreviewed only", value=True)
    exclude_nogo = st.sidebar.checkbox("Exclude NO-GO", value=True)

    # Advanced filters (in expander)
    with st.sidebar.expander("More filters", expanded=False):
        grades = sorted(df["grade"].dropna().unique().tolist()) if "grade" in df.columns else []
        caps = sorted(df["cap"].dropna().unique().tolist()) if "cap" in df.columns else []
        setups = sorted(df["setup_type"].dropna().unique().tolist()) if "setup_type" in df.columns else []
        recs = sorted(df["recommendation"].dropna().unique().tolist()) if "recommendation" in df.columns else []
        filter_grade = st.multiselect("Grade", grades, default=[])
        filter_cap = st.multiselect("Cap", caps, default=[])
        filter_setup = st.multiselect("Setup", setups, default=[])
        filter_rec = st.multiselect("Rec", recs, default=[])

    filter_key = (show_unreviewed, exclude_nogo, tuple(filter_grade), tuple(filter_cap),
                  tuple(filter_setup), tuple(filter_rec))
    if st.session_state.get("_prev_filters") != filter_key:
        st.session_state._prev_filters = filter_key
        st.session_state.trade_idx = 0
        st.session_state.pop("trade_select", None)

    # Apply filters
    view_df = df.copy()
    view_df["_orig_idx"] = range(len(view_df))
    if show_unreviewed:
        view_df = view_df[view_df["review_status"].str.strip() == ""]
    if exclude_nogo and "recommendation" in view_df.columns:
        view_df = view_df[view_df["recommendation"] != "NO-GO"]
    if filter_grade:
        view_df = view_df[view_df["grade"].isin(filter_grade)]
    if filter_cap:
        view_df = view_df[view_df["cap"].isin(filter_cap)]
    if filter_setup:
        view_df = view_df[view_df["setup_type"].isin(filter_setup)]
    if filter_rec:
        view_df = view_df[view_df["recommendation"].isin(filter_rec)]
    view_df = view_df.reset_index(drop=True)

    if view_df.empty:
        st.info("No trades match the current filters.")
        return

    # Navigation
    st.sidebar.markdown("---")
    nav_cols = st.sidebar.columns([1, 1])
    with nav_cols[0]:
        st.button("Prev", on_click=go_prev, use_container_width=True)
    with nav_cols[1]:
        st.button("Next", on_click=go_next, args=(len(view_df) - 1,),
                  use_container_width=True)

    if "trade_idx" not in st.session_state:
        st.session_state.trade_idx = 0
    st.session_state.trade_idx = min(max(0, _safe_idx()), len(view_df) - 1)

    labels = []
    for i, (_, r) in enumerate(view_df.iterrows()):
        score = r.get("score", "?")
        grade = r.get("grade", "?")
        labels.append(
            f"{i+1}/{len(view_df)}  {r['ticker']}  {r['date']}  [{grade}({score})]")

    st.sidebar.selectbox(
        "Trade", range(len(labels)), format_func=lambda i: labels[i],
        index=st.session_state.trade_idx,
        key="trade_select", on_change=on_select,
    )

    selected_idx = st.session_state.trade_idx
    trade = view_df.iloc[selected_idx]
    orig_idx = int(trade["_orig_idx"])
    api_date = normalize_to_api_date(str(trade["date"]))

    # ── Main content ──────────────────────────────────────────────────────

    # Header card with ticker, badges, and metrics
    render_header_card(trade, mode)

    # Charts
    with st.spinner(f"Loading {trade['ticker']}..."):
        daily_df = fetch_daily(trade["ticker"], api_date)
        intraday_df = fetch_intraday(trade["ticker"], api_date)

    col_daily, col_intra = st.columns(2)

    with col_daily:
        if daily_df is not None and not daily_df.empty:
            fig_daily = build_daily_chart(daily_df, trade["ticker"], api_date, mode)
            st.plotly_chart(fig_daily, use_container_width=True)
        else:
            st.error("Could not fetch daily data")

    with col_intra:
        if intraday_df is not None and not intraday_df.empty:
            gap_val = trade.get("gap_pct")
            gap_pct = float(gap_val) if pd.notna(gap_val) else None
            fig_intra = build_intraday_chart(
                intraday_df, trade["ticker"], api_date, mode, gap_pct=gap_pct)
            st.plotly_chart(fig_intra, use_container_width=True)
        else:
            st.error("Could not fetch intraday data")

    # ── Action bar ────────────────────────────────────────────────────────
    st.markdown('<div class="action-bar">', unsafe_allow_html=True)
    act_cols = st.columns([1.2, 2, 0.2, 1, 1, 1])

    with act_cols[0]:
        grade_options = ["A", "B", "C", "D"]
        selected_grade = st.selectbox("Grade", grade_options, key="tag_grade",
                                      label_visibility="collapsed")

    with act_cols[1]:
        default_setup = str(trade.get("setup_type", "")) if pd.notna(trade.get("setup_type")) else ""
        selected_setup = st.text_input("Setup Type", value=default_setup,
                                       key="tag_setup", label_visibility="collapsed",
                                       placeholder="Setup type...")

    with act_cols[3]:
        if st.button("ADD", type="primary", use_container_width=True):
            ticker = trade["ticker"]
            date = str(trade["date"])
            cap = str(trade.get("cap", ""))
            setup = selected_setup or default_setup
            if mode == "Bounce":
                ok, msg = append_to_bounce_data(ticker, date, selected_grade, cap, setup)
            else:
                ok, msg = append_to_reversal_data(ticker, date, selected_grade, cap, setup)
            if ok:
                df.at[orig_idx, "review_status"] = "ADD"
                save_backscanner_csv(df, csv_path)
                st.session_state.trade_idx = min(_safe_idx() + 1, len(view_df) - 1)
                st.rerun()
            else:
                st.warning(msg)

    with act_cols[4]:
        if st.button("SKIP", use_container_width=True):
            df.at[orig_idx, "review_status"] = "SKIP"
            save_backscanner_csv(df, csv_path)
            st.session_state.trade_idx = min(_safe_idx() + 1, len(view_df) - 1)
            st.rerun()

    with act_cols[5]:
        if st.button("REJECT", use_container_width=True):
            df.at[orig_idx, "review_status"] = "REJECT"
            save_backscanner_csv(df, csv_path)
            st.session_state.trade_idx = min(_safe_idx() + 1, len(view_df) - 1)
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Keyboard shortcut hint
    st.markdown(
        '<div class="shortcut-hint">'
        '<span class="kbd">&larr;</span> <span class="kbd">&rarr;</span> navigate'
        ' &nbsp;&nbsp;&middot;&nbsp;&nbsp; '
        '<span class="kbd">1</span> ADD'
        ' &nbsp;&nbsp; '
        '<span class="kbd">2</span> SKIP'
        ' &nbsp;&nbsp; '
        '<span class="kbd">3</span> REJECT'
        '</div>',
        unsafe_allow_html=True,
    )

    # Inject keyboard shortcuts (invisible JS component)
    render_keyboard_shortcuts()


if __name__ == "__main__":
    main()
