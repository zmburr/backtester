"""
Shared CSS and Plotly template for the charting app.
Financial terminal aesthetic — readable, clean, functional.
"""

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
CYAN = "#4fc3f7"
GREEN = "#6ee7b7"
AMBER = "#fbbf24"
RED = "#f87171"
LAVENDER = "#c084fc"
TANGERINE = "#fb923c"

CHART_COLORS = [CYAN, GREEN, AMBER, RED, LAVENDER, TANGERINE, "#67e8f9", "#a78bfa"]


# ---------------------------------------------------------------------------
# Plotly template — dark terminal theme
# ---------------------------------------------------------------------------
TRADING_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#8a94a6", size=11),
        title=dict(font=dict(family="JetBrains Mono, monospace", size=14, color="#c0c8d8")),
        xaxis=dict(
            gridcolor="rgba(30,41,59,0.5)",
            zerolinecolor="#1e293b",
            linecolor="#1e293b",
            tickfont=dict(size=9, color="#5a6578"),
        ),
        yaxis=dict(
            gridcolor="rgba(30,41,59,0.5)",
            zerolinecolor="#1e293b",
            linecolor="#1e293b",
            tickfont=dict(size=9, color="#5a6578"),
        ),
        colorway=CHART_COLORS,
        hoverlabel=dict(
            bgcolor="#111827",
            bordercolor="#2a3548",
            font=dict(family="JetBrains Mono, monospace", size=11, color="#c0c8d8"),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=10, color="#8a94a6"),
        ),
        margin=dict(l=48, r=12, t=40, b=32),
    )
)

pio.templates["trading_dark"] = TRADING_TEMPLATE
pio.templates.default = "trading_dark"


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------
def inject_css():
    """Inject the financial terminal CSS."""
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

/* ── Base ─────────────────────────────────────────────────────── */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Outfit', sans-serif !important;
    -webkit-font-smoothing: antialiased;
}
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stText"],
.stApp [data-testid="stCaption"],
.stApp label,
.stApp p,
.stApp span:not([class*="st-"]),
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    font-family: 'Outfit', sans-serif !important;
}

.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}

/* ── Sidebar ──────────────────────────────────────────────────── */
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
    font-size: 1.3rem !important;
}

/* ── Sidebar section headings ────────────────────────────────── */
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #6b7a90 !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0.3rem !important;
}

/* ── Sidebar labels ──────────────────────────────────────────── */
[data-testid="stSidebar"] label {
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #6b7a90 !important;
    font-weight: 500 !important;
}

/* ── Sidebar inputs ──────────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stTextInput"] input,
[data-testid="stSidebar"] [data-testid="stDateInput"] input {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #c0c8d8 !important;
}

/* ── Sidebar checkboxes ──────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stCheckbox"] label span {
    font-size: 0.78rem !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    color: #8a94a6 !important;
}

/* ── Sidebar dividers ────────────────────────────────────────── */
[data-testid="stSidebar"] hr {
    border-color: #1c2333 !important;
    margin: 0.5rem 0 !important;
}

/* ── Sidebar buttons (watchlist chips) ───────────────────────── */
[data-testid="stSidebar"] .stButton > button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 500;
    padding: 4px 8px !important;
    min-height: 28px !important;
    background: #111827 !important;
    border: 1px solid #1e293b !important;
    border-radius: 4px !important;
    color: #8a94a6 !important;
    transition: all 0.15s ease;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #4fc3f7 !important;
    color: #4fc3f7 !important;
    background: rgba(79,195,247,0.08) !important;
}

/* ── Scrollbar ───────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2a3548; }

/* ── Metrics ─────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #111827 0%, #0d1117 100%);
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 12px 16px;
    transition: border-color 0.2s ease;
}
[data-testid="stMetric"]:hover {
    border-color: #4fc3f7;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 1.5rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Outfit', sans-serif !important;
    text-transform: uppercase !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    opacity: 0.55;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Tabs ────────────────────────────────────────────────────── */
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

/* ── Headings ────────────────────────────────────────────────── */
h1, h2, h3, h4 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
}

/* ── Tables / DataFrames ─────────────────────────────────────── */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ── Recommendation badges ───────────────────────────────────── */
.rec-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.75rem;
    letter-spacing: 0.04em;
}
.rec-go       { background: #064e3b; color: #6ee7b7; border: 1px solid #10b981; }
.rec-caution  { background: #78350f; color: #fcd34d; border: 1px solid #f59e0b; }
.rec-nogo     { background: #7f1d1d; color: #fca5a5; border: 1px solid #ef4444; }

/* ── Dividers ────────────────────────────────────────────────── */
hr { border-color: #1e293b !important; }

/* ── Expanders ───────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border-color: #1e293b !important;
    border-radius: 8px;
}
[data-testid="stExpander"] details > summary {
    display: flex !important;
    align-items: center !important;
    gap: 8px;
    overflow: visible !important;
    white-space: nowrap;
}
[data-testid="stExpander"] details > summary > span {
    overflow: visible !important;
    text-overflow: unset !important;
    white-space: nowrap !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.85rem !important;
}
[data-testid="stExpander"] details > summary svg {
    flex-shrink: 0 !important;
}

/* ── Plotly transparent bg ───────────────────────────────────── */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* ── Main-area buttons ───────────────────────────────────────── */
.block-container .stButton > button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 500;
    letter-spacing: 0.02em;
    border-radius: 4px;
    border: 1px solid #1e293b;
    transition: all 0.15s ease;
}
.block-container .stButton > button:hover {
    border-color: #4fc3f7 !important;
    color: #c0c8d8 !important;
}

/* ── Forms ────────────────────────────────────────────────────── */
[data-testid="stForm"] {
    border-color: #1e293b !important;
    border-radius: 6px;
}

/* ── Criteria table (custom HTML) ────────────────────────────── */
.criteria-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    margin: 0.5rem 0;
}
.criteria-table th {
    text-align: left;
    padding: 6px 10px;
    color: #6b7a90;
    font-weight: 500;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-bottom: 1px solid #1e293b;
}
.criteria-table td {
    padding: 6px 10px;
    color: #c0c8d8;
    border-bottom: 1px solid rgba(30,41,59,0.4);
}
.criteria-table tr:last-child td { border-bottom: none; }
.criteria-table tr:hover td { background: rgba(79,195,247,0.03); }
.status-pass { color: #6ee7b7; font-weight: 600; }
.status-fail { color: #f87171; font-weight: 600; }
.status-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
}
.dot-pass { background: #6ee7b7; box-shadow: 0 0 4px rgba(110,231,183,0.4); }
.dot-fail { background: #f87171; box-shadow: 0 0 4px rgba(248,113,113,0.4); }

/* ── Intensity meter ─────────────────────────────────────────── */
.intensity-meter {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
}
.intensity-bar-bg {
    flex: 1;
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
    overflow: hidden;
}
.intensity-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.4s ease;
}
.intensity-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    min-width: 52px;
    text-align: right;
}

/* ── News summary block ──────────────────────────────────────── */
.news-summary {
    padding: 8px 12px;
    background: rgba(15,20,30,0.5);
    border-left: 2px solid #4fc3f7;
    border-radius: 0 4px 4px 0;
    margin: 6px 0;
    font-size: 0.85rem;
    color: #c0c8d8;
    line-height: 1.5;
}
.news-summary .news-date {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #4fc3f7;
    margin-bottom: 3px;
}

/* ── Chart header ────────────────────────────────────────────── */
.chart-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    font-weight: 600;
    color: #c0c8d8;
    margin-bottom: 4px;
}
.chart-header .ticker-label { color: #4fc3f7; }
.chart-header .price-label { color: #6ee7b7; margin-left: 8px; }
.chart-header .change-positive { color: #6ee7b7; font-size: 0.78rem; }
.chart-header .change-negative { color: #f87171; font-size: 0.78rem; }

/* ── Alerts ──────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 4px;
}

/* ── Hide branding ───────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""",
        unsafe_allow_html=True,
    )


def rec_badge_html(rec: str) -> str:
    """Return a styled HTML badge for GO / CAUTION / NO-GO."""
    cls = {"GO": "rec-go", "CAUTION": "rec-caution", "NO-GO": "rec-nogo"}.get(
        str(rec).upper(), "rec-nogo"
    )
    return f'<span class="rec-badge {cls}">{rec}</span>'


def criteria_table_html(rows: list, columns: list) -> str:
    """Build a styled HTML criteria table."""
    header = "".join(f"<th>{c}</th>" for c in columns)
    body = ""
    for row in rows:
        cells = ""
        for col in columns:
            val = row.get(col, "")
            if col == "Status":
                if val == "PASS":
                    cells += '<td><span class="status-dot dot-pass"></span><span class="status-pass">PASS</span></td>'
                else:
                    cells += '<td><span class="status-dot dot-fail"></span><span class="status-fail">FAIL</span></td>'
            else:
                cells += f"<td>{val}</td>"
        body += f"<tr>{cells}</tr>"
    return f'<table class="criteria-table"><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>'


def intensity_meter_html(value: float) -> str:
    """Render a horizontal intensity meter with value label."""
    if value >= 70:
        color = "#6ee7b7"
    elif value >= 40:
        color = "#fbbf24"
    else:
        color = "#f87171"
    pct = min(max(value, 0), 100)
    return f"""
    <div class="intensity-meter">
        <div class="intensity-bar-bg">
            <div class="intensity-bar-fill" style="width:{pct}%; background:{color};"></div>
        </div>
        <div class="intensity-value" style="color:{color};">{value:.0f}</div>
    </div>
    """


def news_summary_html(date_str: str, summary: str) -> str:
    """Render a styled news summary block."""
    return f"""
    <div class="news-summary">
        <div class="news-date">{date_str}</div>
        {summary}
    </div>
    """
