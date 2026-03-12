"""
Copper & Ink — design system shared across journal_dash.py and setup_playbook.py.

Exports: C, CUSTOM_CSS, _card, _label, _mono, _serif, dark_graph, dark_graph_inline,
         TAB_STYLE, TAB_SELECTED, DROPDOWN_STYLE
"""

from dash import dcc, html

# ── Colour palette ─────────────────────────────────────────
C = {
    "bg":        "#0c0e12",       # deep ink
    "surface":   "#13161c",       # card surface
    "elevated":  "#1a1e26",       # raised / hover
    "overlay":   "#21262f",       # modal overlay

    "text":      "#e2dbd0",       # warm cream
    "text2":     "#8c8577",       # muted warm
    "text3":     "#4d4a44",       # very muted

    "profit":    "#5fb88a",       # sage green
    "loss":      "#c95555",       # brick red

    "gold":      "#c8a46e",       # copper / gold accent
    "steel":     "#5889a0",       # steel blue (secondary)

    "border":    "#1f232b",       # hairline
    "divider":   "#191c23",       # section line

    "star_lit":  "#d4a84b",       # active star
    "star_dim":  "#2a2720",       # dim star
}

# ── Custom CSS ──────────────────────────────────────────────
CUSTOM_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {{
    --bg: {C["bg"]};
    --surface: {C["surface"]};
    --elevated: {C["elevated"]};
    --text: {C["text"]};
    --text2: {C["text2"]};
    --text3: {C["text3"]};
    --gold: {C["gold"]};
    --border: {C["border"]};
    --profit: {C["profit"]};
    --loss: {C["loss"]};
}}

* {{ box-sizing: border-box; }}

body {{
    margin: 0;
    background: var(--bg);
    font-family: 'Outfit', sans-serif;
    color: var(--text);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}

/* ── Grain texture overlay ─────────────────────────── */
body::after {{
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.025'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 9999;
}}

/* ── Scrollbar ─────────────────────────────────────── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {C["border"]}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {C["text3"]}; }}

/* ── Entrance animations ───────────────────────────── */
@keyframes fadeSlideUp {{
    from {{ opacity: 0; transform: translateY(14px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeIn {{
    from {{ opacity: 0; }}
    to   {{ opacity: 1; }}
}}

.anim-stagger > * {{
    animation: fadeSlideUp 0.45s cubic-bezier(0.23, 1, 0.32, 1) backwards;
}}
.anim-stagger > *:nth-child(1) {{ animation-delay: 0.04s; }}
.anim-stagger > *:nth-child(2) {{ animation-delay: 0.09s; }}
.anim-stagger > *:nth-child(3) {{ animation-delay: 0.14s; }}
.anim-stagger > *:nth-child(4) {{ animation-delay: 0.19s; }}
.anim-stagger > *:nth-child(5) {{ animation-delay: 0.24s; }}
.anim-stagger > *:nth-child(6) {{ animation-delay: 0.29s; }}

/* ── DataTable typography override ─────────────────── */
.dash-spreadsheet-container .dash-spreadsheet-inner td,
.dash-spreadsheet-container .dash-spreadsheet-inner th {{
    font-family: 'Outfit', sans-serif !important;
}}
.dash-spreadsheet-container .dash-spreadsheet-inner td .dash-cell-value {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11.5px !important;
}}
.dash-spreadsheet-container .dash-spreadsheet-inner th .column-header-name {{
    font-family: 'Outfit', sans-serif !important;
}}

/* ── Date picker dark ──────────────────────────────── */
.DateRangePickerInput, .DateInput, .DateInput_input {{
    background: {C["surface"]} !important;
    color: {C["text"]} !important;
    border-color: {C["border"]} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}}
.DateRangePickerInput_arrow_svg {{ fill: {C["text3"]} !important; }}
.DateInput_input__focused {{ border-bottom-color: {C["gold"]} !important; }}

/* ── Dropdown dark ─────────────────────────────────── */
.Select-control, .Select-menu-outer {{
    background-color: {C["surface"]} !important;
    border-color: {C["border"]} !important;
    color: {C["text"]} !important;
}}
.Select-value-label, .Select-placeholder {{
    color: {C["text2"]} !important;
    font-family: 'Outfit', sans-serif !important;
}}
.Select-arrow {{ border-color: {C["text3"]} transparent transparent !important; }}
.VirtualizedSelectOption {{
    background-color: {C["surface"]} !important;
    color: {C["text"]} !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 13px !important;
}}
.VirtualizedSelectFocusedOption {{
    background-color: {C["elevated"]} !important;
}}
.Select-multi-value-wrapper .Select-value {{
    background-color: {C["elevated"]} !important;
    border-color: {C["border"]} !important;
    color: {C["gold"]} !important;
    font-size: 12px !important;
}}
.Select-value-icon {{ border-right-color: {C["border"]} !important; }}
.Select-value-icon:hover {{ background-color: {C["overlay"]} !important; color: {C["loss"]} !important; }}

/* ── Textarea ──────────────────────────────────────── */
textarea {{
    font-family: 'Outfit', sans-serif !important;
    transition: border-color 0.2s ease;
}}
textarea:focus {{
    outline: none;
    border-color: {C["gold"]} !important;
    box-shadow: 0 0 0 1px {C["gold"]}22;
}}

/* ── Buttons ───────────────────────────────────────── */
button {{
    transition: all 0.15s ease;
    font-family: 'Outfit', sans-serif;
}}
button:hover {{ filter: brightness(1.12); }}
button:active {{ transform: scale(0.97); }}

/* ── Filter input ──────────────────────────────────── */
.dash-spreadsheet .dash-filter input {{
    background-color: {C["elevated"]} !important;
    color: {C["text"]} !important;
    border: 1px solid {C["border"]} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    border-radius: 2px !important;
}}

/* ── Auto-save pulse ───────────────────────────────── */
@keyframes softPulse {{
    0%   {{ opacity: 1; }}
    60%  {{ opacity: 0.4; }}
    100% {{ opacity: 0; }}
}}

/* ── Tab underline slide ───────────────────────────── */
.custom-tabs .tab--selected {{
    transition: border-color 0.3s ease;
}}

/* ── Subtle glow behind stat numbers ───────────────── */
@keyframes numberGlow {{
    from {{ text-shadow: 0 0 8px currentColor; }}
    to   {{ text-shadow: none; }}
}}

/* ── Input focus ring ──────────────────────────────── */
input:focus {{
    outline: none;
    border-color: {C["gold"]} !important;
}}

/* ── Chip hover ────────────────────────────────────── */
.chip-btn:hover {{
    filter: brightness(1.25) !important;
    transform: translateY(-1px);
}}
"""


# ── Reusable style helpers ──────────────────────────────────

def _card(extra=None, accent_left=None):
    """Ledger-style card with optional left accent bar."""
    s = {
        "backgroundColor": C["surface"],
        "borderRadius": "4px",
        "padding": "20px",
        "marginBottom": "14px",
        "border": f"1px solid {C['border']}",
        "position": "relative",
    }
    if accent_left:
        s["borderLeft"] = f"3px solid {accent_left}"
    if extra:
        s.update(extra)
    return s


def _label():
    """Uppercase tracking label style."""
    return {
        "color": C["text3"],
        "fontSize": "10px",
        "fontWeight": "700",
        "textTransform": "uppercase",
        "letterSpacing": "1.2px",
        "fontFamily": "'Outfit', sans-serif",
        "marginBottom": "6px",
        "display": "block",
    }


def _mono(size="13px", weight="500", color=None):
    """JetBrains Mono style dict."""
    return {
        "fontFamily": "'JetBrains Mono', monospace",
        "fontSize": size,
        "fontWeight": weight,
        "color": color or C["text"],
    }


def _serif(size="24px", color=None):
    """DM Serif Display style dict."""
    return {
        "fontFamily": "'DM Serif Display', serif",
        "fontSize": size,
        "color": color or C["text"],
        "fontWeight": "400",
        "letterSpacing": "-0.3px",
    }


DROPDOWN_STYLE = {
    "backgroundColor": C["surface"],
    "color": C["text"],
    "border": f"1px solid {C['border']}",
    "borderRadius": "3px",
    "fontFamily": "'Outfit', sans-serif",
    "fontSize": "13px",
}

TAB_STYLE = {
    "backgroundColor": "transparent",
    "color": C["text3"],
    "border": "none",
    "borderBottom": "2px solid transparent",
    "padding": "10px 18px",
    "fontWeight": "600",
    "fontSize": "11px",
    "fontFamily": "'Outfit', sans-serif",
    "letterSpacing": "1.5px",
    "textTransform": "uppercase",
}

TAB_SELECTED = {
    **TAB_STYLE,
    "color": C["gold"],
    "borderBottom": f"2px solid {C['gold']}",
}


# ── Dark theme for plotly figures ───────────────────────────

def dark_graph(figure, graph_id=None, height="400px"):
    """Apply dark theme to a plotly figure, return dcc.Graph."""
    figure.update_layout(
        paper_bgcolor=C["surface"],
        plot_bgcolor=C["bg"],
        font=dict(family="Outfit, sans-serif", color=C["text2"], size=11),
        title_font=dict(family="DM Serif Display, serif", color=C["text"], size=15),
        legend_font_color=C["text2"],
        margin=dict(l=40, r=20, t=50, b=40),
    )
    figure.update_xaxes(gridcolor=C["border"], zerolinecolor=C["border"])
    figure.update_yaxes(gridcolor=C["border"], zerolinecolor=C["border"])
    kwargs = {"figure": figure, "style": {"height": height}}
    if graph_id:
        kwargs["id"] = graph_id
    return dcc.Graph(**kwargs)


def dark_graph_inline(fig):
    """Apply dark theme to a plotly figure, return dcc.Graph (used by writeup insights + news)."""
    fig.update_layout(
        paper_bgcolor=C["surface"], plot_bgcolor=C["bg"],
        font=dict(family="Outfit, sans-serif", color=C["text2"], size=11),
        title_font=dict(family="DM Serif Display, serif", color=C["text"], size=15),
        legend_font_color=C["text2"],
        margin=dict(l=50, r=20, t=50, b=40),
    )
    fig.update_xaxes(gridcolor=C["border"], zerolinecolor=C["border"])
    fig.update_yaxes(gridcolor=C["border"], zerolinecolor=C["border"])
    return dcc.Graph(figure=fig, style={"height": "400px"})
