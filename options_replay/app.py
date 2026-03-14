"""
Options Replay — Dash dashboard for replaying options chain behavior
around headline-driven equity trades.

Run: python -m options_replay.app
"""

import sys
import os
import logging
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update

# Ensure backtester root is on path for polygon imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from options_replay.theme import C, CUSTOM_CSS, _card, _label, _mono, _serif, DROPDOWN_STYLE
from options_replay.trade_loader import load_trades, get_trade_options, parse_manual_input
from options_replay.theta_client import (
    check_terminal_running, get_chain_snapshot, get_option_ohlc,
    get_option_quotes, get_option_greeks, ThetaTerminalOfflineError,
)
from options_replay.chain_analyzer import (
    filter_chain, compute_option_returns, score_options,
    compute_ideal_play_summary, contract_key, contract_label,
)
from options_replay.contract_picker import pick_contracts, PickerResult
from options_replay.charts import (
    fig_underlying_candlestick, fig_chain_heatmap, fig_top_options_table,
    fig_option_price_grid, fig_return_comparison, fig_liquidity_comparison,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Load trades ─────────────────────────────────────────────
try:
    TRADES_DF = load_trades()
    TRADE_OPTIONS = get_trade_options(TRADES_DF)
    logger.info("Loaded %d trades from CSV", len(TRADES_DF))
except Exception as e:
    logger.error("Failed to load trades: %s", e)
    TRADES_DF = pd.DataFrame()
    TRADE_OPTIONS = []

# ── Batch state (shared with background thread) ────────────
_batch_state = {
    "running": False,
    "completed": 0,
    "total": 0,
    "current_symbol": "",
    "results_path": None,
    "error": None,
}

# ── Dash App ────────────────────────────────────────────────
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Options Replay"

app.index_string = f"""<!DOCTYPE html>
<html>
<head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
    <style>{CUSTOM_CSS}</style>
</head>
<body>
    {{%app_entry%}}
    <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body>
</html>"""


def _input_style():
    return {
        "backgroundColor": C["surface"],
        "color": C["text"],
        "border": f"1px solid {C['border']}",
        "borderRadius": "3px",
        "padding": "8px 10px",
        "fontFamily": "'JetBrains Mono', monospace",
        "fontSize": "12px",
    }


TAB_STYLE = {
    "backgroundColor": C["surface"],
    "color": C["text3"],
    "border": f"1px solid {C['border']}",
    "borderBottom": "none",
    "padding": "10px 20px",
    "fontFamily": "Outfit, sans-serif",
    "fontSize": "12px",
    "fontWeight": "600",
    "letterSpacing": "1px",
    "textTransform": "uppercase",
}
TAB_SELECTED = {
    **TAB_STYLE,
    "backgroundColor": C["bg"],
    "color": C["gold"],
    "borderTop": f"2px solid {C['gold']}",
}


# ── Layout ──────────────────────────────────────────────────
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Options Replay", style={
            **_serif("32px", C["text"]),
            "margin": "0",
            "background": f"linear-gradient(135deg, {C['gold']}, {C['text']})",
            "-webkit-background-clip": "text",
            "-webkit-text-fill-color": "transparent",
        }),
        html.Span("Replay headline trades through the options chain",
                   style={**_mono("12px", "400", C["text3"]), "marginLeft": "16px"}),
    ], style={
        "padding": "20px 30px",
        "borderBottom": f"1px solid {C['border']}",
        "display": "flex",
        "alignItems": "baseline",
        "gap": "12px",
    }),

    # Tabs
    dcc.Tabs(id="main-tabs", value="single", children=[
        dcc.Tab(label="SINGLE TRADE", value="single",
                style=TAB_STYLE, selected_style=TAB_SELECTED),
        dcc.Tab(label="BATCH ANALYSIS", value="batch",
                style=TAB_STYLE, selected_style=TAB_SELECTED),
        dcc.Tab(label="SYSTEMS", value="systems",
                style=TAB_STYLE, selected_style=TAB_SELECTED),
        dcc.Tab(label="CAPITULATION", value="capitulation",
                style=TAB_STYLE, selected_style=TAB_SELECTED),
    ], style={"margin": "0 30px"}),

    # Tab content
    html.Div(id="tab-content"),

], style={"backgroundColor": C["bg"], "minHeight": "100vh"})


# ── Tab switching ───────────────────────────────────────────
@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
)
def render_tab(tab):
    if tab == "batch":
        return _batch_layout()
    if tab == "systems":
        return _systems_layout()
    if tab == "capitulation":
        return _cap_layout()
    return _single_trade_layout()


# ═══════════════════════════════════════════════════════════
# TAB 1: SINGLE TRADE
# ═══════════════════════════════════════════════════════════

def _single_trade_layout():
    return html.Div([
        # Trade Selector
        html.Div([
            html.Div([
                html.Label("Select a Trade", style=_label()),
                dcc.Dropdown(
                    id="trade-dropdown",
                    options=TRADE_OPTIONS,
                    placeholder="Pick a trade from your history...",
                    style={"fontSize": "12px"},
                ),
            ], style={"flex": "1"}),

            html.Div("OR", style={
                **_mono("12px", "600", C["text3"]),
                "padding": "24px 16px 0 16px",
            }),

            html.Div([
                html.Label("Manual Entry", style=_label()),
                html.Div([
                    dcc.Input(id="manual-ticker", placeholder="Ticker",
                             style={**_input_style(), "width": "80px"}),
                    dcc.Input(id="manual-date", placeholder="2025-03-12",
                             style={**_input_style(), "width": "110px"}),
                    dcc.Input(id="manual-time", placeholder="14:30",
                             style={**_input_style(), "width": "70px"}),
                    dcc.Dropdown(id="manual-direction",
                                options=[{"label": "Long", "value": "LONG"},
                                         {"label": "Short", "value": "SHORT"}],
                                value="LONG",
                                style={"width": "90px", "fontSize": "12px"}),
                    html.Button("Analyze", id="manual-go-btn", style={
                        "backgroundColor": C["gold"],
                        "color": C["bg"],
                        "border": "none",
                        "borderRadius": "3px",
                        "padding": "8px 16px",
                        "fontWeight": "600",
                        "fontSize": "12px",
                        "cursor": "pointer",
                    }),
                ], style={"display": "flex", "gap": "6px", "alignItems": "center"}),
            ]),
        ], style={
            **_card(),
            "display": "flex",
            "alignItems": "flex-start",
            "gap": "16px",
            "margin": "16px 30px",
        }),

        # Window slider
        html.Div([
            html.Label("Analysis Window", style=_label()),
            dcc.Slider(
                id="window-slider",
                min=15, max=45, step=15, value=30,
                marks={15: "15 min", 30: "30 min", 45: "45 min"},
            ),
        ], style={
            **_card(),
            "margin": "0 30px 16px 30px",
            "padding": "12px 20px",
        }),

        # Risk budget
        html.Div([
            html.Label("Risk Budget", style=_label()),
            html.Div([
                html.Span("$", style=_mono("14px", "600", C["text3"])),
                dcc.Input(
                    id="risk-budget-input",
                    type="number",
                    value=500,
                    min=100,
                    max=10000,
                    step=100,
                    style={**_input_style(), "width": "100px"},
                ),
                html.Span("max premium at risk per trade",
                           style=_mono("11px", "400", C["text3"])),
            ], style={"display": "flex", "gap": "8px", "alignItems": "center"}),
        ], style={
            **_card(),
            "margin": "0 30px 16px 30px",
            "padding": "12px 20px",
        }),

        # Results
        dcc.Loading(
            id="loading-results",
            type="default",
            color=C["gold"],
            children=html.Div(id="results-container", style={"padding": "0 30px 30px 30px"}),
        ),
    ])


# ── Single trade callback ──────────────────────────────────
@app.callback(
    Output("results-container", "children"),
    [Input("trade-dropdown", "value"),
     Input("manual-go-btn", "n_clicks")],
    [State("manual-ticker", "value"),
     State("manual-date", "value"),
     State("manual-time", "value"),
     State("manual-direction", "value"),
     State("window-slider", "value"),
     State("risk-budget-input", "value")],
    prevent_initial_call=True,
)
def analyze_trade(dropdown_value, n_clicks, manual_ticker, manual_date,
                  manual_time, manual_direction, hold_minutes, risk_budget):
    """Main analysis callback — triggered by trade selection or manual entry."""
    ctx = callback_context
    if not ctx.triggered:
        return no_update

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    hold_minutes = hold_minutes or 30

    # Determine trade parameters
    if trigger_id == "trade-dropdown" and dropdown_value is not None:
        row = TRADES_DF.iloc[dropdown_value]
        trade = row.to_dict()
    elif trigger_id == "manual-go-btn":
        if not all([manual_ticker, manual_date, manual_time]):
            return _error_card("Please fill in Ticker, Date, and Time.")
        try:
            trade = parse_manual_input(manual_ticker, manual_date, manual_time,
                                       manual_direction or "LONG")
        except Exception as e:
            return _error_card(f"Invalid input: {e}")
    else:
        return no_update

    symbol = trade["symbol"]
    date_str = str(trade["date"])
    entry_time = trade["entry_time"]
    if isinstance(entry_time, str):
        entry_time = pd.Timestamp(entry_time)
    time_of_day = entry_time.strftime("%H:%M:%S")
    side = int(trade.get("side", 1))

    if not check_terminal_running():
        return _error_card(
            "Theta Terminal is not running. Please start the Java process "
            "and ensure it's listening on localhost:25503."
        )

    # Fetch underlying
    try:
        from data_queries.polygon_queries import get_intraday
        underlying_df = get_intraday(symbol, date_str, 1, "minute")
        if underlying_df is None:
            underlying_df = pd.DataFrame()
    except Exception as e:
        logger.warning("Failed to fetch underlying for %s: %s", symbol, e)
        underlying_df = pd.DataFrame()

    underlying_price = float(trade.get("avg_price", 0))
    if underlying_price <= 0 and not underlying_df.empty:
        try:
            entry_tz = entry_time
            if hasattr(underlying_df.index, 'tz') and underlying_df.index.tz is not None:
                if entry_tz.tzinfo is None:
                    from pytz import timezone
                    entry_tz = timezone("US/Eastern").localize(entry_tz)
            idx = underlying_df.index.get_indexer([entry_tz], method="nearest")[0]
            underlying_price = float(underlying_df.iloc[idx]["close"])
        except Exception:
            underlying_price = float(underlying_df["close"].iloc[0])

    if underlying_price <= 0:
        return _error_card(f"Could not determine underlying price for {symbol} on {date_str}.")

    # Chain snapshot
    try:
        chain_df = get_chain_snapshot(symbol, date_str, time_of_day)
    except ThetaTerminalOfflineError as e:
        return _error_card(str(e))
    except Exception as e:
        return _error_card(f"Failed to fetch chain snapshot: {e}")

    if chain_df.empty:
        return _error_card(f"No options chain data for {symbol} on {date_str} at {time_of_day}.")

    filtered = filter_chain(chain_df, underlying_price, side, date_str)
    if filtered.empty:
        return _error_card(
            f"No liquid options found for {symbol} at ${underlying_price:.2f}. "
            "The chain may have had insufficient liquidity."
        )

    # Per-contract data (parallel)
    ohlc_dict = {}
    quotes_dict = {}
    greeks_dict = {}

    def _fetch_contract(row):
        key = contract_key(row)
        ohlc = get_option_ohlc(symbol, row["expiration"], row["strike"],
                               row["right"], date_str)
        quotes = get_option_quotes(symbol, row["expiration"], row["strike"],
                                   row["right"], date_str)
        greeks = get_option_greeks(symbol, row["expiration"], row["strike"],
                                   row["right"], date_str)
        return key, ohlc, quotes, greeks

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_fetch_contract, row): idx
                   for idx, row in filtered.iterrows()}
        for future in as_completed(futures):
            try:
                key, ohlc, quotes, greeks = future.result()
                ohlc_dict[key] = ohlc
                quotes_dict[key] = quotes
                greeks_dict[key] = greeks
            except Exception as e:
                logger.warning("Contract fetch failed: %s", e)

    # Compute returns and score
    returns_df = compute_option_returns(filtered, ohlc_dict, quotes_dict,
                                        entry_time.to_pydatetime() if hasattr(entry_time, 'to_pydatetime') else entry_time,
                                        hold_minutes, greeks_dict=greeks_dict)
    scored_df = score_options(returns_df)

    if scored_df.empty:
        return _error_card("Scoring produced no results. The contracts may have had no price movement.")

    # Contract picker
    risk_budget = risk_budget or 500
    picker_result = pick_contracts(scored_df, risk_budget=risk_budget)

    # Ideal play
    underlying_max = underlying_price
    if not underlying_df.empty:
        try:
            snap = entry_time
            if hasattr(underlying_df.index, 'tz') and underlying_df.index.tz is not None and snap.tzinfo is None:
                from pytz import timezone
                snap = timezone("US/Eastern").localize(snap)
            end = snap + timedelta(minutes=hold_minutes)
            window = underlying_df[(underlying_df.index >= snap) & (underlying_df.index <= end)]
            if not window.empty:
                if side == 1:
                    underlying_max = float(window["high"].max())
                else:
                    underlying_max = float(window["low"].min())
        except Exception:
            pass

    ideal = compute_ideal_play_summary(scored_df.iloc[0], underlying_price, underlying_max, side)

    # Build figures
    entry_dt = entry_time.to_pydatetime() if hasattr(entry_time, 'to_pydatetime') else entry_time

    exit_time_raw = trade.get("exit_time")
    exit_dt = None
    if exit_time_raw:
        try:
            exit_dt = pd.Timestamp(exit_time_raw).to_pydatetime()
        except Exception:
            pass

    fig_underlying = fig_underlying_candlestick(
        underlying_df, entry_dt, exit_dt, underlying_price, side, hold_minutes)
    fig_heatmap = fig_chain_heatmap(filtered, underlying_price)
    fig_table = fig_top_options_table(scored_df)
    fig_returns = fig_return_comparison(scored_df)
    fig_liquidity = fig_liquidity_comparison(scored_df)
    option_figs = fig_option_price_grid(scored_df, quotes_dict, ohlc_dict,
                                         entry_dt, hold_minutes)

    # Assemble layout
    direction = "LONG" if side == 1 else "SHORT"
    net_pnl = trade.get("net_pnl", 0)
    pnl_color = C["profit"] if net_pnl >= 0 else C["loss"]
    pnl_str = f"+${net_pnl:,.0f}" if net_pnl >= 0 else f"-${abs(net_pnl):,.0f}"

    children = [
        # Trade context card
        html.Div([
            html.Div([
                html.Span(symbol, style={**_mono("20px", "700", C["gold"])}),
                html.Span(f"  {date_str}", style=_mono("14px", "400", C["text2"])),
                html.Span(f"  {time_of_day} ET", style=_mono("14px", "400", C["text2"])),
                html.Span(f"  {direction}", style=_mono("14px", "600",
                          C["profit"] if side == 1 else C["loss"])),
                html.Span(f"  {pnl_str}", style=_mono("14px", "600", pnl_color)),
            ], style={"marginBottom": "6px"}),
            html.Div([
                html.Span(f"Underlying: ${underlying_price:.2f}", style=_mono("11px", "400", C["text3"])),
                html.Span(f"  |  LQA: {trade.get('lqa_score', 'N/A')}", style=_mono("11px", "400", C["text3"])),
                html.Span(f"  |  {trade.get('news_summary', '')[:80]}", style=_mono("11px", "400", C["text3"])),
            ]),
        ], style=_card(accent_left=C["gold"])),

        # Underlying chart
        dcc.Graph(figure=fig_underlying, style={"height": "450px"}),

        # Contract Picker card
        _contract_picker_card(picker_result),

        # Ranked table
        dcc.Graph(figure=fig_table, style={"height": f"{max(200, 50 + len(scored_df) * 32)}px"}),

        # Returns vs Liquidity
        html.Div([
            html.Div([dcc.Graph(figure=fig_returns)], style={"flex": "1"}),
            html.Div([dcc.Graph(figure=fig_liquidity)], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "12px"}),

        # Chain heatmap
        dcc.Graph(figure=fig_heatmap, style={"height": "300px"}),

        # Option price charts
        html.Div("OPTION PRICE CHARTS", style={**_label(), "marginTop": "20px", "marginBottom": "10px"}),
    ]

    for i in range(0, len(option_figs), 2):
        row_children = []
        for j in range(2):
            if i + j < len(option_figs):
                row_children.append(
                    html.Div([dcc.Graph(figure=option_figs[i + j])],
                             style={"flex": "1"})
                )
        children.append(
            html.Div(row_children, style={"display": "flex", "gap": "8px"})
        )

    return html.Div(children, className="anim-stagger")


# ═══════════════════════════════════════════════════════════
# TAB 2: BATCH ANALYSIS
# ═══════════════════════════════════════════════════════════

def _batch_filter_options():
    """Build batch filter dropdown options with counts."""
    opts = [
        {"label": f"All $10K+ trades ({len(TRADE_OPTIONS)})", "value": "all"},
        {"label": "Long only", "value": "long"},
        {"label": "Short only", "value": "short"},
        {"label": "Top 50 by P&L", "value": "top50"},
        {"label": "Top 100 by P&L", "value": "top100"},
    ]
    if not TRADES_DF.empty and "setup_type" in TRADES_DF.columns:
        news = TRADES_DF[TRADES_DF["setup_type"] == "news"]
        n5k = len(news[news["net_pnl"] >= 5000])
        n1k = len(news[news["net_pnl"] >= 1000])
        nall = len(news[news["net_pnl"] > 0])
        opts.extend([
            {"label": f"News Winners $5K+ ({n5k})", "value": "news_5k"},
            {"label": f"News Winners $1K+ ({n1k})", "value": "news_1k"},
            {"label": f"News Winners All ({nall})", "value": "news_all"},
        ])
    return opts


def _batch_layout():
    return html.Div([
        # Controls
        html.Div([
            html.Div([
                html.Label("Trade Filter", style=_label()),
                dcc.Dropdown(
                    id="batch-filter",
                    options=_batch_filter_options(),
                    value="top50",
                    style={"fontSize": "12px"},
                ),
            ], style={"flex": "1"}),

            html.Div([
                html.Label("Hold Windows", style=_label()),
                dcc.Checklist(
                    id="batch-windows",
                    options=[
                        {"label": " 5m", "value": 5},
                        {"label": " 15m", "value": 15},
                        {"label": " 30m", "value": 30},
                    ],
                    value=[5, 15, 30],
                    inline=True,
                    style={**_mono("12px", "400", C["text"]), "marginTop": "8px"},
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "12px"},
                ),
            ], style={"flex": "1"}),

            html.Div([
                html.Button("Run Batch", id="batch-run-btn", style={
                    "backgroundColor": C["gold"],
                    "color": C["bg"],
                    "border": "none",
                    "borderRadius": "3px",
                    "padding": "10px 20px",
                    "fontWeight": "700",
                    "fontSize": "12px",
                    "cursor": "pointer",
                    "marginTop": "20px",
                }),
                html.Button("Load Last", id="batch-load-btn", style={
                    "backgroundColor": C["surface"],
                    "color": C["text2"],
                    "border": f"1px solid {C['border']}",
                    "borderRadius": "3px",
                    "padding": "10px 16px",
                    "fontWeight": "600",
                    "fontSize": "12px",
                    "cursor": "pointer",
                    "marginTop": "20px",
                    "marginLeft": "8px",
                }),
            ], style={"display": "flex"}),
        ], style={
            **_card(),
            "display": "flex",
            "alignItems": "flex-start",
            "gap": "20px",
            "margin": "16px 30px",
        }),

        # Progress
        html.Div([
            html.Div(id="batch-progress-text",
                     style={**_mono("12px", "400", C["text2"]), "marginBottom": "6px"}),
            html.Div([
                html.Div(id="batch-progress-bar", style={
                    "width": "0%",
                    "height": "4px",
                    "backgroundColor": C["gold"],
                    "borderRadius": "2px",
                    "transition": "width 0.3s ease",
                }),
            ], style={
                "backgroundColor": C["border"],
                "borderRadius": "2px",
                "height": "4px",
                "overflow": "hidden",
            }),
        ], id="batch-progress-container", style={
            **_card(),
            "margin": "0 30px 16px 30px",
            "display": "none",
        }),

        # Results
        html.Div(id="batch-results-container", style={"padding": "0 30px 30px 30px"}),

        # Polling interval
        dcc.Interval(id="batch-poll", interval=2000, disabled=True),

    ], style={"padding": "0"})


def _get_trade_indices(filter_value: str) -> list:
    """Get trade indices based on filter selection."""
    if TRADES_DF.empty:
        return []

    df = TRADES_DF.copy()

    # News-specific filters (no $10K minimum)
    if filter_value == "news_5k":
        mask = (df["net_pnl"] >= 5000) & (df["setup_type"] == "news")
        return df[mask].index.tolist()
    elif filter_value == "news_1k":
        mask = (df["net_pnl"] >= 1000) & (df["setup_type"] == "news")
        return df[mask].index.tolist()
    elif filter_value == "news_all":
        mask = (df["net_pnl"] > 0) & (df["setup_type"] == "news")
        return df[mask].index.tolist()

    # Standard filters: start from $10K+ base
    mask = df["net_pnl"].abs() >= 10_000

    if filter_value == "long":
        mask = mask & (df["side"] == 1)
    elif filter_value == "short":
        mask = mask & (df["side"] == -1)

    filtered = df[mask].copy()

    if filter_value == "top50":
        filtered = filtered.nlargest(50, "net_pnl")
    elif filter_value == "top100":
        filtered = filtered.nlargest(100, "net_pnl")

    return filtered.index.tolist()


# ── Batch: Start ───────────────────────────────────────────
@app.callback(
    [Output("batch-poll", "disabled", allow_duplicate=True),
     Output("batch-progress-container", "style", allow_duplicate=True),
     Output("batch-progress-text", "children", allow_duplicate=True),
     Output("batch-progress-bar", "style", allow_duplicate=True)],
    Input("batch-run-btn", "n_clicks"),
    [State("batch-filter", "value"),
     State("batch-windows", "value")],
    prevent_initial_call=True,
)
def start_batch(n_clicks, filter_value, windows):
    if _batch_state["running"]:
        return no_update, no_update, no_update, no_update

    trade_indices = _get_trade_indices(filter_value or "top50")
    if not trade_indices:
        return True, no_update, "No trades match the filter.", no_update

    windows = windows or [5, 15, 30]

    _batch_state.update({
        "running": True,
        "completed": 0,
        "total": len(trade_indices),
        "current_symbol": "",
        "results_path": None,
        "error": None,
    })

    def _progress_cb(completed, total, symbol):
        _batch_state["completed"] = completed
        _batch_state["current_symbol"] = symbol

    def _run():
        try:
            from options_replay.batch_analyzer import run_batch, save_batch_results
            results = run_batch(TRADES_DF, trade_indices,
                               hold_windows=windows,
                               progress_callback=_progress_cb)
            if not results.empty:
                path = save_batch_results(results)
                _batch_state["results_path"] = str(path)
            else:
                _batch_state["error"] = "No results produced — all trades may have failed."
        except Exception as e:
            logger.error("Batch failed: %s", e)
            _batch_state["error"] = str(e)
        finally:
            _batch_state["running"] = False

    threading.Thread(target=_run, daemon=True).start()

    progress_style = {
        **_card(),
        "margin": "0 30px 16px 30px",
        "display": "block",
    }
    bar_style = {"width": "0%", "height": "4px",
                 "backgroundColor": C["gold"], "borderRadius": "2px",
                 "transition": "width 0.3s ease"}

    return False, progress_style, f"Starting batch (0/{len(trade_indices)})...", bar_style


# ── Batch: Poll progress ──────────────────────────────────
@app.callback(
    [Output("batch-progress-text", "children", allow_duplicate=True),
     Output("batch-progress-bar", "style", allow_duplicate=True),
     Output("batch-results-container", "children"),
     Output("batch-poll", "disabled"),
     Output("batch-progress-container", "style")],
    Input("batch-poll", "n_intervals"),
    prevent_initial_call=True,
)
def poll_batch(n):
    completed = _batch_state["completed"]
    total = _batch_state["total"] or 1
    symbol = _batch_state["current_symbol"]
    pct = (completed / total) * 100

    bar_style = {"width": f"{pct:.0f}%", "height": "4px",
                 "backgroundColor": C["gold"], "borderRadius": "2px",
                 "transition": "width 0.3s ease"}
    progress_style = {**_card(), "margin": "0 30px 16px 30px", "display": "block"}
    text = f"Processing: {completed}/{total}  {symbol}  ({pct:.0f}%)"

    if not _batch_state["running"]:
        # Done
        if _batch_state["error"]:
            return (
                f"Error: {_batch_state['error']}",
                bar_style,
                _error_card(_batch_state["error"]),
                True,
                progress_style,
            )

        # Load and render results
        results_children = _render_batch_results(_batch_state["results_path"])
        return (
            f"Complete: {total} trades processed",
            {"width": "100%", "height": "4px",
             "backgroundColor": C["profit"], "borderRadius": "2px"},
            results_children,
            True,
            progress_style,
        )

    return text, bar_style, no_update, False, progress_style


# ── Batch: Load last ──────────────────────────────────────
@app.callback(
    Output("batch-results-container", "children", allow_duplicate=True),
    Input("batch-load-btn", "n_clicks"),
    prevent_initial_call=True,
)
def load_last_batch(n_clicks):
    from options_replay.batch_analyzer import load_batch_results
    df = load_batch_results()
    if df is None:
        return _error_card("No saved batch results found in data/batch_results/.")
    return _render_batch_results_from_df(df)


def _render_batch_results(results_path: str):
    """Load a CSV path and render batch results."""
    try:
        df = pd.read_csv(results_path)
        return _render_batch_results_from_df(df)
    except Exception as e:
        return _error_card(f"Failed to load results: {e}")


def _render_batch_results_from_df(df: pd.DataFrame):
    """Build the full batch results dashboard from a DataFrame."""
    from options_replay.batch_aggregator import (
        add_buckets, compute_category_stats,
        compute_cross_stats, compute_hold_window_stats, compute_summary,
    )
    from options_replay.batch_charts import (
        fig_return_by_delta_bucket, fig_return_by_dte_bucket,
        fig_return_by_moneyness, fig_delta_vs_return_scatter,
        fig_iv_vs_return_scatter, fig_moneyness_dte_heatmap,
        fig_hold_window_comparison, fig_spread_cost_by_category,
        fig_return_distribution, fig_win_rate_by_category,
        fig_summary_stats_table,
    )

    df = add_buckets(df)
    summary = compute_summary(df)

    # Stats at 30-min window (or largest available)
    default_window = 30 if 30 in df["hold_window"].unique() else df["hold_window"].max()

    delta_agg = compute_category_stats(df, "delta_bucket", hold_window=default_window)
    dte_agg = compute_category_stats(df, "dte_bucket", hold_window=default_window)
    moneyness_agg = compute_category_stats(df, "moneyness_5", hold_window=default_window)
    window_stats = compute_hold_window_stats(df, "moneyness_5")

    values_pivot, count_pivot = compute_cross_stats(
        df, "moneyness_5", "dte_bucket", hold_window=default_window
    )

    children = [
        # Summary table
        dcc.Graph(figure=fig_summary_stats_table(summary), style={"height": "310px"}),

        # Row 1: Scatters
        html.Div([
            html.Div([dcc.Graph(figure=fig_delta_vs_return_scatter(
                df[df["hold_window"] == default_window] if "hold_window" in df.columns else df
            ))], style={"flex": "1"}),
            html.Div([dcc.Graph(figure=fig_iv_vs_return_scatter(
                df[df["hold_window"] == default_window] if "hold_window" in df.columns else df
            ))], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "12px"}),

        # Row 2: Return by delta / DTE
        html.Div([
            html.Div([dcc.Graph(figure=fig_return_by_delta_bucket(delta_agg))], style={"flex": "1"}),
            html.Div([dcc.Graph(figure=fig_return_by_dte_bucket(dte_agg))], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "12px"}),

        # Row 3: Moneyness / Win rate
        html.Div([
            html.Div([dcc.Graph(figure=fig_return_by_moneyness(moneyness_agg))], style={"flex": "1"}),
            html.Div([dcc.Graph(figure=fig_win_rate_by_category(delta_agg, moneyness_agg))], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "12px"}),

        # Row 4: Heatmap (full width)
        dcc.Graph(figure=fig_moneyness_dte_heatmap(values_pivot, count_pivot)),

        # Row 5: Hold window / Spread cost
        html.Div([
            html.Div([dcc.Graph(figure=fig_hold_window_comparison(window_stats))], style={"flex": "1"}),
            html.Div([dcc.Graph(figure=fig_spread_cost_by_category(delta_agg, moneyness_agg))], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "12px"}),

        # Row 6: Distribution (full width)
        dcc.Graph(figure=fig_return_distribution(
            df[df["hold_window"] == default_window] if "hold_window" in df.columns else df
        )),
    ]

    return html.Div(children, className="anim-stagger")


# ═══════════════════════════════════════════════════════════
# TAB 3: SYSTEMS
# ═══════════════════════════════════════════════════════════

def _systems_layout():
    return html.Div([
        # Controls
        html.Div([
            html.Div([
                html.Label("Setup Type", style=_label()),
                dcc.Dropdown(
                    id="systems-setup-filter",
                    options=[
                        {"label": "All Setups", "value": "all"},
                        {"label": "News Only", "value": "news"},
                        {"label": "Momentum", "value": "momo"},
                        {"label": "Capitulation", "value": "capitulation"},
                    ],
                    value="all",
                    style={"fontSize": "12px", "minWidth": "160px"},
                ),
            ], style={"flex": "0 0 auto"}),

            html.Div([
                html.Button("Load Batch Data", id="systems-load-btn", style={
                    "backgroundColor": C["gold"],
                    "color": C["bg"],
                    "border": "none",
                    "borderRadius": "3px",
                    "padding": "10px 20px",
                    "fontWeight": "700",
                    "fontSize": "12px",
                    "cursor": "pointer",
                    "marginTop": "20px",
                }),
                html.Span(id="systems-status",
                          style={**_mono("12px", "400", C["text3"]), "marginLeft": "12px"}),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={
            **_card(),
            "display": "flex",
            "alignItems": "flex-start",
            "gap": "20px",
            "margin": "16px 30px",
        }),

        # Results
        html.Div(id="systems-results-container", style={"padding": "0 30px 30px 30px"}),
    ], style={"padding": "0"})


@app.callback(
    [Output("systems-results-container", "children"),
     Output("systems-status", "children")],
    Input("systems-load-btn", "n_clicks"),
    State("systems-setup-filter", "value"),
    prevent_initial_call=True,
)
def load_systems_data(n_clicks, setup_filter):
    from options_replay.batch_analyzer import load_batch_results
    from options_replay.systems_analyzer import (
        prepare_systems_data, compute_hotkey_grid,
        compute_price_otm_grid, recommend_hotkeys, compute_delta_profile,
    )
    from options_replay.systems_charts import (
        fig_hotkey_heatmap, fig_price_otm_grid, fig_hotkey_comparison_table,
        fig_delta_distribution, fig_return_distribution_by_hotkey,
        fig_price_vs_return_scatter,
    )

    df = load_batch_results()
    if df is None:
        return _error_card("No batch results found. Run a batch analysis first."), "No data"

    setup_type = setup_filter if setup_filter and setup_filter != "all" else None
    sys_df = prepare_systems_data(df, setup_type=setup_type)
    if sys_df.empty:
        return _error_card("No OTM/ATM contracts found in batch results."), "No data"

    grid_df = compute_hotkey_grid(sys_df)
    price_otm = compute_price_otm_grid(sys_df)
    recs = recommend_hotkeys(grid_df)

    # Build recommendation cards
    rec_cards = []
    for rec in recs:
        label = rec.get("label", "")
        label_color = C["loss"] if label == "AGGRESSIVE" else C["steel"] if label == "CONSERVATIVE" else C["gold"]
        profile = compute_delta_profile(sys_df, rec["max_price"], rec["max_otm"])
        delta_range = f"{profile.get('delta_q25', 0):.2f}–{profile.get('delta_q75', 0):.2f}"

        rec_cards.append(html.Div([
            html.Div(label, style={
                **_mono("10px", "700", label_color),
                "letterSpacing": "1.5px",
                "marginBottom": "6px",
            }),
            html.Div(f"<${rec['max_price']:.2f}  &  <{rec['max_otm']:.0f}% OTM",
                      style=_mono("14px", "700", C["text"])),
            html.Div([
                _stat_pill("Count", str(int(rec["count"]))),
                _stat_pill("Avg Return", f"{rec['avg_return']:.0%}",
                           color=C["profit"] if rec["avg_return"] > 0 else C["loss"]),
                _stat_pill("Win Rate", f"{rec['win_rate']:.0%}",
                           color=C["profit"] if rec["win_rate"] > 0.7 else C["text"]),
                _stat_pill("Edge", f"{rec['edge']:.1f}"),
                _stat_pill("Spread Cost", f"{rec['avg_spread_cost']:.0%}", color=C["loss"]),
                _stat_pill("Delta Range", delta_range),
            ], style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginTop": "8px"}),
            html.P(rec.get("rationale", ""), style={
                **_mono("10px", "400", C["text3"]),
                "margin": "6px 0 0 0",
            }),
        ], style={
            **_card(),
            "flex": "1",
            "borderLeft": f"3px solid {label_color}",
        }))

    children = [
        # Recommended hotkeys header
        html.Div("RECOMMENDED HOTKEYS", style={**_label(), "marginBottom": "10px"}),

        # Hotkey cards row
        html.Div(rec_cards, style={"display": "flex", "gap": "12px", "marginBottom": "16px"}),

        # Row 1: Heatmaps
        html.Div([
            html.Div([dcc.Graph(figure=fig_hotkey_heatmap(grid_df, "edge"))], style={"flex": "1"}),
            html.Div([dcc.Graph(figure=fig_price_otm_grid(price_otm))], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "12px"}),

        # Row 2: Distributions
        html.Div([
            html.Div([dcc.Graph(figure=fig_delta_distribution(sys_df, recs))], style={"flex": "1"}),
            html.Div([dcc.Graph(figure=fig_return_distribution_by_hotkey(sys_df, recs))], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "12px"}),

        # Row 3: Price vs Return scatter (full width)
        dcc.Graph(figure=fig_price_vs_return_scatter(sys_df)),

        # Row 4: Comparison table (full width)
        dcc.Graph(figure=fig_hotkey_comparison_table(recs)),
    ]

    filter_label = f" [{setup_filter}]" if setup_type else ""
    status = f"{len(sys_df)} OTM/ATM contracts loaded ({len(df)} total in batch){filter_label}"
    return html.Div(children, className="anim-stagger"), status


# ═══════════════════════════════════════════════════════════
# TAB 4: CAPITULATION
# ═══════════════════════════════════════════════════════════

def _cap_layout():
    return html.Div([
        # Controls
        html.Div([
            html.Div([
                html.Label("Trade Type", style=_label()),
                dcc.Dropdown(
                    id="cap-type-filter",
                    options=[
                        {"label": "All", "value": "all"},
                        {"label": "Bounce (Longs)", "value": "bounce"},
                        {"label": "Reversal (Shorts)", "value": "reversal"},
                    ],
                    value="all",
                    style={"fontSize": "12px", "minWidth": "150px"},
                ),
            ], style={"flex": "0 0 auto"}),

            html.Div([
                html.Label("Entry Offset", style=_label()),
                dcc.Dropdown(
                    id="cap-offset-filter",
                    options=[
                        {"label": "All Offsets", "value": "all"},
                        {"label": "Low+0 (at low)", "value": "low+0"},
                        {"label": "Low+2", "value": "low+2"},
                        {"label": "Low+5", "value": "low+5"},
                        {"label": "High+0 (at high)", "value": "high+0"},
                        {"label": "Open+0", "value": "open+0"},
                        {"label": "Open+15", "value": "open+15"},
                    ],
                    value="all",
                    style={"fontSize": "12px", "minWidth": "140px"},
                ),
            ], style={"flex": "0 0 auto"}),

            html.Div([
                html.Button("Load Cap Data", id="cap-load-btn", style={
                    "backgroundColor": C["gold"],
                    "color": C["bg"],
                    "border": "none",
                    "borderRadius": "3px",
                    "padding": "10px 20px",
                    "fontWeight": "700",
                    "fontSize": "12px",
                    "cursor": "pointer",
                    "marginTop": "20px",
                }),
                html.Span(id="cap-status",
                          style={**_mono("12px", "400", C["text3"]), "marginLeft": "12px"}),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={
            **_card(),
            "display": "flex",
            "alignItems": "flex-start",
            "gap": "20px",
            "margin": "16px 30px",
        }),

        # Results
        html.Div(id="cap-results-container", style={"padding": "0 30px 30px 30px"}),
    ], style={"padding": "0"})


@app.callback(
    [Output("cap-results-container", "children"),
     Output("cap-status", "children")],
    Input("cap-load-btn", "n_clicks"),
    [State("cap-type-filter", "value"),
     State("cap-offset-filter", "value")],
    prevent_initial_call=True,
)
def load_cap_data(n_clicks, type_filter, offset_filter):
    from options_replay.cap_batch_analyzer import load_cap_results
    from options_replay.cap_systems_analyzer import (
        prepare_cap_data, compute_target_hit_rates,
        compute_entry_offset_comparison, compute_cap_hotkey_grid,
        recommend_cap_hotkeys, compute_iv_analysis, compute_cap_summary,
    )
    from options_replay.cap_charts import (
        fig_target_hit_heatmap, fig_entry_offset_comparison,
        fig_time_to_target, fig_cap_hotkey_heatmap,
        fig_iv_analysis, fig_bounce_vs_reversal,
    )

    df = load_cap_results()
    if df is None:
        return _error_card("No cap batch results found. Run a capitulation batch first."), "No data"

    trade_type = type_filter if type_filter and type_filter != "all" else None
    entry_offset = offset_filter if offset_filter and offset_filter != "all" else None

    cap_df = prepare_cap_data(df, trade_type=trade_type, entry_offset=entry_offset)
    if cap_df.empty:
        return _error_card("No OTM/ATM contracts found in cap batch results."), "No data"

    summary = compute_cap_summary(cap_df)
    target_rates = compute_target_hit_rates(cap_df)
    offset_comp = compute_entry_offset_comparison(df)  # use full df for offset comparison
    grid_df = compute_cap_hotkey_grid(cap_df)
    recs = recommend_cap_hotkeys(grid_df)
    iv_data = compute_iv_analysis(cap_df)

    # Summary cards
    summary_items = [
        _stat_pill("Contracts", str(summary.get("total_contracts", 0))),
        _stat_pill("Trades", str(summary.get("unique_trades", 0))),
        _stat_pill("Symbols", str(summary.get("unique_symbols", 0))),
        _stat_pill("Win Rate", f"{summary.get('win_rate', 0):.0%}",
                   color=C["profit"] if summary.get("win_rate", 0) > 0.5 else C["loss"]),
        _stat_pill("Avg Return", f"{summary.get('avg_return', 0):.0%}",
                   color=C["profit"] if summary.get("avg_return", 0) > 0 else C["loss"]),
    ]
    # Add target hit rates
    for target in ["0.5x", "1.0x", "1.5x"]:
        hr = summary.get(f"target_{target}_hit_rate")
        if hr is not None:
            summary_items.append(_stat_pill(f"{target} ATR Hit", f"{hr:.0%}",
                                           color=C["profit"] if hr > 0.5 else C["text2"]))

    # Recommendation cards
    rec_cards = []
    for rec in recs:
        label = rec.get("label", "")
        label_color = C["loss"] if label == "AGGRESSIVE" else C["steel"] if label == "BALANCED" else C["gold"]
        rec_cards.append(html.Div([
            html.Div(label, style={
                **_mono("10px", "700", label_color),
                "letterSpacing": "1.5px",
                "marginBottom": "6px",
            }),
            html.Div(f"<${rec['max_price']:.2f}  &  <{rec['max_otm']:.0f}% OTM",
                      style=_mono("14px", "700", C["text"])),
            html.Div([
                _stat_pill("Count", str(int(rec["count"]))),
                _stat_pill("Avg Return", f"{rec['avg_return']:.0%}",
                           color=C["profit"] if rec["avg_return"] > 0 else C["loss"]),
                _stat_pill("Win Rate", f"{rec['win_rate']:.0%}",
                           color=C["profit"] if rec["win_rate"] > 0.5 else C["text"]),
                _stat_pill("Edge", f"{rec['edge']:.2f}"),
            ], style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginTop": "8px"}),
            html.P(rec.get("rationale", ""), style={
                **_mono("10px", "400", C["text3"]),
                "margin": "6px 0 0 0",
            }),
        ], style={
            **_card(),
            "flex": "1",
            "borderLeft": f"3px solid {label_color}",
        }))

    children = [
        # Summary
        html.Div("CAPITULATION OVERVIEW", style={**_label(), "marginBottom": "10px"}),
        html.Div(summary_items, style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "16px"}),

        # Hotkey recommendations
        html.Div("RECOMMENDED HOTKEYS", style={**_label(), "marginBottom": "10px"}),
        html.Div(rec_cards, style={"display": "flex", "gap": "12px", "marginBottom": "16px"}),

        # Row 1: Target heatmap + hotkey heatmap
        html.Div([
            html.Div([dcc.Graph(figure=fig_target_hit_heatmap(target_rates))], style={"flex": "1"}),
            html.Div([dcc.Graph(figure=fig_cap_hotkey_heatmap(grid_df))], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "12px"}),

        # Row 2: Entry offset comparison + time to target
        html.Div([
            html.Div([dcc.Graph(figure=fig_entry_offset_comparison(offset_comp))], style={"flex": "1"}),
            html.Div([dcc.Graph(figure=fig_time_to_target(target_rates))], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "12px"}),

        # Row 3: IV analysis + bounce vs reversal
        html.Div([
            html.Div([dcc.Graph(figure=fig_iv_analysis(iv_data))], style={"flex": "1"}),
            html.Div([dcc.Graph(figure=fig_bounce_vs_reversal(cap_df))], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "12px"}),
    ]

    filter_parts = []
    if trade_type:
        filter_parts.append(trade_type)
    if entry_offset:
        filter_parts.append(entry_offset)
    filter_label = f" [{', '.join(filter_parts)}]" if filter_parts else ""
    status = f"{len(cap_df)} OTM/ATM contracts loaded ({len(df)} total){filter_label}"
    return html.Div(children, className="anim-stagger"), status


# ═══════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════

def _badge(text: str) -> html.Span:
    """Small inline badge for contract attributes."""
    return html.Span(text, style={
        "backgroundColor": C["elevated"],
        "color": C["text2"],
        "borderRadius": "3px",
        "padding": "2px 6px",
        "fontSize": "10px",
        "fontFamily": "'JetBrains Mono', monospace",
        "fontWeight": "500",
        "border": f"1px solid {C['border']}",
    })


def _contract_picker_card(picker: PickerResult) -> html.Div:
    """Build the Contract Picker card with top pick, alternatives, and sizing."""
    children = []

    children.append(html.Div("CONTRACT PICKER", style=_label()))

    if picker.top_pick is None:
        children.append(html.P(
            f"No contracts passed quality filters ({picker.total_candidates} candidates evaluated).",
            style=_mono("12px", "400", C["loss"])
        ))
        return html.Div(children, style=_card(accent_left=C["loss"]))

    top = picker.top_pick

    # TOP PICK header
    children.append(html.Div("TOP PICK", style={
        **_mono("9px", "700", C["gold"]),
        "letterSpacing": "1.5px",
        "marginBottom": "4px",
    }))

    # Contract name + badges
    badges = []
    if top.delta is not None:
        badges.append(_badge(f"{abs(top.delta):.2f} delta"))
    badges.append(_badge(top.moneyness_label))
    badges.append(_badge(f"{top.dte} DTE"))

    children.append(html.Div([
        html.Span(top.label, style=_mono("18px", "700", C["gold"])),
        *badges,
    ], style={"marginBottom": "8px", "display": "flex", "gap": "8px", "alignItems": "center"}))

    # Metrics row 1: pricing and return
    children.append(html.Div([
        _stat_pill("Entry (Ask)", f"${top.entry_ask:.2f}"),
        _stat_pill("Max (Bid)", f"${top.max_bid:.2f}"),
        _stat_pill("Realistic", f"{top.realistic_return_pct:.0%}",
                   color=C["profit"] if top.realistic_return_pct > 0 else C["loss"]),
        _stat_pill("Spread Cost", f"{top.spread_cost_pct:.0%}", color=C["loss"]),
        _stat_pill("Score", f"{top.composite_score:.1f}"),
    ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "8px"}))

    # Metrics row 2: position sizing + historical context
    children.append(html.Div([
        _stat_pill("Contracts", str(top.contracts)),
        _stat_pill("Total Risk", f"${top.total_risk:,.0f}"),
        _stat_pill("Hist WR",
                   f"{top.hist_win_rate:.0%}" if top.hist_win_rate is not None else "—",
                   color=C["profit"] if (top.hist_win_rate or 0) > 0.7 else C["text"]),
        _stat_pill("Hist Avg",
                   f"{top.hist_avg_return:.0%}" if top.hist_avg_return is not None else "—",
                   color=C["profit"] if (top.hist_avg_return or 0) > 0 else C["text"]),
        _stat_pill("Sample",
                   str(top.hist_sample_count) if top.hist_sample_count else "—"),
    ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "8px"}))

    # Greeks row
    children.append(html.Div([
        _stat_pill("Delta", f"{top.delta:.2f}" if top.delta is not None else "—"),
        _stat_pill("IV", f"{top.implied_vol:.0%}" if top.implied_vol is not None else "—"),
    ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "8px"}))

    # Rationale
    children.append(html.P(top.rationale, style={
        **_mono("11px", "400", C["text2"]),
        "margin": "0 0 12px 0",
        "lineHeight": "1.5",
    }))

    # Divider + Alternatives
    for alt in [picker.conservative_pick, picker.aggressive_pick]:
        if alt is not None:
            type_label = "CONSERVATIVE" if alt.pick_type == "conservative" else "AGGRESSIVE"
            type_color = C["steel"] if alt.pick_type == "conservative" else C["loss"]

            children.append(html.Hr(style={
                "border": "none",
                "borderTop": f"1px solid {C['border']}",
                "margin": "6px 0",
            }))
            children.append(html.Div([
                html.Span(type_label, style={
                    **_mono("9px", "700", type_color),
                    "letterSpacing": "1px",
                    "minWidth": "110px",
                }),
                html.Span(alt.label, style=_mono("12px", "600", C["text"])),
                html.Span(f"Δ{abs(alt.delta):.2f}" if alt.delta else "",
                          style=_mono("11px", "400", C["text2"])),
                html.Span(f"{alt.contracts} ct", style=_mono("11px", "400", C["text2"])),
                html.Span(f"${alt.total_risk:,.0f} risk", style=_mono("11px", "400", C["text2"])),
                html.Span(f"{alt.realistic_return_pct:.0%}",
                          style=_mono("11px", "600",
                                      C["profit"] if alt.realistic_return_pct > 0 else C["loss"])),
                html.Span(f"— {alt.rationale}", style=_mono("10px", "400", C["text3"])),
            ], style={
                "display": "flex",
                "gap": "12px",
                "alignItems": "center",
                "padding": "4px 0",
                "flexWrap": "wrap",
            }))

    # Risk guidance
    rg = picker.risk_guidance
    if rg and rg.get("summary"):
        children.append(html.Hr(style={
            "border": "none",
            "borderTop": f"1px solid {C['border']}",
            "margin": "6px 0",
        }))
        children.append(html.Div([
            html.Span("RISK", style={
                **_mono("9px", "700", C["text3"]),
                "letterSpacing": "1px",
                "minWidth": "40px",
            }),
            html.Span(rg["summary"], style=_mono("10px", "400", C["text3"])),
        ], style={"display": "flex", "gap": "12px", "alignItems": "center"}))

    accent = C["profit"] if top.realistic_return_pct > 0 else C["gold"]
    return html.Div(children, style=_card(accent_left=accent))


def _stat_pill(label: str, value: str, color: str = None) -> html.Div:
    """Small stat display: label on top, value below."""
    return html.Div([
        html.Div(label, style={
            "fontSize": "9px", "fontWeight": "700", "textTransform": "uppercase",
            "color": C["text3"], "letterSpacing": "0.8px", "marginBottom": "2px",
        }),
        html.Div(value, style={
            **_mono("13px", "600", color or C["text"]),
        }),
    ], style={
        "backgroundColor": C["elevated"],
        "borderRadius": "4px",
        "padding": "6px 10px",
        "border": f"1px solid {C['border']}",
    })


def _error_card(message: str):
    """Return an error display card."""
    return html.Div([
        html.Div("ERROR", style=_label()),
        html.P(message, style={**_mono("13px", "400", C["loss"]), "margin": "0"}),
    ], style=_card(accent_left=C["loss"]))


# ── Run ─────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("OPTIONS_REPLAY_PORT", 8060))
    logger.info("Starting Options Replay on port %d", port)
    app.run(debug=True, port=port)
