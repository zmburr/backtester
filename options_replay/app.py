"""
Options Replay — Dash dashboard for replaying options chain behavior
around headline-driven equity trades.

Run: python -m options_replay.app
"""

import sys
import os
import logging
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

    # Trade Selector
    html.Div([
        # CSV trade dropdown
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

        # Manual input
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

    # Status / Loading
    dcc.Loading(
        id="loading-results",
        type="default",
        color=C["gold"],
        children=html.Div(id="results-container", style={"padding": "0 30px 30px 30px"}),
    ),

], style={"backgroundColor": C["bg"], "minHeight": "100vh"})


# ── Main callback ───────────────────────────────────────────
@app.callback(
    Output("results-container", "children"),
    [Input("trade-dropdown", "value"),
     Input("manual-go-btn", "n_clicks")],
    [State("manual-ticker", "value"),
     State("manual-date", "value"),
     State("manual-time", "value"),
     State("manual-direction", "value"),
     State("window-slider", "value")],
    prevent_initial_call=True,
)
def analyze_trade(dropdown_value, n_clicks, manual_ticker, manual_date,
                  manual_time, manual_direction, hold_minutes):
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

    # ── Step 1: Check Theta Terminal ─────────────────────
    if not check_terminal_running():
        return _error_card(
            "Theta Terminal is not running. Please start the Java process "
            "and ensure it's listening on localhost:25503."
        )

    # ── Step 2: Fetch underlying data via Polygon ────────
    try:
        from data_queries.polygon_queries import get_intraday
        underlying_df = get_intraday(symbol, date_str, 1, "minute")
        if underlying_df is None:
            underlying_df = pd.DataFrame()
    except Exception as e:
        logger.warning("Failed to fetch underlying for %s: %s", symbol, e)
        underlying_df = pd.DataFrame()

    # Get underlying price at entry for chain filtering
    underlying_price = float(trade.get("avg_price", 0))
    if underlying_price <= 0 and not underlying_df.empty:
        # Find price closest to entry time
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

    # ── Step 3: Fetch options chain snapshot ──────────────
    try:
        chain_df = get_chain_snapshot(symbol, date_str, time_of_day)
    except ThetaTerminalOfflineError as e:
        return _error_card(str(e))
    except Exception as e:
        return _error_card(f"Failed to fetch chain snapshot: {e}")

    if chain_df.empty:
        return _error_card(f"No options chain data for {symbol} on {date_str} at {time_of_day}.")

    # ── Step 4: Filter chain ─────────────────────────────
    filtered = filter_chain(chain_df, underlying_price, side, date_str)
    if filtered.empty:
        return _error_card(
            f"No liquid options found for {symbol} at ${underlying_price:.2f}. "
            "The chain may have had insufficient liquidity."
        )

    # ── Step 5: Fetch per-contract data (parallel) ───────
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

    # ── Step 6: Compute returns and score ────────────────
    returns_df = compute_option_returns(filtered, ohlc_dict, quotes_dict,
                                        entry_time.to_pydatetime() if hasattr(entry_time, 'to_pydatetime') else entry_time,
                                        hold_minutes, greeks_dict=greeks_dict)
    scored_df = score_options(returns_df)

    if scored_df.empty:
        return _error_card("Scoring produced no results. The contracts may have had no price movement.")

    # ── Step 7: Compute ideal play ───────────────────────
    # Underlying max during window
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

    # ── Step 8: Build all figures ────────────────────────
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

    # ── Step 9: Assemble layout ──────────────────────────
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

        # Ideal play card
        html.Div([
            html.Div("IDEAL PLAY", style=_label()),
            html.Div([
                html.Span(f"{ideal['label']}", style=_mono("18px", "700", C["gold"])),
            ], style={"marginBottom": "8px"}),
            html.Div([
                _stat_pill("Entry", f"${ideal['entry_mid']:.2f}"),
                _stat_pill("Max", f"${ideal['max_mid']:.2f}"),
                _stat_pill("Return", f"{ideal['return_pct']:.0%}",
                          color=C["profit"] if ideal["return_pct"] > 0 else C["loss"]),
                _stat_pill("$/Contract", f"${ideal['return_per_contract']:.0f}"),
                _stat_pill("Spread", f"{ideal['spread_pct']:.1%}"),
                _stat_pill("Volume", f"{ideal['volume']:,}"),
                _stat_pill("Leverage", f"{ideal['leverage']:.0f}x"),
                _stat_pill("DTE", str(ideal.get("dte", "?"))),
                _stat_pill("Fillability", ideal["fillability"],
                          color=C["profit"] if ideal["fillability"] == "High"
                          else C["gold"] if ideal["fillability"] == "Medium"
                          else C["loss"]),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "10px"}),
            # Greeks row
            html.Div([
                _stat_pill("Delta", f"{ideal['delta']:.2f}" if ideal.get("delta") is not None else "—"),
                _stat_pill("Theta", f"{ideal['theta']:.3f}" if ideal.get("theta") is not None else "—"),
                _stat_pill("Vega", f"{ideal['vega']:.3f}" if ideal.get("vega") is not None else "—"),
                _stat_pill("Rho", f"{ideal['rho']:.3f}" if ideal.get("rho") is not None else "—"),
                _stat_pill("IV", f"{ideal['implied_vol']:.0%}" if ideal.get("implied_vol") is not None else "—"),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "10px"}),
            html.P(ideal["verdict"], style={
                **_mono("11px", "400", C["text2"]),
                "margin": "0",
                "lineHeight": "1.5",
            }),
        ], style=_card(accent_left=C["profit"] if ideal["return_pct"] > 0 else C["loss"])),

        # Ranked table
        dcc.Graph(figure=fig_table, style={"height": f"{max(200, 50 + len(scored_df) * 32)}px"}),

        # Returns vs Liquidity side by side
        html.Div([
            html.Div([dcc.Graph(figure=fig_returns)], style={"flex": "1"}),
            html.Div([dcc.Graph(figure=fig_liquidity)], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "12px"}),

        # Chain heatmap
        dcc.Graph(figure=fig_heatmap, style={"height": "300px"}),

        # Option price charts — 2x4 grid
        html.Div("OPTION PRICE CHARTS", style={**_label(), "marginTop": "20px", "marginBottom": "10px"}),
    ]

    # Add option charts in 2-column grid
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
