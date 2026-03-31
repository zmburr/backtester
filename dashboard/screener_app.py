"""
Factor Screener Dashboard — Dash app showing watchlist ranked by bounce/reversal metrics.

Two tables (Bounce + Reversal) with raw values and percentile-based color coding.
Run:  python -m dashboard.screener_app
"""

import sys
import os
import datetime
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from scipy.stats import percentileofscore

import dash
from dash import html, dcc, dash_table, Input, Output, State
from dash.exceptions import PreventUpdate

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from options_replay.theme import C, CUSTOM_CSS
from scanners.stock_screener import watchlist, get_all_stocks_data
from dashboard.data.report_engine import (
    route_playbook,
    get_ticker_cap,
    get_pretrade_metrics,
    score_pretrade_setup,
    compute_bounce_intensity,
    ReportCache,
    BOUNCE_DF_WEAK, BOUNCE_DF_STRONG,
)
from analyzers.bounce_scorer import BouncePretrade, fetch_bounce_metrics, classify_stock
from analyzers.reversal_scorer import compute_reversal_intensity
from data_queries.polygon_queries import get_atr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

_MAX_WORKERS = 8

# ---------------------------------------------------------------------------
# Historical reference data for percentile ranking
# ---------------------------------------------------------------------------
_DATA_DIR = REPO_ROOT / 'data'
_bounce_ref = pd.read_csv(_DATA_DIR / 'bounce_data.csv')
_reversal_ref = pd.read_csv(_DATA_DIR / 'reversal_data.csv')

# ---------------------------------------------------------------------------
# Metric column definitions
# ---------------------------------------------------------------------------
BOUNCE_METRICS = [
    # (column_id, display_name, higher_is_better, format_type)
    ('selloff_total_pct',    'Selloff%',   False, 'pct'),
    ('pct_off_30d_high',     'Off30dH%',   False, 'pct'),
    ('gap_pct',              'Gap%',       False, 'pct'),
    ('prior_day_range_atr',  'Range/ATR',  True,  'x'),
    ('pct_change_3',         '3dChg%',     False, 'pct'),
    ('pct_off_52wk_high',    'Off52wk%',   False, 'pct'),
    ('pct_change_15',        '15dChg%',    False, 'pct'),
]

REVERSAL_METRICS = [
    ('pct_from_9ema',        '9EMA%',      True,  'pct'),
    ('prior_day_range_atr',  'Range/ATR',  True,  'x'),
    ('prior_day_rvol',       'RVOL',       True,  'x'),
    ('pct_change_3',         '3dChg%',     True,  'pct'),
    ('gap_pct',              'Gap%',       True,  'pct'),
    ('pct_from_50mav',       'From50MA%',  True,  'pct'),
]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _fmt(val, fmt_type):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return '—'
    if fmt_type == 'pct':
        return f'{val * 100:.1f}%'
    elif fmt_type == 'x':
        return f'{val:.2f}x'
    return str(val)


def _pctile(ref_series, val, higher_is_better):
    """Compute percentile of val within ref_series, oriented so higher = better."""
    clean = ref_series.dropna().values
    if len(clean) == 0 or val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    raw = percentileofscore(clean, val, kind='weak')
    return raw if higher_is_better else 100.0 - raw


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def build_screener_data(date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build bounce and reversal DataFrames with metrics + percentiles."""
    if date is None:
        date = datetime.datetime.now().strftime('%Y-%m-%d')

    cache = ReportCache()
    cache.install()
    try:
        # Phase 1: Screener metrics (parallel)
        log.info("Phase 1: Fetching screener metrics...")
        all_data = get_all_stocks_data(watchlist)

        # Phase 2: Route each ticker
        log.info("Phase 2: Routing tickers...")
        bucket_map = {}
        for ticker in watchlist:
            td = all_data.get(ticker, {})
            bucket, _ = route_playbook(
                td.get('pct_data', {}) or {},
                td.get('mav_data', {}) or {},
            )
            bucket_map[ticker] = bucket

        bounce_tickers = [t for t in watchlist if bucket_map[t] == 'bounce']
        reversal_tickers = [t for t in watchlist if bucket_map[t] == 'reversal']

        # Phase 3: Fetch setup-specific metrics (parallel)
        log.info(f"Phase 3: Fetching metrics ({len(bounce_tickers)} bounce, {len(reversal_tickers)} reversal)...")
        bounce_metrics_all = {}
        reversal_metrics_all = {}

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            # Bounce metrics
            bounce_futs = {executor.submit(fetch_bounce_metrics, t, date): t for t in bounce_tickers}
            # Reversal metrics
            reversal_futs = {executor.submit(get_pretrade_metrics, t, date): t for t in reversal_tickers}

            for f in as_completed(bounce_futs):
                tk = bounce_futs[f]
                try:
                    bounce_metrics_all[tk] = f.result()
                except Exception as e:
                    log.warning(f"{tk} bounce metrics failed: {e}")

            for f in as_completed(reversal_futs):
                tk = reversal_futs[f]
                try:
                    reversal_metrics_all[tk] = f.result()
                except Exception as e:
                    log.warning(f"{tk} reversal metrics failed: {e}")

        # Phase 4: Score and build DataFrames
        log.info("Phase 4: Scoring...")
        bounce_rows = _build_bounce_rows(bounce_tickers, bounce_metrics_all, date)
        reversal_rows = _build_reversal_rows(reversal_tickers, reversal_metrics_all, all_data, date)

        bounce_df = pd.DataFrame(bounce_rows) if bounce_rows else pd.DataFrame()
        reversal_df = pd.DataFrame(reversal_rows) if reversal_rows else pd.DataFrame()

        # Sort by intensity descending
        if not bounce_df.empty and 'intensity_raw' in bounce_df.columns:
            bounce_df = bounce_df.sort_values('intensity_raw', ascending=False, na_position='last')
        if not reversal_df.empty and 'intensity_raw' in reversal_df.columns:
            reversal_df = reversal_df.sort_values('intensity_raw', ascending=False, na_position='last')

        return bounce_df, reversal_df
    finally:
        cache.uninstall()


def _build_bounce_rows(tickers, metrics_all, date):
    checker = BouncePretrade()
    rows = []
    for ticker in tickers:
        bm = metrics_all.get(ticker, {})
        if not bm:
            continue
        cap = get_ticker_cap(ticker)
        result = checker.validate(ticker, bm, cap=cap)
        setup_type, _ = classify_stock(bm)
        ref_df = BOUNCE_DF_WEAK if setup_type == 'GapFade_weakstock' else BOUNCE_DF_STRONG
        intensity = compute_bounce_intensity(bm, ref_df=ref_df)

        row = {
            'ticker': ticker,
            'cap': cap,
            'setup_type': setup_type.replace('GapFade_', ''),
            'score': f'{result.score}/{result.max_score}',
            'score_raw': result.score,
            'rec': result.recommendation,
            'intensity': intensity.get('composite'),
            'intensity_raw': intensity.get('composite') or 0,
        }

        # Add raw values and percentiles for each metric
        for col_id, _, higher_is_better, fmt_type in BOUNCE_METRICS:
            raw_val = bm.get(col_id)
            row[col_id] = _fmt(raw_val, fmt_type)
            row[f'{col_id}_raw'] = raw_val if raw_val is not None else float('nan')
            ref_col = col_id
            if ref_col in _bounce_ref.columns:
                row[f'{col_id}_pctile'] = _pctile(_bounce_ref[ref_col], raw_val, higher_is_better)
            else:
                row[f'{col_id}_pctile'] = None

        rows.append(row)
    return rows


def _build_reversal_rows(tickers, metrics_all, all_data, date):
    rows = []
    for ticker in tickers:
        pm = metrics_all.get(ticker, {})
        if not pm:
            continue
        cap = get_ticker_cap(ticker)
        score_result = score_pretrade_setup(ticker, pm, cap=cap)

        # Merge screener mav_data for pct_from_50mav
        td = all_data.get(ticker, {})
        mav_data = td.get('mav_data', {}) or {}
        if 'pct_from_50mav' not in pm and 'pct_from_50mav' in mav_data:
            pm['pct_from_50mav'] = mav_data['pct_from_50mav']

        # Get atr_pct for intensity calculation
        range_data = td.get('range_data', {}) or {}
        pct_data = td.get('pct_data', {}) or {}
        if 'atr_pct' not in pm:
            try:
                atr = get_atr(ticker, date)
                price = pm.get('gap_pct', 0)  # dummy
                # Get reference price from screener data
                live_price = None
                vol_data = td.get('volume_data', {}) or {}
                # Calculate atr_pct from ATR and a reference price
                if atr and atr > 0:
                    # Use prior_close approximation: prior_ema9 / (1 + pct_from_9ema)
                    ema_dist = pm.get('pct_from_9ema', 0)
                    if ema_dist is not None and ema_dist != 0:
                        approx_price = atr / 0.05  # rough fallback
                    # Better: get from pct_change_3 reference
                    pm['atr_pct'] = atr / (atr / max(pm.get('prior_day_range_atr', 1) or 1, 0.01)) if pm.get('prior_day_range_atr') else None
            except Exception:
                pass

        # Map rvol_score for intensity (it expects 'rvol_score' not 'prior_day_rvol')
        if 'rvol_score' not in pm and 'prior_day_rvol' in pm:
            pm['rvol_score'] = pm['prior_day_rvol']

        intensity = compute_reversal_intensity(pm, cap=cap)

        row = {
            'ticker': ticker,
            'cap': cap,
            'setup_type': '—',
            'score': f'{score_result["score"]}/{score_result["max_score"]}',
            'score_raw': score_result['score'],
            'rec': score_result['recommendation'],
            'intensity': intensity.get('composite'),
            'intensity_raw': intensity.get('composite') or 0,
        }

        for col_id, _, higher_is_better, fmt_type in REVERSAL_METRICS:
            raw_val = pm.get(col_id)
            # Fallback to screener mav_data
            if raw_val is None and col_id in mav_data:
                raw_val = mav_data[col_id]
            row[col_id] = _fmt(raw_val, fmt_type)
            row[f'{col_id}_raw'] = raw_val if raw_val is not None else float('nan')
            ref_col = col_id
            # Map to reversal_data.csv column names
            if ref_col == 'prior_day_rvol':
                ref_col = 'percent_of_vol_one_day_before'
            if ref_col in _reversal_ref.columns:
                row[f'{col_id}_pctile'] = _pctile(_reversal_ref[ref_col], raw_val, higher_is_better)
            else:
                row[f'{col_id}_pctile'] = None

        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Color coding
# ---------------------------------------------------------------------------
def _build_color_styles(metric_specs):
    """Build style_data_conditional rules for percentile-colored metric cells."""
    styles = []
    bands = [
        (0,    20, C['loss'],   0.40),
        (20,   40, C['loss'],   0.15),
        (40,   60, C['text3'],  0.05),
        (60,   80, C['profit'], 0.15),
        (80, 101,  C['profit'], 0.40),
    ]
    for col_id, _, _, _ in metric_specs:
        pctile_col = f'{col_id}_pctile'
        for lo, hi, color, alpha in bands:
            # Convert hex color to rgba
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            styles.append({
                'if': {
                    'filter_query': f'{{{pctile_col}}} >= {lo} && {{{pctile_col}}} < {hi}',
                    'column_id': col_id,
                },
                'backgroundColor': f'rgba({r},{g},{b},{alpha})',
                'color': C['text'],
            })
    # Rec column coloring
    styles.extend([
        {'if': {'filter_query': '{rec} = "GO"', 'column_id': 'rec'},
         'backgroundColor': f'rgba(95,184,138,0.35)', 'color': C['profit'], 'fontWeight': '600'},
        {'if': {'filter_query': '{rec} = "CAUTION"', 'column_id': 'rec'},
         'backgroundColor': f'rgba(200,164,110,0.25)', 'color': C['gold'], 'fontWeight': '600'},
        {'if': {'filter_query': '{rec} = "NO-GO"', 'column_id': 'rec'},
         'backgroundColor': f'rgba(201,85,85,0.30)', 'color': C['loss'], 'fontWeight': '600'},
    ])
    return styles


def _build_columns(metric_specs):
    """Build visible DataTable columns."""
    cols = [
        {'name': 'Ticker', 'id': 'ticker'},
        {'name': 'Cap', 'id': 'cap'},
        {'name': 'Setup', 'id': 'setup_type'},
        {'name': 'Score', 'id': 'score', 'type': 'text'},
        {'name': 'Rec', 'id': 'rec'},
        {'name': 'Intensity', 'id': 'intensity', 'type': 'numeric'},
    ]
    for col_id, display_name, _, _ in metric_specs:
        cols.append({'name': display_name, 'id': col_id, 'type': 'text'})
    return cols


def _hidden_columns(metric_specs):
    """List of hidden columns (raw values + percentiles for sorting/filtering)."""
    hidden = ['score_raw', 'intensity_raw']
    for col_id, _, _, _ in metric_specs:
        hidden.append(f'{col_id}_raw')
        hidden.append(f'{col_id}_pctile')
    return hidden


# ---------------------------------------------------------------------------
# Dash App
# ---------------------------------------------------------------------------
app = dash.Dash(__name__, title='Factor Screener')

app.index_string = f'''
<!DOCTYPE html>
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
</html>
'''

# Base table styles
TABLE_STYLE_HEADER = {
    'backgroundColor': C['surface'],
    'color': C['gold'],
    'fontWeight': '600',
    'fontFamily': "'Outfit', sans-serif",
    'fontSize': '12px',
    'textTransform': 'uppercase',
    'letterSpacing': '0.04em',
    'borderBottom': f'1px solid {C["border"]}',
    'padding': '10px 12px',
    'textAlign': 'center',
}
TABLE_STYLE_CELL = {
    'backgroundColor': C['bg'],
    'color': C['text'],
    'fontFamily': "'JetBrains Mono', monospace",
    'fontSize': '12px',
    'borderBottom': f'1px solid {C["border"]}',
    'borderRight': 'none',
    'borderLeft': 'none',
    'padding': '8px 12px',
    'textAlign': 'center',
    'minWidth': '75px',
    'maxWidth': '120px',
}
TABLE_STYLE_DATA = {
    'backgroundColor': C['bg'],
    'color': C['text'],
}

SECTION_STYLE = {
    'marginBottom': '40px',
}

app.layout = html.Div(
    style={'backgroundColor': C['bg'], 'minHeight': '100vh', 'padding': '24px 32px'},
    children=[
        # Header
        html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '20px', 'marginBottom': '28px'}, children=[
            html.H1('Factor Screener', style={
                'fontFamily': "'DM Serif Display', serif",
                'fontSize': '28px',
                'color': C['text'],
                'margin': '0',
            }),
            html.Span(
                datetime.datetime.now().strftime('%A, %B %d, %Y'),
                style={'fontFamily': "'Outfit', sans-serif", 'fontSize': '14px', 'color': C['text2']},
            ),
            html.Button('Refresh', id='refresh-btn', style={
                'marginLeft': 'auto',
                'backgroundColor': C['surface'],
                'color': C['gold'],
                'border': f'1px solid {C["border"]}',
                'borderRadius': '6px',
                'padding': '8px 20px',
                'fontFamily': "'Outfit', sans-serif",
                'fontSize': '13px',
                'fontWeight': '500',
                'cursor': 'pointer',
                'letterSpacing': '0.03em',
            }),
        ]),

        # Data store
        dcc.Store(id='screener-store'),
        # Auto-load on page open
        dcc.Interval(id='initial-load', interval=500, max_intervals=1),

        # Loading wrapper
        dcc.Loading(
            type='default',
            color=C['gold'],
            children=[
                # Bounce table
                html.Div(id='bounce-section', style=SECTION_STYLE, children=[
                    html.Div(style={'display': 'flex', 'alignItems': 'baseline', 'gap': '12px', 'marginBottom': '12px'}, children=[
                        html.H2('BOUNCE SETUPS', style={
                            'fontFamily': "'Outfit', sans-serif",
                            'fontSize': '16px',
                            'fontWeight': '600',
                            'color': C['profit'],
                            'letterSpacing': '0.06em',
                            'margin': '0',
                        }),
                        html.Span(id='bounce-count', style={
                            'fontFamily': "'Outfit', sans-serif",
                            'fontSize': '12px',
                            'color': C['text3'],
                        }),
                    ]),
                    dash_table.DataTable(
                        id='bounce-table',
                        sort_action='native',
                        sort_by=[{'column_id': 'intensity', 'direction': 'desc'}],
                        style_header=TABLE_STYLE_HEADER,
                        style_cell=TABLE_STYLE_CELL,
                        style_data=TABLE_STYLE_DATA,
                        style_table={'overflowX': 'auto'},
                        style_as_list_view=True,
                        page_size=50,
                    ),
                ]),

                # Reversal table
                html.Div(id='reversal-section', style=SECTION_STYLE, children=[
                    html.Div(style={'display': 'flex', 'alignItems': 'baseline', 'gap': '12px', 'marginBottom': '12px'}, children=[
                        html.H2('REVERSAL SETUPS', style={
                            'fontFamily': "'Outfit', sans-serif",
                            'fontSize': '16px',
                            'fontWeight': '600',
                            'color': C['steel'],
                            'letterSpacing': '0.06em',
                            'margin': '0',
                        }),
                        html.Span(id='reversal-count', style={
                            'fontFamily': "'Outfit', sans-serif",
                            'fontSize': '12px',
                            'color': C['text3'],
                        }),
                    ]),
                    dash_table.DataTable(
                        id='reversal-table',
                        sort_action='native',
                        sort_by=[{'column_id': 'intensity', 'direction': 'desc'}],
                        style_header=TABLE_STYLE_HEADER,
                        style_cell=TABLE_STYLE_CELL,
                        style_data=TABLE_STYLE_DATA,
                        style_table={'overflowX': 'auto'},
                        style_as_list_view=True,
                        page_size=50,
                    ),
                ]),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output('screener-store', 'data'),
    Input('refresh-btn', 'n_clicks'),
    Input('initial-load', 'n_intervals'),
    prevent_initial_call=False,
)
def load_data(n_clicks, n_intervals):
    log.info("Loading screener data...")
    bounce_df, reversal_df = build_screener_data()
    return {
        'bounce': bounce_df.to_dict('records') if not bounce_df.empty else [],
        'reversal': reversal_df.to_dict('records') if not reversal_df.empty else [],
    }


@app.callback(
    Output('bounce-table', 'data'),
    Output('bounce-table', 'columns'),
    Output('bounce-table', 'hidden_columns'),
    Output('bounce-table', 'style_data_conditional'),
    Output('bounce-count', 'children'),
    Output('reversal-table', 'data'),
    Output('reversal-table', 'columns'),
    Output('reversal-table', 'hidden_columns'),
    Output('reversal-table', 'style_data_conditional'),
    Output('reversal-count', 'children'),
    Input('screener-store', 'data'),
)
def render_tables(store_data):
    if store_data is None:
        raise PreventUpdate

    bounce_data = store_data.get('bounce', [])
    reversal_data = store_data.get('reversal', [])

    # Build columns and styles
    bounce_cols = _build_columns(BOUNCE_METRICS)
    bounce_hidden = _hidden_columns(BOUNCE_METRICS)
    bounce_styles = _build_color_styles(BOUNCE_METRICS)

    reversal_cols = _build_columns(REVERSAL_METRICS)
    reversal_hidden = _hidden_columns(REVERSAL_METRICS)
    reversal_styles = _build_color_styles(REVERSAL_METRICS)

    # Add hidden columns to column definitions (needed for filter_query)
    for col_id in bounce_hidden:
        bounce_cols.append({'name': col_id, 'id': col_id, 'type': 'numeric'})
    for col_id in reversal_hidden:
        reversal_cols.append({'name': col_id, 'id': col_id, 'type': 'numeric'})

    bounce_count = f'{len(bounce_data)} tickers — sorted by intensity'
    reversal_count = f'{len(reversal_data)} tickers — sorted by intensity'

    return (
        bounce_data, bounce_cols, bounce_hidden, bounce_styles, bounce_count,
        reversal_data, reversal_cols, reversal_hidden, reversal_styles, reversal_count,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, port=8051)
