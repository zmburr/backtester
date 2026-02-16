"""Generate a textual watch-list report plus daily charts.

The report logic re-uses helper functions defined in *scanners.stock_screener* so we don't
repeat code.  It produces a giant string suitable for plain-text e-mail (or logging) and
saves one PNG chart per ticker in the *charts/* directory.
"""

from pathlib import Path
from textwrap import indent
from typing import List, Dict, Optional, Tuple
from support.config import send_email
import pandas as pd
import base64
import datetime
import os
import shutil
import logging

try:
    from weasyprint import HTML
    _PDF_AVAILABLE = True
except (ImportError, OSError):
    # Import failed either because WeasyPrint isn't installed or its runtime
    # dependencies (e.g., GTK/Pango/Cairo) are missing on this system.
    _PDF_AVAILABLE = False
    HTML = None  # type: ignore

# Try pdfkit / wkhtmltopdf as a lighter-weight alternative (single exe on Windows)
try:
    import pdfkit  # type: ignore
    _PDFKIT_AVAILABLE = True
except ImportError:
    _PDFKIT_AVAILABLE = False

# Resolve wkhtmltopdf location once so we can reuse it
_WKHTMLTOPDF_PATH = os.getenv("WKHTMLTOPDF_PATH")  # user override

if not _WKHTMLTOPDF_PATH:
    # Try typical Windows install location
    default_win_path = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
    if os.path.isfile(default_win_path):
        _WKHTMLTOPDF_PATH = default_win_path
    else:
        # Fallback to `which` / `where` search in PATH
        _WKHTMLTOPDF_PATH = shutil.which("wkhtmltopdf")

# None -> still unresolved

from scanners import stock_screener as ss
from data_collectors.combined_data_collection import reversal_df
from analyzers.charter import create_daily_chart, cleanup_charts
from data_queries.polygon_queries import get_levels_data, get_daily, get_atr, get_actual_current_price as get_actual_current_price_polygon
from data_queries.trillium_queries import get_actual_current_price_trill
from analyzers.exit_targets import get_exit_framework, calculate_exit_targets, format_exit_targets_html
from analyzers.bounce_exit_targets import calculate_bounce_exit_targets, format_bounce_exit_targets_html
from analyzers.bounce_scorer import (
    BouncePretrade,
    fetch_bounce_metrics,
    SETUP_PROFILES,
    classify_from_setup_column,
    classify_stock,
)

# Load bounce data and split by setup type for percentile comparisons
_bounce_csv = Path(__file__).resolve().parent.parent / "data" / "bounce_data.csv"
_bounce_df_all = pd.read_csv(_bounce_csv).dropna(subset=['ticker', 'date'])

# ---------------------------------------------------------------------------
# Column normalization (historical bounce CSV vs live metrics naming)
# ---------------------------------------------------------------------------
# Live bounce metrics (from `fetch_bounce_metrics`) expose `prior_day_rvol` (x ADV),
# but the historical `bounce_data.csv` stores the same concept as
# `percent_of_vol_one_day_before`. Create/populate prior_day_rvol so the
# bounce-intensity percentile table doesn't show N/A.
if 'prior_day_rvol' not in _bounce_df_all.columns:
    # Primary source: percent_of_vol_one_day_before (already RVOL ratio)
    if 'percent_of_vol_one_day_before' in _bounce_df_all.columns:
        _bounce_df_all['prior_day_rvol'] = pd.to_numeric(
            _bounce_df_all['percent_of_vol_one_day_before'], errors='coerce'
        )
    # Fallback: compute from raw volumes
    elif {'vol_one_day_before', 'avg_daily_vol'}.issubset(_bounce_df_all.columns):
        _bounce_df_all['prior_day_rvol'] = (
            pd.to_numeric(_bounce_df_all['vol_one_day_before'], errors='coerce')
            / pd.to_numeric(_bounce_df_all['avg_daily_vol'], errors='coerce')
        )
    else:
        _bounce_df_all['prior_day_rvol'] = pd.NA
else:
    # Column exists but may have gaps — fill from percent_of_vol_one_day_before
    _bounce_df_all['prior_day_rvol'] = pd.to_numeric(_bounce_df_all['prior_day_rvol'], errors='coerce')
    if 'percent_of_vol_one_day_before' in _bounce_df_all.columns:
        _bounce_df_all['prior_day_rvol'] = _bounce_df_all['prior_day_rvol'].fillna(
            pd.to_numeric(_bounce_df_all['percent_of_vol_one_day_before'], errors='coerce')
        )

_bounce_df_all['_setup_profile'] = _bounce_df_all['Setup'].apply(classify_from_setup_column)
BOUNCE_DF_WEAK = _bounce_df_all[_bounce_df_all['_setup_profile'] == 'GapFade_weakstock'].copy()
BOUNCE_DF_STRONG = _bounce_df_all[_bounce_df_all['_setup_profile'] == 'GapFade_strongstock'].copy()

# Debug: verify prior_day_rvol was populated from historical data
_pdr_valid = _bounce_df_all['prior_day_rvol'].notna().sum()
print(f"[generate_report] Historical bounce data: {len(_bounce_df_all)} rows, "
      f"prior_day_rvol has {_pdr_valid} non-null values")
if _pdr_valid == 0:
    print(f"  WARNING: prior_day_rvol is all NaN! Columns available: {list(_bounce_df_all.columns)[:20]}...")

# ---------------------------------------------------------------------------
# Bounce Intensity Score — continuous 0-100 ranking for bounce candidates
# Uses percentile rank against all 52 historical bounce trades.
# Higher = more extreme setup = better bounce candidate.
# ---------------------------------------------------------------------------
from scipy.stats import percentileofscore as _pctrank

# Metrics, direction (True = higher actual = higher score, False = lower/more-negative = higher score), weight
_BOUNCE_INTENSITY_SPEC = [
    ('selloff_total_pct',              False, 0.30),  # deeper selloff = better
    ('consecutive_down_days',          True,  0.10),  # more days = better
    ('prior_day_rvol',                 True,  0.15),  # higher prior day vol = better
    ('pct_off_30d_high',               False, 0.20),  # further off high = better
    ('gap_pct',                        False, 0.25),  # bigger gap down = better
]


def compute_bounce_intensity(metrics: Dict, ref_df: pd.DataFrame = None) -> Dict:
    """
    Compute a weighted composite bounce intensity score (0-100).

    For each metric, percentile rank the candidate against all 52 historical
    bounce trades.  "More extreme = higher score" so negative-direction metrics
    are inverted (100 - pctrank).

    Returns dict with per-metric percentiles, weights, and the composite score.
    """
    if ref_df is None:
        ref_df = _bounce_df_all

    details = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for col, higher_is_better, weight in _BOUNCE_INTENSITY_SPEC:
        actual = metrics.get(col)
        ref_vals = ref_df[col].dropna().values if col in ref_df.columns else []

        if actual is None or pd.isna(actual) or len(ref_vals) == 0:
            details[col] = {'pctile': None, 'weight': weight, 'actual': actual}
            continue

        raw_pctile = _pctrank(ref_vals, actual, kind='rank')
        # Invert for "lower = better" metrics so 100 = most extreme
        pctile = raw_pctile if higher_is_better else 100.0 - raw_pctile

        details[col] = {'pctile': round(pctile, 1), 'weight': weight, 'actual': actual}
        weighted_sum += pctile * weight
        total_weight += weight

    composite = round(weighted_sum / total_weight, 1) if total_weight > 0 else 0.0

    return {'composite': composite, 'details': details}


def format_bounce_intensity_html(intensity: Dict) -> str:
    """Format bounce intensity score as compact HTML."""
    score = intensity['composite']

    if score >= 70:
        color = '#28a745'
    elif score >= 40:
        color = '#ffc107'
    else:
        color = '#dc3545'

    lines = [
        f'<div style="margin: 6px 0;">',
        f'<strong>Bounce Intensity: <span style="color: {color}; font-size: 1.1em;">{score:.0f}/100</span></strong>',
        '<table style="font-size: 0.85em; margin-top: 4px;">',
        '<tr style="color: #888;"><td style="padding-right: 10px;">Metric</td>'
        '<td style="padding-right: 10px;">Percentile</td><td>Weight</td></tr>',
    ]

    labels = {
        'selloff_total_pct': 'Selloff depth',
        'consecutive_down_days': 'Down days',
        'prior_day_rvol': 'Prior day RVOL',
        'pct_off_30d_high': '30d high discount',
        'gap_pct': 'Gap down',
    }

    for col, _, _ in _BOUNCE_INTENSITY_SPEC:
        d = intensity['details'].get(col, {})
        pctile = d.get('pctile')
        weight = d.get('weight', 0)
        label = labels.get(col, col)
        pctile_str = f'{pctile:.0f}' if pctile is not None else 'N/A'
        lines.append(
            f'<tr><td style="padding-right: 10px;">{label}</td>'
            f'<td style="padding-right: 10px;">{pctile_str}</td>'
            f'<td>{weight*100:.0f}%</td></tr>'
        )

    lines.append('</table></div>')
    return '\n'.join(lines)


# Columns to compare for bounce percentiles (pct_change + MA distances, not range)
BOUNCE_COLUMNS_TO_COMPARE = [
    'pct_change_120', 'pct_change_90', 'pct_change_30', 'pct_change_15', 'pct_change_3',
    'pct_from_9ema', 'pct_from_10mav', 'pct_from_20mav', 'pct_from_50mav', 'pct_from_200mav',
]

# -------------------------------------------------
# Pre-Trade Reversal Scoring (5 of 6 criteria - excludes reversal_pct since pre-trade)
# -------------------------------------------------

# Cap-adjusted thresholds for pre-trade screening
# Same as reversal_scorer.py but without reversal_pct
PRETRADE_THRESHOLDS = {
    'Micro': {'pct_from_9ema': 0.80, 'prior_day_range_atr': 3.0, 'prior_day_rvol': 2.0, 'premarket_rvol': 0.05, 'consecutive_up_days': 3, 'gap_pct': 0.15},
    'Small': {'pct_from_9ema': 0.40, 'prior_day_range_atr': 2.0, 'prior_day_rvol': 2.0, 'premarket_rvol': 0.05, 'consecutive_up_days': 2, 'gap_pct': 0.10},
    'Medium': {'pct_from_9ema': 0.15, 'prior_day_range_atr': 1.0, 'prior_day_rvol': 1.5, 'premarket_rvol': 0.05, 'consecutive_up_days': 2, 'gap_pct': 0.05},
    'Large': {'pct_from_9ema': 0.08, 'prior_day_range_atr': 0.8, 'prior_day_rvol': 1.0, 'premarket_rvol': 0.05, 'consecutive_up_days': 1, 'gap_pct': 0.00},
    'ETF': {'pct_from_9ema': 0.04, 'prior_day_range_atr': 1.0, 'prior_day_rvol': 1.5, 'premarket_rvol': 0.05, 'consecutive_up_days': 1, 'gap_pct': 0.00},
}

# Known ETFs (type detection from Polygon can be unreliable)
KNOWN_ETFS = {'GLD', 'SLV', 'GDXJ', 'QQQ', 'SPY', 'XLE', 'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU',
              'IWM', 'DIA', 'VTI', 'VOO', 'VXX', 'UVXY', 'SQQQ', 'TQQQ', 'SPXU', 'SPXL', 'TLT', 'HYG', 'LQD',
              'EEM', 'EWZ', 'EWJ', 'FXI', 'KWEB', 'SMH', 'XBI', 'IBB', 'ARKK', 'ARKG'}

# Cache for market cap lookups (avoid repeated API calls)
_market_cap_cache: Dict[str, str] = {}


def get_ticker_cap(ticker: str) -> str:
    """
    Get market cap category for a ticker by fetching from Polygon API.

    Categories:
        - Large: >= $100B
        - Medium: $2B - $100B
        - Small: $300M - $2B
        - Micro: < $300M
        - ETF: Exchange-traded funds
    """
    # Check cache first
    if ticker in _market_cap_cache:
        return _market_cap_cache[ticker]

    # Check known ETFs
    if ticker.upper() in KNOWN_ETFS:
        _market_cap_cache[ticker] = 'ETF'
        return 'ETF'

    try:
        from polygon.rest import RESTClient
        client = RESTClient('pcwUY7TnSF66nYAPIBCApPMyVrXTckJY')

        details = client.get_ticker_details(ticker)

        # Check if ETF by type
        if hasattr(details, 'type') and details.type in ('ETF', 'ETN'):
            _market_cap_cache[ticker] = 'ETF'
            return 'ETF'

        mc = details.market_cap

        if mc is None:
            cap = 'Medium'  # Default if no market cap data
        elif mc >= 100_000_000_000:  # >= $100B
            cap = 'Large'
        elif mc >= 2_000_000_000:    # >= $2B
            cap = 'Medium'
        elif mc >= 300_000_000:      # >= $300M
            cap = 'Small'
        else:
            cap = 'Micro'

        _market_cap_cache[ticker] = cap
        logging.info(f"{ticker}: Market cap ${mc/1e9:.1f}B -> {cap}")
        return cap

    except Exception as e:
        logging.warning(f"Failed to get market cap for {ticker}: {e}, defaulting to Medium")
        _market_cap_cache[ticker] = 'Medium'
        return 'Medium'


def get_pretrade_metrics(ticker: str, date: str) -> Dict:
    """
    Fetch additional metrics needed for pre-trade scoring that aren't in stock_screener.
    Returns: dict with pct_from_9ema, consecutive_up_days, gap_pct, prior_day_range_atr, rvol_score
    """
    metrics = {}
    try:
        # Get historical data (35 days for calculations)
        df = get_levels_data(ticker, date, 35, 1, 'day')
        if df is None or len(df) < 5:
            return metrics

        # Calculate 9-day EMA
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()

        # Determine if today's bar is in the data or if we're in premarket
        today_date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        last_bar_date = df.index[-1].date()
        is_premarket = last_bar_date != today_date

        if is_premarket and len(df) >= 1:
            # Premarket: last bar is yesterday
            prior_close = df['close'].iloc[-1]
            prior_high = df['high'].iloc[-1]
            prior_low = df['low'].iloc[-1]
            prior_ema9 = df['ema_9'].iloc[-1]
        elif len(df) >= 2:
            prior_close = df['close'].iloc[-2]
            prior_high = df['high'].iloc[-2]
            prior_low = df['low'].iloc[-2]
            prior_ema9 = df['ema_9'].iloc[-2]
        else:
            prior_close = prior_high = prior_low = prior_ema9 = None

        # Fetch live price for current-price calculations
        live_price = None
        try:
            live_price = get_actual_current_price_trill(ticker)
        except Exception:
            pass
        reference_price = live_price if live_price is not None else prior_close

        if prior_close is not None:
            # 1. Distance from 9EMA (using live price)
            if prior_ema9 and prior_ema9 > 0:
                metrics['pct_from_9ema'] = (reference_price - prior_ema9) / prior_ema9

            # 2. Gap % (live price vs prior close)
            if reference_price and prior_close > 0:
                metrics['gap_pct'] = (reference_price - prior_close) / prior_close

            # 3. Prior day range as multiple of ATR
            prior_range = prior_high - prior_low
            atr = get_atr(ticker, date)
            if atr and atr > 0:
                metrics['prior_day_range_atr'] = prior_range / atr

            # 4. Consecutive up days
            consecutive_up = 0
            for i in range(len(df) - 2, -1, -1):  # Start from prior day going back
                if df['close'].iloc[i] > df['open'].iloc[i]:
                    consecutive_up += 1
                else:
                    break
            metrics['consecutive_up_days'] = consecutive_up

            # 5. Volume signal (prior day RVOL + premarket RVOL)
            if len(df) >= 20:
                avg_vol = df['volume'].iloc[-21:-1].mean()  # 20 days before today
                if avg_vol and avg_vol > 0:
                    if is_premarket:
                        # Premarket: last bar is yesterday
                        prior_day_vol = df['volume'].iloc[-1]
                    else:
                        # Market hours: second-to-last bar is yesterday
                        prior_day_vol = df['volume'].iloc[-2] if len(df) >= 2 else 0
                    metrics['prior_day_rvol'] = prior_day_vol / avg_vol if prior_day_vol else 0
                    # Premarket volume from Polygon intraday
                    try:
                        from data_queries.polygon_queries import get_intraday
                        pm_data = get_intraday(ticker, date, 1, 'minute')
                        if pm_data is not None and not pm_data.empty:
                            pm_vol = pm_data.between_time('06:00:00', '09:30:00')['volume'].sum()
                            metrics['premarket_rvol'] = pm_vol / avg_vol if pm_vol else 0
                        else:
                            metrics['premarket_rvol'] = 0
                    except Exception:
                        metrics['premarket_rvol'] = 0

    except Exception as e:
        logging.warning(f"Error getting pretrade metrics for {ticker}: {e}")

    return metrics


def score_pretrade_setup(ticker: str, metrics: Dict, cap: str = None) -> Dict:
    """
    Score a potential reversal setup using 5 pre-trade criteria.

    Returns dict with:
        - score: 0-5
        - max_score: 5
        - recommendation: GO/CAUTION/NO-GO
        - criteria: list of (name, passed, threshold, actual) tuples
    """
    if cap is None:
        cap = get_ticker_cap(ticker)

    if cap not in PRETRADE_THRESHOLDS:
        cap = 'Medium'

    thresholds = PRETRADE_THRESHOLDS[cap]

    criteria = []
    score = 0

    # Check each criterion
    criteria_checks = [
        ('9EMA Distance', 'pct_from_9ema', thresholds['pct_from_9ema']),
        ('Range (ATR)', 'prior_day_range_atr', thresholds['prior_day_range_atr']),
        ('Consec Up Days', 'consecutive_up_days', thresholds['consecutive_up_days']),
        ('Gap Up', 'gap_pct', thresholds['gap_pct']),
    ]

    for name, key, threshold in criteria_checks:
        actual = metrics.get(key)
        if actual is not None and not pd.isna(actual):
            passed = actual >= threshold
        else:
            passed = False
            actual = None

        if passed:
            score += 1

        criteria.append({
            'name': name,
            'key': key,
            'passed': passed,
            'threshold': threshold,
            'actual': actual,
        })

    # Vol signal: either prior day RVOL or premarket RVOL meets threshold
    prior_rvol = metrics.get('prior_day_rvol')
    pm_rvol = metrics.get('premarket_rvol')
    prior_thresh = thresholds['prior_day_rvol']
    pm_thresh = thresholds['premarket_rvol']
    prior_pass = prior_rvol is not None and not pd.isna(prior_rvol) and prior_rvol >= prior_thresh
    pm_pass = pm_rvol is not None and not pd.isna(pm_rvol) and pm_rvol >= pm_thresh
    vol_passed = prior_pass or pm_pass
    if vol_passed:
        score += 1

    prior_str = f"{prior_rvol:.1f}x" if prior_rvol is not None and not pd.isna(prior_rvol) else "N/A"
    pm_str = f"{pm_rvol:.2f}x" if pm_rvol is not None and not pd.isna(pm_rvol) else "N/A"
    criteria.append({
        'name': 'Vol Signal',
        'key': 'vol_signal',
        'passed': vol_passed,
        'threshold': prior_thresh,
        'actual': prior_rvol,
        'display_actual': f"Prior: {prior_str} | PM: {pm_str}",
        'display_threshold': f"{prior_thresh:.1f}x / {pm_thresh:.2f}x",
    })

    # Determine recommendation (adjusted for 5 criteria)
    if score >= 4:
        recommendation = 'GO'
    elif score == 3:
        recommendation = 'CAUTION'
    else:
        recommendation = 'NO-GO'

    return {
        'ticker': ticker,
        'cap': cap,
        'score': score,
        'max_score': 5,
        'recommendation': recommendation,
        'criteria': criteria,
    }


# Historical performance statistics by score (from 61 Grade A trades analysis)
SCORE_STATISTICS = {
    5: {'trades': 25, 'win_rate': 100.0, 'avg_pnl': 13.3},
    4: {'trades': 20, 'win_rate': 85.0, 'avg_pnl': 15.9},
    3: {'trades': 6, 'win_rate': 83.3, 'avg_pnl': 10.5},
    2: {'trades': 1, 'win_rate': 0.0, 'avg_pnl': -4.6},
    1: {'trades': 0, 'win_rate': 0.0, 'avg_pnl': 0.0},
    0: {'trades': 0, 'win_rate': 0.0, 'avg_pnl': 0.0},
}


# Historical bounce performance statistics by recommendation (from 54 trades, setup-based scoring with range expansion)
BOUNCE_SCORE_STATISTICS = {
    'GO': {'trades': 33, 'win_rate': 100.0, 'avg_pnl': 14.2},
    'CAUTION': {'trades': 3, 'win_rate': 66.7, 'avg_pnl': -13.1},
    'NO-GO': {'trades': 18, 'win_rate': 22.2, 'avg_pnl': -5.4},
}


def get_exit_target_data(ticker: str, date: str, prefer_open: bool = False) -> Dict:
    """
    Fetch data needed for exit target calculations.
    Returns: dict with:
      - open_price: reference price for targets (OPEN if available and prefer_open=True, else live price)
      - open_price_source: 'open' | 'live' | 'prior_close'
      - atr: ATR (computed as-of the prior completed daily bar when possible)
      - prior_close, prior_low, prior_high: prior completed daily bar levels
      - ema_4: 4-day EMA as-of prior completed daily bar

    Note: For consistency with the historical target framework, targets are measured
    from a single reference price. When the official OPEN is available, using it
    makes targets stable and comparable to bounce_data.csv analysis.
    """
    data = {}
    try:
        # Get historical data for EMA and prior levels
        df = get_levels_data(ticker, date, 10, 1, 'day')
        if df is None or len(df) < 2:
            return data

        # Calculate 4-day EMA
        df['ema_4'] = df['close'].ewm(span=4, adjust=False).mean()

        # Determine if Polygon included today's (possibly in-progress) daily bar
        today_date = None
        has_today_bar = False
        try:
            today_date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
            last_bar_date = df.index[-1].date()
            has_today_bar = (today_date is not None and last_bar_date == today_date)
        except Exception:
            has_today_bar = False

        # Prior completed daily bar index:
        # - If today's bar is present, prior day is -2
        # - Otherwise (premarket / no today bar), prior day is -1
        prior_idx = -2 if has_today_bar and len(df) >= 2 else -1

        data['prior_close'] = float(df['close'].iloc[prior_idx])
        data['prior_low'] = float(df['low'].iloc[prior_idx])
        data['prior_high'] = float(df['high'].iloc[prior_idx])
        data['ema_4'] = float(df['ema_4'].iloc[prior_idx])  # EMA as of prior completed day
        try:
            data['prior_bar_date'] = df.index[prior_idx].date().strftime('%Y-%m-%d')
        except Exception:
            data['prior_bar_date'] = None

        # Reference price selection:
        # Prefer official OPEN (once available), else fall back to live price.
        ref_price = None
        ref_source = None

        if prefer_open:
            open_candidate = None

            # 1) Polygon daily open/close agg (may be unavailable premarket)
            try:
                daily = get_daily(ticker, date)
                open_candidate = getattr(daily, 'open', None) if daily else None
            except Exception:
                open_candidate = None

            # 2) If today's daily bar exists in levels data, use its open
            if (open_candidate is None or (isinstance(open_candidate, float) and pd.isna(open_candidate))) and has_today_bar:
                try:
                    open_candidate = df['open'].iloc[-1]
                except Exception:
                    open_candidate = None

            try:
                if open_candidate is not None and not (isinstance(open_candidate, float) and pd.isna(open_candidate)) and float(open_candidate) > 0:
                    ref_price = float(open_candidate)
                    ref_source = 'open'
            except Exception:
                pass

        if ref_price is None:
            # Live price: Trillium first, Polygon intraday fallback
            try:
                ref_price = get_actual_current_price_trill(ticker)
                ref_source = 'live'
            except Exception:
                try:
                    ref_price = get_actual_current_price_polygon(ticker, date)
                    ref_source = 'live'
                except Exception:
                    # Last resort: prior close
                    if data.get('prior_close') is not None:
                        ref_price = data['prior_close']
                        ref_source = 'prior_close'

        if ref_price is not None:
            data['open_price'] = ref_price
            data['open_price_source'] = ref_source

        # Get ATR
        # Use ATR as-of the prior completed daily bar when possible (stable, objective).
        atr_date = data.get('prior_bar_date') or date
        atr = get_atr(ticker, atr_date)
        if atr:
            data['atr'] = atr

    except Exception as e:
        logging.warning(f"Error getting exit target data for {ticker}: {e}")

    return data


def format_pretrade_score_html(score_result: Dict) -> str:
    """Format pre-trade score as HTML for the report."""
    rec = score_result['recommendation']
    score = score_result['score']
    cap = score_result['cap']

    # Color coding
    if rec == 'GO':
        color = '#28a745'  # green
        rec_text = 'GO'
    elif rec == 'CAUTION':
        color = '#ffc107'  # yellow
        rec_text = 'CAUTION'
    else:
        color = '#dc3545'  # red
        rec_text = 'NO-GO'

    # Get historical stats for this score
    stats = SCORE_STATISTICS.get(score, {'trades': 0, 'win_rate': 0.0, 'avg_pnl': 0.0})
    stats_text = f"Historical: {stats['win_rate']:.0f}% win rate, {stats['avg_pnl']:+.1f}% avg P&L (n={stats['trades']})"

    lines = [
        f'<div style="border: 2px solid {color}; padding: 10px; margin: 10px 0; border-radius: 5px;">',
        f'<strong style="color: {color}; font-size: 1.2em;">{rec_text}</strong> ',
        f'<span>Score: {score}/5 ({cap} Cap)</span>',
        f'<br><span style="font-size: 0.85em; color: #666;">{stats_text}</span>',
        '<table style="margin-top: 8px; font-size: 0.9em;">',
    ]

    for c in score_result['criteria']:
        status = '✓' if c['passed'] else '✗'
        status_color = '#28a745' if c['passed'] else '#dc3545'

        # Format values
        if c['key'] == 'vol_signal':
            actual_str = c.get('display_actual', 'N/A')
            thresh_str = c.get('display_threshold', '')
        elif c['actual'] is not None:
            if c['key'] in ['pct_from_9ema', 'gap_pct']:
                actual_str = f"{c['actual']*100:.1f}%"
                thresh_str = f"{c['threshold']*100:.0f}%"
            elif c['key'] == 'consecutive_up_days':
                actual_str = f"{int(c['actual'])}"
                thresh_str = f"{int(c['threshold'])}"
            else:
                actual_str = f"{c['actual']:.1f}x"
                thresh_str = f"{c['threshold']:.1f}x"
        else:
            actual_str = 'N/A'
            thresh_str = f"{c['threshold']}"

        lines.append(
            f'<tr><td style="color: {status_color}; padding-right: 10px;">{status}</td>'
            f'<td style="padding-right: 10px;">{c["name"]}</td>'
            f'<td style="padding-right: 10px;">≥{thresh_str}</td>'
            f'<td>{actual_str}</td></tr>'
        )

    lines.append('</table></div>')
    return '\n'.join(lines)


def format_bounce_score_html(result, bounce_metrics: Optional[Dict] = None) -> str:
    """Format bounce pre-trade checklist result as HTML for the report."""
    rec = result.recommendation
    score = result.score
    setup_type = result.setup_type
    profile = SETUP_PROFILES[setup_type]

    # Color coding
    if rec == 'GO':
        color = '#28a745'
    elif rec == 'CAUTION':
        color = '#ffc107'
    else:
        color = '#dc3545'

    cap_label = getattr(result, 'cap', '')
    stats_text = (f"Profile: {profile.name} | {cap_label} Cap | Historical: {profile.historical_win_rate*100:.0f}% WR, "
                  f"+{profile.historical_avg_pnl:.0f}% avg P&L (n={profile.sample_size} Grade A)")

    lines = [
        f'<div style="border: 2px solid {color}; padding: 10px; margin: 10px 0; border-radius: 5px;">',
        f'<strong style="color: {color}; font-size: 1.2em;">BOUNCE {rec}</strong> ',
        f'<span>Score: {score}/{result.max_score} ({cap_label} Cap)</span>',
        f'<br><span style="font-size: 0.85em; color: #666;">{stats_text}</span>',
    ]

    # Classification details
    if result.classification_details.get('signals'):
        signals_str = ' | '.join(result.classification_details['signals'])
        lines.append(f'<br><span style="font-size: 0.8em; color: #888;">Classification: {signals_str}</span>')

    # Bounce stats by recommendation
    bounce_stats = BOUNCE_SCORE_STATISTICS.get(rec, {})
    if bounce_stats:
        lines.append(
            f'<br><span style="font-size: 0.85em; color: #666;">'
            f'Historical {rec}: {bounce_stats["win_rate"]:.0f}% win rate, '
            f'{bounce_stats["avg_pnl"]:+.1f}% avg P&L (n={bounce_stats["trades"]})</span>'
        )

    # Criteria table
    lines.append('<table style="margin-top: 8px; font-size: 0.9em;">')
    for item in result.items:
        status = '✓' if item.passed else '✗'
        status_color = '#28a745' if item.passed else '#dc3545'
        ref_base = item.reference or ''

        # Prefer a clean "A median: 52%" display for drawdown-style metrics
        if item.name in {"selloff_total_pct", "pct_off_30d_high", "gap_pct"}:
            a_med = profile.reference_medians.get(item.name)
            if a_med is not None and not pd.isna(a_med):
                ref_base = f"A median: {abs(float(a_med))*100:.0f}%"

        targets_html = ""
        if bounce_metrics and item.name in {"selloff_total_pct", "pct_off_30d_high", "gap_pct"}:
            targets_html = _format_bounce_price_targets_html(
                item_name=item.name,
                threshold_frac=item.threshold,
                a_median_frac=profile.reference_medians.get(item.name),
                bounce_metrics=bounce_metrics,
            )
            # Targets block already includes A-median context; avoid duplicating it in parentheses.
            if targets_html:
                ref_base = ""

        ref_parts = []
        if ref_base:
            ref_parts.append(f'<span style="color: #999;">({ref_base})</span>')
        if targets_html:
            ref_parts.append(targets_html)
        ref_str = "<br>".join(ref_parts) if ref_parts else ''

        lines.append(
            f'<tr><td style="color: {status_color}; padding-right: 10px;">{status}</td>'
            f'<td style="padding-right: 10px;">{item.description}</td>'
            f'<td style="padding-right: 10px;">{item.actual_display}</td>'
            f'<td>{ref_str}</td></tr>'
        )

    lines.append('</table>')

    # Bonuses
    if result.bonuses:
        lines.append('<div style="margin-top: 6px; font-size: 0.85em; color: #28a745;">')
        for bonus in result.bonuses:
            lines.append(f'<br>✦ {bonus}')
        lines.append('</div>')

    # Warnings
    if result.warnings:
        lines.append('<div style="margin-top: 6px; font-size: 0.85em; color: #dc3545;">')
        for warning in result.warnings:
            lines.append(f'<br>⚠ {warning}')
        lines.append('</div>')

    lines.append('</div>')
    return '\n'.join(lines)


# Columns we want to compare (same list used inside stock_screener)
COLUMNS_TO_COMPARE = ss.columns_to_compare

# ---------------------------------------------
# Report header (trading rules & daily checklist)
# ---------------------------------------------

HEADER_HTML = """<h1 style="text-align:center;">Daily Trading Rules & Checklist</h1>

<h2>Reversal Setup Scoring Guide</h2>
<p>Each stock is scored on <strong>5 pre-trade criteria</strong> (cap-adjusted thresholds):</p>
<ol>
  <li><strong>9EMA Distance</strong> - Price elevated above 9-day EMA</li>
  <li><strong>Range (ATR)</strong> - Prior day range vs ATR</li>
  <li><strong>RVOL</strong> - Volume vs 20-day average</li>
  <li><strong>Consecutive Up Days</strong> - Momentum into the top</li>
  <li><strong>Gap Up</strong> - Gap up on reversal day</li>
</ol>

<h3>Historical Performance by Score (61 Grade A Trades)</h3>
<table border="1" cellpadding="8" style="border-collapse: collapse; margin: 10px 0;">
<tr style="background-color: #f0f0f0;"><th>Score</th><th>Trades</th><th>Win Rate</th><th>Avg P&L</th><th>Recommendation</th></tr>
<tr style="background-color: #d4edda;"><td><strong>5/5</strong></td><td>25</td><td>100%</td><td>+13.3%</td><td style="color: #28a745;"><strong>GO</strong></td></tr>
<tr style="background-color: #d4edda;"><td><strong>4/5</strong></td><td>20</td><td>85%</td><td>+15.9%</td><td style="color: #28a745;"><strong>GO</strong></td></tr>
<tr style="background-color: #fff3cd;"><td><strong>3/5</strong></td><td>6</td><td>83%</td><td>+10.5%</td><td style="color: #ffc107;"><strong>CAUTION</strong></td></tr>
<tr style="background-color: #f8d7da;"><td><strong>&lt;3</strong></td><td>1</td><td>0%</td><td>-4.6%</td><td style="color: #dc3545;"><strong>NO-GO</strong></td></tr>
</table>

<p><strong style="color: #28a745;">GO (4-5/5)</strong>: 45 trades, 93% win rate, +14.5% avg |
<strong style="color: #ffc107;">CAUTION (3/5)</strong>: 6 trades, 83% win, +10.5% avg |
<strong style="color: #dc3545;">NO-GO (&lt;3)</strong>: Skip</p>

<h3>Target Price LEVELS (52 Grade A Trades - Measured from OPEN)</h3>
<p><strong>These are fixed price levels from OPEN - mark on chart at 9:30 AM.</strong> Exit 1/3 at each tier:</p>
<table border="1" cellpadding="6" style="border-collapse: collapse; margin: 10px 0; font-size: 0.9em;">
<tr style="background-color: #f0f0f0;"><th>Cap</th><th>Tier 1 (33%)</th><th>Tier 2 (33%)</th><th>Tier 3 (34%)</th></tr>
<tr><td><strong>Large</strong></td><td>Gap Fill (100%)</td><td>4-Day EMA (71%)</td><td>Prior Day Low (86%)</td></tr>
<tr><td><strong>ETF</strong></td><td>1.0x ATR (80%)</td><td>Gap Fill (60%)</td><td>1.5x ATR (80%)</td></tr>
<tr><td><strong>Medium</strong></td><td>Gap Fill (81%)</td><td>1.5x ATR (65%)</td><td>2.0x ATR (45%)</td></tr>
<tr><td><strong>Small</strong></td><td>1.0x ATR (80%)</td><td>1.5x ATR (80%)</td><td>2.0x ATR (80%)</td></tr>
<tr><td><strong>Micro</strong></td><td>1.5x ATR (100%)</td><td>2.0x ATR (100%)</td><td>2.5x ATR (100%)</td></tr>
</table>
<p style="font-size: 0.85em; color: #666;"><em>Squeeze Risk: ETF +0.4% | Large +2% | Small/Medium +10% | Micro +14% above open before reversal</em></p>
<hr>

<h2>Bounce Setup Scoring Guide</h2>
<h3>Bounce Target Price LEVELS (52 Trades - Measured ABOVE Open)</h3>
<p><strong>These are fixed price levels ABOVE open for long bounce trades. Gap Fill = Red-to-Green move.</strong> Exit 1/3 at each tier:</p>
<table border="1" cellpadding="6" style="border-collapse: collapse; margin: 10px 0; font-size: 0.9em;">
<tr style="background-color: #c8e6c9;"><th>Cap</th><th>Tier 1 (33%)</th><th>Tier 2 (33%)</th><th>Tier 3 (34%)</th><th>n</th></tr>
<tr><td><strong>ETF</strong></td><td>0.5x ATR (83%)</td><td>1.0x ATR (83%)</td><td>Gap Fill (42%)</td><td>12</td></tr>
<tr><td><strong>Medium</strong></td><td>0.5x ATR (72%)</td><td>Gap Fill (48%)</td><td>1.0x ATR (60%)</td><td>25</td></tr>
<tr><td><strong>Small</strong></td><td>0.5x ATR (75%)</td><td>1.0x ATR (75%)</td><td>Gap Fill (86%)</td><td>8</td></tr>
<tr><td><strong>Large</strong></td><td>0.5x ATR (86%)</td><td>1.0x ATR (86%)</td><td>Gap Fill (29%)</td><td>7</td></tr>
<tr><td><strong>Micro</strong></td><td colspan="3"><em>Uses Small defaults</em></td><td>0</td></tr>
</table>
<p style="font-size: 0.85em; color: #666;"><em>Dip Risk: Median -1.0 ATR drawdown below open before bounce. All trades median high = 1.4 ATR.</em></p>
<p>Stocks <strong>not above all major moving averages</strong> (10/20/50 and 200 if available) are evaluated as bounce candidates. Auto-classified into two profiles:</p>
<table border="1" cellpadding="6" style="border-collapse: collapse; margin: 10px 0; font-size: 0.9em;">
<tr style="background-color: #f0f0f0;"><th>Profile</th><th>Description</th><th>Trades</th><th>Win Rate</th><th>Avg P&L</th></tr>
<tr><td><strong>GapFade_weakstock</strong></td><td>Stock already in downtrend, deep multi-day selloff</td><td>18</td><td>78%</td><td>+13.5%</td></tr>
<tr><td><strong>GapFade_strongstock</strong></td><td>Healthy stock hit by sudden selloff</td><td>34</td><td>74%</td><td>+3.0%</td></tr>
</table>
<p style="font-size: 0.85em; color: #dc3545;"><strong>WARNING: IntradayCapitch pattern = AVOID.</strong> n=9, 11% WR, -10.2% avg. GapFade trades: 88% WR, +10.1% avg.</p>

<h3>6 Pre-Trade Criteria (profile-adjusted thresholds)</h3>
<ol>
  <li><strong>Deep Selloff</strong> &mdash; Total % decline over consecutive down days</li>
  <li><strong>Consecutive Down Days</strong> &mdash; Multi-day selling pressure</li>
  <li><strong>Discount from 30d High</strong> &mdash; How far off recent highs</li>
  <li><strong>Capitulation Gap Down</strong> &mdash; Gap down on bounce day</li>
  <li><strong>Prior Day Range Expansion</strong> &mdash; Prior day range &ge; 1.0x ATR (75.6% WR vs 55.6%)</li>
  <li><strong>Volume Climax</strong> &mdash; Prior day or premarket volume vs ADV</li>
</ol>

<h3>Historical Performance by Recommendation (54 Trades)</h3>
<table border="1" cellpadding="8" style="border-collapse: collapse; margin: 10px 0;">
<tr style="background-color: #f0f0f0;"><th>Recommendation</th><th>Trades</th><th>Win Rate</th><th>Avg P&L</th></tr>
<tr style="background-color: #d4edda;"><td style="color: #28a745;"><strong>GO (5-6/6)</strong></td><td>33</td><td>100%</td><td>+14.2%</td></tr>
<tr style="background-color: #fff3cd;"><td style="color: #ffc107;"><strong>CAUTION (4/6)</strong></td><td>3</td><td>67%</td><td>-13.1%</td></tr>
<tr style="background-color: #f8d7da;"><td style="color: #dc3545;"><strong>NO-GO (&lt;4)</strong></td><td>18</td><td>22%</td><td>-5.4%</td></tr>
</table>

<p><strong>Routing Logic:</strong> Above 10/20/50MA (and 200MA if available) &rarr; Reversal | Otherwise &rarr; Bounce</p>

<h2>Bounce Day Cheat Sheet (n=52, cluster days n=28)</h2>
<p style="font-size: 0.85em; color: #666;"><em>All targets use only pre-entry information. Cluster days = multiple names bouncing same day.</em></p>

<h3>1. ATR-Based Targets</h3>
<table border="1" cellpadding="6" style="border-collapse: collapse; margin: 10px 0; font-size: 0.9em;">
<tr style="background-color: #c8e6c9;"><th></th><th>25th pct</th><th>Median</th><th>75th pct</th></tr>
<tr><td><strong>High (target) — Cluster</strong></td><td>1.3 ATR</td><td>2.4 ATR</td><td>3.5 ATR</td></tr>
<tr><td><strong>High (target) — All</strong></td><td>0.7 ATR</td><td>1.4 ATR</td><td>3.1 ATR</td></tr>
<tr><td><strong>Close — Cluster</strong></td><td>0.1 ATR</td><td>1.0 ATR</td><td>2.6 ATR</td></tr>
<tr><td><strong>Close — All</strong></td><td>0.0 ATR</td><td>0.9 ATR</td><td>1.8 ATR</td></tr>
<tr><td><strong>Drawdown — All</strong></td><td>-2.0 ATR</td><td>-1.0 ATR</td><td>-0.3 ATR</td></tr>
</table>
<p style="font-size: 0.85em;"><strong>Scale out starting at 1 ATR, aggressive target 2-3 ATR. Cluster days run ~70% further than solo trades.</strong></p>

<h3>2. Selloff Retrace Targets</h3>
<table border="1" cellpadding="6" style="border-collapse: collapse; margin: 10px 0; font-size: 0.9em;">
<tr style="background-color: #f0f0f0;"><th>Selloff Depth</th><th>Bounce High Retraces</th><th>Close Retraces</th></tr>
<tr><td>5-20%</td><td>60% of selloff</td><td>40%</td></tr>
<tr><td>20-40%</td><td><strong>68%</strong> of selloff</td><td><strong>51%</strong></td></tr>
<tr><td>40%+</td><td><strong>64%</strong> of selloff</td><td>45%</td></tr>
</table>
<p style="font-size: 0.85em;">Target ~60-68% retrace of prior selloff for the high. Close holds ~40-51%.</p>

<h3>3. Gap Fill</h3>
<ul style="font-size: 0.9em;">
<li>78% fill &gt;50% of the gap. 49% fill 100%+.</li>
<li>Median high fills 95% of gap. Median close fills 64%.</li>
<li>Only 33% close above full gap fill.</li>
<li><strong>50% gap fill = bread-and-butter first target. Full gap fill = stretch.</strong></li>
</ul>

<h3>4. Key Decision Rules</h3>
<table border="1" cellpadding="6" style="border-collapse: collapse; margin: 10px 0; font-size: 0.9em;">
<tr style="background-color: #f0f0f0;"><th>Rule</th><th>Data</th></tr>
<tr><td><strong>Take profits on the way up</strong></td><td>Only 49% of open-to-high retained at close. Only 31% close above 75% of high.</td></tr>
<tr><td><strong>First 30-min low = CRITICAL</strong></td><td>100% close green when low is in first 30 min (n=31). After 10 AM: 45% WR, -8.4% avg close for after-12 lows.</td></tr>
<tr><td><strong>Cluster days &gt; solo</strong></td><td>Cluster: 79% WR, +11.1% avg. Solo: 71% WR, +1.4% avg.</td></tr>
<tr><td><strong>Exhaustion gap = much better</strong></td><td>With exhaustion gap: 82% WR, +15.6% avg close. Without: 0% WR, -10.3% avg.</td></tr>
<tr><td><strong>5 consec down days = 100% WR</strong></td><td>5 days: 100% WR, +13.3% avg. 4 days: 75% WR, +15.3% avg. 0 days (IntradayCapitch): 0% WR.</td></tr>
<tr><td><strong>Weak stock setups bounce harder</strong></td><td>Weakstock: med high +14.6%, med close +10.7%. Strongstock: med high +9.9%, med close +4.0%.</td></tr>
<tr><td><strong>Closed outside lower BB = edge</strong></td><td>Outside BB: 85% WR, +9.7% avg. Inside BB: 65% WR, +3.5% avg.</td></tr>
<tr><td><strong>Prior day closed near lows = capitulation</strong></td><td>Closed near lows (&le;15%): 86% WR, +8.6% avg. Not near lows: 55% WR, +3.7% avg.</td></tr>
<tr><td><strong>Near 52-week low = bigger bounce</strong></td><td>Near 52wk low: +10.3% avg. Not near: +4.3% avg.</td></tr>
<tr><td><strong>ETFs highest WR (92%) but lower upside</strong></td><td>ETF: 92% WR, +6.3% avg. Small: 88% WR, +6.7%. Medium: 68% WR, +7.5%. Large: 57% WR, +3.9%.</td></tr>
</table>

<h3>5. Overnight Hold (cluster days: 86% gapped up next morning)</h3>
<table border="1" cellpadding="6" style="border-collapse: collapse; margin: 10px 0; font-size: 0.9em;">
<tr style="background-color: #f0f0f0;"><th>Metric</th><th>Cluster Days</th><th>All Trades</th></tr>
<tr><td>Overnight positive %</td><td><strong>86%</strong> (24/28)</td><td>75%</td></tr>
<tr><td>Median overnight</td><td><strong>+11.0%</strong></td><td>+9.2%</td></tr>
</table>
<p style="font-size: 0.85em;"><strong>Hold a portion overnight on cluster bounce days.</strong></p>

<h3>6. What Predicts Bigger Bounces (check pre-entry, n=52)</h3>
<ol style="font-size: 0.9em;">
<li><strong>High vol trend 3d</strong> (r=+0.62 pnl, r=+0.73 high) &mdash; volume accelerating into bounce day = strongest signal</li>
<li><strong>Low down-day volume ratio</strong> (r=-0.46 pnl, r=-0.51 high) &mdash; selling exhausted on declining volume</li>
<li><strong>High prior-day vol expansion</strong> (r=+0.56 pnl, r=+0.63 high) &mdash; prior day volume spike = capitulation</li>
<li><strong>High premarket vol %</strong> (r=+0.34 pnl, r=+0.40 high) &mdash; early attention/participation</li>
<li><strong>Deeper selloff</strong> (r=-0.40 pnl) &mdash; &gt;30% selloff: +18.9% avg. 15-30%: +12.0%. &lt;5%: +0.0%</li>
<li><strong>Further off 52wk high</strong> (r=-0.32 pnl, r=-0.39 high) &mdash; deeper fall = bigger bounce</li>
<li><strong>Lower BB position</strong> (r=-0.32 pnl) &mdash; lower on Bollinger Band = more oversold</li>
<li><strong>Higher ATR%</strong> (r=+0.25 pnl) &mdash; more volatile names bounce harder</li>
<li><strong>Low in first 30 min</strong> (r=-0.42 pnl) &mdash; earlier low = better close. Late low = disaster.</li>
</ol>
<hr>

<h2>Rules</h2>
<ol>
  <li>Quality in everything – end day with quality & take breaks to maintain quality</li>
  <li>Push size in liquid names</li>
  <li>Start orderpipe</li>
  <li>Start cnbc</li>
  <li>It's ok to consciously risk 30-40K on bread and butter / ETF aggression</li>
  <li>Let the upside take care of itself</li>
  <li>Selectivity - trust your instincts - reactive trades always best (don't change tiers)</li>
  <li>Use the 2-minute bar for high volume good news / 1 min for scalp - after VOLUME</li>
  <li>Liquidity focus</li>
  <li>Who gets paid? → That's my trade.</li>
  <li>Expected Value over First Prints - Push size in your bread and butter</li>
  <li>Every single trade was not within .2% of reference after a minute unless it was a dissem issue</li>
  <li>Single stocks - 50% of them last 21.5 minutes - On my biggest trades - 50% = 35 mins</li>
  <li>If it breaks upper or lower bound trend- hold until it fails trend as it's a pos signal (good RRR to see if it goes para)</li>
</ol>

<h2>News Rules / Reminders</h2>
<ol>
  <li>CP on canada deal with US / CAR on any car tariff changes / STZ+EWW or TNA on Mexico / XLE short / MT LONG / KYIV on Russia Deal</li>
  <li>XRT SWK for Trump tariffs</li>
  <li>SILJ PAAS on Silver </li>

</ol>

<h2>Morning Checklist</h2>
<ul>
  <li>Read overnight news</li>
  <li>Look at all stocks gapping up or down 5 %+ (Stockfetcher, MAT, NLRTs)</li>
  <li>Go through rules and reminders</li>
  <li>Check Trump schedule</li>
  <li>Create one explicit process-oriented goal for the day</li>
  <li>Go through all events / ECO for the day – call out important ones to group</li>
  <li>Write down any tasks you want to accomplish today</li>
</ul>"""

# Custom ordering for percentile keys
PERCENTILE_ORDER = [
    "pct_change_120",
    "pct_change_90",
    "pct_change_30",
    "pct_change_15",
    "pct_change_3",
    "pct_from_10mav", "pct_from_20mav", "pct_from_50mav", "pct_from_200mav",
]

def _format_percentile_dict(d: dict) -> str:
    """Return nicely-formatted percentile dict respecting the desired order."""
    if not d:
        return "    None"
    lines: List[str] = []
    # Add keys in the specified order first
    for key in PERCENTILE_ORDER:
        if key in d:
            lines.append(f"    {key}: {d[key]:.1f}")
    # Append any remaining (non-PCT) keys alphabetically
    for key in sorted(k for k in d if k not in PERCENTILE_ORDER):
        lines.append(f"    {key}: {d[key]:.1f}")
    return "\n".join(lines)

# +++++ NEW helper ++++++++++++++++++++++++++++++++++++++++++++++++++++++
def _fmt_pct(val):
    """
    Convert a fractional distance-from-MA (e.g. 0.042) to a percentage
    string like '4.2%'.  Falls back to str(val) if conversion fails.
    """
    try:
        return f"{float(val) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(val)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def _fmt_dollars(val) -> str:
    """Format a number as dollars (e.g. $12.34)."""
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "N/A"
        return f"${float(val):,.2f}"
    except Exception:
        return "N/A"


def _calc_price_from_frac(base_price, frac) -> Optional[float]:
    """
    Convert a fractional metric into a boundary price.

    Many bounce metrics are defined as: frac = (price - base) / base
    -> price = base * (1 + frac)
    """
    try:
        if base_price is None or frac is None:
            return None
        if (isinstance(base_price, float) and pd.isna(base_price)) or (isinstance(frac, float) and pd.isna(frac)):
            return None
        base_f = float(base_price)
        frac_f = float(frac)
        if base_f == 0:
            return None
        return base_f * (1.0 + frac_f)
    except Exception:
        return None


def _format_bounce_price_targets_html(
    item_name: str,
    threshold_frac: Optional[float],
    a_median_frac: Optional[float],
    bounce_metrics: Optional[Dict],
) -> str:
    """
    Return compact HTML for "what price hits threshold/median" for key bounce criteria.

    Only supports:
      - pct_off_30d_high: base = high_30d
      - gap_pct: base = prior_close
      - selloff_total_pct: base = selloff_start_open
    """
    if not bounce_metrics:
        return ""

    if item_name == "pct_off_30d_high":
        base_label = "30d high"
        base_price = bounce_metrics.get("high_30d")
        suffix = "off 30d high"
    elif item_name == "gap_pct":
        base_label = "prior close"
        base_price = bounce_metrics.get("prior_close")
        suffix = "gap down"
    elif item_name == "selloff_total_pct":
        base_label = "selloff start open"
        base_price = bounce_metrics.get("selloff_start_open")
        suffix = "selloff"
    else:
        return ""

    if base_price is None or (isinstance(base_price, float) and pd.isna(base_price)):
        return ""
    try:
        if float(base_price) == 0:
            return ""
    except Exception:
        return ""

    threshold_price = _calc_price_from_frac(base_price, threshold_frac)
    median_price = _calc_price_from_frac(base_price, a_median_frac)

    threshold_pct = None
    if threshold_frac is not None and not pd.isna(threshold_frac):
        threshold_pct = abs(float(threshold_frac)) * 100
    median_pct = None
    if a_median_frac is not None and not pd.isna(a_median_frac):
        median_pct = abs(float(a_median_frac)) * 100
    median_pct_str = None
    if median_pct is not None:
        median_pct_str = f"{median_pct:.1f}".rstrip("0").rstrip(".")

    lines = [
        f'<div style="margin-top: 2px; color: #888; font-size: 0.82em; line-height: 1.25em;">'
        f'<div>{base_label}: <strong>{_fmt_dollars(base_price)}</strong></div>'
    ]

    if threshold_price is not None and threshold_pct is not None:
        lines.append(
            f'<div>price @ {threshold_pct:.0f}% {suffix}: <strong>{_fmt_dollars(threshold_price)}</strong></div>'
        )

    if median_price is not None and median_pct_str is not None:
        lines.append(
            f'<div>A median price @ {median_pct_str}% {suffix}: <strong>{_fmt_dollars(median_price)}</strong></div>'
        )

    lines.append("</div>")
    return "\n".join(lines)


# -------------------------------------------------
# Bucket routing: Bounce vs Reversal
# -------------------------------------------------

ROUTING_MA_KEYS = ("pct_from_10mav", "pct_from_20mav", "pct_from_50mav", "pct_from_200mav")


def _safe_num(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(v) else v


def route_playbook(pct_data_in: Dict, mav_data_in: Dict) -> Tuple[str, str]:
    """
    Return (bucket, reason) where bucket is: 'reversal' | 'bounce'.

    Logic:
      - If price is above all major moving averages (10/20/50 and 200 if available) -> reversal
      - Otherwise -> bounce

    Notes:
      - Requires 10/20/50 MA distances to be present to label a reversal.
      - If MA data is incomplete (common for new IPOs), defaults to bounce.
    """
    pct_data_in = pct_data_in or {}
    mav_data_in = mav_data_in or {}

    ma_vals: Dict[str, float] = {}
    for k in ROUTING_MA_KEYS:
        v = _safe_num(mav_data_in.get(k))
        if v is not None:
            ma_vals[k] = v

    required = ("pct_from_10mav", "pct_from_20mav", "pct_from_50mav")
    if all(k in ma_vals for k in required):
        keys_to_check = list(required)
        if "pct_from_200mav" in ma_vals:
            keys_to_check.append("pct_from_200mav")

        above_all = all(ma_vals[k] > 0 for k in keys_to_check)
        if above_all:
            suffix = " (+200MA)" if "pct_from_200mav" in ma_vals else ""
            return "reversal", f"Above 10/20/50MA{suffix}"

        below = [k for k in keys_to_check if ma_vals.get(k, 0) <= 0]
        below_lbl = ", ".join([b.replace("pct_from_", "").replace("mav", "MA") for b in below]) or "N/A"
        return "bounce", f"Not above all MAs (below: {below_lbl})"

    return "bounce", "MA data incomplete (default bounce)"


def _generate_ticker_section(ticker: str, data: dict, charts_dir: str, pretrade_metrics: dict = None, bounce_metrics: dict = None) -> str:
    """Return formatted string section for one ticker and create its chart."""

    pct_data = data.get("pct_data", {})
    range_data = data.get("range_data", {})
    mav_data = data.get("mav_data", {})

    # Build section text
    lines: List[str] = [f"<h2>Ticker: {ticker}</h2>"]

    # --- Routing: MA positioning (preferred) with pct-change fallback ---
    raw_pct_3 = pct_data.get("pct_change_3")
    raw_pct_120 = pct_data.get("pct_change_120")
    has_pct_3 = raw_pct_3 is not None and not pd.isna(raw_pct_3)
    has_pct_120 = raw_pct_120 is not None and not pd.isna(raw_pct_120)

    bucket, bucket_reason = route_playbook(pct_data, mav_data)
    is_reversal = bucket == "reversal"
    is_bounce = bucket == "bounce"

    # Show routing decision near the top for quick scanning/debugging
    bucket_color = {"reversal": "#0d6efd", "bounce": "#198754", "filtered": "#6c757d"}.get(bucket, "#6c757d")
    lines.append(
        f'<div style="margin: 6px 0 10px 0; padding: 8px 10px; border-radius: 6px; border: 1px solid {bucket_color};">'
        f'<strong style="color: {bucket_color};">Bucket:</strong> <strong>{bucket.upper()}</strong>'
        f'<span style="color:#666;"> — {bucket_reason}</span>'
        f'</div>'
    )

    cap = None
    if is_reversal:
        # REVERSAL path (by MA routing): show reversal checklist + exit levels
        if pretrade_metrics:
            score_result = score_pretrade_setup(ticker, pretrade_metrics)
            cap = score_result.get('cap')
            lines.append("<strong>Reversal Setup Score:</strong>")
            lines.append(format_pretrade_score_html(score_result))

        # Add data-informed exit target LEVELS (measured from last close)
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        exit_data = get_exit_target_data(ticker, today)
        if exit_data.get('open_price') and exit_data.get('atr'):
            if cap is None:
                cap = get_ticker_cap(ticker)
            targets = calculate_exit_targets(
                cap=cap,
                entry_price=exit_data['open_price'],
                atr=exit_data['atr'],
                prior_close=exit_data.get('prior_close'),
                prior_low=exit_data.get('prior_low'),
                ema_4=exit_data.get('ema_4')
            )
            lines.append("<strong>Target Price Levels (from Live Price):</strong>")
            lines.append(format_exit_targets_html(targets))

    elif is_bounce:
        # BOUNCE path (by MA routing): show bounce checklist (if available) + exit levels
        if cap is None:
            cap = get_ticker_cap(ticker)

        # Default bounce profile for percentiles even if bounce_metrics is missing
        try:
            bounce_setup_type, _ = classify_stock(
                {
                    "pct_from_200mav": mav_data.get("pct_from_200mav"),
                    "pct_change_30": pct_data.get("pct_change_30"),
                }
            )
        except Exception:
            bounce_setup_type = "GapFade_strongstock"

        if bounce_metrics:
            # Supplement bounce_metrics with screener data (only if missing)
            if bounce_metrics.get('pct_from_200mav') is None and mav_data.get('pct_from_200mav') is not None:
                bounce_metrics['pct_from_200mav'] = mav_data['pct_from_200mav']
            volume_data = data.get("volume_data", {})
            if bounce_metrics.get('percent_of_vol_on_breakout_day') is None and volume_data.get('percent_of_vol_on_breakout_day') is not None:
                bounce_metrics['percent_of_vol_on_breakout_day'] = volume_data['percent_of_vol_on_breakout_day']

            checker = BouncePretrade()
            bounce_result = checker.validate(ticker, bounce_metrics, cap=cap)
            bounce_setup_type = bounce_result.setup_type
            lines.append("<strong>Bounce Setup Score:</strong>")
            lines.append(format_bounce_score_html(bounce_result, bounce_metrics=bounce_metrics))

            # Bounce intensity ranking score
            intensity = compute_bounce_intensity(bounce_metrics)
            lines.append(format_bounce_intensity_html(intensity))
        else:
            lines.append(
                '<div style="border: 2px solid #6c757d; padding: 10px; margin: 10px 0; border-radius: 5px;">'
                '<strong style="color: #6c757d;">BOUNCE</strong> '
                '<span>Bounce metrics unavailable; skipping bounce checklist.</span>'
                '</div>'
            )

        # Add bounce exit target LEVELS (measured ABOVE open for long trades)
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        exit_data = get_exit_target_data(ticker, today, prefer_open=True)
        if exit_data.get('open_price') and exit_data.get('atr'):
            bounce_targets = calculate_bounce_exit_targets(
                cap=cap,
                entry_price=exit_data['open_price'],
                atr=exit_data['atr'],
                prior_close=exit_data.get('prior_close'),
                prior_high=exit_data.get('prior_high'),
            )
            # Let the HTML formatter reflect whether we're using OPEN or a live fallback.
            bounce_targets['entry_price_source'] = exit_data.get('open_price_source')
            src = (exit_data.get('open_price_source') or '').lower()
            if src == 'open':
                label = "from OPEN"
            elif src == 'live':
                label = "from LIVE (open not available)"
            else:
                label = "from reference price"
            lines.append(f"<strong>Bounce Target Levels ({label}):</strong>")
            lines.append(format_bounce_exit_targets_html(bounce_targets))

    else:
        # FILTERED: no clear reversal or bounce signal
        pct_3_str = f"{raw_pct_3 * 100:.1f}%" if has_pct_3 else "N/A"
        pct_120_str = f"{raw_pct_120 * 100:.1f}%" if has_pct_120 else "N/A"
        lines.append(
            '<div style="border: 2px solid #6c757d; padding: 10px; margin: 10px 0; border-radius: 5px;">'
            f'<strong style="color: #6c757d;">FILTERED</strong> '
            f'<span>{bucket_reason} | 3-day: {pct_3_str} | 120-day: {pct_120_str}</span>'
            '</div>'
        )

    # --- Extension from Moving Averages (always shown) ---
    if mav_data:
        mav_label = lambda k: k.removeprefix("pct_from_").removesuffix("mav") + " MA"
        lines.append('<h3 style="margin: 4px 0 2px 0;">Extension from Moving Averages</h3>')
        lines.append('<table style="font-size: 0.9em; margin-top: 0;">')
        for k, v in mav_data.items():
            lines.append(
                f'<tr><td style="padding-right: 12px;">{mav_label(k)}</td>'
                f'<td>{_fmt_pct(v)}</td></tr>'
            )
        lines.append('</table>')

    # --- Percentiles: compare against the right dataset based on bucket ---
    if is_bounce:
        bounce_ref_df = BOUNCE_DF_WEAK if bounce_setup_type == 'GapFade_weakstock' else BOUNCE_DF_STRONG
        pcts = ss.calculate_percentiles(bounce_ref_df, data, BOUNCE_COLUMNS_TO_COMPARE)
        # Invert percentiles for bounce candidates: more negative = deeper selloff = better setup.
        # percentileofscore returns % of values <= x, so -13% scores 100% when history is -40%.
        # Flip so 100% = most extreme (most negative) bounce candidate.
        pcts = {k: round(100 - v, 1) for k, v in pcts.items()}
        setup_label = bounce_setup_type.replace('GapFade_', '')
        lines.append(f"<strong>Bounce Percentiles ({setup_label}, n={len(bounce_ref_df)}):</strong>")
    else:
        pcts = ss.calculate_percentiles(reversal_df, data, COLUMNS_TO_COMPARE)
        lines.append("<strong>Reversal Percentiles:</strong>")
    lines.append(f'<pre style="margin: 2px 0; font-size: 0.9em;">{_format_percentile_dict(pcts)}</pre>')

    def _fmt(val):
        try:
            return f"{float(val):.2f}"
        except (TypeError, ValueError):
            return str(val)

    if pct_data:
        lines.append("<strong>Absolute PCT Changes:</strong>")
        lines.append(f'<pre style="margin: 2px 0; font-size: 0.9em;">{indent(chr(10).join([f"{k}: {_fmt(v)}" for k, v in pct_data.items()]), "    ")}</pre>')

    if range_data:
        lines.append('<br><strong>Range Data:</strong>')
        lines.append(f'<pre style="margin: 2px 0; font-size: 0.9em;">{indent(chr(10).join([f"{k}: {_fmt(v)}" for k, v in range_data.items()]), "    ")}</pre>')

    # Generate chart and embed inline
    try:
        chart_path = Path(create_daily_chart(ticker, output_dir=charts_dir))
        img_tag = f'<img src="{_png_to_data_uri(chart_path)}" alt="{ticker} chart" style="max-width:800px;">'
        lines.append(img_tag)
    except Exception as e:
        print(f"Failed to generate chart for {ticker}: {e}")
        lines.append(f"<p><em>Chart unavailable for {ticker}</em></p>")

    # Return HTML block
    return "\n".join(lines)


def _png_to_data_uri(path: Path) -> str:
    """Return a data URI string for embedding PNG inline."""
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{encoded}"

# -------------------------------------------------
# PDF export helper
# -------------------------------------------------

def _save_report_pdf(html_report: str, output_dir: str = "reports") -> str | None:
    """Save the given HTML report to *output_dir* as a timestamped PDF.

    Uses WeasyPrint if available (pip install weasyprint).  If WeasyPrint is
    not installed the function prints a warning and does nothing.
    Returns the path of the written file or *None* if skipped/failed.
    """

    if not _PDF_AVAILABLE and not _PDFKIT_AVAILABLE:
        print("No PDF backend available (install `weasyprint` or `pdfkit` + wkhtmltopdf); skipping PDF export.")
        return None

    Path(output_dir).mkdir(exist_ok=True)
    filename = f"watchlist_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    path = Path(output_dir) / filename

    if _PDF_AVAILABLE:
        try:
            HTML(string=html_report).write_pdf(str(path))
            print(f"PDF report saved to {path} using WeasyPrint")
            return str(path)
        except Exception as e:
            print(f"WeasyPrint failed ({e}). Attempting pdfkit fallback...")

    # Fallback to pdfkit if available
    if _PDFKIT_AVAILABLE:
        if not _WKHTMLTOPDF_PATH or not os.path.isfile(_WKHTMLTOPDF_PATH):
            print("wkhtmltopdf executable not found. Set WKHTMLTOPDF_PATH environment variable to its location.")
        try:
            # pdfkit requires wkhtmltopdf binary; if not on PATH you can set pdfkit.configuration()
            try:
                # First try default invocation (assumes binary on PATH)
                pdfkit.from_string(html_report, str(path))
            except (OSError, IOError):
                # Fallback to explicit path if provided or default Windows install location
                if _WKHTMLTOPDF_PATH:
                    config = pdfkit.configuration(wkhtmltopdf=_WKHTMLTOPDF_PATH)
                    pdfkit.from_string(html_report, str(path), configuration=config)
                else:
                    raise
            print(f"PDF report saved to {path} using pdfkit/wkhtmltopdf")
            return str(path)
        except Exception as e:
            print(f"pdfkit failed to write PDF: {e}")

    return None

#TODO - add historical context to the report - "largest volume day since xyz" or "highest PCT change since abc"
#TODO - add news overnight from context and google into llm summary to report.
def generate_report() -> str:
    """Return a giant string report and create all charts under *charts/*."""

    watchlist = ss.watchlist
    charts_dir = "charts"  # same default as charter
    Path(charts_dir).mkdir(exist_ok=True)

    # Collect new metrics for watchlist tickers
    all_data = ss.get_all_stocks_data(watchlist)

    # Collect pre-trade metrics for reversal scoring
    print("Fetching pre-trade reversal metrics...")
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    pretrade_metrics_all = {}
    for ticker in watchlist:
        try:
            pretrade_metrics_all[ticker] = get_pretrade_metrics(ticker, today)
            print(f"  {ticker}: {pretrade_metrics_all[ticker]}")
        except Exception as e:
            print(f"  {ticker}: Failed to get pretrade metrics - {e}")
            pretrade_metrics_all[ticker] = {}

    # Route tickers into bounce vs reversal buckets (MA-based routing)
    print("Routing tickers into bounce vs reversal buckets...")
    bucket_map: Dict[str, str] = {}
    for ticker in watchlist:
        ticker_data = all_data.get(ticker, {})
        bucket, _reason = route_playbook(
            ticker_data.get("pct_data", {}) or {},
            ticker_data.get("mav_data", {}) or {},
        )
        bucket_map[ticker] = bucket

    bounce_ct = sum(1 for b in bucket_map.values() if b == "bounce")
    rev_ct = sum(1 for b in bucket_map.values() if b == "reversal")
    print(f"Bucket counts — bounce: {bounce_ct} | reversal: {rev_ct}")

    # Collect bounce pre-trade metrics for tickers in the bounce bucket
    print("Fetching bounce pre-trade metrics (bounce bucket)...")
    bounce_metrics_all = {}
    for ticker in watchlist:
        try:
            if bucket_map.get(ticker) == "bounce":
                bounce_metrics_all[ticker] = fetch_bounce_metrics(ticker, today)
                # Debug: show prior_day_rvol value
                pdr = bounce_metrics_all[ticker].get('prior_day_rvol')
                print(f"  {ticker}: bounce metrics fetched (prior_day_rvol={pdr})")
        except Exception as e:
            print(f"  {ticker}: Failed to get bounce metrics - {e}")

    # ---- Split watchlist into bounce vs reversal/other, sort each independently ----
    def _safe_float(x, default=float('-inf')):
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    bounce_tickers = []
    other_tickers = []
    _bounce_intensity_cache = {}

    for ticker in watchlist:
        if bucket_map.get(ticker) == "bounce":
            if ticker in bounce_metrics_all:
                intensity = compute_bounce_intensity(bounce_metrics_all[ticker])
                _bounce_intensity_cache[ticker] = intensity['composite']
            else:
                _bounce_intensity_cache[ticker] = 0
            bounce_tickers.append(ticker)
        else:
            other_tickers.append(ticker)

    # Sort bounce candidates by intensity score (highest first)
    bounce_tickers.sort(key=lambda t: _bounce_intensity_cache.get(t, 0), reverse=True)

    # Sort reversal/other by pct_change_15 reversal percentile (descending)
    _rev_pcts_cache = {}
    for ticker in other_tickers:
        try:
            _rev_pcts_cache[ticker] = ss.calculate_percentiles(
                reversal_df, all_data.get(ticker, {}), COLUMNS_TO_COMPARE
            )
        except Exception:
            _rev_pcts_cache[ticker] = {}

    other_tickers.sort(
        key=lambda t: _safe_float(_rev_pcts_cache.get(t, {}).get("pct_change_15")),
        reverse=True
    )

    # Bounce candidates first, then reversal/other
    sorted_watchlist = bounce_tickers + other_tickers
    # -----------------------------------------------------------------------

    sections = [
        _generate_ticker_section(ticker, all_data[ticker], charts_dir,
                                 pretrade_metrics_all.get(ticker, {}),
                                 bounce_metrics_all.get(ticker))
        for ticker in sorted_watchlist
    ]

    # Put the long scoring guides at the bottom (tickers first).
    html_report = "<br><br>\n".join(sections) + "<hr>" + HEADER_HTML

    # Persist a copy of the report locally before (or even if) e-mailing
    _save_report_pdf(html_report)

    # Send email as HTML
    send_email(
        to_email="zburr@trlm.com",
        subject="Daily Watchlist Report",
        body=html_report,
        is_html=True,
    )

    return html_report

def project_choice():
    send_email(
        to_email="zburr@trlm.com",
        subject="Your One Current Focus Project",
        body="""Jupiter improvements - focus on making the pipeline cleaner, should review the code and understand it.""",
        is_html=True,
    )
if __name__ == "__main__":
    rep = generate_report()
    # project_choice()
    # Print plain-text fallback (strip HTML tags) if desired
    print("Report generated, saved, and (attempted) e-mailed.")
    cleanup_charts() 