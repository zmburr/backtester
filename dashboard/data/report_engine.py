"""
Report computation engine — extracted from scripts/generate_report.py.

Produces structured TickerReportData objects (NOT HTML) that can be rendered
by any frontend: Streamlit dashboard, email report, etc.

Both generate_report.py and the dashboard import from here so computation
logic stays in one place.
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from scipy.stats import percentileofscore as _pctrank

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import data_queries.polygon_queries as _pq_module
from analyzers.bounce_exit_targets import calculate_bounce_exit_targets
from analyzers.bounce_scorer import (
    BouncePretrade,
    SETUP_PROFILES,
    classify_from_setup_column,
    classify_stock,
    fetch_bounce_metrics,
)
from analyzers.exit_targets import calculate_exit_targets
from data_queries.ticker_cache import TickerCache
from scanners import stock_screener as ss

try:
    from data_queries.trillium_queries import get_actual_current_price_trill
except ImportError:
    get_actual_current_price_trill = None

# ---------------------------------------------------------------------------
# Save original polygon functions before any monkey-patching
# ---------------------------------------------------------------------------
_orig_get_levels_data = _pq_module.get_levels_data
_orig_get_daily = _pq_module.get_daily
_orig_get_atr = _pq_module.get_atr
_orig_get_actual_current_price = _pq_module.get_actual_current_price


# ---------------------------------------------------------------------------
# ReportCache — thread-safe API response cache
# ---------------------------------------------------------------------------
class ReportCache:
    """Thread-safe cache for polygon_queries API calls.

    Monkey-patches module-level functions so all downstream code
    (bounce_scorer, charter, exit_targets, etc.) transparently hits cache.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._levels = {}
        self._daily = {}
        self._atr = {}
        self._price = {}
        self._ticker_cache = TickerCache()

    def get_levels_data(self, ticker, date, window, multiplier, timespan):
        if timespan == 'day' and multiplier == 1:
            key = (ticker, date, window, multiplier, timespan)
            with self._lock:
                if key in self._levels:
                    return self._levels[key]
            result = self._ticker_cache.get_daily_bars(ticker, date, window)
            with self._lock:
                self._levels[key] = result
            return result
        key = (ticker, date, window, multiplier, timespan)
        with self._lock:
            if key in self._levels:
                return self._levels[key]
        result = _orig_get_levels_data(ticker, date, window, multiplier, timespan)
        with self._lock:
            self._levels[key] = result
        return result

    def get_daily(self, ticker, date):
        key = (ticker, date)
        with self._lock:
            if key in self._daily:
                return self._daily[key]
        result = _orig_get_daily(ticker, date)
        with self._lock:
            self._daily[key] = result
        return result

    def get_atr(self, ticker, date):
        key = (ticker, date)
        with self._lock:
            if key in self._atr:
                return self._atr[key]
        result = _orig_get_atr(ticker, date)
        with self._lock:
            self._atr[key] = result
        return result

    def get_actual_current_price(self, ticker, date=None):
        with self._lock:
            if ticker in self._price:
                return self._price[ticker]
        result = _orig_get_actual_current_price(ticker, date) if date else _orig_get_actual_current_price(ticker)
        with self._lock:
            self._price[ticker] = result
        return result

    def install(self):
        """Monkey-patch polygon_queries to use cached versions."""
        _pq_module.get_levels_data = self.get_levels_data
        _pq_module.get_daily = self.get_daily
        _pq_module.get_atr = self.get_atr
        _pq_module.get_actual_current_price = self.get_actual_current_price

    def uninstall(self):
        """Restore original polygon_queries functions."""
        _pq_module.get_levels_data = _orig_get_levels_data
        _pq_module.get_daily = _orig_get_daily
        _pq_module.get_atr = _orig_get_atr
        _pq_module.get_actual_current_price = _orig_get_actual_current_price


# ---------------------------------------------------------------------------
# Bounce reference data — loaded once at module level
# ---------------------------------------------------------------------------
_bounce_csv = REPO_ROOT / "data" / "bounce_data.csv"
_bounce_df_all = pd.read_csv(_bounce_csv).dropna(subset=['ticker', 'date'])

# Normalize prior_day_rvol column
if 'prior_day_rvol' not in _bounce_df_all.columns:
    if 'percent_of_vol_one_day_before' in _bounce_df_all.columns:
        _bounce_df_all['prior_day_rvol'] = pd.to_numeric(
            _bounce_df_all['percent_of_vol_one_day_before'], errors='coerce'
        )
    elif {'vol_one_day_before', 'avg_daily_vol'}.issubset(_bounce_df_all.columns):
        _bounce_df_all['prior_day_rvol'] = (
            pd.to_numeric(_bounce_df_all['vol_one_day_before'], errors='coerce')
            / pd.to_numeric(_bounce_df_all['avg_daily_vol'], errors='coerce')
        )
    else:
        _bounce_df_all['prior_day_rvol'] = pd.NA
else:
    _bounce_df_all['prior_day_rvol'] = pd.to_numeric(_bounce_df_all['prior_day_rvol'], errors='coerce')
    if 'percent_of_vol_one_day_before' in _bounce_df_all.columns:
        _bounce_df_all['prior_day_rvol'] = _bounce_df_all['prior_day_rvol'].fillna(
            pd.to_numeric(_bounce_df_all['percent_of_vol_one_day_before'], errors='coerce')
        )

_bounce_df_all['_setup_profile'] = _bounce_df_all['Setup'].apply(classify_from_setup_column)
_bounce_df_all = _bounce_df_all[_bounce_df_all['_setup_profile'] != 'IntradayCapitch'].copy()
BOUNCE_DF_WEAK = _bounce_df_all[_bounce_df_all['_setup_profile'] == 'GapFade_weakstock'].copy()
BOUNCE_DF_STRONG = _bounce_df_all[_bounce_df_all['_setup_profile'] == 'GapFade_strongstock'].copy()
BOUNCE_DF_ALL = _bounce_df_all


# ---------------------------------------------------------------------------
# Bounce Intensity Score
# ---------------------------------------------------------------------------
_BOUNCE_INTENSITY_SPEC = [
    ('pct_change_3',         False, 0.30),
    ('pct_change_15',        False, 0.20),
    ('selloff_total_pct',    False, 0.15),
    ('gap_pct',              False, 0.15),
    ('pct_off_30d_high',     False, 0.15),
    ('pct_off_52wk_high',    False, 0.05),
]


def compute_bounce_intensity(metrics: Dict, ref_df: pd.DataFrame = None) -> Dict:
    """Compute weighted composite bounce intensity score (0-100)."""
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
        pctile = raw_pctile if higher_is_better else 100.0 - raw_pctile

        details[col] = {'pctile': round(pctile, 1), 'weight': weight, 'actual': actual}
        weighted_sum += pctile * weight
        total_weight += weight

    composite = round(weighted_sum / total_weight, 1) if total_weight > 0 else 0.0
    return {'composite': composite, 'details': details}


# ---------------------------------------------------------------------------
# Reversal pre-trade scoring
# ---------------------------------------------------------------------------
PRETRADE_THRESHOLDS = {
    'Micro': {'pct_from_9ema': 0.80, 'prior_day_range_atr': 3.0, 'prior_day_rvol': 2.0, 'premarket_rvol': 0.05, 'pct_change_3': 0.50, 'gap_pct': 0.15},
    'Small': {'pct_from_9ema': 0.40, 'prior_day_range_atr': 2.0, 'prior_day_rvol': 2.0, 'premarket_rvol': 0.05, 'pct_change_3': 0.25, 'gap_pct': 0.10},
    'Medium': {'pct_from_9ema': 0.15, 'prior_day_range_atr': 1.0, 'prior_day_rvol': 1.5, 'premarket_rvol': 0.05, 'pct_change_3': 0.10, 'gap_pct': 0.05},
    'Large': {'pct_from_9ema': 0.08, 'prior_day_range_atr': 0.8, 'prior_day_rvol': 1.0, 'premarket_rvol': 0.05, 'pct_change_3': 0.05, 'gap_pct': 0.00},
    'ETF': {'pct_from_9ema': 0.04, 'prior_day_range_atr': 1.0, 'prior_day_rvol': 1.5, 'premarket_rvol': 0.05, 'pct_change_3': 0.03, 'gap_pct': 0.00},
}

KNOWN_ETFS = {
    'GLD', 'SLV', 'GDXJ', 'QQQ', 'SPY', 'XLE', 'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU',
    'IWM', 'DIA', 'VTI', 'VOO', 'VXX', 'UVXY', 'SQQQ', 'TQQQ', 'SPXU', 'SPXL', 'TLT', 'HYG', 'LQD',
    'EEM', 'EWZ', 'EWJ', 'FXI', 'KWEB', 'SMH', 'XBI', 'IBB', 'ARKK', 'ARKG',
}

_market_cap_cache: Dict[str, str] = {}


def get_ticker_cap(ticker: str) -> str:
    """Get market cap category for a ticker from Polygon API."""
    if ticker in _market_cap_cache:
        return _market_cap_cache[ticker]

    if ticker.upper() in KNOWN_ETFS:
        _market_cap_cache[ticker] = 'ETF'
        return 'ETF'

    try:
        from polygon.rest import RESTClient
        client = RESTClient(api_key=os.getenv("POLYGON_API_KEY"))
        details = client.get_ticker_details(ticker)

        if hasattr(details, 'type') and details.type in ('ETF', 'ETN'):
            _market_cap_cache[ticker] = 'ETF'
            return 'ETF'

        mc = details.market_cap
        if mc is None:
            cap = 'Medium'
        elif mc >= 100_000_000_000:
            cap = 'Large'
        elif mc >= 2_000_000_000:
            cap = 'Medium'
        elif mc >= 300_000_000:
            cap = 'Small'
        else:
            cap = 'Micro'

        _market_cap_cache[ticker] = cap
        return cap
    except Exception as e:
        logging.warning(f"Failed to get market cap for {ticker}: {e}, defaulting to Medium")
        _market_cap_cache[ticker] = 'Medium'
        return 'Medium'


def get_pretrade_metrics(ticker: str, date: str) -> Dict:
    """Fetch metrics needed for pre-trade scoring (9EMA distance, gap%, range/ATR, etc.)."""
    metrics = {}
    try:
        df = _pq_module.get_levels_data(ticker, date, 35, 1, 'day')
        if df is None or len(df) < 5:
            return metrics

        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()

        today_date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        last_bar_date = df.index[-1].date()
        is_premarket = last_bar_date != today_date

        if is_premarket and len(df) >= 1:
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

        # Live price
        live_price = None
        if get_actual_current_price_trill is not None:
            try:
                live_price = get_actual_current_price_trill(ticker)
            except Exception:
                pass
        if live_price is None:
            try:
                live_price = _pq_module.get_actual_current_price(ticker, date)
            except Exception:
                pass
        reference_price = live_price if live_price is not None else prior_close

        if prior_close is not None:
            if prior_ema9 and prior_ema9 > 0:
                metrics['pct_from_9ema'] = (reference_price - prior_ema9) / prior_ema9
            if reference_price and prior_close > 0:
                metrics['gap_pct'] = (reference_price - prior_close) / prior_close
            prior_range = prior_high - prior_low
            atr = _pq_module.get_atr(ticker, date)
            if atr and atr > 0:
                metrics['prior_day_range_atr'] = prior_range / atr
            if len(df) >= 4:
                if is_premarket:
                    close_now = df['close'].iloc[-1]
                    close_3ago = df['close'].iloc[-4] if len(df) >= 4 else df['close'].iloc[0]
                else:
                    close_now = df['close'].iloc[-2] if len(df) >= 2 else df['close'].iloc[-1]
                    close_3ago = df['close'].iloc[-5] if len(df) >= 5 else df['close'].iloc[0]
                if close_3ago and close_3ago > 0:
                    metrics['pct_change_3'] = (close_now - close_3ago) / close_3ago
            if len(df) >= 20:
                avg_vol = df['volume'].iloc[-21:-1].mean()
                if avg_vol and avg_vol > 0:
                    if is_premarket:
                        prior_day_vol = df['volume'].iloc[-1]
                    else:
                        prior_day_vol = df['volume'].iloc[-2] if len(df) >= 2 else 0
                    metrics['prior_day_rvol'] = prior_day_vol / avg_vol if prior_day_vol else 0
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
    """Score a reversal setup using 5 pre-trade criteria. Returns dict with score, recommendation, criteria."""
    if cap is None:
        cap = get_ticker_cap(ticker)
    if cap not in PRETRADE_THRESHOLDS:
        cap = 'Medium'

    thresholds = PRETRADE_THRESHOLDS[cap]
    criteria = []
    score = 0

    criteria_checks = [
        ('9EMA Distance', 'pct_from_9ema', thresholds['pct_from_9ema']),
        ('Range (ATR)', 'prior_day_range_atr', thresholds['prior_day_range_atr']),
        ('3-Day Run-Up', 'pct_change_3', thresholds['pct_change_3']),
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
            'name': name, 'key': key, 'passed': passed,
            'threshold': threshold, 'actual': actual,
        })

    # Vol signal
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
        'name': 'Vol Signal', 'key': 'vol_signal', 'passed': vol_passed,
        'threshold': prior_thresh, 'actual': prior_rvol,
        'display_actual': f"Prior: {prior_str} | PM: {pm_str}",
        'display_threshold': f"{prior_thresh:.1f}x / {pm_thresh:.2f}x",
    })

    if score >= 4:
        recommendation = 'GO'
    elif score == 3:
        recommendation = 'CAUTION'
    else:
        recommendation = 'NO-GO'

    return {
        'ticker': ticker, 'cap': cap, 'score': score,
        'max_score': 5, 'recommendation': recommendation, 'criteria': criteria,
    }


# Historical performance stats
SCORE_STATISTICS = {
    5: {'trades': 24, 'win_rate': 100.0, 'avg_pnl': 15.5},
    4: {'trades': 14, 'win_rate': 92.9, 'avg_pnl': 14.6},
    3: {'trades': 8, 'win_rate': 87.5, 'avg_pnl': 14.6},
    2: {'trades': 4, 'win_rate': 50.0, 'avg_pnl': 1.1},
    1: {'trades': 0, 'win_rate': 0.0, 'avg_pnl': 0.0},
    0: {'trades': 0, 'win_rate': 0.0, 'avg_pnl': 0.0},
}

BOUNCE_SCORE_STATISTICS = {
    'GO': {'trades': 45, 'win_rate': 95.6, 'avg_pnl': 14.8},
    'CAUTION': {'trades': 19, 'win_rate': 100.0, 'avg_pnl': 9.3},
    'NO-GO': {'trades': 20, 'win_rate': 70.0, 'avg_pnl': 2.5},
}


def get_exit_target_data(ticker: str, date: str, prefer_open: bool = False) -> Dict:
    """Fetch data needed for exit target calculations."""
    data = {}
    try:
        df = _pq_module.get_levels_data(ticker, date, 10, 1, 'day')
        if df is None or len(df) < 2:
            return data

        df['ema_4'] = df['close'].ewm(span=4, adjust=False).mean()

        has_today_bar = False
        try:
            today_date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
            last_bar_date = df.index[-1].date()
            has_today_bar = (last_bar_date == today_date)
        except Exception:
            pass

        prior_idx = -2 if has_today_bar and len(df) >= 2 else -1
        data['prior_close'] = float(df['close'].iloc[prior_idx])
        data['prior_low'] = float(df['low'].iloc[prior_idx])
        data['prior_high'] = float(df['high'].iloc[prior_idx])
        data['ema_4'] = float(df['ema_4'].iloc[prior_idx])
        try:
            data['prior_bar_date'] = df.index[prior_idx].date().strftime('%Y-%m-%d')
        except Exception:
            data['prior_bar_date'] = None

        ref_price = None
        ref_source = None

        if prefer_open:
            open_candidate = None
            try:
                daily = _pq_module.get_daily(ticker, date)
                open_candidate = getattr(daily, 'open', None) if daily else None
            except Exception:
                pass
            if (open_candidate is None or (isinstance(open_candidate, float) and pd.isna(open_candidate))) and has_today_bar:
                try:
                    open_candidate = df['open'].iloc[-1]
                except Exception:
                    pass
            try:
                if open_candidate is not None and not (isinstance(open_candidate, float) and pd.isna(open_candidate)) and float(open_candidate) > 0:
                    ref_price = float(open_candidate)
                    ref_source = 'open'
            except Exception:
                pass

        if ref_price is None:
            if get_actual_current_price_trill is not None:
                try:
                    ref_price = get_actual_current_price_trill(ticker)
                    ref_source = 'live'
                except Exception:
                    pass
            if ref_price is None:
                try:
                    ref_price = _pq_module.get_actual_current_price(ticker, date)
                    ref_source = 'live'
                except Exception:
                    if data.get('prior_close') is not None:
                        ref_price = data['prior_close']
                        ref_source = 'prior_close'

        if ref_price is not None:
            data['open_price'] = ref_price
            data['open_price_source'] = ref_source

        atr_date = data.get('prior_bar_date') or date
        atr = _pq_module.get_atr(ticker, atr_date)
        if atr:
            data['atr'] = atr

    except Exception as e:
        logging.warning(f"Error getting exit target data for {ticker}: {e}")

    return data


# ---------------------------------------------------------------------------
# Routing: Bounce vs Reversal
# ---------------------------------------------------------------------------
ROUTING_MA_KEYS = ("pct_from_10mav", "pct_from_20mav", "pct_from_50mav", "pct_from_200mav")


def _safe_num(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(v) else v


def route_playbook(pct_data_in: Dict, mav_data_in: Dict) -> Tuple[str, str]:
    """Return (bucket, reason) where bucket is 'reversal' | 'bounce'."""
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


# ---------------------------------------------------------------------------
# Percentile comparison columns
# ---------------------------------------------------------------------------
BOUNCE_COLUMNS_TO_COMPARE = [
    'pct_change_120', 'pct_change_90', 'pct_change_30', 'pct_change_15', 'pct_change_3',
    'pct_from_9ema', 'pct_from_10mav', 'pct_from_20mav', 'pct_from_50mav', 'pct_from_200mav',
]

COLUMNS_TO_COMPARE = ss.columns_to_compare

PERCENTILE_ORDER = [
    "pct_change_120", "pct_change_90", "pct_change_30", "pct_change_15", "pct_change_3",
    "pct_from_10mav", "pct_from_20mav", "pct_from_50mav", "pct_from_200mav",
]


# ---------------------------------------------------------------------------
# TickerReportData — structured output (replaces HTML generation)
# ---------------------------------------------------------------------------
@dataclass
class ExitTargetData:
    """Computed exit target prices for a ticker."""
    entry_price: Optional[float] = None
    entry_source: Optional[str] = None  # 'open' | 'live' | 'prior_close'
    atr: Optional[float] = None
    prior_close: Optional[float] = None
    prior_high: Optional[float] = None
    prior_low: Optional[float] = None
    ema_4: Optional[float] = None
    targets: Optional[Dict] = None  # output of calculate_exit_targets / calculate_bounce_exit_targets


@dataclass
class TickerReportData:
    """All computed data for one ticker in the report."""
    ticker: str
    bucket: str  # 'bounce' | 'reversal'
    bucket_reason: str
    cap: Optional[str] = None

    # Scoring
    score_result: Optional[Dict] = None          # reversal score dict
    bounce_result: Optional[Any] = None           # ChecklistResult from BouncePretrade
    bounce_metrics: Optional[Dict] = None         # raw bounce metrics
    bounce_setup_type: Optional[str] = None       # 'GapFade_weakstock' | 'GapFade_strongstock'
    bounce_intensity: Optional[Dict] = None       # {composite, details}

    # Exit targets
    exit_targets: Optional[ExitTargetData] = field(default_factory=lambda: None)

    # Percentiles
    percentiles: Optional[Dict[str, float]] = None
    percentile_ref_label: Optional[str] = None    # e.g. "weakstock, n=36"

    # MA distances
    mav_data: Optional[Dict[str, float]] = None

    # Raw data
    pct_data: Optional[Dict] = None
    range_data: Optional[Dict] = None

    # Chart overlay lines: [(price, color, label), ...]
    chart_hlines: List[Tuple[float, str, str]] = field(default_factory=list)


def _build_ticker_report(
    ticker: str,
    data: dict,
    pretrade_metrics: dict = None,
    bounce_metrics: dict = None,
) -> TickerReportData:
    """Build structured TickerReportData for one ticker."""
    from data_collectors.combined_data_collection import reversal_df

    pct_data = data.get("pct_data", {})
    range_data = data.get("range_data", {})
    mav_data = data.get("mav_data", {})

    bucket, bucket_reason = route_playbook(pct_data, mav_data)

    report = TickerReportData(
        ticker=ticker,
        bucket=bucket,
        bucket_reason=bucket_reason,
        pct_data=pct_data,
        range_data=range_data,
        mav_data=mav_data,
    )

    cap = None
    today = datetime.datetime.now().strftime('%Y-%m-%d')

    if bucket == "reversal":
        if pretrade_metrics:
            score_result = score_pretrade_setup(ticker, pretrade_metrics)
            cap = score_result.get('cap')
            report.score_result = score_result
            report.cap = cap

        exit_data = get_exit_target_data(ticker, today)
        if exit_data.get('open_price') and exit_data.get('atr'):
            if cap is None:
                cap = get_ticker_cap(ticker)
                report.cap = cap
            targets = calculate_exit_targets(
                cap=cap,
                entry_price=exit_data['open_price'],
                atr=exit_data['atr'],
                prior_close=exit_data.get('prior_close'),
                prior_low=exit_data.get('prior_low'),
                ema_4=exit_data.get('ema_4'),
            )
            report.exit_targets = ExitTargetData(
                entry_price=exit_data['open_price'],
                entry_source=exit_data.get('open_price_source'),
                atr=exit_data['atr'],
                prior_close=exit_data.get('prior_close'),
                prior_high=exit_data.get('prior_high'),
                prior_low=exit_data.get('prior_low'),
                ema_4=exit_data.get('ema_4'),
                targets=targets,
            )

            # Chart overlay lines for GO reversals
            if report.score_result and report.score_result.get('recommendation') == 'GO':
                op = exit_data['open_price']
                a = exit_data['atr']
                report.chart_hlines = [
                    (op,            'blue',    f'Open ${op:.2f}'),
                    (op - 1.0 * a, 'orange',  f'-1 ATR ${op - 1.0*a:.2f}'),
                    (op - 2.0 * a, 'red',     f'-2 ATR ${op - 2.0*a:.2f}'),
                    (op - 3.0 * a, 'darkred', f'-3 ATR ${op - 3.0*a:.2f}'),
                ]
                pc = exit_data.get('prior_close')
                if pc and pc < op:
                    report.chart_hlines.append((pc, 'green', f'Prior Close ${pc:.2f}'))

        # Reversal percentiles
        pcts = ss.calculate_percentiles(reversal_df, data, COLUMNS_TO_COMPARE)
        report.percentiles = pcts
        report.percentile_ref_label = "reversal"

    elif bucket == "bounce":
        if cap is None:
            cap = get_ticker_cap(ticker)
            report.cap = cap

        # Classify setup type
        try:
            bounce_setup_type, _ = classify_stock({
                "pct_from_200mav": mav_data.get("pct_from_200mav"),
                "pct_change_30": pct_data.get("pct_change_30"),
            })
        except Exception:
            bounce_setup_type = "GapFade_strongstock"
        report.bounce_setup_type = bounce_setup_type

        if bounce_metrics:
            # Supplement with screener data
            if bounce_metrics.get('pct_from_200mav') is None and mav_data.get('pct_from_200mav') is not None:
                bounce_metrics['pct_from_200mav'] = mav_data['pct_from_200mav']
            volume_data = data.get("volume_data", {})
            if bounce_metrics.get('percent_of_vol_on_breakout_day') is None and volume_data.get('percent_of_vol_on_breakout_day') is not None:
                bounce_metrics['percent_of_vol_on_breakout_day'] = volume_data['percent_of_vol_on_breakout_day']

            checker = BouncePretrade()
            bounce_result = checker.validate(ticker, bounce_metrics, cap=cap)
            report.bounce_result = bounce_result
            report.bounce_metrics = bounce_metrics
            report.bounce_setup_type = bounce_result.setup_type

            # Intensity score
            bounce_ref = BOUNCE_DF_WEAK if bounce_result.setup_type == 'GapFade_weakstock' else BOUNCE_DF_STRONG
            report.bounce_intensity = compute_bounce_intensity(bounce_metrics, ref_df=bounce_ref)

        # Bounce exit targets
        exit_data = get_exit_target_data(ticker, today, prefer_open=True)
        if exit_data.get('open_price') and exit_data.get('atr'):
            bounce_targets = calculate_bounce_exit_targets(
                cap=cap,
                entry_price=exit_data['open_price'],
                atr=exit_data['atr'],
                prior_close=exit_data.get('prior_close'),
                prior_high=exit_data.get('prior_high'),
            )
            bounce_targets['entry_price_source'] = exit_data.get('open_price_source')
            report.exit_targets = ExitTargetData(
                entry_price=exit_data['open_price'],
                entry_source=exit_data.get('open_price_source'),
                atr=exit_data['atr'],
                prior_close=exit_data.get('prior_close'),
                prior_high=exit_data.get('prior_high'),
                targets=bounce_targets,
            )

        # Bounce percentiles (inverted: more negative = better)
        bounce_ref_df = BOUNCE_DF_WEAK if bounce_setup_type == 'GapFade_weakstock' else BOUNCE_DF_STRONG
        pcts = ss.calculate_percentiles(bounce_ref_df, data, BOUNCE_COLUMNS_TO_COMPARE)
        pcts = {k: round(100 - v, 1) for k, v in pcts.items()}
        setup_label = bounce_setup_type.replace('GapFade_', '')
        report.percentiles = pcts
        report.percentile_ref_label = f"bounce {setup_label}, n={len(bounce_ref_df)}"

    return report


# ---------------------------------------------------------------------------
# Main entry point: compute full report
# ---------------------------------------------------------------------------
_MAX_WORKERS = 8


def compute_report(
    watchlist: List[str],
    date: str = None,
    progress_callback=None,
) -> List[TickerReportData]:
    """
    Compute the full watchlist report and return structured data.

    Args:
        watchlist: List of ticker symbols
        date: Date string (YYYY-MM-DD), defaults to today
        progress_callback: Optional callable(current, total, message) for progress updates

    Returns:
        List of TickerReportData sorted by: bounce (by intensity desc) then reversal (by percentile desc)
    """
    if date is None:
        date = datetime.datetime.now().strftime('%Y-%m-%d')

    def _progress(current, total, msg):
        if progress_callback:
            progress_callback(current, total, msg)

    cache = ReportCache()
    cache.install()

    try:
        # Phase 1: Collect screener metrics
        _progress(0, 5, "Fetching screener data...")
        all_data = ss.get_all_stocks_data(watchlist)

        # Phase 2: Collect pre-trade reversal metrics (parallel)
        _progress(1, 5, "Fetching reversal pre-trade metrics...")
        pretrade_metrics_all = {}
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = {executor.submit(get_pretrade_metrics, t, date): t for t in watchlist}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    pretrade_metrics_all[ticker] = future.result()
                except Exception:
                    pretrade_metrics_all[ticker] = {}

        # Route tickers
        _progress(2, 5, "Routing tickers (bounce vs reversal)...")
        bucket_map: Dict[str, str] = {}
        for ticker in watchlist:
            ticker_data = all_data.get(ticker, {})
            bucket, _ = route_playbook(
                ticker_data.get("pct_data", {}) or {},
                ticker_data.get("mav_data", {}) or {},
            )
            bucket_map[ticker] = bucket

        # Phase 3: Collect bounce metrics (parallel)
        _progress(3, 5, "Fetching bounce pre-trade metrics...")
        bounce_tickers_to_fetch = [t for t in watchlist if bucket_map.get(t) == "bounce"]
        bounce_metrics_all = {}
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = {executor.submit(fetch_bounce_metrics, t, date): t for t in bounce_tickers_to_fetch}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    bounce_metrics_all[ticker] = future.result()
                except Exception:
                    pass

        # Phase 4: Build report data (parallel)
        _progress(4, 5, "Computing scores and targets...")
        ticker_reports: Dict[str, TickerReportData] = {}
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    _build_ticker_report, ticker, all_data.get(ticker, {}),
                    pretrade_metrics_all.get(ticker, {}),
                    bounce_metrics_all.get(ticker),
                ): ticker
                for ticker in watchlist
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    ticker_reports[ticker] = future.result()
                except Exception as e:
                    logging.warning(f"Error building report for {ticker}: {e}")
                    ticker_reports[ticker] = TickerReportData(
                        ticker=ticker, bucket="bounce",
                        bucket_reason=f"Error: {e}",
                    )

        # Sort: bounce by intensity desc, then reversal by percentile desc
        bounce_tickers = []
        other_tickers = []
        for ticker in watchlist:
            r = ticker_reports.get(ticker)
            if r and r.bucket == "bounce":
                bounce_tickers.append(ticker)
            else:
                other_tickers.append(ticker)

        bounce_tickers.sort(
            key=lambda t: (ticker_reports[t].bounce_intensity or {}).get('composite', 0),
            reverse=True,
        )
        other_tickers.sort(
            key=lambda t: (ticker_reports[t].percentiles or {}).get('pct_change_15', 0),
            reverse=True,
        )

        sorted_list = [ticker_reports[t] for t in bounce_tickers + other_tickers]

        _progress(5, 5, "Done!")
        return sorted_list

    finally:
        cache.uninstall()
