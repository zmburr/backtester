"""Unified local metric computation from raw daily OHLCV bars.

Replaces 15+ per-ticker Polygon API calls in ``stock_screener`` with pure
pandas math on the DataFrame already fetched by ``TickerCache``.

Each public function returns the **exact same dict structure** as the
original API-based code so downstream display / percentile logic is
unchanged.
"""

from typing import Dict, Optional

import pandas as pd
import numpy as np

from data_queries.ticker_cache import is_today_finalized


# ======================================================================
# Public API
# ======================================================================

def compute_screener_metrics(
    daily_bars: pd.DataFrame,
    date: str,
    current_price: float = None,
    intraday: pd.DataFrame = None,
) -> Dict:
    """Compute every metric ``stock_screener.get_stock_data`` returns.

    Args:
        daily_bars: Daily OHLCV DataFrame from ``TickerCache.get_daily_bars``
                    (indexed by tz-aware timestamp).
        date: Reference date in ``YYYY-MM-DD`` format.
        current_price: Live / most-recent price.  When *None*, the latest
                       bar's close is used.
        intraday: Optional 1-minute intraday DataFrame for the same ticker
                  and date.  When provided, premarket and first-N-minute
                  volume metrics are computed from it.

    Returns:
        ``{pct_data, volume_data, range_data, mav_data}`` — identical in
        structure to the dict returned by the original ``get_stock_data``.
    """
    empty = {'pct_data': {}, 'volume_data': {}, 'range_data': {}, 'mav_data': {}}
    if daily_bars is None or daily_bars.empty:
        return empty

    today_date = pd.to_datetime(date).date()

    # Detect whether today's (possibly partial) bar is present.
    has_today_bar = False
    try:
        last_bar_date = daily_bars.index[-1].date()
        has_today_bar = (last_bar_date == today_date)
    except Exception:
        pass

    # Completed bars only — exclude today's bar when the session is still
    # open (partial data).  When the market has closed, today's bar is
    # finalized and should be included in MA / range / volume calculations.
    today_complete = is_today_finalized(today_date)
    if has_today_bar and not today_complete and len(daily_bars) > 1:
        hist = daily_bars.iloc[:-1]
    else:
        hist = daily_bars
    if len(hist) < 2:
        return empty

    # Determine reference prices.
    if current_price is None:
        current_price = daily_bars.iloc[-1]['close']

    # For MA extension: original code uses HIGH of day when today's bar
    # exists, prior close otherwise.  If we have intraday data we can
    # derive high-of-day; otherwise fall back to current_price.
    if has_today_bar:
        mav_ref = daily_bars.iloc[-1]['high']
    elif intraday is not None and not intraday.empty:
        mav_ref = intraday['high'].max()
    else:
        mav_ref = current_price  # best available

    pct_data = _compute_pct_data(daily_bars, hist, date, current_price)
    mav_data = _compute_mav_data(hist, mav_ref)
    range_data = _compute_range_data(daily_bars, has_today_bar)
    volume_data = _compute_volume_data(daily_bars, hist, has_today_bar, intraday)

    return {
        'pct_data': pct_data,
        'volume_data': volume_data,
        'range_data': range_data,
        'mav_data': mav_data,
    }


# ======================================================================
# Private helpers
# ======================================================================

def _compute_pct_data(
    daily_bars: pd.DataFrame,
    hist: pd.DataFrame,
    date: str,
    current_price: float,
) -> Dict:
    """Replaces ``get_ticker_pct_move`` (5 API calls → 0).

    Uses *calendar-day* lookback to match the original
    ``adjust_date_to_market(date, N)`` behaviour.
    """
    result: Dict[str, Optional[float]] = {}

    for days, key in [
        (120, 'pct_change_120'),
        (90, 'pct_change_90'),
        (30, 'pct_change_30'),
        (15, 'pct_change_15'),
        (3, 'pct_change_3'),
    ]:
        old_close = _close_n_calendar_days_ago(daily_bars, date, days)
        if old_close is not None and old_close != 0:
            result[key] = (current_price - old_close) / old_close
        else:
            result[key] = None

    return result


def _close_n_calendar_days_ago(
    daily_bars: pd.DataFrame, date: str, days: int,
) -> Optional[float]:
    """Find the close price of the bar closest to *date - days* calendar days.

    Mirrors ``adjust_date_to_market(date, days)`` + ``get_daily().close``.
    """
    target_date = (pd.to_datetime(date) - pd.Timedelta(days=days)).date()
    bar_dates = daily_bars.index.date  # numpy array of datetime.date
    mask = bar_dates >= target_date
    if not mask.any():
        return None
    first_idx = np.argmax(mask)
    return float(daily_bars.iloc[first_idx]['close'])


def _compute_mav_data(hist: pd.DataFrame, reference_price: float) -> Dict:
    """Replaces ``get_ticker_mavs_open`` (7+ API calls → 0).

    Computes EMA-9, SMA-10/20/50/200 and ATR distance from SMA-50
    locally from completed daily bars.
    """
    result: Dict[str, Optional[float]] = {}
    closes = hist['close']

    # 9-day EMA
    if len(closes) >= 9:
        ema_9 = closes.ewm(span=9, adjust=False).mean().iloc[-1]
        if ema_9 and ema_9 != 0:
            result['pct_from_9ema'] = (reference_price - ema_9) / ema_9

    # SMAs
    for window, key in [
        (10, 'pct_from_10mav'),
        (20, 'pct_from_20mav'),
        (50, 'pct_from_50mav'),
        (200, 'pct_from_200mav'),
    ]:
        if len(closes) >= window:
            sma = closes.rolling(window).mean().iloc[-1]
            if sma and sma != 0 and not pd.isna(sma):
                result[key] = (reference_price - sma) / sma

    # ATR distance from 50-day SMA
    if 'pct_from_50mav' in result and len(hist) >= 2:
        atr = _compute_atr(hist)
        sma_50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else None
        if atr and atr > 0 and sma_50 and not pd.isna(sma_50):
            result['atr_distance_from_50mav'] = (reference_price - sma_50) / atr

    return result


def _compute_atr(hist: pd.DataFrame, window: int = 14) -> Optional[float]:
    """Compute ATR from completed daily bars."""
    if len(hist) < 2:
        return None
    hl = hist['high'] - hist['low']
    hpc = abs(hist['high'] - hist['close'].shift(1))
    lpc = abs(hist['low'] - hist['close'].shift(1))
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    atr_window = min(window, max(1, len(tr)))
    atr = tr.rolling(window=atr_window, min_periods=1).mean().iloc[-1]
    return float(atr) if atr and not pd.isna(atr) else None


def _compute_range_data(
    daily_bars: pd.DataFrame, has_today_bar: bool,
) -> Dict:
    """Replaces ``add_range_data`` (1 API call → 0).

    Returns formatted strings like ``"(1,234 / 5,678) = 1.23"`` to match
    the display format used by stock_screener.
    """
    result: Dict[str, str] = {}
    if daily_bars is None or daily_bars.empty or len(daily_bars) < 2:
        return result

    df = daily_bars.copy()

    # True Range
    df['high-low'] = df['high'] - df['low']
    df['high-previous_close'] = abs(df['high'] - df['close'].shift())
    df['low-previous_close'] = abs(df['low'] - df['close'].shift())
    df['TR'] = df[['high-low', 'high-previous_close', 'low-previous_close']].max(axis=1)

    atr_window = 14 if len(df) >= 14 else max(1, len(df))
    df['ATR'] = df['TR'].rolling(window=atr_window, min_periods=1).mean()

    adv_window = 20 if len(df) >= 20 else max(1, len(df))
    df['20Day_Avg_Volume'] = df['volume'].rolling(window=adv_window, min_periods=1).mean()

    # Get dates by recency
    available_dates = df.index.sort_values(ascending=False)

    def nth(n):
        return available_dates[n] if len(available_dates) > n else None

    d0 = nth(0)
    d1 = nth(1)
    d2 = nth(2)
    d3 = nth(3)

    def _safe(series, idx):
        try:
            return series.loc[idx] if idx is not None and idx in series.index else None
        except (KeyError, TypeError):
            return None

    def _pct_str(num, den):
        if num is None or den in (None, 0):
            return "N/A"
        pct = num / den
        return f"({num:,.0f} / {den:,.0f}) = {pct:.2f}"

    # Volume comparisons
    result['day_of_vol_pct'] = _pct_str(
        _safe(df['volume'], d0), _safe(df['20Day_Avg_Volume'], d0))
    result['percent_of_vol_one_day_before'] = _pct_str(
        _safe(df['volume'], d1), _safe(df['20Day_Avg_Volume'], d1))
    result['percent_of_vol_two_day_before'] = _pct_str(
        _safe(df['volume'], d2), _safe(df['20Day_Avg_Volume'], d2))
    result['percent_of_vol_three_day_before'] = _pct_str(
        _safe(df['volume'], d3), _safe(df['20Day_Avg_Volume'], d3))

    # Range comparisons (TR / ATR)
    result['day_of_range_pct'] = _pct_str(
        _safe(df['TR'], d0), _safe(df['ATR'], d0))
    result['one_day_before_range_pct'] = _pct_str(
        _safe(df['TR'], d1), _safe(df['ATR'], d1))
    result['two_day_before_range_pct'] = _pct_str(
        _safe(df['TR'], d2), _safe(df['ATR'], d2))
    result['three_day_before_range_pct'] = _pct_str(
        _safe(df['TR'], d3), _safe(df['ATR'], d3))

    return result


def _compute_volume_data(
    daily_bars: pd.DataFrame,
    hist: pd.DataFrame,
    has_today_bar: bool,
    intraday: pd.DataFrame = None,
) -> Dict:
    """Replaces ``fetch_and_calculate_volumes`` + ``add_percent_of_adv_columns``
    (2 API calls → 0 for daily portion, intraday still required separately).
    """
    result: Dict = {}
    if hist is None or hist.empty:
        return result

    # Average daily volume (20-day rolling or whatever is available)
    adv_window = min(20, len(hist))
    avg_daily_vol = hist['volume'].tail(adv_window).mean()
    result['avg_daily_vol'] = avg_daily_vol

    # Prior-day volumes — use offsets relative to the full daily_bars
    # to match original fetch_and_calculate_volumes indexing.
    n = len(daily_bars)
    result['vol_one_day_before'] = float(daily_bars.iloc[-2]['volume']) if n >= 2 else None
    result['vol_two_day_before'] = float(daily_bars.iloc[-3]['volume']) if n >= 3 else None
    result['vol_three_day_before'] = float(daily_bars.iloc[-4]['volume']) if n >= 4 else None

    # Intraday metrics (require minute-level data)
    if intraday is not None and not intraday.empty:
        result['vol_on_breakout_day'] = intraday['volume'].sum()
        result['premarket_vol'] = intraday.between_time('06:00:00', '09:30:00')['volume'].sum()
        result['vol_in_first_5_min'] = intraday.between_time('09:30:00', '09:35:00')['volume'].sum()
        result['vol_in_first_10_min'] = intraday.between_time('09:30:00', '09:40:00')['volume'].sum()
        result['vol_in_first_15_min'] = intraday.between_time('09:30:00', '09:45:00')['volume'].sum()
        result['vol_in_first_30_min'] = intraday.between_time('09:30:00', '10:00:00')['volume'].sum()
    else:
        result['vol_on_breakout_day'] = None
        result['premarket_vol'] = None
        result['vol_in_first_5_min'] = None
        result['vol_in_first_10_min'] = None
        result['vol_in_first_15_min'] = None
        result['vol_in_first_30_min'] = None

    # Percent-of-ADV columns (mirrors add_percent_of_adv_columns)
    _add_percent_of_adv(result)

    return result


def _add_percent_of_adv(volume_data: Dict) -> None:
    """Add percent_of_* columns in-place (same as stock_screener.add_percent_of_adv_columns)."""
    adv = volume_data.get('avg_daily_vol', 0) or 0
    for col in [
        'premarket_vol', 'vol_in_first_5_min', 'vol_in_first_15_min',
        'vol_in_first_10_min', 'vol_in_first_30_min', 'vol_on_breakout_day',
    ]:
        pct_key = f'percent_of_{col}'
        val = volume_data.get(col)
        if val is not None and adv > 0:
            volume_data[pct_key] = val / adv
        else:
            volume_data[pct_key] = None
