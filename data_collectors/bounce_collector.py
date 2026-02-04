"""
Bounce data collector — fills bounce_data.csv with computed features.
Follows the same pattern as combined_data_collection.py.
"""
from types import SimpleNamespace
from data_queries.polygon_queries import (
    get_daily, adjust_date_forward, get_levels_data,
    adjust_date_to_market, get_intraday, check_pct_move
)
import data_queries.trillium_queries as trlm
from data_collectors.combined_data_collection import (
    get_volume, get_pct_volume, get_pct_from_mavs, get_range_vol_expansion,
    get_spy, get_market_context, get_volume_profile
)
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

bounce_df = pd.read_csv("C:\\Users\\zmbur\\PycharmProjects\\backtester\\data\\bounce_data.csv")
bounce_df = bounce_df.dropna(subset=['ticker'])
bounce_df = bounce_df.dropna(subset=['date'])
bounce_df['ticker'] = bounce_df['ticker'].str.strip()


# ---------------------------------------------------------------------------
# Helper: parse date from row
# ---------------------------------------------------------------------------
def _parse_date(row):
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    return datetime.strftime(wrong_date, '%Y-%m-%d')


# ---------------------------------------------------------------------------
# Trillium routing for Canadian (.T) tickers
# ---------------------------------------------------------------------------
def _is_canadian(ticker):
    """Check if ticker is a Canadian .T ticker."""
    return ticker.upper().rstrip().endswith('.T')


def _get_daily(ticker, date):
    """Get daily OHLC, routing to trillium for .T tickers."""
    if _is_canadian(ticker):
        try:
            result = trlm.get_daily(ticker, date)
            return SimpleNamespace(**result)
        except Exception as e:
            logging.error(f"Trillium get_daily error for {ticker} on {date}: {e}")
            return None
    else:
        return get_daily(ticker, date)


def _get_intraday_1min(ticker, date):
    """Get 1-min intraday bars, routing to trillium for .T tickers."""
    if _is_canadian(ticker):
        return trlm.get_intraday(ticker, date, 'bar-1min')
    else:
        return get_intraday(ticker, date, multiplier=1, timespan='minute')


def _get_levels_daily(ticker, date, window):
    """Get daily historical bars, routing to trillium for .T tickers.

    Note: polygon treats `window` as calendar days, but trillium's
    get_levels_data treats it as trading days. To keep behaviour consistent
    we compute the start date with polygon's calendar logic and call
    trillium's _shel_request directly.
    """
    if _is_canadian(ticker):
        import pytz
        end_date = date if isinstance(date, str) else date.strftime('%Y-%m-%d')
        start_date = adjust_date_to_market(end_date, window)
        df = trlm._shel_request(ticker, start_date, end_date, ['bar-1day'])
        if "close-time" in df.columns:
            df["close-time"] = (
                pd.to_datetime(df["close-time"], unit="ns")
                  .dt.tz_localize("UTC")
                  .dt.tz_convert(pytz.timezone("US/Eastern"))
            )
            df.set_index("close-time", inplace=True)
        return df
    else:
        return get_levels_data(ticker, date, window, 1, 'day')


# ---------------------------------------------------------------------------
# 1. Bounce day stats
# ---------------------------------------------------------------------------
def get_bounce_stats(row, analysis_type):
    ticker = row['ticker']
    date = _parse_date(row)
    logging.info(f'Running get_bounce_stats for {ticker} on {date}')

    try:
        daily_data = _get_daily(ticker, date)
        open_price = daily_data.open
        close_price = daily_data.close
        high_price = daily_data.high
        low_price = daily_data.low

        day_before_data = _get_daily(ticker, adjust_date_to_market(date, 1))
        prev_close = day_before_data.close

        day_after_data = _get_daily(ticker, adjust_date_forward(date, 1))
        day_after_open = day_after_data.open

        row['gap_pct'] = (open_price - prev_close) / prev_close
        row['bounce_open_high_pct'] = (high_price - open_price) / open_price
        row['bounce_open_close_pct'] = (close_price - open_price) / open_price
        row['bounce_open_low_pct'] = (low_price - open_price) / open_price
        row['bounce_open_to_day_after_open_pct'] = (day_after_open - open_price) / open_price

    except Exception as e:
        logging.error(f"Error in get_bounce_stats for {ticker} on {date}: {e}")

    return row


# ---------------------------------------------------------------------------
# 2. Bounce conditionals
# ---------------------------------------------------------------------------
def get_bounce_conditionals(row, analysis_type):
    ticker = row['ticker']
    date = _parse_date(row)
    logging.info(f'Running get_bounce_conditionals for {ticker} on {date}')

    try:
        daily_data = _get_daily(ticker, date)
        open_price = daily_data.open
        close_price = daily_data.close
        high_price = daily_data.high
        low_price = daily_data.low

        day_before_data = _get_daily(ticker, adjust_date_to_market(date, 1))
        prev_close = day_before_data.close
        prev_low = day_before_data.low

        # 52-week low
        hist = _get_levels_daily(ticker, adjust_date_to_market(date, 1), 365)
        fifty_two_week_low = hist['low'].min() if hist is not None and not hist.empty else None

        if fifty_two_week_low is not None:
            row['near_52wk_low'] = low_price <= fifty_two_week_low * 1.05
            row['breaks_52wk_low'] = low_price < fifty_two_week_low

        row['close_at_highs'] = abs(close_price - high_price) / high_price <= 0.02
        row['close_above_prior_close'] = close_price > prev_close
        row['close_green_red'] = close_price > open_price
        row['hit_prior_day_hilo'] = low_price < prev_low

    except Exception as e:
        logging.error(f"Error in get_bounce_conditionals for {ticker} on {date}: {e}")

    return row


# ---------------------------------------------------------------------------
# 3. Bollinger bands (both upper AND lower for bounce context)
# ---------------------------------------------------------------------------
def get_bounce_bollinger(row, analysis_type):
    ticker = row['ticker']
    date = _parse_date(row)
    logging.info(f'Running get_bounce_bollinger for {ticker} on {date}')

    try:
        df = _get_levels_daily(ticker, date, 35)
        if df is None or len(df) < 21:
            logging.warning(f'Insufficient data for Bollinger Bands: {ticker} on {date}')
            return row

        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['sma_20'] + (2 * df['std_20'])
        df['lower_band'] = df['sma_20'] - (2 * df['std_20'])

        prior_close = df['close'].iloc[-2]
        upper_band = df['upper_band'].iloc[-2]
        lower_band = df['lower_band'].iloc[-2]
        middle_band = df['sma_20'].iloc[-2]

        if pd.isna(upper_band) or pd.isna(lower_band) or pd.isna(middle_band):
            logging.warning(f'NaN Bollinger values for {ticker} on {date}')
            return row

        row['lower_band_distance'] = (prior_close - lower_band) / lower_band
        row['closed_outside_lower_band'] = prior_close < lower_band
        row['upper_band_distance'] = (prior_close - upper_band) / upper_band
        row['bollinger_width'] = (upper_band - lower_band) / middle_band
        row['bb_position'] = (prior_close - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) != 0 else None

        # Days since upper BB: count backwards from bounce date to last close > upper band
        days_since_upper = None
        for i in range(len(df) - 2, 0, -1):
            if not pd.isna(df['upper_band'].iloc[i]) and df['close'].iloc[i] > df['upper_band'].iloc[i]:
                days_since_upper = (len(df) - 2) - i
                break
        row['days_since_upper_bb'] = days_since_upper

        # Upper BB to lower BB days: days from last upper BB touch to first lower BB touch
        last_upper_idx = None
        first_lower_idx = None
        for i in range(len(df) - 2, 0, -1):
            if not pd.isna(df['upper_band'].iloc[i]) and df['close'].iloc[i] > df['upper_band'].iloc[i]:
                last_upper_idx = i
                break
        if last_upper_idx is not None:
            for i in range(last_upper_idx + 1, len(df) - 1):
                if not pd.isna(df['lower_band'].iloc[i]) and df['close'].iloc[i] < df['lower_band'].iloc[i]:
                    first_lower_idx = i
                    break
        if last_upper_idx is not None and first_lower_idx is not None:
            row['upper_bb_to_lower_bb_days'] = first_lower_idx - last_upper_idx
        else:
            row['upper_bb_to_lower_bb_days'] = None

    except Exception as e:
        logging.error(f'Error in get_bounce_bollinger for {ticker} on {date}: {e}')

    return row


# ---------------------------------------------------------------------------
# 4. Selloff context
# ---------------------------------------------------------------------------
def get_selloff_context(row, analysis_type):
    ticker = row['ticker']
    date = _parse_date(row)
    logging.info(f'Running get_selloff_context for {ticker} on {date}')

    try:
        daily_data = _get_daily(ticker, date)
        open_price = daily_data.open

        hist = _get_levels_daily(ticker, adjust_date_to_market(date, 1), 60)
        if hist is None or hist.empty:
            return row

        # Consecutive down days before bounce
        consecutive_down = 0
        for i in range(len(hist) - 1, 0, -1):
            if hist['close'].iloc[i] < hist['open'].iloc[i]:
                consecutive_down += 1
            else:
                break
        row['consecutive_down_days'] = consecutive_down

        # Pct off 30-day high
        recent_30 = hist.tail(30) if len(hist) >= 30 else hist
        high_30d = recent_30['high'].max()
        row['pct_off_30d_high'] = (open_price - high_30d) / high_30d

        # Pct off 52-week high
        hist_52wk = _get_levels_daily(ticker, adjust_date_to_market(date, 1), 365)
        if hist_52wk is not None and not hist_52wk.empty:
            high_52wk = hist_52wk['high'].max()
            row['pct_off_52wk_high'] = (open_price - high_52wk) / high_52wk

        # Selloff total pct over consecutive down days
        if consecutive_down > 0 and len(hist) > consecutive_down:
            start_idx = len(hist) - consecutive_down
            selloff_start_close = hist['close'].iloc[start_idx - 1] if start_idx > 0 else hist['open'].iloc[start_idx]
            selloff_end_close = hist['close'].iloc[-1]
            row['selloff_total_pct'] = (selloff_end_close - selloff_start_close) / selloff_start_close
        else:
            row['selloff_total_pct'] = 0

        # Prior day close vs low: (close - low) / (high - low)
        prior_high = hist['high'].iloc[-1]
        prior_low = hist['low'].iloc[-1]
        prior_close = hist['close'].iloc[-1]
        if prior_high != prior_low:
            row['prior_day_close_vs_low_pct'] = (prior_close - prior_low) / (prior_high - prior_low)
        else:
            row['prior_day_close_vs_low_pct'] = None

    except Exception as e:
        logging.error(f'Error in get_selloff_context for {ticker} on {date}: {e}')

    return row


# ---------------------------------------------------------------------------
# 5. Volume climax
# ---------------------------------------------------------------------------
def get_volume_climax(row, analysis_type):
    ticker = row['ticker']
    date = _parse_date(row)
    logging.info(f'Running get_volume_climax for {ticker} on {date}')

    try:
        hist = _get_levels_daily(ticker, date, 10)
        if hist is None or len(hist) < 4:
            return row

        # Vol trend 3d: today vol / vol 3 days ago
        today_vol = hist['volume'].iloc[-1]
        three_ago_vol = hist['volume'].iloc[-4]
        if three_ago_vol > 0:
            row['vol_trend_3d'] = today_vol / three_ago_vol

        # Vol trend direction: consecutive days where vol > prior day vol
        consecutive_vol_up = 0
        for i in range(len(hist) - 1, 0, -1):
            if hist['volume'].iloc[i] > hist['volume'].iloc[i - 1]:
                consecutive_vol_up += 1
            else:
                break
        row['vol_trend_direction'] = consecutive_vol_up

        # Down day vol ratio: avg vol on down days / avg vol on up days (5-day window)
        recent = hist.tail(5)
        down_vols = recent.loc[recent['close'] < recent['open'], 'volume']
        up_vols = recent.loc[recent['close'] >= recent['open'], 'volume']
        if not up_vols.empty and up_vols.mean() > 0:
            row['down_day_vol_ratio'] = down_vols.mean() / up_vols.mean() if not down_vols.empty else 0
        else:
            row['down_day_vol_ratio'] = None

    except Exception as e:
        logging.error(f'Error in get_volume_climax for {ticker} on {date}: {e}')

    return row


# ---------------------------------------------------------------------------
# 6. Bounce intraday timing
# ---------------------------------------------------------------------------
def get_bounce_intraday_timing(row, analysis_type):
    ticker = row['ticker']
    date = _parse_date(row)
    logging.info(f'Running get_bounce_intraday_timing for {ticker} on {date}')

    try:
        data = _get_intraday_1min(ticker, date)
        if data is None or data.empty:
            return row

        premarket_data = data.between_time('06:00:00', '09:29:59')
        regular_session = data.between_time('09:30:00', '16:00:00')

        if regular_session.empty:
            return row

        # Time of low price
        low_time = regular_session['low'].idxmin()
        row['time_of_low_price'] = low_time

        # Time of low bucket
        hour = low_time.hour
        minute = low_time.minute
        if hour == 9 or (hour == 10 and minute == 0):
            row['time_of_low_bucket'] = 1  # 9:30-10:00
        elif hour == 10:
            row['time_of_low_bucket'] = 2  # 10:00-11:00
        elif hour == 11:
            row['time_of_low_bucket'] = 3  # 11:00-12:00
        else:
            row['time_of_low_bucket'] = 4  # After 12:00

        # High to low duration (minutes from HOD to LOD)
        high_time = regular_session['high'].idxmax()
        duration_td = low_time - high_time
        row['high_to_low_duration_min'] = duration_td.total_seconds() / 60.0

        # Gap from premarket low
        if not premarket_data.empty:
            pm_low = premarket_data['low'].min()
            open_price = regular_session['open'].iloc[0]
            if pm_low > 0:
                row['gap_from_pm_low'] = (open_price - pm_low) / pm_low

    except Exception as e:
        logging.error(f'Error in get_bounce_intraday_timing for {ticker} on {date}: {e}')

    return row


# ---------------------------------------------------------------------------
# 7. ATR for bounce
# ---------------------------------------------------------------------------
def calculate_bounce_atr(row, analysis_type, period=30):
    ticker = row['ticker']
    date = _parse_date(row)
    logging.info(f'Running calculate_bounce_atr for {ticker} on {date}')

    try:
        stock_data = _get_levels_daily(ticker, adjust_date_to_market(date, 1), period)
        if stock_data is not None and not stock_data.empty:
            stock_data['tr'] = stock_data[['high', 'low', 'close']].apply(
                lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close']), abs(x['low'] - x['close'])),
                axis=1
            )
            atr = stock_data['tr'].mean() / stock_data['close'].mean()
            row['atr_pct'] = atr

            pct_key = 'bounce_open_close_pct'
            if pct_key in row and row[pct_key] and not pd.isna(row[pct_key]):
                row['atr_pct_move'] = float(row[pct_key]) / atr
    except Exception as e:
        logging.error(f'Error in calculate_bounce_atr for {ticker} on {date}: {e}')

    return row


# ---------------------------------------------------------------------------
# 8. Bounce duration (time from break of PM high to HOD)
# ---------------------------------------------------------------------------
def get_bounce_duration(row, analysis_type):
    ticker = row['ticker']
    date = _parse_date(row)
    logging.info(f'Running get_bounce_duration for {ticker} on {date}')

    try:
        data = _get_intraday_1min(ticker, date)
        if data is None or data.empty:
            return row

        premarket_data = data.between_time('06:00:00', '09:29:59')
        regular_session = data.between_time('09:30:00', '16:00:00')

        if premarket_data.empty or regular_session.empty:
            return row

        pm_high = premarket_data['high'].max()
        time_of_hod = regular_session['high'].idxmax()

        # Find first break above premarket high
        break_above = regular_session[regular_session['close'] > pm_high].first_valid_index()

        if break_above is not None:
            duration = time_of_hod - break_above
            row['bounce_duration'] = duration
            row['time_of_bounce'] = break_above
        else:
            row['bounce_duration'] = None
            row['time_of_bounce'] = None

    except Exception as e:
        logging.error(f'Error in get_bounce_duration for {ticker} on {date}: {e}')

    return row


# ---------------------------------------------------------------------------
# Trillium-aware replacements for reused combined_data_collection functions
# ---------------------------------------------------------------------------
def _get_volume_bounce(row):
    """Trillium-aware replacement for get_volume."""
    ticker = row['ticker']
    if not _is_canadian(ticker):
        return get_volume(row)
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
    logging.info(f'Running _get_volume_bounce (trillium) for {ticker} on {date}')
    try:
        metrics = trlm.fetch_and_calculate_volumes(ticker, date)
        for key, value in metrics.items():
            row[key] = value
    except Exception as e:
        logging.error(f'Error in _get_volume_bounce for {ticker} on {date}: {e}')
    return row


def _get_range_vol_expansion_bounce(row):
    """Trillium-aware replacement for get_range_vol_expansion."""
    ticker = row['ticker']
    if not _is_canadian(ticker):
        return get_range_vol_expansion(row)
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
    logging.info(f'Running _get_range_vol_expansion_bounce (trillium) for {ticker} on {date}')
    try:
        df = _get_levels_daily(ticker, date, 40)
        if df is None or len(df) < 4:
            return row
        df['high-low'] = df['high'] - df['low']
        df['high-previous_close'] = abs(df['high'] - df['close'].shift())
        df['low-previous_close'] = abs(df['low'] - df['close'].shift())
        df['TR'] = df[['high-low', 'high-previous_close', 'low-previous_close']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14, min_periods=1).mean()
        df['PCT_ATR'] = df['TR'] / df['ATR']
        rolling_window = min(len(df) - 4, 20)
        df['30Day_Avg_Volume'] = df['volume'].rolling(window=rolling_window).mean()
        df['pct_avg_volume'] = df['volume'] / df['30Day_Avg_Volume']
        metrics = {
            'day_of_range_pct': df['PCT_ATR'].iloc[-1],
            'one_day_before_range_pct': df['PCT_ATR'].iloc[-2],
            'two_day_before_range_pct': df['PCT_ATR'].iloc[-3],
            'three_day_before_range_pct': df['PCT_ATR'].iloc[-4],
            'percent_of_vol_one_day_before': df['pct_avg_volume'].iloc[-2],
            'percent_of_vol_two_day_before': df['pct_avg_volume'].iloc[-3],
            'percent_of_vol_three_day_before': df['pct_avg_volume'].iloc[-4]
        }
        for key, value in metrics.items():
            row[key] = value
    except Exception as e:
        logging.error(f'Error in _get_range_vol_expansion_bounce for {ticker} on {date}: {e}')
    return row


def _get_pct_from_mavs_bounce(row):
    """Trillium-aware replacement for get_pct_from_mavs. Computes MAs from daily bars."""
    ticker = row['ticker']
    if not _is_canadian(ticker):
        return get_pct_from_mavs(row)
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
    logging.info(f'Running _get_pct_from_mavs_bounce (trillium) for {ticker} on {date}')
    try:
        daily_data = _get_daily(ticker, date)
        if daily_data is None:
            return row
        high_of_day = daily_data.high
        hist = _get_levels_daily(ticker, adjust_date_to_market(date, 1), 250)
        if hist is None or hist.empty:
            return row
        closes = hist['close']
        for window, key in [(10, 'pct_from_10mav'), (20, 'pct_from_20mav'),
                            (50, 'pct_from_50mav'), (200, 'pct_from_200mav')]:
            if len(closes) >= window:
                sma = closes.tail(window).mean()
                row[key] = (high_of_day - sma) / sma
        if len(closes) >= 9:
            ema_9 = closes.ewm(span=9, adjust=False).mean().iloc[-1]
            row['pct_from_9ema'] = (high_of_day - ema_9) / ema_9
        if len(closes) >= 50:
            sma_50 = closes.tail(50).mean()
            highs = hist['high']
            lows = hist['low']
            tr = pd.concat([
                highs - lows,
                (highs - closes.shift()).abs(),
                (lows - closes.shift()).abs()
            ], axis=1).max(axis=1)
            atr_val = tr.tail(14).mean()
            if atr_val > 0:
                row['atr_distance_from_50mav'] = (high_of_day - sma_50) / atr_val
    except Exception as e:
        logging.error(f'Error in _get_pct_from_mavs_bounce for {ticker} on {date}: {e}')
    return row


def _check_pct_move_bounce(row):
    """Trillium-aware replacement for check_pct_move."""
    ticker = row['ticker']
    if not _is_canadian(ticker):
        return check_pct_move(row)
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
    logging.info(f'Running _check_pct_move_bounce (trillium) for {ticker} on {date}')
    try:
        daily = _get_daily(ticker, date)
        if daily is None:
            return row
        current_price = daily.open
        pct_return_dict = trlm.get_ticker_pct_move(ticker, date, current_price)
        for key, value in pct_return_dict.items():
            row[key] = value
    except Exception as e:
        logging.error(f"Error in _check_pct_move_bounce for {ticker} on {date}: {e}")
    return row


# ---------------------------------------------------------------------------
# fill_functions_bounce
# ---------------------------------------------------------------------------
fill_functions_bounce = {
    # Volume (trillium-aware)
    'avg_daily_vol': _get_volume_bounce,
    'vol_on_breakout_day': _get_volume_bounce,
    'premarket_vol': _get_volume_bounce,
    'vol_in_first_5_min': _get_volume_bounce,
    'vol_in_first_10_min': _get_volume_bounce,
    'vol_in_first_15_min': _get_volume_bounce,
    'vol_in_first_30_min': _get_volume_bounce,
    'vol_one_day_before': _get_volume_bounce,
    'vol_two_day_before': _get_volume_bounce,
    'vol_three_day_before': _get_volume_bounce,

    # Pct volume (pure math, no API call)
    'percent_of_premarket_vol': get_pct_volume,
    'percent_of_vol_in_first_5_min': get_pct_volume,
    'percent_of_vol_in_first_10_min': get_pct_volume,
    'percent_of_vol_in_first_15_min': get_pct_volume,
    'percent_of_vol_in_first_30_min': get_pct_volume,
    'percent_of_vol_on_breakout_day': get_pct_volume,

    # Range/vol expansion (trillium-aware)
    'percent_of_vol_one_day_before': _get_range_vol_expansion_bounce,
    'percent_of_vol_two_day_before': _get_range_vol_expansion_bounce,
    'percent_of_vol_three_day_before': _get_range_vol_expansion_bounce,
    'day_of_range_pct': _get_range_vol_expansion_bounce,
    'one_day_before_range_pct': _get_range_vol_expansion_bounce,
    'two_day_before_range_pct': _get_range_vol_expansion_bounce,
    'three_day_before_range_pct': _get_range_vol_expansion_bounce,

    # Pct changes (trillium-aware)
    'pct_change_120': _check_pct_move_bounce,
    'pct_change_90': _check_pct_move_bounce,
    'pct_change_30': _check_pct_move_bounce,
    'pct_change_15': _check_pct_move_bounce,
    'pct_change_3': _check_pct_move_bounce,

    # Moving averages (trillium-aware)
    'pct_from_10mav': _get_pct_from_mavs_bounce,
    'pct_from_20mav': _get_pct_from_mavs_bounce,
    'pct_from_50mav': _get_pct_from_mavs_bounce,
    'pct_from_200mav': _get_pct_from_mavs_bounce,
    'atr_distance_from_50mav': _get_pct_from_mavs_bounce,

    # Bounce day stats (new)
    'gap_pct': get_bounce_stats,
    'bounce_open_high_pct': get_bounce_stats,
    'bounce_open_close_pct': get_bounce_stats,
    'bounce_open_low_pct': get_bounce_stats,
    'bounce_open_to_day_after_open_pct': get_bounce_stats,

    # Selloff context (new)
    'consecutive_down_days': get_selloff_context,
    'pct_off_30d_high': get_selloff_context,
    'pct_off_52wk_high': get_selloff_context,
    'selloff_total_pct': get_selloff_context,
    'prior_day_close_vs_low_pct': get_selloff_context,

    # Bollinger bands (new)
    'lower_band_distance': get_bounce_bollinger,
    'closed_outside_lower_band': get_bounce_bollinger,
    'upper_band_distance': get_bounce_bollinger,
    'bollinger_width': get_bounce_bollinger,
    'bb_position': get_bounce_bollinger,
    'days_since_upper_bb': get_bounce_bollinger,
    'upper_bb_to_lower_bb_days': get_bounce_bollinger,

    # Volume climax (new)
    'vol_trend_3d': get_volume_climax,
    'vol_trend_direction': get_volume_climax,
    'down_day_vol_ratio': get_volume_climax,

    # Volume profile (reuse)
    'rvol_score': get_volume_profile,
    'vol_ratio_5min_to_pm': get_volume_profile,

    # ATR (new bounce-specific)
    'atr_pct': calculate_bounce_atr,
    'atr_pct_move': calculate_bounce_atr,

    # Bounce conditionals (new)
    'near_52wk_low': get_bounce_conditionals,
    'breaks_52wk_low': get_bounce_conditionals,
    'close_at_highs': get_bounce_conditionals,
    'close_above_prior_close': get_bounce_conditionals,
    'close_green_red': get_bounce_conditionals,
    'hit_prior_day_hilo': get_bounce_conditionals,

    # Intraday timing (new)
    'time_of_low_price': get_bounce_intraday_timing,
    'time_of_low_bucket': get_bounce_intraday_timing,
    'high_to_low_duration_min': get_bounce_intraday_timing,
    'gap_from_pm_low': get_bounce_intraday_timing,

    # Bounce duration (new)
    'bounce_duration': get_bounce_duration,
    'time_of_bounce': get_bounce_duration,

    # SPY (reuse)
    'spy_open_close_pct': get_spy,
    'move_together': get_spy,

    # Market context (reuse)
    'spy_5day_return': get_market_context,
    'uvxy_close': get_market_context,
}


# ---------------------------------------------------------------------------
# fill_data — local copy with 'bounce' support
# ---------------------------------------------------------------------------
def fill_data(df, analysis_type, fill_functions):
    for column, fill_function in fill_functions.items():
        try:
            if column not in df.columns:
                df[column] = pd.NA
                logging.info(f'Added new column: {column}')

            if analysis_type in ['momentum', 'reversal', 'bounce'] and callable(fill_function):
                try:
                    df = df.apply(
                        lambda row, col=column, fn=fill_function: fn(row, analysis_type) if pd.isna(row[col]) else row,
                        axis=1
                    )
                except TypeError:
                    df = df.apply(
                        lambda row, col=column, fn=fill_function: fn(row) if pd.isna(row[col]) else row,
                        axis=1
                    )
            else:
                df = df.apply(
                    lambda row, col=column, fn=fill_function: fn(row) if pd.isna(row[col]) else row,
                    axis=1
                )
        except Exception as e:
            logging.error(f'Error processing column {column}: {e}')
    return df


# ---------------------------------------------------------------------------
# Column order
# ---------------------------------------------------------------------------
BOUNCE_COLUMN_ORDER = [
    # Hand-entered / existing
    'date', 'ticker', 'trade_grade', 'cap', 'Day', 'Setup',
    'run_points', 'run_pct', 'absolute_val_run_pct', 'run_duration',
    'price_over_time', 'absolute_val_price_over_time',
    'reversal_pts', 'reversal_pct', 'absolute_val_reversal_pct', 'reversal_duration',
    'short_long', 'exhaustion_gap',

    # Bounce day stats
    'gap_pct', 'bounce_open_high_pct', 'bounce_open_close_pct',
    'bounce_open_low_pct', 'bounce_open_to_day_after_open_pct',

    # Selloff context
    'consecutive_down_days', 'pct_off_30d_high', 'pct_off_52wk_high',
    'selloff_total_pct', 'prior_day_close_vs_low_pct',

    # Bollinger bands
    'lower_band_distance', 'closed_outside_lower_band',
    'upper_band_distance', 'bollinger_width', 'bb_position',
    'days_since_upper_bb', 'upper_bb_to_lower_bb_days',

    # Volume
    'avg_daily_vol', 'vol_on_breakout_day', 'premarket_vol',
    'vol_in_first_5_min', 'vol_in_first_10_min', 'vol_in_first_15_min', 'vol_in_first_30_min',
    'vol_one_day_before', 'vol_two_day_before', 'vol_three_day_before',
    'percent_of_premarket_vol',
    'percent_of_vol_in_first_5_min', 'percent_of_vol_in_first_10_min',
    'percent_of_vol_in_first_15_min', 'percent_of_vol_in_first_30_min',
    'percent_of_vol_on_breakout_day',
    'percent_of_vol_one_day_before', 'percent_of_vol_two_day_before', 'percent_of_vol_three_day_before',

    # Volume climax
    'vol_trend_3d', 'vol_trend_direction', 'down_day_vol_ratio',
    'rvol_score', 'vol_ratio_5min_to_pm',

    # Pct changes
    'pct_change_120', 'pct_change_90', 'pct_change_30', 'pct_change_15', 'pct_change_3',

    # Moving averages
    'pct_from_10mav', 'pct_from_20mav', 'pct_from_50mav', 'pct_from_200mav',
    'atr_distance_from_50mav',

    # Range expansion
    'day_of_range_pct', 'one_day_before_range_pct', 'two_day_before_range_pct', 'three_day_before_range_pct',

    # ATR
    'atr_pct', 'atr_pct_move',

    # Conditionals
    'near_52wk_low', 'breaks_52wk_low', 'close_at_highs',
    'close_above_prior_close', 'close_green_red', 'hit_prior_day_hilo',

    # Intraday timing
    'time_of_low_price', 'time_of_low_bucket',
    'high_to_low_duration_min', 'gap_from_pm_low',
    'bounce_duration', 'time_of_bounce',

    # Market context
    'spy_open_close_pct', 'move_together', 'spy_5day_return', 'uvxy_close',
]


if __name__ == '__main__':
    df_bounce = fill_data(bounce_df, 'bounce', fill_functions_bounce)

    # Reorder columns: defined order first, then any extra columns at the end
    existing_cols = [col for col in BOUNCE_COLUMN_ORDER if col in df_bounce.columns]
    extra_cols = [col for col in df_bounce.columns if col not in BOUNCE_COLUMN_ORDER]
    df_bounce = df_bounce[existing_cols + extra_cols]

    df_bounce.to_csv("C:\\Users\\zmbur\\PycharmProjects\\backtester\\data\\bounce_data.csv", index=False)
    logging.info(f'Bounce data saved. {len(df_bounce)} rows, {len(df_bounce.columns)} columns.')
