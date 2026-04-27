from functools import lru_cache
from pathlib import Path
from data_queries.polygon_queries import get_daily, adjust_date_forward, get_levels_data, get_price_with_fallback, \
    adjust_date_to_market, get_intraday, check_pct_move, fetch_and_calculate_volumes, get_ticker_mavs_open, get_range_vol_expansion_data, get_ipo_date, get_atr
import pandas as pd
import logging
from tabulate import tabulate
from datetime import datetime, timedelta
from support.date_utils import csv_date_to_iso, parse_row_date
from support.market_session import PREMARKET_START, PREMARKET_END, MARKET_OPEN, MARKET_CLOSE, AFTERHOURS_END
from support.csv_utils import load_csv, save_csv_atomic
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / 'data'


@lru_cache(maxsize=1)
def _load_momentum_df():
    return load_csv(_DATA_DIR / 'breakout_data.csv', 'breakout')


@lru_cache(maxsize=1)
def _load_reversal_df():
    return load_csv(_DATA_DIR / 'reversal_data.csv', 'reversal')


def __getattr__(name):
    """Lazy-load DataFrames on first access instead of at import time."""
    if name == 'momentum_df':
        return _load_momentum_df()
    if name == 'reversal_df':
        return _load_reversal_df()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def find_time_of_high_price(data):
    time_of_high_price = data['high'].idxmax()
    idx_high_price = data.index.get_loc(time_of_high_price)
    if idx_high_price != len(data.index) - 1:  # check if it's not the last index
        time_of_high_price = data.index[idx_high_price + 1]
    return time_of_high_price

def _fill_time_of_high(row, output_column):
    ticker = row['ticker']
    date = parse_row_date(row)
    logging.info(f'Running time-of-high fill for {ticker} on {date}')
    data = get_intraday(ticker, date, multiplier=1, timespan='minute')
    if data is None or data.empty:
        row[output_column] = None
        return row
    time_of_high_price = find_time_of_high_price(data)
    row[output_column] = time_of_high_price
    return row


def fill_function_time_of_high_price(row):
    return _fill_time_of_high(row, 'time_of_high_price')


def fill_function_time_of_high(row):
    return _fill_time_of_high(row, 'time_of_high')

def find_time_of_low_price(data):
    time_of_low_price = data['low'].idxmin()
    idx_low_price = data.index.get_loc(time_of_low_price)
    if idx_low_price != len(data.index) - 1:  # check if it's not the last index
        time_of_low_price = data.index[idx_low_price + 1]
    return time_of_low_price


def get_current_price(ticker, date):
    date = csv_date_to_iso(date)
    data = get_daily(ticker, date)
    return data.open


def get_pct_volume(row):
    volume_columns = ['premarket_vol', 'vol_in_first_5_min', 'vol_in_first_15_min', 'vol_in_first_10_min',
                      'vol_in_first_30_min','vol_on_breakout_day']
    for col in volume_columns:
        if col in row:
            row[f'percent_of_{col}'] = row[col] / row['avg_daily_vol']
    return row

def get_pct_from_mavs(row):
    ticker = row['ticker']
    date = parse_row_date(row)
    logging.info(f'Running get_pct_from_mavs for {ticker} on {date}')
    try:
        metrics = get_ticker_mavs_open(ticker, date)

        for key, value in metrics.items():
            row[key] = value
        return row
    except Exception as e:
        print(f"Data doesn't exist for {ticker} or an error occurred: {e}")
        return row


def get_volume(row):
    ticker = row['ticker']
    date = parse_row_date(row)
    logging.info(f'Running get_volume for {ticker} on {date}')

    metrics = fetch_and_calculate_volumes(ticker, date)

    for key, value in metrics.items():
        row[key] = value
    return row

def get_range_vol_expansion(row):
    ticker = row['ticker']
    date = parse_row_date(row)
    logging.info(f'Running get_range_expansion for {ticker} on {date}')

    metrics = get_range_vol_expansion_data(ticker, date)

    for key, value in metrics.items():
        row[key] = value
    return row


def check_breakout_stats(row, analysis_type):
    """
    Checks breakout or reversal stats based on the analysis type.

    Parameters:
    - row: A dictionary or Series containing the ticker and date information.
    - analysis_type: A string that specifies the type of analysis ('momentum' or 'reversal').

    Returns:
    - row: The original row updated with the calculated stats.
    """
    ticker = row['ticker']
    date = parse_row_date(row)
    logging.info(f'Running check_breakout_stats for {ticker} on {date} for {analysis_type} analysis')

    try:
        daily_data = get_daily(ticker, date)
        open_price = daily_data.open
        close_price = daily_data.close
        day_before_data = get_daily(ticker, adjust_date_to_market(date, 1))
        prev_close = day_before_data.close
        day_after_data = get_daily(ticker, adjust_date_forward(date, 1))
        day_after_open = day_after_data.open

        if analysis_type == 'momentum':
            high_price = daily_data.high
            post_high_data = get_intraday(ticker, date, multiplier=1, timespan='minute')
            post_high = post_high_data.between_time(MARKET_CLOSE, AFTERHOURS_END).high.max()

            # Calculating percentages for momentum
            row['gap_pct'] = (open_price - prev_close) / prev_close
            row['breakout_open_high_pct'] = (high_price - open_price) / open_price
            row['breakout_open_close_pct'] = (close_price - open_price) / open_price
            row['breakout_open_post_high_pct'] = (post_high - open_price) / open_price
            row['breakout_open_to_day_after_open_pct'] = (day_after_open - open_price) / open_price

        elif analysis_type == 'reversal':
            low_price = daily_data.low
            post_low_data = get_intraday(ticker, date, multiplier=1, timespan='minute')
            post_low = post_low_data.between_time(MARKET_CLOSE, AFTERHOURS_END).low.min()

            # Calculating percentages for reversal
            row['gap_pct'] = (open_price - prev_close) / prev_close
            row['reversal_open_low_pct'] = (low_price - open_price) / open_price
            row['reversal_open_close_pct'] = (close_price - open_price) / open_price
            row['reversal_open_post_low_pct'] = (post_low - open_price) / open_price
            row['reversal_open_to_day_after_open_pct'] = (day_after_open - open_price) / open_price

        else:
            raise ValueError("Invalid analysis type. Please specify 'momentum' or 'reversal'.")

    except Exception as e:
        print(f"Data doesn't exist for {ticker} or an error occurred: {e}")

    return row


def get_spy(row):
    ticker = 'SPY'
    date = parse_row_date(row)
    row_ticker = row['ticker']
    logging.info(f'Running get_spy for {row_ticker} on {date}')

    daily_data = get_daily(ticker, date)
    if daily_data is None:
        row['spy_open_close_pct'] = None
        row['move_together'] = None
        return row

    open_price = daily_data.open
    close = daily_data.close

    try:
        spy_open_close_pct = (close - open_price) / open_price
    except TypeError:
        spy_open_close_pct = None
    row['spy_open_close_pct'] = spy_open_close_pct
    try:
        row['move_together'] = True if spy_open_close_pct < 0 else False
    except TypeError:
        row['move_together'] = None
    return row


def get_conditionals(row, analysis_type):
    """
    Computes conditionals for a given row based on the analysis type (momentum or reversal).

    Parameters:
    - row: A dictionary or Series with ticker and date information.
    - analysis_type: A string specifying the type of analysis ('momentum' or 'reversal').

    Returns:
    - row: The input row updated with the computed conditionals.
    """
    ticker = row['ticker']
    date = parse_row_date(row)
    logging.info(f'Running get_conditionals for {ticker} on {date} for {analysis_type} analysis')

    try:
        daily_data = get_daily(ticker, date)
        fifty_two_week_high = get_levels_data(ticker, adjust_date_to_market(date, 1), 365, 1, 'day').high.max()
        all_time_high = get_levels_data(ticker, adjust_date_to_market(date, 1), 1200, 1, 'day').high.max()
        high_price = daily_data.high
        low_price = daily_data.low
        close_price = daily_data.close
        breaks_fifty_two_week_high = high_price > fifty_two_week_high
        breaks_all_time_high = high_price > all_time_high
        row['breaks_fifty_two_wk'] = breaks_fifty_two_week_high
        row['breaks_ath'] = breaks_all_time_high

        if analysis_type == 'momentum':
            close_at_highs = abs(close_price - high_price) / high_price <= 0.02
            row['close_at_highs'] = close_at_highs

        elif analysis_type == 'reversal':
            day_before_data = get_daily(ticker, adjust_date_to_market(date, 1))
            prev_close = day_before_data.close
            close_at_lows = abs(close_price - low_price) / low_price <= 0.02
            close_green_red = close_price < prev_close
            hit_green_red = low_price < prev_close
            hit_prior_day_hilo = low_price < day_before_data.low

            row['close_at_lows'] = close_at_lows
            row['close_green_red'] = close_green_red
            row['hit_green_red'] = hit_green_red
            row['hit_prior_day_hilo'] = hit_prior_day_hilo

        else:
            raise ValueError("Invalid analysis type. Please specify 'momentum' or 'reversal'.")

    except Exception as e:
        print(f"Data doesn't exist for {ticker} or an error occurred: {e}")

    return row


def calculate_atr(row, analysis_type, period=30):
    """
    Calculate the Average True Range (ATR) for a given stock data.

    Parameters:
    - row: Input data containing ticker and date.
    - analysis_type: Type of analysis ('momentum' or 'reversal').
    - period: Number of periods for ATR calculation (default 14).

    Returns:
    - Updated row with ATR calculations.
    """
    ticker = row['ticker']
    date = parse_row_date(row)
    logging.info(f'Running calculate_atr for {ticker} on {date}')
    try:
        stock_data = get_levels_data(ticker, adjust_date_to_market(date, 1), period, 1, 'day')

        if not stock_data.empty:
            # Calculate True Range
            stock_data['tr'] = stock_data[['high', 'low', 'close']].apply(
                lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close']), abs(x['low'] - x['close'])), axis=1)
            # Calculate ATR
            atr = stock_data['tr'].mean() / stock_data['close'].mean()

            pct_key = 'reversal_open_close_pct' if analysis_type == 'reversal' else 'breakout_open_close_pct'
            if pct_key in row and row[pct_key]:
                row['atr_pct'] = atr
                row['atr_pct_move'] = float(row[pct_key]) / atr

        else:
            logging.warning(f'No data available for {ticker} on {date}.')

    except Exception as e:
        logging.error(f'Error calculating ATR for {ticker} on {date}: {e}')

    return row


def get_duration(row, analysis_type):
    """
    Calculate the duration from the breakout or reversal point to the high or low point of the day.

    Parameters:
    - row: A dictionary or Series with ticker and date information.
    - analysis_type: A string specifying the type of analysis ('breakout' or 'reversal').

    Returns:
    - row: The input row updated with the time and duration of the breakout or reversal.
    """
    ticker = row['ticker']
    date = parse_row_date(row)
    logging.info(f'Running get_duration for {ticker} on {date} for {analysis_type} analysis')

    data = get_intraday(ticker, date, multiplier=1, timespan='minute')
    premarket_data = data.between_time(PREMARKET_START, PREMARKET_END)
    regular_session_data = data.between_time(MARKET_OPEN, MARKET_CLOSE)

    if analysis_type == 'momentum':
        premarket_extreme = premarket_data.high.max()
        time_of_extreme = find_time_of_high_price(regular_session_data)
        breakout_row = regular_session_data[regular_session_data['close'] > premarket_extreme].first_valid_index()
    elif analysis_type == 'reversal':
        premarket_extreme = premarket_data.low.min()
        time_of_extreme = find_time_of_low_price(regular_session_data)
        breakout_row = regular_session_data[regular_session_data['close'] < premarket_extreme].first_valid_index()

    else:
        raise ValueError("Invalid analysis type. Please specify 'breakout' or 'reversal'.")

    if breakout_row is not None:
        if isinstance(regular_session_data.index, pd.DatetimeIndex):
            index_pos = regular_session_data.index.get_loc(breakout_row)
            breakout_time = regular_session_data.index[index_pos]
        else:
            breakout_time = regular_session_data.index[breakout_row]

        duration = time_of_extreme - breakout_time
        key_prefix = 'breakout' if analysis_type == 'momentum' else 'reversal'
        row[f'time_of_{key_prefix}'] = breakout_time
        row[f'{key_prefix}_duration'] = duration

    else:
        row[f'time_of_{analysis_type}'] = None
        row[f'{analysis_type}_duration'] = None

    return row


def get_bollinger_bands(row, analysis_type):
    """
    Calculate Bollinger Band indicators for reversal analysis.
    - upper_band_distance: % distance from upper band (positive = above)
    - bollinger_width: Band width as % of middle band (squeeze indicator)
    - closed_outside_upper_band: True if prior day closed outside upper band
    """
    ticker = row['ticker']
    date = parse_row_date(row)
    logging.info(f'Running get_bollinger_bands for {ticker} on {date}')

    try:
        # Get 30+ days of historical data ending at trade date for Bollinger calculation
        df = get_levels_data(ticker, date, 35, 1, 'day')
        if df is None or len(df) < 21:
            logging.warning(f'Insufficient data for Bollinger Bands: {ticker} on {date}, got {len(df) if df is not None else 0} rows')
            return row

        # Calculate 20-day SMA and standard deviation
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['sma_20'] + (2 * df['std_20'])
        df['lower_band'] = df['sma_20'] - (2 * df['std_20'])

        # Get prior day's values (second to last row, last row is trade date)
        prior_close = df['close'].iloc[-2]
        upper_band = df['upper_band'].iloc[-2]
        lower_band = df['lower_band'].iloc[-2]
        middle_band = df['sma_20'].iloc[-2]

        # Check for valid values before calculating
        if pd.isna(upper_band) or pd.isna(lower_band) or pd.isna(middle_band):
            logging.warning(f'NaN values in Bollinger calculation for {ticker} on {date}')
            return row

        # Calculate indicators
        row['upper_band_distance'] = (prior_close - upper_band) / upper_band
        row['bollinger_width'] = (upper_band - lower_band) / middle_band
        row['closed_outside_upper_band'] = prior_close > upper_band

    except Exception as e:
        logging.error(f'Error calculating Bollinger Bands for {ticker} on {date}: {e}')

    return row


def get_volume_profile(row, analysis_type):
    """
    Calculate volume profile indicators:
    - vol_ratio_5min_to_pm: First 5 min volume / premarket volume
    - rvol_score: Relative volume score (day's vol / 20-day avg, normalized)
    """
    ticker = row['ticker']
    date = parse_row_date(row)
    logging.info(f'Running get_volume_profile for {ticker} on {date}')

    try:
        # Use existing volume data if available
        premarket_vol = row.get('premarket_vol')
        vol_first_5 = row.get('vol_in_first_5_min')
        avg_daily_vol = row.get('avg_daily_vol')
        vol_on_day = row.get('vol_on_breakout_day')

        # Calculate 5min to premarket ratio
        if premarket_vol and vol_first_5 and premarket_vol > 0:
            row['vol_ratio_5min_to_pm'] = vol_first_5 / premarket_vol

        # Calculate relative volume score (normalized 0-10 scale)
        if avg_daily_vol and vol_on_day and avg_daily_vol > 0:
            rvol = vol_on_day / avg_daily_vol
            # Cap at 10x ADV and normalize to 0-10 scale
            row['rvol_score'] = min(rvol, 10.0)

    except Exception as e:
        logging.error(f'Error calculating volume profile for {ticker} on {date}: {e}')

    return row


def get_prior_day_context(row, analysis_type):
    """
    Calculate prior day context indicators:
    - prior_day_close_vs_high_pct: How close to highs prior day closed (0-1, 1=at high)
    - consecutive_up_days: Count of consecutive up days before trade
    - prior_day_range_atr: Prior day's range as multiple of ATR
    """
    ticker = row['ticker']
    date = parse_row_date(row)
    logging.info(f'Running get_prior_day_context for {ticker} on {date}')

    try:
        # Get historical daily data (10 days for consecutive count)
        df = get_levels_data(ticker, adjust_date_to_market(date, 1), 15, 1, 'day')
        if df is None or len(df) < 2:
            return row

        # Prior day close vs high (1 = closed at high, 0 = closed at low)
        prior_high = df['high'].iloc[-1]
        prior_low = df['low'].iloc[-1]
        prior_close = df['close'].iloc[-1]
        if prior_high != prior_low:
            row['prior_day_close_vs_high_pct'] = (prior_close - prior_low) / (prior_high - prior_low)

        # Count consecutive up days
        consecutive_up = 0
        for i in range(len(df) - 1, 0, -1):
            if df['close'].iloc[i] > df['close'].iloc[i - 1]:
                consecutive_up += 1
            else:
                break
        row['consecutive_up_days'] = consecutive_up

        # Prior day range as ATR multiple
        atr_pct = row.get('atr_pct')
        if atr_pct and atr_pct > 0:
            prior_range_pct = (prior_high - prior_low) / prior_close
            row['prior_day_range_atr'] = prior_range_pct / atr_pct

    except Exception as e:
        logging.error(f'Error calculating prior day context for {ticker} on {date}: {e}')

    return row


def get_intraday_timing(row, analysis_type):
    """
    Calculate intraday timing indicators:
    - time_of_high_bucket: Categorical (1=9:30-10, 2=10-11, 3=11-12, 4=after 12)
    - gap_from_pm_high: Open price vs premarket high (negative = opened below PM high)
    """
    ticker = row['ticker']
    date = parse_row_date(row)
    logging.info(f'Running get_intraday_timing for {ticker} on {date}')

    try:
        data = get_intraday(ticker, date, multiplier=1, timespan='minute')
        if data is None or data.empty:
            return row

        # Get premarket high and open price
        premarket_data = data.between_time(PREMARKET_START, PREMARKET_END)
        regular_session = data.between_time(MARKET_OPEN, MARKET_CLOSE)

        if not premarket_data.empty and not regular_session.empty:
            pm_high = premarket_data['high'].max()
            open_price = regular_session['open'].iloc[0]
            row['gap_from_pm_high'] = (open_price - pm_high) / pm_high

        # Determine time of high bucket
        if not regular_session.empty:
            high_time = regular_session['high'].idxmax()
            hour = high_time.hour
            minute = high_time.minute

            if hour == 9 or (hour == 10 and minute == 0):
                row['time_of_high_bucket'] = 1  # 9:30-10:00
            elif hour == 10:
                row['time_of_high_bucket'] = 2  # 10:00-11:00
            elif hour == 11:
                row['time_of_high_bucket'] = 3  # 11:00-12:00
            else:
                row['time_of_high_bucket'] = 4  # After 12:00

    except Exception as e:
        logging.error(f'Error calculating intraday timing for {ticker} on {date}: {e}')

    return row


def get_market_context(row, analysis_type):
    """
    Calculate market context indicators:
    - spy_5day_return: SPY return over past 5 trading days
    - uvxy_close: UVXY close on prior day (volatility proxy, since VIX not on Polygon)
    """
    date = parse_row_date(row)
    row_ticker = row['ticker']
    logging.info(f'Running get_market_context for {row_ticker} on {date}')

    try:
        # SPY 5-day return
        spy_df = get_levels_data('SPY', adjust_date_to_market(date, 1), 10, 1, 'day')
        if spy_df is not None and len(spy_df) >= 5:
            spy_5d_ago = spy_df['close'].iloc[-5]
            spy_prior = spy_df['close'].iloc[-1]
            row['spy_5day_return'] = (spy_prior - spy_5d_ago) / spy_5d_ago

        # UVXY close (prior day) - volatility ETF proxy for VIX
        uvxy_data = get_daily('UVXY', adjust_date_to_market(date, 1))
        if uvxy_data:
            row['uvxy_close'] = uvxy_data.close

    except Exception as e:
        logging.error(f'Error calculating market context for {row_ticker} on {date}: {e}')

    return row


def _trading_days_between(start_date_str, end_date_str):
    """Approximate trading days between two YYYY-MM-DD dates using NYSE calendar.
    Returns None on failure."""
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar('NYSE')
        days = nyse.valid_days(start_date=start_date_str, end_date=end_date_str)
        return len(days)
    except Exception:
        return None


def _resolve_pivot(setup_type_str, ticker, date):
    """
    Resolve the breakout pivot price for outcome measurement based on setup_type.

    Returns: (pivot_price, source_label) or (None, None) on failure.

    Priority order from setup_type tags:
      ATH_breakout      → max high in 1200d window before trade date
      52wk_breakout     → max high in last 252 trading days before trade date
      IPO_high_breakout → max high since IPO date
      news_day2         → prior trading day's high
      news_day1         → prior trading day's close
      other             → prior trading day's close (fallback)

    Sanity check: if a high-water pivot ends up >25% above prior close, the
    flag is stale/incorrect (volatile names can legitimately gap 10-20% on news
    to break a level just above them, so 25% is the conservative threshold).
    Fall back to prior close and log a warning so outcome metrics stay sane.
    """
    if not setup_type_str:
        return None, None

    tags = [t.strip() for t in str(setup_type_str).split(',')]
    primary = tags[0] if tags else None

    try:
        prior_date = adjust_date_to_market(date, 1)
        prior_daily = get_daily(ticker, prior_date)
        prior_close = float(prior_daily.close) if prior_daily is not None else None

        def _high_water_pivot(window_days, source_label):
            hist = get_levels_data(ticker, prior_date, window_days, 1, 'day')
            if hist is None or hist.empty:
                return None, None
            pivot = float(hist['high'].max())
            if prior_close is not None and pivot > prior_close * 1.25:
                logging.warning(
                    f'{ticker} on {date}: setup={primary} resolves pivot={pivot:.2f} '
                    f'but prior_close={prior_close:.2f} — flag likely stale. '
                    f'Falling back to prior_close.'
                )
                return prior_close, f'prior_close_fallback_from_{source_label}'
            return pivot, source_label

        if primary == 'ATH_breakout':
            return _high_water_pivot(1200, 'ath')

        if primary == '52wk_breakout':
            return _high_water_pivot(252, '52wk_high')

        if primary == 'IPO_high_breakout':
            ipo_str = get_ipo_date(ticker)
            if not ipo_str:
                return (prior_close, 'prior_close') if prior_close is not None else (None, None)
            window_days = max(1, _trading_days_between(ipo_str, csv_date_to_iso(prior_date)) or 30)
            return _high_water_pivot(window_days, 'ipo_high')

        if primary == 'news_day2':
            d1 = get_daily(ticker, prior_date)
            if d1 is None:
                return None, None
            return float(d1.high), 'd1_high'

        # news_day1 and other → prior day close
        if prior_close is not None:
            return prior_close, 'prior_close'
        return None, None

    except Exception as e:
        logging.error(f'_resolve_pivot error for {ticker} on {date}: {e}')
        return None, None


def get_breakout_outcome(row, analysis_type):
    """
    Outcome labels (Y) — measures whether the breakout worked.

    Computes:
      - pivot_to_high_pct        : (day_high - pivot) / pivot
      - pivot_to_close_pct       : (close - pivot) / pivot
      - atr_max_extension        : (day_high - pivot) / atr_dollars
      - atr_at_close             : (close - pivot) / atr_dollars
      - held_above_pivot_at_close: bool — close > pivot
      - pulled_back_to_pivot_intraday: bool — touched pivot from above after breaking
      - pivot_source             : label describing which pivot was used
    """
    ticker = row['ticker']
    date = parse_row_date(row)
    logging.info(f'Running get_breakout_outcome for {ticker} on {date}')

    try:
        setup_type = row.get('setup_type')
        pivot, source = _resolve_pivot(setup_type, ticker, date)
        if pivot is None or pivot <= 0:
            return row

        daily = get_daily(ticker, date)
        if daily is None:
            return row
        day_high = float(daily.high)
        day_close = float(daily.close)

        atr_dollars = get_atr(ticker, date)

        row['pivot_to_high_pct'] = (day_high - pivot) / pivot
        row['pivot_to_close_pct'] = (day_close - pivot) / pivot
        row['held_above_pivot_at_close'] = day_close > pivot
        row['pivot_source'] = source

        if atr_dollars and atr_dollars > 0:
            row['atr_max_extension'] = (day_high - pivot) / atr_dollars
            row['atr_at_close'] = (day_close - pivot) / atr_dollars

        # Pullback to pivot: did intraday low (after first cross above pivot) revisit the pivot?
        try:
            intraday = get_intraday(ticker, date, multiplier=1, timespan='minute')
            if intraday is not None and not intraday.empty:
                regular = intraday.between_time(MARKET_OPEN, MARKET_CLOSE)
                above = regular[regular['high'] >= pivot]
                if not above.empty:
                    first_break_idx = above.index[0]
                    after_break = regular.loc[first_break_idx:]
                    row['pulled_back_to_pivot_intraday'] = bool((after_break['low'] <= pivot).any())
                else:
                    row['pulled_back_to_pivot_intraday'] = False
        except Exception as e:
            logging.warning(f'pulled_back check failed for {ticker} on {date}: {e}')

    except Exception as e:
        logging.error(f'Error in get_breakout_outcome for {ticker} on {date}: {e}')

    return row


def get_consolidation_quality(row, analysis_type):
    """
    Forward-looking base/squeeze metrics, all measured at PRIOR close.

      - consolidation_days        : trading days the stock has held within 10% of its 52wk high
      - range_contraction_atr     : last-5d avg TR / last-30d avg TR (<1 = contraction)
      - bb_width_percentile_6mo   : percentile of prior-day BB width over last 126 days (low = tight squeeze)
      - vol_dry_up_ratio          : last-5d avg vol / last-20d avg vol (low = dry-up)
      - up_down_vol_ratio_30d     : vol on green-close days / vol on red-close days, last 30d
    """
    ticker = row['ticker']
    date = parse_row_date(row)
    logging.info(f'Running get_consolidation_quality for {ticker} on {date}')

    try:
        prior_date = adjust_date_to_market(date, 1)
        # 380 calendar days ≈ 270 trading days — gives us full 252d (52wk) coverage
        hist = get_levels_data(ticker, prior_date, 380, 1, 'day')
        if hist is None or len(hist) < 30:
            return row

        # Compute true range
        hist = hist.sort_index().copy()
        hist['prev_close'] = hist['close'].shift(1)
        hist['tr'] = hist[['high', 'low', 'prev_close']].apply(
            lambda x: max(
                x['high'] - x['low'],
                abs(x['high'] - x['prev_close']) if pd.notna(x['prev_close']) else x['high'] - x['low'],
                abs(x['low'] - x['prev_close']) if pd.notna(x['prev_close']) else x['high'] - x['low'],
            ), axis=1
        )

        # consolidation_days: count trailing days where close >= 0.90 * 52wk high
        wk52_high = hist['high'].max()
        if wk52_high > 0:
            threshold = 0.90 * wk52_high
            cd = 0
            for i in range(len(hist) - 1, -1, -1):
                if hist['close'].iloc[i] >= threshold:
                    cd += 1
                else:
                    break
            row['consolidation_days'] = cd

        # range_contraction_atr: 5d / 30d TR
        if len(hist) >= 30:
            tr_5 = hist['tr'].tail(5).mean()
            tr_30 = hist['tr'].tail(30).mean()
            if tr_30 > 0:
                row['range_contraction_atr'] = tr_5 / tr_30

        # bb_width_percentile_6mo: prior-day BB width vs distribution over last 126 days
        if len(hist) >= 126:
            sma20 = hist['close'].rolling(window=20).mean()
            std20 = hist['close'].rolling(window=20).std()
            bb_w = ((sma20 + 2 * std20) - (sma20 - 2 * std20)) / sma20
            recent = bb_w.tail(126).dropna()
            current = bb_w.iloc[-1]
            if pd.notna(current) and len(recent) > 10:
                row['bb_width_percentile_6mo'] = (recent <= current).sum() / len(recent)

        # vol_dry_up_ratio: 5d / 20d
        if len(hist) >= 20:
            v5 = hist['volume'].tail(5).mean()
            v20 = hist['volume'].tail(20).mean()
            if v20 > 0:
                row['vol_dry_up_ratio'] = v5 / v20

        # up_down_vol_ratio_30d
        recent_30 = hist.tail(30)
        up_vol = recent_30.loc[recent_30['close'] > recent_30['prev_close'], 'volume'].sum()
        dn_vol = recent_30.loc[recent_30['close'] < recent_30['prev_close'], 'volume'].sum()
        if dn_vol > 0:
            row['up_down_vol_ratio_30d'] = up_vol / dn_vol

    except Exception as e:
        logging.error(f'Error in get_consolidation_quality for {ticker} on {date}: {e}')

    return row


@lru_cache(maxsize=64)
def _benchmark_history(symbol, prior_date, window):
    """Cached fetch of benchmark daily history (SPY/QQQ) for RS calculations."""
    return get_levels_data(symbol, prior_date, window, 1, 'day')


def get_relative_strength(row, analysis_type):
    """
    Relative strength vs SPY and QQQ over multiple lookbacks (forward-looking — uses prior close).

      - rs_vs_spy_30d, rs_vs_spy_90d, rs_vs_spy_252d
      - rs_vs_qqq_30d, rs_vs_qqq_90d
    """
    ticker = row['ticker']
    date = parse_row_date(row)
    logging.info(f'Running get_relative_strength for {ticker} on {date}')

    try:
        prior_date = adjust_date_to_market(date, 1)
        # 380 calendar days ≈ 270 trading days for full 252d (12mo) RS calculation
        hist = get_levels_data(ticker, prior_date, 380, 1, 'day')
        if hist is None or len(hist) < 30:
            return row
        hist = hist.sort_index()

        spy_hist = _benchmark_history('SPY', prior_date, 380)
        qqq_hist = _benchmark_history('QQQ', prior_date, 380)

        def _ret(df, n):
            if df is None or len(df) <= n:
                return None
            df = df.sort_index()
            return (df['close'].iloc[-1] - df['close'].iloc[-1 - n]) / df['close'].iloc[-1 - n]

        for n, key_spy, key_qqq in [(30, 'rs_vs_spy_30d', 'rs_vs_qqq_30d'),
                                    (90, 'rs_vs_spy_90d', 'rs_vs_qqq_90d')]:
            stk = _ret(hist, n)
            spy = _ret(spy_hist, n)
            qqq = _ret(qqq_hist, n)
            if stk is not None and spy is not None:
                row[key_spy] = stk - spy
            if stk is not None and qqq is not None:
                row[key_qqq] = stk - qqq

        # 252-day vs SPY only (12mo IBD-style relative return)
        stk_252 = _ret(hist, 251) if len(hist) >= 252 else None
        spy_252 = _ret(spy_hist, 251) if spy_hist is not None and len(spy_hist) >= 252 else None
        if stk_252 is not None and spy_252 is not None:
            row['rs_vs_spy_252d'] = stk_252 - spy_252

    except Exception as e:
        logging.error(f'Error in get_relative_strength for {ticker} on {date}: {e}')

    return row


def get_trend_structure(row, analysis_type):
    """
    Trend health metrics at PRIOR close.

      - ma_stack_aligned          : bool — 10 SMA > 20 SMA > 50 SMA > 200 SMA
      - ma_50_slope_5d            : (50ma_t - 50ma_t-5) / 50ma_t-5
      - consecutive_days_above_50ma: count of consecutive days at end with close > 50ma
    """
    ticker = row['ticker']
    date = parse_row_date(row)
    logging.info(f'Running get_trend_structure for {ticker} on {date}')

    try:
        prior_date = adjust_date_to_market(date, 1)
        # 290 calendar days ≈ 205 trading days — needed for full 200MA
        hist = get_levels_data(ticker, prior_date, 290, 1, 'day')
        if hist is None or len(hist) < 50:
            return row
        hist = hist.sort_index()
        closes = hist['close']

        sma_10 = closes.rolling(10).mean()
        sma_20 = closes.rolling(20).mean()
        sma_50 = closes.rolling(50).mean()
        sma_200 = closes.rolling(200).mean() if len(closes) >= 200 else None

        last10 = sma_10.iloc[-1]
        last20 = sma_20.iloc[-1]
        last50 = sma_50.iloc[-1]
        last200 = sma_200.iloc[-1] if sma_200 is not None else None

        # ma_stack_aligned
        if all(pd.notna(v) for v in [last10, last20, last50] + ([last200] if last200 is not None else [])):
            stack_ok = last10 > last20 > last50
            if last200 is not None:
                stack_ok = stack_ok and last50 > last200
            row['ma_stack_aligned'] = bool(stack_ok)

        # ma_50_slope_5d
        if len(sma_50.dropna()) >= 6:
            sma50_now = sma_50.iloc[-1]
            sma50_then = sma_50.iloc[-6]
            if pd.notna(sma50_now) and pd.notna(sma50_then) and sma50_then > 0:
                row['ma_50_slope_5d'] = (sma50_now - sma50_then) / sma50_then

        # consecutive_days_above_50ma
        cdays = 0
        for i in range(len(closes) - 1, -1, -1):
            sma_val = sma_50.iloc[i]
            if pd.isna(sma_val):
                break
            if closes.iloc[i] > sma_val:
                cdays += 1
            else:
                break
        row['consecutive_days_above_50ma'] = cdays

    except Exception as e:
        logging.error(f'Error in get_trend_structure for {ticker} on {date}: {e}')

    return row


def get_d2_continuation_features(row, analysis_type):
    """
    Day-2 continuation features. Only computes when row['t'] == 1.

      - d1_close_at_high_pct       : (D1 close - D1 low) / (D1 high - D1 low) — 1 = closed at HOD
      - d1_rvol                    : D1 volume / 20d avg vol
      - d1_range_atr               : D1 range as multiple of ATR (>2 = wide thrust, exhaustion risk)
      - d1_thrust_atr              : (D1 close - D1 open) / ATR (positive = up-day, magnitude = strength)
      - overnight_gap_d1_to_d2_pct : (D2 open - D1 close) / D1 close
      - pm_d2_holds_above_d1_high  : bool — premarket price held above D1 high
    """
    ticker = row['ticker']
    date = parse_row_date(row)

    t_val = row.get('t')
    try:
        t_int = int(t_val) if pd.notna(t_val) else None
    except (ValueError, TypeError):
        t_int = None
    if t_int != 1:
        return row

    logging.info(f'Running get_d2_continuation_features for {ticker} on {date}')

    try:
        d1_date = adjust_date_to_market(date, 1)
        d1 = get_daily(ticker, d1_date)
        if d1 is None:
            return row

        d1_high = float(d1.high)
        d1_low = float(d1.low)
        d1_open = float(d1.open)
        d1_close = float(d1.close)

        rng = d1_high - d1_low
        if rng > 0:
            row['d1_close_at_high_pct'] = (d1_close - d1_low) / rng

        # ATR (in dollars) computed at the trade date — use it consistently
        atr_dollars = get_atr(ticker, date)
        if atr_dollars and atr_dollars > 0:
            row['d1_range_atr'] = rng / atr_dollars
            row['d1_thrust_atr'] = (d1_close - d1_open) / atr_dollars

        # D1 RVOL: D1 volume / 20-day average volume excluding D1
        # NOTE: get_levels_data window is in CALENDAR days (~0.71x trading days),
        # so 40 calendar days ≈ 28 trading days, giving us 21+ days reliably.
        hist = get_levels_data(ticker, d1_date, 40, 1, 'day')
        if hist is not None and len(hist) >= 21:
            d1_vol = hist['volume'].iloc[-1]
            avg20 = hist['volume'].iloc[-21:-1].mean()
            if avg20 > 0:
                row['d1_rvol'] = d1_vol / avg20

        # Overnight gap D1 → D2
        d2 = get_daily(ticker, date)
        if d2 is not None:
            d2_open = float(d2.open)
            row['overnight_gap_d1_to_d2_pct'] = (d2_open - d1_close) / d1_close

        # Premarket of D2: did price hold above D1 high
        intraday = get_intraday(ticker, date, multiplier=1, timespan='minute')
        if intraday is not None and not intraday.empty:
            pm = intraday.between_time(PREMARKET_START, PREMARKET_END)
            if not pm.empty:
                pm_low_after_first_break = None
                # First time PM crosses above D1 high
                above = pm[pm['high'] > d1_high]
                if not above.empty:
                    first_idx = above.index[0]
                    pm_after = pm.loc[first_idx:]
                    pm_low_after_first_break = pm_after['low'].min()
                    row['pm_d2_holds_above_d1_high'] = bool(pm_low_after_first_break >= d1_high)
                else:
                    row['pm_d2_holds_above_d1_high'] = False

    except Exception as e:
        logging.error(f'Error in get_d2_continuation_features for {ticker} on {date}: {e}')

    return row


def get_ipo_features(row, analysis_type):
    """
    IPO-specific features. Only computes when days_since_ipo < 365.

      - days_since_ipo
      - pct_to_ipo_high  : (prior close - max high since IPO) / max high since IPO
      - pct_off_ipo_low  : (prior close - min low since IPO) / min low since IPO
    """
    ticker = row['ticker']
    date = parse_row_date(row)

    try:
        ipo_str = get_ipo_date(ticker)
        if not ipo_str:
            return row

        date_iso = csv_date_to_iso(date)
        days_since = _trading_days_between(ipo_str, date_iso)
        if days_since is None or days_since >= 365:
            return row

        row['days_since_ipo'] = days_since

        prior_date = adjust_date_to_market(date, 1)
        # Pull all history since IPO (cap at days_since trading days)
        window = max(5, days_since + 5)
        hist = get_levels_data(ticker, prior_date, window, 1, 'day')
        if hist is None or hist.empty:
            return row

        prior_close = float(hist['close'].iloc[-1])
        ipo_high = float(hist['high'].max())
        ipo_low = float(hist['low'].min())

        if ipo_high > 0:
            row['pct_to_ipo_high'] = (prior_close - ipo_high) / ipo_high
        if ipo_low > 0:
            row['pct_off_ipo_low'] = (prior_close - ipo_low) / ipo_low

        logging.info(f'Running get_ipo_features for {ticker} on {date} (days_since_ipo={days_since})')

    except Exception as e:
        logging.error(f'Error in get_ipo_features for {ticker} on {date}: {e}')

    return row


def derive_setup_type(row, analysis_type):
    """
    Multi-label setup classification (comma-joined when multiple apply).

    Tags:
      - ATH_breakout       : breaks_ath True AND t in {0, 1}
      - 52wk_breakout      : breaks_fifty_two_wk True (and not ATH) AND t in {0, 1}
      - IPO_high_breakout  : days_since_ipo < 365 AND breaks_fifty_two_wk True
      - news_day1          : t == 0
      - news_day2          : t == 1
      - other              : fallback if no tags applied
    """
    ticker = row['ticker']
    date = parse_row_date(row)
    logging.info(f'Running derive_setup_type for {ticker} on {date}')

    tags = []

    # Read news-day indicator
    t_val = row.get('t')
    try:
        t_int = int(t_val) if pd.notna(t_val) else None
    except (ValueError, TypeError):
        t_int = None

    # Pull conditionals already set by get_conditionals
    breaks_ath = row.get('breaks_ath')
    breaks_52wk = row.get('breaks_fifty_two_wk')

    def _is_true(v):
        if pd.isna(v):
            return False
        if isinstance(v, str):
            return v.strip().lower() in ('true', '1', 'yes')
        return bool(v)

    is_ath = _is_true(breaks_ath)
    is_52wk = _is_true(breaks_52wk)

    # Breakout-level tags (only meaningful on news days {0, 1})
    if t_int in (0, 1):
        if is_ath:
            tags.append('ATH_breakout')
        elif is_52wk:
            tags.append('52wk_breakout')

    # IPO breakout — independent dimension, can co-occur
    try:
        ipo_date_str = get_ipo_date(ticker)
        if ipo_date_str:
            days_since = _trading_days_between(ipo_date_str, csv_date_to_iso(date))
            if days_since is not None and days_since < 365 and is_52wk:
                tags.append('IPO_high_breakout')
    except Exception as e:
        logging.warning(f'IPO check failed for {ticker} on {date}: {e}')

    # News-day tags (always applied if t is set)
    if t_int == 0:
        tags.append('news_day1')
    elif t_int == 1:
        tags.append('news_day2')

    if not tags:
        tags.append('other')

    row['setup_type'] = ','.join(tags)
    return row


def get_breakout_proximity(row, analysis_type):
    """
    Distance to key levels at PRIOR close (forward-looking — measured before the trade day).
    For breakouts: small/zero values indicate the stock is near a trigger level.

    Computes:
      - pct_to_52wk_high, pct_to_ath
      - pct_to_30d_high, pct_to_60d_high, pct_to_90d_high
      - days_since_52wk_high, days_since_ath
    """
    ticker = row['ticker']
    date = parse_row_date(row)
    logging.info(f'Running get_breakout_proximity for {ticker} on {date}')

    try:
        # Pull 1200 trading days ending at PRIOR close (so we exclude the trade day's high)
        prior_date = adjust_date_to_market(date, 1)
        hist = get_levels_data(ticker, prior_date, 1200, 1, 'day')
        if hist is None or hist.empty:
            return row

        prior_close = hist['close'].iloc[-1]

        # All-time high in the available window
        ath = hist['high'].max()
        if ath > 0:
            row['pct_to_ath'] = (prior_close - ath) / ath
            ath_idx = hist['high'].idxmax()
            row['days_since_ath'] = len(hist.loc[ath_idx:]) - 1

        # 52-week high (last ~252 trading days)
        recent_252 = hist.tail(252) if len(hist) >= 252 else hist
        wk52_high = recent_252['high'].max()
        if wk52_high > 0:
            row['pct_to_52wk_high'] = (prior_close - wk52_high) / wk52_high
            wk52_idx = recent_252['high'].idxmax()
            row['days_since_52wk_high'] = len(recent_252.loc[wk52_idx:]) - 1

        # 30 / 60 / 90 day highs
        for window, key in [(30, 'pct_to_30d_high'), (60, 'pct_to_60d_high'), (90, 'pct_to_90d_high')]:
            window_df = hist.tail(window) if len(hist) >= window else hist
            window_high = window_df['high'].max()
            if window_high > 0:
                row[key] = (prior_close - window_high) / window_high

    except Exception as e:
        logging.error(f'Error in get_breakout_proximity for {ticker} on {date}: {e}')

    return row


fill_functions_momentum = {
    # Volume (extended to include prior 3 days)
    'avg_daily_vol': get_volume,
    'vol_on_breakout_day': get_volume,
    'premarket_vol': get_volume,
    'vol_in_first_15_min': get_volume,
    'vol_in_first_5_min': get_volume,
    'vol_in_first_10_min': get_volume,
    'vol_in_first_30_min': get_volume,
    'vol_one_day_before': get_volume,
    'vol_two_day_before': get_volume,
    'vol_three_day_before': get_volume,
    'percent_of_premarket_vol': get_pct_volume,
    'percent_of_vol_in_first_5_min': get_pct_volume,
    'percent_of_vol_in_first_10_min': get_pct_volume,
    'percent_of_vol_in_first_15_min': get_pct_volume,
    'percent_of_vol_in_first_30_min': get_pct_volume,
    'percent_of_vol_on_breakout_day': get_pct_volume,

    # Range expansion / contraction (used inverted for breakouts: contraction is bullish)
    'percent_of_vol_one_day_before': get_range_vol_expansion,
    'percent_of_vol_two_day_before': get_range_vol_expansion,
    'percent_of_vol_three_day_before': get_range_vol_expansion,
    'day_of_range_pct': get_range_vol_expansion,
    'one_day_before_range_pct': get_range_vol_expansion,
    'two_day_before_range_pct': get_range_vol_expansion,
    'three_day_before_range_pct': get_range_vol_expansion,

    # Pct changes
    'pct_change_120': check_pct_move,
    'pct_change_90': check_pct_move,
    'pct_change_30': check_pct_move,
    'pct_change_15': check_pct_move,
    'pct_change_3': check_pct_move,

    # Moving averages (ported from reversal — uses high-of-day as reference)
    'pct_from_9ema': get_pct_from_mavs,
    'pct_from_10mav': get_pct_from_mavs,
    'pct_from_20mav': get_pct_from_mavs,
    'pct_from_50mav': get_pct_from_mavs,
    'pct_from_200mav': get_pct_from_mavs,
    'atr_distance_from_50mav': get_pct_from_mavs,

    # Breakout day stats
    'breakout_open_high_pct': check_breakout_stats,
    'breakout_open_close_pct': check_breakout_stats,
    'breakout_open_post_high_pct': check_breakout_stats,
    'breakout_open_to_day_after_open_pct': check_breakout_stats,
    'gap_pct': check_breakout_stats,

    # SPY
    'spy_open_close_pct': get_spy,
    'move_together': get_spy,

    # Breakout duration
    'breakout_duration': get_duration,
    'time_of_breakout': get_duration,
    'time_of_high': fill_function_time_of_high,

    # Conditionals
    'breaks_fifty_two_wk': get_conditionals,
    'breaks_ath': get_conditionals,
    'close_at_highs': get_conditionals,

    # ATR
    'atr_pct': calculate_atr,
    'atr_pct_move': calculate_atr,

    # Bollinger bands (upper-only for breakouts; squeeze metric is bollinger_width)
    'upper_band_distance': get_bollinger_bands,
    'bollinger_width': get_bollinger_bands,

    # Volume profile
    'vol_ratio_5min_to_pm': get_volume_profile,
    'rvol_score': get_volume_profile,

    # Prior day context (consecutive_up_days + closing strength + range vs ATR)
    'prior_day_close_vs_high_pct': get_prior_day_context,
    'consecutive_up_days': get_prior_day_context,
    'prior_day_range_atr': get_prior_day_context,

    # Intraday timing
    'time_of_high_bucket': get_intraday_timing,
    'gap_from_pm_high': get_intraday_timing,

    # Market context
    'spy_5day_return': get_market_context,
    'uvxy_close': get_market_context,

    # Consolidation / squeeze
    'consolidation_days': get_consolidation_quality,
    'range_contraction_atr': get_consolidation_quality,
    'bb_width_percentile_6mo': get_consolidation_quality,
    'vol_dry_up_ratio': get_consolidation_quality,
    'up_down_vol_ratio_30d': get_consolidation_quality,

    # Relative strength
    'rs_vs_spy_30d': get_relative_strength,
    'rs_vs_spy_90d': get_relative_strength,
    'rs_vs_spy_252d': get_relative_strength,
    'rs_vs_qqq_30d': get_relative_strength,
    'rs_vs_qqq_90d': get_relative_strength,

    # Trend structure
    'ma_stack_aligned': get_trend_structure,
    'ma_50_slope_5d': get_trend_structure,
    'consecutive_days_above_50ma': get_trend_structure,

    # Day-2 continuation (only fills when t == 1)
    'd1_close_at_high_pct': get_d2_continuation_features,
    'd1_rvol': get_d2_continuation_features,
    'd1_range_atr': get_d2_continuation_features,
    'd1_thrust_atr': get_d2_continuation_features,
    'overnight_gap_d1_to_d2_pct': get_d2_continuation_features,
    'pm_d2_holds_above_d1_high': get_d2_continuation_features,

    # IPO features (only fills when days_since_ipo < 365)
    'days_since_ipo': get_ipo_features,
    'pct_to_ipo_high': get_ipo_features,
    'pct_off_ipo_low': get_ipo_features,

    # Proximity to key levels (forward-looking — distance at prior close)
    'pct_to_52wk_high': get_breakout_proximity,
    'pct_to_ath': get_breakout_proximity,
    'pct_to_30d_high': get_breakout_proximity,
    'pct_to_60d_high': get_breakout_proximity,
    'pct_to_90d_high': get_breakout_proximity,
    'days_since_52wk_high': get_breakout_proximity,
    'days_since_ath': get_breakout_proximity,

    # Setup type — must come AFTER breaks_ath/breaks_fifty_two_wk so they're populated first
    'setup_type': derive_setup_type,

    # Outcome labels (Y) — must come AFTER setup_type so the pivot can be resolved
    'pivot_to_high_pct': get_breakout_outcome,
    'pivot_to_close_pct': get_breakout_outcome,
    'atr_max_extension': get_breakout_outcome,
    'atr_at_close': get_breakout_outcome,
    'held_above_pivot_at_close': get_breakout_outcome,
    'pulled_back_to_pivot_intraday': get_breakout_outcome,
    'pivot_source': get_breakout_outcome,
}

fill_functions_reversal = {
    'avg_daily_vol': get_volume,
    'vol_on_breakout_day': get_volume,
    'premarket_vol': get_volume,
    'vol_in_first_15_min': get_volume,
    'vol_in_first_5_min': get_volume,
    'vol_in_first_10_min': get_volume,
    'vol_in_first_30_min': get_volume,
    'vol_two_day_before': get_volume,
    'vol_one_day_before': get_volume,
    'vol_three_day_before': get_volume,
    'percent_of_premarket_vol': get_pct_volume,
    'percent_of_vol_in_first_5_min': get_pct_volume,
    'percent_of_vol_in_first_10_min': get_pct_volume,
    'percent_of_vol_in_first_15_min': get_pct_volume,
    'percent_of_vol_in_first_30_min': get_pct_volume,
    'percent_of_vol_on_breakout_day': get_pct_volume,


    'pct_change_120': check_pct_move,
    'pct_change_90': check_pct_move,
    'pct_change_30': check_pct_move,
    'pct_change_15': check_pct_move,
    'pct_change_3': check_pct_move,

     'percent_of_vol_two_day_before': get_range_vol_expansion,
    'percent_of_vol_one_day_before': get_range_vol_expansion,
    'percent_of_vol_three_day_before': get_range_vol_expansion,
    'day_of_range_pct': get_range_vol_expansion,
    'one_day_before_range_pct': get_range_vol_expansion,
    'two_day_before_range_pct': get_range_vol_expansion,
    'three_day_before_range_pct': get_range_vol_expansion,

    'pct_from_10mav': get_pct_from_mavs,
    'pct_from_20mav': get_pct_from_mavs,
    'pct_from_50mav': get_pct_from_mavs,
    'pct_from_200mav': get_pct_from_mavs,
    'atr_distance_from_50mav': get_pct_from_mavs,

    'gap_pct':check_breakout_stats,
    'reversal_open_low_pct': check_breakout_stats,
    'reversal_open_close_pct': check_breakout_stats,
    'reversal_open_post_low_pct': check_breakout_stats,
    'reversal_open_to_day_after_open_pct': check_breakout_stats,

    'spy_open_close_pct': get_spy,
    'move_together': get_spy,

    'reversal_duration': get_duration,
    'time_of_reversal': get_duration,
    'time_of_high_price': fill_function_time_of_high_price,

    # For conditional data
    'breaks_fifty_two_wk': get_conditionals,
    'breaks_ath': get_conditionals,
    'close_at_lows': get_conditionals,
    'close_green_red': get_conditionals,
    'hit_green_red': get_conditionals,
    'hit_prior_day_hilo': get_conditionals,

    # For ATR related data
    'atr_pct': calculate_atr,
    'atr_pct_move': calculate_atr,

    # Bollinger Band indicators
    'upper_band_distance': get_bollinger_bands,
    'bollinger_width': get_bollinger_bands,
    'closed_outside_upper_band': get_bollinger_bands,

    # Volume profile indicators
    'vol_ratio_5min_to_pm': get_volume_profile,
    'rvol_score': get_volume_profile,

    # Prior day context indicators
    'prior_day_close_vs_high_pct': get_prior_day_context,
    'consecutive_up_days': get_prior_day_context,
    'prior_day_range_atr': get_prior_day_context,

    # Intraday timing indicators
    'time_of_high_bucket': get_intraday_timing,
    'gap_from_pm_high': get_intraday_timing,

    # Market context indicators
    'spy_5day_return': get_market_context,
    'uvxy_close': get_market_context
}


def fill_data(df, analysis_type, fill_functions):
    """
    Fill dataframe with calculated data based on the specified analysis type.

    Parameters:
    - df: DataFrame to be filled.
    - analysis_type: 'breakout' or 'reversal' to specify the type of analysis.
    - fill_functions: Dictionary mapping column names to functions.
    """
    for column, fill_function in fill_functions.items():
        try:
            # Add column if it doesn't exist
            if column not in df.columns:
                df[column] = pd.NA
                logging.info(f'Added new column: {column}')

            # Adjust the lambda function to pass analysis_type if required
            if analysis_type in ['momentum', 'reversal'] and callable(fill_function):
                try:
                    df = df.apply(lambda row: fill_function(row, analysis_type) if pd.isna(row[column]) else row, axis=1)
                except TypeError:
                    df = df.apply(lambda row: fill_function(row) if pd.isna(row[column]) else row, axis=1)
            else:
                df = df.apply(lambda row: fill_function(row) if pd.isna(row[column]) else row, axis=1)
        except ValueError as e:
            print(f'Error processing column {column}: {e}')
    return df


REVERSAL_COLUMN_ORDER = [
    'date', 'ticker', 'trade_grade', 'cap', 'intraday_setup', 'setup', 'atr_pct', 'atr_pct_move',
    'avg_daily_vol', 'breaks_ath', 'breaks_fifty_two_wk', 'close_at_lows', 'close_green_red',
    'day_of_range_pct', 'gap_pct', 'hit_green_red', 'hit_prior_day_hilo', 'move_together',
    'one_day_before_range_pct', 'pct_change_120', 'pct_change_15', 'pct_change_3', 'pct_change_30',
    'pct_change_90', 'pct_from_10mav', 'pct_from_200mav', 'pct_from_20mav', 'pct_from_50mav', 'atr_distance_from_50mav',
    'percent_of_premarket_vol', 'percent_of_vol_in_first_10_min', 'percent_of_vol_in_first_15_min',
    'percent_of_vol_in_first_30_min', 'percent_of_vol_in_first_5_min', 'percent_of_vol_on_breakout_day',
    'percent_of_vol_one_day_before', 'percent_of_vol_three_day_before', 'percent_of_vol_two_day_before',
    'premarket_vol', 'reversal_duration', 'reversal_open_close_pct', 'reversal_open_low_pct',
    'reversal_open_post_low_pct', 'reversal_open_to_day_after_open_pct', 'spy_open_close_pct',
    'three_day_before_range_pct', 'time_of_high_price', 'time_of_low', 'time_of_reversal',
    'two_day_before_range_pct', 'vol_in_first_10_min', 'vol_in_first_15_min', 'vol_in_first_30_min',
    'vol_in_first_5_min', 'vol_on_breakout_day', 'vol_one_day_before', 'vol_three_day_before',
    'vol_two_day_before',
    # New indicators - Bollinger Bands
    'upper_band_distance', 'bollinger_width', 'closed_outside_upper_band',
    # New indicators - Volume Profile
    'vol_ratio_5min_to_pm', 'rvol_score',
    # New indicators - Prior Day Context
    'prior_day_close_vs_high_pct', 'consecutive_up_days', 'prior_day_range_atr',
    # New indicators - Intraday Timing
    'time_of_high_bucket', 'gap_from_pm_high',
    # New indicators - Market Context
    'spy_5day_return', 'uvxy_close',
    # Original trailing columns
    'bp', 'npl', 'size'
]

BREAKOUT_COLUMN_ORDER = [
    # Identification
    'date', 't', 'ticker', 'trade_grade', 'news_type', 'setup_type', 'cap', 'size', 'bp', 'npl',

    # Setup quality (forward-looking X) — proximity / consolidation
    'pct_to_52wk_high', 'pct_to_ath', 'pct_to_30d_high', 'pct_to_60d_high', 'pct_to_90d_high',
    'days_since_52wk_high', 'days_since_ath',
    'consolidation_days', 'range_contraction_atr', 'bb_width_percentile_6mo',
    'bollinger_width', 'upper_band_distance',
    'vol_dry_up_ratio', 'up_down_vol_ratio_30d',
    'prior_day_range_atr', 'prior_day_close_vs_high_pct', 'consecutive_up_days',

    # Trend (forward-looking X)
    'pct_from_9ema', 'pct_from_10mav', 'pct_from_20mav', 'pct_from_50mav', 'pct_from_200mav',
    'atr_distance_from_50mav',
    'ma_stack_aligned', 'ma_50_slope_5d', 'consecutive_days_above_50ma',

    # Relative strength (forward-looking X)
    'rs_vs_spy_30d', 'rs_vs_spy_90d', 'rs_vs_spy_252d',
    'rs_vs_qqq_30d', 'rs_vs_qqq_90d',

    # Day-2 / IPO (conditional X)
    'd1_close_at_high_pct', 'd1_rvol', 'd1_range_atr', 'd1_thrust_atr',
    'overnight_gap_d1_to_d2_pct', 'pm_d2_holds_above_d1_high',
    'days_since_ipo', 'pct_to_ipo_high', 'pct_off_ipo_low',

    # Volume (X)
    'avg_daily_vol', 'premarket_vol',
    'vol_in_first_5_min', 'vol_in_first_10_min', 'vol_in_first_15_min', 'vol_in_first_30_min',
    'vol_on_breakout_day',
    'vol_one_day_before', 'vol_two_day_before', 'vol_three_day_before',
    'percent_of_premarket_vol',
    'percent_of_vol_in_first_5_min', 'percent_of_vol_in_first_10_min',
    'percent_of_vol_in_first_15_min', 'percent_of_vol_in_first_30_min',
    'percent_of_vol_on_breakout_day',
    'percent_of_vol_one_day_before', 'percent_of_vol_two_day_before', 'percent_of_vol_three_day_before',
    'vol_ratio_5min_to_pm', 'rvol_score',

    # Range expansion (X — same window as reversal but used inverted for breakouts)
    'day_of_range_pct', 'one_day_before_range_pct', 'two_day_before_range_pct', 'three_day_before_range_pct',

    # Premarket / intraday (X)
    'gap_pct', 'gap_from_pm_high', 'time_of_high_bucket',

    # ATR (X)
    'atr_pct',

    # Pct change (X)
    'pct_change_3', 'pct_change_15', 'pct_change_30', 'pct_change_90', 'pct_change_120',

    # Market context (X)
    'spy_open_close_pct', 'move_together', 'spy_5day_return', 'uvxy_close',

    # Float / SI (X — populated separately by breakout_bloomberg.py)
    'float_shares', 'short_interest_pct', 'days_to_cover',

    # Outcome labels (Y)
    'breakout_duration', 'breakout_open_high_pct', 'breakout_open_close_pct',
    'breakout_open_post_high_pct', 'breakout_open_to_day_after_open_pct',
    'breaks_fifty_two_wk', 'breaks_ath', 'close_at_highs',
    'pivot_to_high_pct', 'pivot_to_close_pct',
    'atr_max_extension', 'atr_at_close',
    'held_above_pivot_at_close', 'pulled_back_to_pivot_intraday', 'pivot_source',
    'time_of_breakout', 'time_of_high',
    'atr_pct_move',
]

if __name__ == '__main__':
    # Process breakout data
    df_momentum = fill_data(_load_momentum_df(), 'momentum', fill_functions_momentum)
    # Reorder columns to match expected format, keeping any extra columns at the end
    existing_cols = [col for col in BREAKOUT_COLUMN_ORDER if col in df_momentum.columns]
    extra_cols = [col for col in df_momentum.columns if col not in BREAKOUT_COLUMN_ORDER]
    df_momentum = df_momentum[existing_cols + extra_cols]
    save_csv_atomic(df_momentum, _DATA_DIR / 'breakout_data.csv')

    # Process reversal data
    df_reversal = fill_data(_load_reversal_df(), 'reversal', fill_functions_reversal)
    # Reorder columns to match expected format, keeping any extra columns at the end
    existing_cols = [col for col in REVERSAL_COLUMN_ORDER if col in df_reversal.columns]
    extra_cols = [col for col in df_reversal.columns if col not in REVERSAL_COLUMN_ORDER]
    df_reversal = df_reversal[existing_cols + extra_cols]
    save_csv_atomic(df_reversal, _DATA_DIR / 'reversal_data.csv')
