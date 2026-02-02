from data_queries.polygon_queries import get_daily, adjust_date_forward, get_levels_data, get_price_with_fallback, \
    adjust_date_to_market, get_intraday, check_pct_move, fetch_and_calculate_volumes, get_ticker_mavs_open, get_range_vol_expansion_data
import pandas as pd
import logging
from tabulate import tabulate
from datetime import datetime, timedelta
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

momentum_df = pd.read_csv("C:\\Users\\zmbur\\PycharmProjects\\backtester\\data\\breakout_data.csv")
momentum_df = momentum_df.dropna(subset=['ticker'])
momentum_df = momentum_df.dropna(subset=['date'])
reversal_df = pd.read_csv("C:\\Users\\zmbur\\PycharmProjects\\backtester\\data\\reversal_data.csv")
reversal_df = reversal_df.dropna(subset=['ticker'])
reversal_df = reversal_df.dropna(subset=['date'])


def find_time_of_high_price(data):
    time_of_high_price = data['high'].idxmax()
    idx_high_price = data.index.get_loc(time_of_high_price)
    if idx_high_price != len(data.index) - 1:  # check if it's not the last index
        time_of_high_price = data.index[idx_high_price + 1]
    return time_of_high_price

def fill_function_time_of_high_price(row):
    ticker = row['ticker']
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
    logging.info(f'Running fill_function_time_of_high_price for {ticker} on {date}')
    data = get_intraday(ticker, date, multiplier=1, timespan='minute')
    time_of_high_price = find_time_of_high_price(data)
    row['time_of_high_price'] = time_of_high_price
    return row

def find_time_of_low_price(data):
    time_of_low_price = data['low'].idxmin()
    idx_low_price = data.index.get_loc(time_of_low_price)
    if idx_low_price != len(data.index) - 1:  # check if it's not the last index
        time_of_low_price = data.index[idx_low_price + 1]
    return time_of_low_price


def get_current_price(ticker, date):
    wrong_date = datetime.strptime(date, '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
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
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
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
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
    logging.info(f'Running get_volume for {ticker} on {date}')

    metrics = fetch_and_calculate_volumes(ticker, date)

    for key, value in metrics.items():
        row[key] = value
    return row

def get_range_vol_expansion(row):
    ticker = row['ticker']
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
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
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
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
            post_high = post_high_data.between_time('16:00:00', '20:00:00').high.max()

            # Calculating percentages for momentum
            row['gap_pct'] = (open_price - prev_close) / prev_close
            row['breakout_open_high_pct'] = (high_price - open_price) / open_price
            row['breakout_open_close_pct'] = (close_price - open_price) / open_price
            row['breakout_open_post_high_pct'] = (post_high - open_price) / open_price
            row['breakout_open_to_day_after_open_pct'] = (day_after_open - open_price) / open_price

        elif analysis_type == 'reversal':
            low_price = daily_data.low
            post_low_data = get_intraday(ticker, date, multiplier=1, timespan='minute')
            post_low = post_low_data.between_time('16:00:00', '20:00:00').low.min()

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
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
    row_ticker = row['ticker']
    logging.info(f'Running get_spy for {row_ticker} on {date}')

    daily_data = get_daily(ticker, date)
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
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
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
    date_str = row['date']
    wrong_date = datetime.strptime(date_str, '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
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
        logging.error(f'Error calculating ATR for {ticker} on {date_str}: {e}')

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
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
    logging.info(f'Running get_duration for {ticker} on {date} for {analysis_type} analysis')

    data = get_intraday(ticker, date, multiplier=1, timespan='minute')
    premarket_data = data.between_time('6:00:00', '09:29:59')
    regular_session_data = data.between_time('09:30:00', '16:00:00')

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
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
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
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
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
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
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
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
    logging.info(f'Running get_intraday_timing for {ticker} on {date}')

    try:
        data = get_intraday(ticker, date, multiplier=1, timespan='minute')
        if data is None or data.empty:
            return row

        # Get premarket high and open price
        premarket_data = data.between_time('06:00:00', '09:29:59')
        regular_session = data.between_time('09:30:00', '16:00:00')

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
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
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


fill_functions_momentum = {
    'avg_daily_vol': get_volume,
    'vol_on_breakout_day': get_volume,
    'premarket_vol': get_volume,
    'vol_in_first_15_min': get_volume,
    'vol_in_first_5_min': get_volume,
    'vol_in_first_10_min': get_volume,
    'vol_in_first_30_min': get_volume,
    'percent_of_premarket_vol': get_pct_volume,
    'percent_of_vol_in_first_5_min': get_pct_volume,
    'percent_of_vol_in_first_10_min': get_pct_volume,
    'percent_of_vol_in_first_15_min': get_pct_volume,
    'percent_of_vol_in_first_30_min': get_pct_volume,

    'pct_change_120': check_pct_move,
    'pct_change_90': check_pct_move,
    'pct_change_30': check_pct_move,
    'pct_change_15': check_pct_move,
    'pct_change_3': check_pct_move,

    'breakout_open_high_pct': check_breakout_stats,
    'breakout_open_close_pct': check_breakout_stats,
    'breakout_open_post_high_pct': check_breakout_stats,
    'breakout_open_to_day_after_open_pct': check_breakout_stats,
    'gap_pct': check_breakout_stats,

    'spy_open_close_pct': get_spy,
    'move_together': get_spy,

    'breakout_duration': get_duration,
    'time_of_breakout': get_duration,

    # For conditional data
    'breaks_fifty_two_wk': get_conditionals,
    'breaks_ath': get_conditionals,
    'close_at_highs': get_conditionals,

    # For ATR related data
    'atr_pct': calculate_atr,
    'atr_pct_move': calculate_atr
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
    'date', 't', 'ticker', 'trade_grade', 'news_type', 'size', 'bp', 'npl', 'atr_pct', 'atr_pct_move',
    'avg_daily_vol', 'breakout_duration', 'breakout_open_close_pct', 'breakout_open_high_pct',
    'breakout_open_post_high_pct', 'breakout_open_to_day_after_open_pct', 'breaks_ath', 'breaks_fifty_two_wk',
    'close_at_highs', 'gap_pct', 'move_together', 'pct_change_120', 'pct_change_15', 'pct_change_3',
    'pct_change_30', 'pct_change_90', 'percent_of_premarket_vol', 'percent_of_vol_in_first_10_min',
    'percent_of_vol_in_first_15_min', 'percent_of_vol_in_first_30_min', 'percent_of_vol_in_first_5_min',
    'percent_of_vol_on_breakout_day', 'premarket_vol', 'spy_open_close_pct', 'time_of_breakout',
    'time_of_high', 'vol_in_first_10_min', 'vol_in_first_15_min', 'vol_in_first_30_min', 'vol_in_first_5_min',
    'vol_on_breakout_day', 'vol_one_day_before', 'vol_three_day_before', 'vol_two_day_before'
]

if __name__ == '__main__':
    # Process breakout data
    df_momentum = fill_data(momentum_df, 'momentum', fill_functions_momentum)
    # Reorder columns to match expected format, keeping any extra columns at the end
    existing_cols = [col for col in BREAKOUT_COLUMN_ORDER if col in df_momentum.columns]
    extra_cols = [col for col in df_momentum.columns if col not in BREAKOUT_COLUMN_ORDER]
    df_momentum = df_momentum[existing_cols + extra_cols]
    df_momentum.to_csv("C:\\Users\\zmbur\\PycharmProjects\\backtester\\data\\breakout_data.csv", index=False)

    # Process reversal data
    df_reversal = fill_data(reversal_df, 'reversal', fill_functions_reversal)
    # Reorder columns to match expected format, keeping any extra columns at the end
    existing_cols = [col for col in REVERSAL_COLUMN_ORDER if col in df_reversal.columns]
    extra_cols = [col for col in df_reversal.columns if col not in REVERSAL_COLUMN_ORDER]
    df_reversal = df_reversal[existing_cols + extra_cols]
    df_reversal.to_csv("C:\\Users\\zmbur\\PycharmProjects\\backtester\\data\\reversal_data.csv", index=False)
