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
    'atr_pct_move': calculate_atr
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
    'vol_two_day_before', 'bp', 'npl', 'size'
]

if __name__ == '__main__':
    # Process breakout data
    df_momentum = fill_data(momentum_df, 'momentum', fill_functions_momentum)
    df_momentum.to_csv("C:\\Users\\zmbur\\PycharmProjects\\backtester\\data\\breakout_data.csv", index=False)

    # Process reversal data
    df_reversal = fill_data(reversal_df, 'reversal', fill_functions_reversal)
    # Reorder columns to match expected format, keeping any extra columns at the end
    existing_cols = [col for col in REVERSAL_COLUMN_ORDER if col in df_reversal.columns]
    extra_cols = [col for col in df_reversal.columns if col not in REVERSAL_COLUMN_ORDER]
    df_reversal = df_reversal[existing_cols + extra_cols]
    df_reversal.to_csv("C:\\Users\\zmbur\\PycharmProjects\\backtester\\data\\reversal_data.csv", index=False)
