from data_queries.polygon_queries import get_daily, get_levels_data, get_price_with_fallback, get_intraday, get_daily, adjust_date_forward, adjust_date_to_market
from momentum_data_collection import get_volume, get_ticker_pct_move, check_pct_move
import pandas as pd
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import os
from pytz import timezone
import logging

df = pd.read_csv("C:\\Users\\zmbur\\PycharmProjects\\backtester\\data\\reversal_data.csv")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


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


def check_breakout_stats(row):
    ticker = row['ticker']
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
    logging.info(f'Running check_breakout_stats for {ticker} on {date}')

    try:
        daily_data = get_daily(ticker, date)
        open_price = daily_data.open
        close_price = daily_data.close
        low_price = daily_data.low

        day_after_data = get_daily(ticker, adjust_date_forward(date, 1))
        day_after_open = day_after_data.open
        day_before_data = get_daily(ticker, adjust_date_to_market(date, 1))

        post_low_data = get_intraday(ticker, date, multiplier=1, timespan='minute')
        post_low = post_low_data.between_time('16:00:00', '20:00:00').low.min()
        prev_close = day_before_data.close

        # Calculating percentages
        row['gap_pct'] = (open_price - prev_close) / prev_close
        row['reversal_open_low_pct'] = (low_price - open_price) / open_price
        row['reversal_open_close_pct'] = (close_price - open_price) / open_price
        row['reversal_open_post_low_pct'] = (post_low - open_price) / open_price
        row['reversal_open_to_day_after_open_pct'] = (day_after_open - open_price) / open_price
        row['price_over_time'] = row['pct_change_15'] / 6.5
    except:
        print(f'Data doesnt exist for {ticker}')
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

    # Calculating percentages
    spy_open_close_pct = (close - open_price) / open_price
    row['spy_open_close_pct'] = spy_open_close_pct
    row['move_together'] = True if spy_open_close_pct < 0 else False
    return row


def get_conditionals(row):
    ticker = row['ticker']
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
    logging.info(f'Running get_conditionals for {ticker} on {date}')
    try:
        daily_data = get_daily(ticker, date)
        fifty_two_week_high = get_levels_data(ticker, adjust_date_to_market(date, 1), 365, 1, 'day')
        fifty_two_week_high = fifty_two_week_high['high'].max()
        all_time_high = get_levels_data(ticker, adjust_date_to_market(date, 1), 1200, 1, 'day')
        all_time_high = all_time_high['high'].max()
        day_before_data = get_daily(ticker, adjust_date_to_market(date, 1))
        prev_close = day_before_data.close
        high_price = daily_data.high
        low_price = daily_data.low
        close_price = daily_data.close
        breaks_fifty_two_week_high = True if high_price > fifty_two_week_high else False
        breaks_all_time_high = True if high_price > all_time_high else False
        close_at_lows = True if abs(close_price - low_price) / low_price <= 0.02 else False
        close_green_red = True if close_price < prev_close else False
        hit_green_red = True if low_price < prev_close else False
        hit_prior_day_hilo = True if low_price < day_before_data.low else False
        row['breaks_fifty_two_wk'] = breaks_fifty_two_week_high
        row['breaks_ath'] = breaks_all_time_high
        row['close_at_lows'] = close_at_lows
        row['close_green_red'] = close_green_red
        row['hit_green_red'] = hit_green_red
        row['hit_prior_day_hilo'] = hit_prior_day_hilo
    except:
        print(f'Data doesnt exist for {ticker}')
    return row


def calculate_atr(row, period=30):
    """
    Calculate the Average True Range (ATR) for a given stock data.

    :param row:
    :param stock_data: DataFrame with columns 'High', 'Low', and 'Close'.
    :param period: Number of periods to use for ATR calculation (default is 14).
    :return: ATR value.
    """
    ticker = row['ticker']
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
    logging.info(f'Running calculate_atr for {ticker} on {date}')

    max_retries = 10  # Set a maximum number of retries
    attempts = 0

    while attempts < max_retries:
        stock_data = get_levels_data(ticker, adjust_date_to_market(date, 1), period, 1, 'day')

        if not stock_data.empty:
            # Calculate True Range (TR)
            stock_data['high_low'] = (stock_data['high'] - stock_data['low']) / stock_data['close']
            # Calculate ATR
            atr = stock_data['high_low'].mean()
            if row['reversal_open_close_pct']:
                row['atr_pct'] = atr
                row['atr_pct_move'] = float(row['reversal_open_close_pct']) / atr
            return row
        else:
            # Increment date by one day and try again
            wrong_date += timedelta(days=1)
            date = datetime.strftime(wrong_date, '%Y-%m-%d')
            logging.info(f'Retrying calculate_atr for {ticker} on {date}')
            attempts += 1

    logging.warning(f'Failed to calculate ATR for {ticker} after {max_retries} attempts.')
    return row


def get_duration(row):
    ticker = row['ticker']
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
    logging.info(f'Running get_duration for {ticker} on {date}')
    data = get_intraday(ticker, date, multiplier=1, timespan='minute')
    open_price = data.between_time('09:30:00', '09:31:00').open.min()
    premarket_low = data.between_time('6:00:00', '09:29:59').low.min()
    data = data.between_time('09:30:00', '16:00:00')

    time_of_low = find_time_of_low_price(data)
    open_breakout_row = data[data['close'] < open_price].first_valid_index()
    breakout_row = data[data['close'] < premarket_low].first_valid_index()

    if breakout_row is not None:
        if isinstance(data.index, pd.DatetimeIndex):
            index_pos = data.index.get_loc(breakout_row)
            breakout_time = data.index[index_pos]
        else:

            breakout_time = data.index[breakout_row]
        duration = time_of_low - breakout_time
    else:
        breakout_time = None
        duration = None

    row['time_of_reversal'] = breakout_time
    row['reversal_duration'] = duration
    return row


fill_functions = {
    'avg_daily_vol': get_volume,
    'vol_on_breakout_day': get_volume,
    'premarket_vol': get_volume,
    'vol_in_first_15_min': get_volume,
    'vol_in_first_5_min': get_volume,
    'vol_in_first_10_min': get_volume,
    'vol_in_first_30_min': get_volume,

    'pct_change_120': check_pct_move,
    'pct_change_90': check_pct_move,
    'pct_change_30': check_pct_move,
    'pct_change_15': check_pct_move,

    'gap_pct':check_breakout_stats,
    'reversal_open_low_pct': check_breakout_stats,
    'reversal_open_close_pct': check_breakout_stats,
    'reversal_open_post_low_pct': check_breakout_stats,
    'reversal_open_to_day_after_open_pct': check_breakout_stats,
    'price_over_time': check_breakout_stats,

    'spy_open_close_pct': get_spy,
    'move_together': get_spy,

    'reversal_duration': get_duration,
    'time_of_reversal': get_duration,

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

if __name__ == '__main__':

        for column, fill_function in fill_functions.items():
            try:
                df = df.apply(lambda row: fill_function(row) if pd.isna(row[column]) else row, axis=1)
            except ValueError:
                print('data doesnt exist')
        df.to_csv("C:\\Users\\zmbur\\PycharmProjects\\InOffice\\data\\reversal_data.csv", index=False)

