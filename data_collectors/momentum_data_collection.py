from data_queries.polygon_queries import get_daily, adjust_date_forward, get_levels_data, get_price_with_fallback, \
    adjust_date_to_market, get_intraday
import pandas as pd
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import os
from pytz import timezone
import logging

df = pd.read_csv("C:\\Users\\zmbur\\PycharmProjects\\InOffice\\data\\breakout_data.csv")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def find_time_of_high_price(data):
    time_of_high_price = data['high'].idxmax()
    idx_high_price = data.index.get_loc(time_of_high_price)
    if idx_high_price != len(data.index) - 1:  # check if it's not the last index
        time_of_high_price = data.index[idx_high_price + 1]
    return time_of_high_price


def get_current_price(ticker, date):
    wrong_date = datetime.strptime(date, '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
    data = get_daily(ticker, date)
    return data.open


def get_ticker_pct_move(ticker, date, current_price):
    price_120dago = get_price_with_fallback(ticker, date, 120)
    price_90dago = get_price_with_fallback(ticker, date, 90)
    price_30dago = get_price_with_fallback(ticker, date, 30)
    price_15dago = get_price_with_fallback(ticker, date, 15)

    pct_change_120 = (current_price - price_120dago) / price_120dago if price_120dago else None
    pct_change_90 = (current_price - price_90dago) / price_90dago if price_90dago else None
    pct_change_30 = (current_price - price_30dago) / price_30dago if price_30dago else None
    pct_change_15 = (current_price - price_15dago) / price_15dago if price_15dago else None

    return {
        "pct_change_120": pct_change_120,
        "pct_change_90": pct_change_90,
        "pct_change_30": pct_change_30,
        "pct_change_15": pct_change_15
    }


def check_pct_move(row):
    ticker = row['ticker']
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
    current_price = get_current_price(ticker, date)
    logging.info(f'Running check_pct_move for {ticker} on {date}')
    try:
        pct_return_dict = get_ticker_pct_move(ticker, date, current_price)
        for key, value in pct_return_dict.items():
            row[key] = value
    except:
        print(f'data doesnt exist for {ticker}')
    return row


from datetime import datetime


def get_volume(row):
    ticker = row['ticker']
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
    data = get_intraday(ticker, date, multiplier=1, timespan='minute')
    adv = get_levels_data(ticker, date, 30, 1, 'day')
    logging.info(f'Running get_volume for {ticker} on {date}')

    # Calculate total volume for the breakout day
    avg_daily_vol = adv['volume'].sum() / len(adv['volume'])
    total_volume = data['volume'].sum()

    # Calculate premarket volume
    premarket_volume = data.between_time('06:00:00', '09:30:00')['volume'].sum()

    # Calculate market volume for the first 15 minutes
    market_volume_first_15_min = data.between_time('09:30:00', '09:45:00')['volume'].sum()

    # Assuming there is a need to calculate volumes for different time intervals within the first 30 minutes
    vol_in_first_5_min = data.between_time('09:30:00', '09:35:00')['volume'].sum()
    vol_in_first_10_min = data.between_time('09:30:00', '09:40:00')['volume'].sum()
    vol_in_first_30_min = data.between_time('09:30:00', '10:00:00')['volume'].sum()

    # Update the row with new values
    row['avg_daily_vol'] = avg_daily_vol
    row['vol_on_breakout_day'] = total_volume
    row['premarket_vol'] = premarket_volume
    row['vol_in_first_15_min'] = market_volume_first_15_min
    row['vol_in_first_5_min'] = vol_in_first_5_min
    row['vol_in_first_10_min'] = vol_in_first_10_min
    row['vol_in_first_30_min'] = vol_in_first_30_min

    return row


def check_breakout_stats(row):
    ticker = row['ticker']
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
    logging.info(f'Running check_breakout_stats for {ticker} on {date}')

    try:
        daily_data = get_daily(ticker, date)
        open_price = daily_data.open
        close_price = daily_data.close
        high_price = daily_data.high

        day_after_data = get_daily(ticker, adjust_date_forward(date, 1))
        day_after_open = day_after_data.open

        post_high_data = get_intraday(ticker, date, multiplier=1, timespan='minute')
        post_high = post_high_data.between_time('16:00:00', '20:00:00').high.max()

        # Calculating percentages
        row['breakout_open_high_pct'] = (high_price - open_price) / open_price
        row['breakout_open_close_pct'] = (close_price - open_price) / open_price
        row['breakout_open_post_high_pct'] = (post_high - open_price) / open_price
        row['breakout_open_to_day_after_open_pct'] = (day_after_open - open_price) / open_price
        row['price_over_time'] = row['breakout_open_high_pct'] / 6.5
    except:
        print(f'Data doesnt exist for {ticker}')
    return row


def get_spy(row):
    ticker = 'SPY'
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
    logging.info(f'Running get_spy for {ticker} on {date}')

    daily_data = get_daily(ticker, date)
    open_price = daily_data.open
    high_price = daily_data.high

    post_high_data = get_intraday(ticker, date, multiplier=1, timespan='minute')
    post_high = post_high_data.between_time('16:00:00', '20:00:00').high.max()

    # Calculating percentages
    spy_open_high_pct = (high_price - open_price) / open_price
    row['spy_open_high_pct'] = spy_open_high_pct
    row['move_together'] = True if spy_open_high_pct > 0 else False
    return row


def get_conditionals(row):
    ticker = row['ticker']
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
    logging.info(f'Running get_conditionals for {ticker} on {date}')
    daily_data = get_daily(ticker, date)
    fifty_two_week_high = get_levels_data(ticker, adjust_date_to_market(date, 1), 365, 1, 'day')
    fifty_two_week_high = fifty_two_week_high['high'].max()
    all_time_high = get_levels_data(ticker, adjust_date_to_market(date, 1), 1200, 1, 'day')
    all_time_high = all_time_high['high'].max()
    high_price = daily_data.high
    close_price = daily_data.close
    breaks_fifty_two_week_high = True if high_price > fifty_two_week_high else False
    breaks_all_time_high = True if high_price > all_time_high else False
    close_at_highs = True if abs(close_price - high_price) / high_price <= 0.02 else False
    row['breaks_fifty_two_wk'] = breaks_fifty_two_week_high
    row['breaks_ath'] = breaks_all_time_high
    row['close_at_highs'] = close_at_highs
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
    stock_data = get_levels_data(ticker, adjust_date_to_market(date, 1), period, 1, 'day')
    # Calculate True Range (TR)
    stock_data['high_low'] = (stock_data['high'] - stock_data['low']) / stock_data['close']
    # Calculate ATR
    atr = stock_data['high_low'].mean()
    if row['breakout_open_close_pct']:
        row['atr_pct'] = atr
        row['ATR_breakout'] = float(row['breakout_open_close_pct']) / atr
    return row


def get_duration(row):
    ticker = row['ticker']
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
    logging.info(f'Running get_duration for {ticker} on {date}')
    data = get_intraday(ticker, date, multiplier=1, timespan='minute')
    premarket_high = data.between_time('6:00:00', '09:29:59').high.max()
    data = data.between_time('09:30:00', '16:00:00')

    time_of_high = find_time_of_high_price(data)

    breakout_row = data[data['close'] > premarket_high].first_valid_index()

    if breakout_row is not None:
        if isinstance(data.index, pd.DatetimeIndex):
            index_pos = data.index.get_loc(breakout_row)
            breakout_time = data.index[index_pos]
        else:
            breakout_time = data.index[breakout_row]
        duration = time_of_high - breakout_time
    else:
        breakout_time = None
        duration = None

    row['time_of_breakout'] = breakout_time
    row['breakout_duration'] = duration
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

    'breakout_open_high_pct': check_breakout_stats,
    'breakout_open_close_pct': check_breakout_stats,
    'breakout_open_post_high_pct': check_breakout_stats,
    'breakout_open_to_day_after_open_pct': check_breakout_stats,
    'price_over_time': check_breakout_stats,

    'spy_open_high_pct': get_spy,
    'move_together': get_spy,

    'breakout_duration': get_duration,
    'time_of_breakout': get_duration,

    # For conditional data
    'breaks_fifty_two_wk': get_conditionals,
    'breaks_ath': get_conditionals,
    'close_at_highs': get_conditionals,

    # For ATR related data
    'atr_pct': calculate_atr,
    'ATR_breakout': calculate_atr
}

if __name__ == '__main__':
    try:
        for column, fill_function in fill_functions.items():
            try:
                df = df.apply(lambda row: fill_function(row) if pd.isna(row[column]) else row, axis=1)
            except ValueError:
                print('data doesnt exist')
        df.to_csv("C:\\Users\\zmbur\\PycharmProjects\\InOffice\\data\\breakout_data.csv", index=False)
    except:
        df.to_csv("C:\\Users\\zmbur\\PycharmProjects\\InOffice\\data\\breakout_data.csv", index=False)
