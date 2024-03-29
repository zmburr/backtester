import pandas as pd
import requests.exceptions
from polygon.rest import RESTClient
import pandas_market_calendars as mcal
from time import sleep
from datetime import datetime
import os
from pytz import timezone
from datetime import datetime, timedelta
from pandas import Timestamp
import logging

poly_client = RESTClient(api_key="b_s_dRysgNN_kZF_nzxwSLdvClTyopGgxtJSqX")


def get_atr(ticker, date):
    # ATR is the greatest of the following: high-low / high - previous close / low - previous close
    df = get_levels_data(ticker, date, 60, 1, 'day')
    df['high-low'] = df['high'] - df['low']
    df['high-previous_close'] = abs(df['high'] - df['close'].shift())
    df['low-previous_close'] = abs(df['low'] - df['close'].shift())
    df['TR'] = df[['high-low', 'high-previous_close', 'low-previous_close']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    atr = (df['ATR'][-1])
    return atr


def add_two_hours(time_str):
    # Parse the string to a datetime object
    time_obj = datetime.strptime(time_str, '%H:%M:%S')

    # Add two hours
    new_time_obj = time_obj + timedelta(hours=2)

    # Format back to a string
    new_time_str = new_time_obj.strftime('%H:%M:%S')

    return new_time_str


def fetch_and_calculate_volumes(ticker, date):
    # Assuming get_intraday and get_levels_data are defined elsewhere and fetch data from an API
    data = get_intraday(ticker, date, multiplier=1, timespan='minute')
    adv = get_levels_data(ticker, date, 30, 1, 'day')
    logging.info(f'Fetching and calculating volumes for {ticker} on {date}')

    # Calculate metrics
    metrics = {
        'avg_daily_vol': adv['volume'].sum() / len(adv['volume']) if len(adv['volume']) > 0 else 0,
        'vol_on_breakout_day': data['volume'].sum(),
        'premarket_vol': data.between_time('06:00:00', '09:30:00')['volume'].sum(),
        'vol_in_first_5_min': data.between_time('09:30:00', '09:35:00')['volume'].sum(),
        'vol_in_first_15_min': data.between_time('09:30:00', '09:45:00')['volume'].sum(),
        'vol_in_first_10_min': data.between_time('09:30:00', '09:40:00')['volume'].sum(),
        'vol_in_first_30_min': data.between_time('09:30:00', '10:00:00')['volume'].sum()
    }

    return metrics


def timestamp_to_string(timestamp_obj):
    if isinstance(timestamp_obj, Timestamp):
        return timestamp_obj.strftime('%Y-%m-%d %H:%M:%S')
    else:
        raise TypeError("The provided object is not a Timestamp.")


def _adjust_date(original_date, days_to_subtract):
    nyse = mcal.get_calendar('NYSE')
    new_date = original_date - pd.Timedelta(days=days_to_subtract)
    trading_days = nyse.valid_days(start_date=new_date, end_date=original_date)

    if not trading_days.empty:
        adjusted_date = trading_days[0].date().strftime("%Y-%m-%d")
        if adjusted_date == original_date.strftime("%Y-%m-%d"):
            return _adjust_date(original_date, days_to_subtract + 1)
        else:
            return adjusted_date
    else:
        return "No valid trading days found"


def adjust_date_to_market(date_string, days_to_subtract):
    date = pd.to_datetime(date_string)
    return _adjust_date(date, days_to_subtract)


def get_levels_data(ticker, date, window, multiplier, timespan):
    aggs = []
    try:
        for a in poly_client.list_aggs(ticker=ticker, multiplier=multiplier, timespan=timespan,
                                       from_=adjust_date_to_market(date, window), to=date, limit=50000):
            aggs.append(a)
    except KeyError:
        print('data doesnt exist')
        return None
    df = pd.DataFrame([vars(a) for a in aggs])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(timezone('US/Eastern'))
    # Set 'timestamp' as index
    df.set_index('timestamp', inplace=True)
    return df


def get_daily(ticker, date):
    request = poly_client.get_daily_open_close_agg(ticker, date)
    return request


def get_intraday(ticker, date, multiplier, timespan):
    """
    function to get price information for a ticker on given trade.date
    :param date:
    :param ticker: stock
    :param trade.date: in 'YYYY-MM-DD format'
    :param multiplier: number of timespan (must be > 0)
    :param timespan: 'minute', 'second', etc.
    :return: df containing price data
    """
    aggs = []
    try:
        for a in poly_client.list_aggs(ticker=ticker, multiplier=multiplier, timespan=timespan, from_=date, to=date,
                                       limit=50000):
            aggs.append(a)
    except KeyError:
        print('data doesnt exist')
        return None
    df = pd.DataFrame([vars(a) for a in aggs])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(timezone('US/Eastern'))
    df.set_index('timestamp', inplace=True)
    return df


def adjust_date_forward(date_string, days_to_add):
    # Validate and convert input date string to pandas.Timestamp
    try:
        date = pd.to_datetime(date_string)
    except ValueError:
        return "Invalid date format"

    # Validate days_to_add is a non-negative integer
    if not isinstance(days_to_add, int) or days_to_add < 0:
        return "days_to_add must be a non-negative integer"

    # Initialize NYSE calendar
    nyse = mcal.get_calendar('NYSE')

    # Add days to date and find the next valid trading day
    new_date = date
    while True:
        new_date += pd.Timedelta(days=1)  # Increment the day
        # Check if the new date is a valid trading day
        trading_days = nyse.valid_days(start_date=new_date, end_date=new_date)
        if not trading_days.empty:
            # Found a valid trading day, format and return it
            adjusted_date = trading_days[0].date().strftime("%Y-%m-%d")
            break

    return adjusted_date


def get_price_with_fallback(ticker, base_date, days_ago):
    while days_ago > 0:
        try:
            price = get_daily(ticker, adjust_date_to_market(base_date, days_ago)).close
            if price is not None:
                return price
        except Exception as e:
            # Handle specific exceptions if necessary
            pass
        days_ago -= 1
    return None  # or handle this case as needed


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


def get_current_price(ticker, date):
    try:
        wrong_date = datetime.strptime(date, '%m/%d/%Y')
        date = datetime.strftime(wrong_date, '%Y-%m-%d')
    except ValueError:
        pass
    data = get_daily(ticker, date)
    return data.open


# TODO - sub in trillium data
def get_actual_current_price(ticker):
    data = get_intraday(ticker, datetime.now().strftime('%Y-%m-%d'), 1, 'second')
    return data.iloc[-1].close


def check_pct_move(row):
    ticker = row['ticker']
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = datetime.strftime(wrong_date, '%Y-%m-%d')
    logging.info(f'Running check_pct_move for {ticker} on {date}')
    current_price = get_current_price(ticker, date)
    try:
        pct_return_dict = get_ticker_pct_move(ticker, date, current_price)
        for key, value in pct_return_dict.items():
            row[key] = value
    except:
        print(f'data doesnt exist for {ticker}')
    return row
