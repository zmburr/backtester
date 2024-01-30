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

poly_client = RESTClient(api_key="b_s_dRysgNN_kZF_nzxwSLdvClTyopGgxtJSqX")

def add_two_hours(time_str):
    # Parse the string to a datetime object
    time_obj = datetime.strptime(time_str, '%H:%M:%S')

    # Add two hours
    new_time_obj = time_obj + timedelta(hours=2)

    # Format back to a string
    new_time_str = new_time_obj.strftime('%H:%M:%S')

    return new_time_str


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


def _adjust_date_forw(original_date, days_to_add):
    nyse = mcal.get_calendar('NYSE')
    new_date = original_date + pd.Timedelta(days=days_to_add)
    trading_days = nyse.valid_days(start_date=new_date, end_date=original_date)

    if not trading_days.empty:
        adjusted_date = trading_days[0].date().strftime("%Y-%m-%d")
        if adjusted_date == original_date.strftime("%Y-%m-%d"):
            return _adjust_date_forw(original_date, days_to_add + 1)
        else:
            return adjusted_date
    else:
        return "No valid trading days found"


def adjust_date_forward(date_string, days_to_add):
    date = pd.to_datetime(date_string)
    return _adjust_date_forw(date, days_to_add)


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