import pandas as pd
import pytz
from pytz import timezone
import journal
import ctxcapmd
from datetime import date, datetime
import logging
import pandas_market_calendars as mcal


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


def get_intraday(ticker, date, bar_type):
    """
    function to get intraday price information for a ticker on given trade.date
    :param ticker: stock
    :param date: in 'YYYY-MM-DD format'
    :param bar_type: 'bar-5s', 'bar-3min', etc.
    :return: df containing price data
    """
    converted_date = datetime.strptime(date, '%Y-%m-%d').date()
    aggs = []
    with ctxcapmd.Session('10.195.0.102', 65500, journal.any_decompress) as session:
        def append_object(obj):
            aggs.append(obj)

        handle = session.request_data(append_object, ticker, converted_date, converted_date, [bar_type])

        handle.wait()

        handle.raise_on_error()

    df = pd.DataFrame(aggs)
    try:
        df['close-time'] = pd.to_datetime(df['close-time'], unit='ns')
        df['close-time'] = df['close-time'].dt.tz_localize('UTC').dt.tz_convert(timezone('US/Eastern'))
        df.set_index('close-time', inplace=True)
    except KeyError:
        pass
    return df


def get_daily(ticker, date):
    """
    Function to get daily price information for a ticker on given trade date.
    :param ticker: Ticker symbol of the stock
    :param date: Date in the format 'YYYY-MM-DD'
    :return: Daily open, close, high, low volume for a given ticker and date
    """
    # Convert date to datetime and set to Eastern Time zone
    eastern = pytz.timezone('US/Eastern')
    open_time = eastern.localize(datetime.strptime(date + ' 09:30:00', '%Y-%m-%d %H:%M:%S'))
    close_time = eastern.localize(datetime.strptime(date + ' 16:00:00', '%Y-%m-%d %H:%M:%S'))

    # Get intraday data
    df = get_intraday(ticker, date, 'bar-1day')

    df.index = df.index.tz_convert(eastern)

    # Get open, close, high, and low prices
    open_price = df['open'].iloc[0]
    close_price = df['close'].iloc[0]
    high_price = df['high'].iloc[0]
    low_price = df['low'].iloc[0]
    volume = df['volume'].iloc[0]
    return {'open': open_price, 'close': close_price, 'high': high_price, 'low': low_price, 'volume' : volume}


def get_vwap(ticker, date):
    """
    Function to get the volume weighted average price for a given ticker
    :param ticker: Ticker symbol of the stock
    :return: VWAP for the given ticker
    """
    converted_date = datetime.strptime(date, '%Y-%m-%d').date()
    aggs = []
    with ctxcapmd.Session('10.195.0.102', 65500, journal.any_decompress) as session:
        def append_object(obj):
            aggs.append(obj)

        handle = session.request_data(append_object, ticker, converted_date, converted_date, ['vwma-1h'])

        handle.wait()

        handle.raise_on_error()
    df = pd.DataFrame(aggs)
    df['time'] = pd.to_datetime(df['time'], unit='ns')
    df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert(timezone('US/Eastern'))
    df.set_index('time', inplace=True)
    return df.iloc[-1].value


def get_mav_data(df):
    mav_data = {}
    for window in ['10D', '20D', '50D', '200D']:
        data = df[window]
        if data:
            mav_data[f'price_{window.lower()}mav'] = data['SMA']
        else:
            mav_data[f'price_{window.lower()}mav'] = None
    return mav_data


def get_levels_data(ticker, date, window, bar_type):
    converted_date = datetime.strptime(date, '%Y-%m-%d').date()
    adjusted_date = adjust_date_to_market(date, window)
    adjusted_date = datetime.strptime(adjusted_date, '%Y-%m-%d').date()

    aggs = []
    with ctxcapmd.Session('10.195.0.102', 65500, journal.any_decompress) as session:
        def append_object(obj):
            aggs.append(obj)

        handle = session.request_data(append_object, ticker, adjusted_date, converted_date, [bar_type])

        handle.wait()

        handle.raise_on_error()

    df = pd.DataFrame(aggs)
    df['close-time'] = pd.to_datetime(df['close-time'], unit='ns')
    df['close-time'] = df['close-time'].dt.tz_localize('UTC').dt.tz_convert(timezone('US/Eastern'))
    df.set_index('close-time', inplace=True)
    return df


from tabulate import tabulate

if __name__ == '__main__':
    get_vwap('AAPL', '2024-02-07')
    # print(get_daily('AAPL', '2024-02-07'))
    df = get_intraday('AAPL', '2024-02-07', 'bar-1min')
    # get_mav_data('AAPL', '2024-02-07')
    print(tabulate(df, headers=df.columns))
