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
    df = get_levels_data(ticker, date, 30, 1, 'day')
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

def get_ticker_mavs_open(ticker, date):
    results = {}

    try:
        # Get the open price for the given date
        open_price = poly_client.get_daily_open_close_agg(ticker, date).open
    except (AttributeError, IndexError, TypeError) as e:
        print(f"Open price missing or unavailable for {ticker} on {date}: {e}")
        return None  # Open price is essential; return None if it's missing.

    # Helper function to get moving average and calculate the percentage difference
    def calculate_pct_mav(window):
        try:
            mav = poly_client.get_sma(
                ticker=ticker, timestamp=date, timespan='day', adjusted=True, window=window, series_type='close'
            ).values[0].value
            return (open_price - mav) / mav
        except (AttributeError, IndexError, TypeError) as e:
            print(f"Moving average (window={window}) missing or unavailable for {ticker} on {date}: {e}")
            return None

    # Calculate percentage differences for each moving average
    results['pct_from_10mav'] = calculate_pct_mav(10)
    results['pct_from_20mav'] = calculate_pct_mav(20)
    results['pct_from_50mav'] = calculate_pct_mav(50)
    results['pct_from_200mav'] = calculate_pct_mav(200)

    # Filter out None values
    results = {key: value for key, value in results.items() if value is not None}

    if not results:
        print(f"No data available for any moving averages for {ticker} on {date}.")
        return None

    return results


def fetch_and_calculate_volumes(ticker, date):
    # Assuming get_intraday and get_levels_data are defined elsewhere and fetch data from an API
    logging.info(f'Fetching and calculating volumes for {ticker} on {date}')
    data = get_intraday(ticker, date, multiplier=1, timespan='minute')
    adv = get_levels_data(ticker, date, 30, 1, 'day')

    # Calculate metrics
    metrics = {
        'avg_daily_vol': adv['volume'].sum() / len(adv['volume']) if len(adv['volume']) > 0 else 0,
        'vol_on_breakout_day': data['volume'].sum(),
        'premarket_vol': data.between_time('06:00:00', '09:30:00')['volume'].sum(),
        'vol_in_first_5_min': data.between_time('09:30:00', '09:35:00')['volume'].sum(),
        'vol_in_first_15_min': data.between_time('09:30:00', '09:45:00')['volume'].sum(),
        'vol_in_first_10_min': data.between_time('09:30:00', '09:40:00')['volume'].sum(),
        'vol_in_first_30_min': data.between_time('09:30:00', '10:00:00')['volume'].sum(),
        'vol_two_day_before': adv.iloc[-3].volume,
        'vol_one_day_before': adv.iloc[-2].volume,
        'vol_three_day_before': adv.iloc[-4].volume
    }

    return metrics


def get_range_vol_expansion_data(ticker, date):
    import logging
    logging.info(f'Fetching and calculating range expansion data for {ticker} on {date}')
    df = get_levels_data(ticker, date, 40, 1, 'day')

    # Calculate True Range (TR) components
    df['high-low'] = df['high'] - df['low']
    df['high-previous_close'] = abs(df['high'] - df['close'].shift())
    df['low-previous_close'] = abs(df['low'] - df['close'].shift())

    # Calculate the True Range (TR)
    df['TR'] = df[['high-low', 'high-previous_close', 'low-previous_close']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14, min_periods=1).mean()

    # Calculate the percentage of ATR
    df['PCT_ATR'] = df['TR'] / df['ATR']

    # Adjust the rolling window for volume if there are fewer than 20 rows
    rolling_window = min(len(df)-4, 20)
    df['30Day_Avg_Volume'] = df['volume'].rolling(window=rolling_window).mean()
    df['pct_avg_volume'] = df['volume'] / df['30Day_Avg_Volume']
    print(df)
    # Ensure there are enough rows to fetch data for the past three days
    if len(df) < 4:
        logging.error(f"Not enough data for {ticker} on {date}.")
        return None

    # Get the latest range and percentage of ATR
    pct_of_atr = df['PCT_ATR'].iloc[-1]
    day_before_pct_of_atr = df['PCT_ATR'].iloc[-2]
    two_d_before_pct_of_atr = df['PCT_ATR'].iloc[-3]
    three_d_before_pct_of_atr = df['PCT_ATR'].iloc[-4]

    # Construct the metrics dictionary
    metrics = {
        'day_of_range_pct': pct_of_atr,
        'one_day_before_range_pct': day_before_pct_of_atr,
        'two_day_before_range_pct': two_d_before_pct_of_atr,
        'three_day_before_range_pct': three_d_before_pct_of_atr,
        'percent_of_vol_one_day_before': df['pct_avg_volume'].iloc[-2],
        'percent_of_vol_two_day_before': df['pct_avg_volume'].iloc[-3],
        'percent_of_vol_three_day_before': df['pct_avg_volume'].iloc[-4]
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
    if days_to_subtract == 3:
        if len(trading_days) < days_to_subtract:
            return _adjust_date(original_date, days_to_subtract + 1)
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
            # Fetch the adjusted market date and get the daily price
            adjusted_date = adjust_date_to_market(base_date, days_ago)
            price = get_daily(ticker, adjusted_date).close
            if price is not None:  # Check if the price exists
                return price
        except Exception as e:
            # Optional: Log the exception for debugging
            # print(f"Error fetching price for {ticker} on {adjusted_date}: {e}")
            pass  # Continue decrementing days if an exception occurs
        days_ago -= 1
    return 0  # or handle this case as needed


def get_ticker_pct_move(ticker, date, current_price):
    price_120dago = get_price_with_fallback(ticker, date, 120)
    price_90dago = get_price_with_fallback(ticker, date, 90)
    price_30dago = get_price_with_fallback(ticker, date, 30)
    price_15dago = get_price_with_fallback(ticker, date, 15)
    price_3dago = get_price_with_fallback(ticker, date, 3)

    pct_change_120 = (current_price - price_120dago) / price_120dago if price_120dago else None
    pct_change_90 = (current_price - price_90dago) / price_90dago if price_90dago else None
    pct_change_30 = (current_price - price_30dago) / price_30dago if price_30dago else None
    pct_change_15 = (current_price - price_15dago) / price_15dago if price_15dago else None
    pct_change_3 = (current_price - price_3dago) / price_3dago if price_3dago else None

    return {
        "pct_change_120": pct_change_120,
        "pct_change_90": pct_change_90,
        "pct_change_30": pct_change_30,
        "pct_change_15": pct_change_15,
        "pct_change_3": pct_change_3
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
    try:
        data = get_intraday(ticker, datetime.now().strftime('%Y-%m-%d'), 1, 'second')
        price = data.iloc[-1].close
        if price:
            return price
        else:
            data = get_intraday(ticker, datetime.now().strftime('%Y-%m-%d'), 1, 'minute')
            price = data.iloc[-1].close
            return price
    except AttributeError:
        return None

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
