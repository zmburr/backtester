from data_queries.polygon_queries import get_daily, adjust_date_forward, get_levels_data, get_price_with_fallback, \
    adjust_date_to_market, get_intraday, check_pct_move, fetch_and_calculate_volumes, get_ticker_mavs_open, get_range_vol_expansion_data
import pandas as pd
import logging
from ref_price import GetReferencePrice
from tabulate import tabulate
from datetime import datetime, timedelta
import math
from xbbg import blp
from dateutil import tz

#max size limits
small_cap_max_size = 190000
large_cap_max_size = 130000
small_cap_BP = small_cap_max_size * 10
single_stock_BP = 11000000
total_BP = 12000000
lockout = 75000
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
def get_ism_intraday(ticker, date):
    """
    Uses blpapi to get intraday candles.

    :param ticker: US equity ticker
    :param date: %Y-%m-%d string representation of date
    :return: a dataframe object containing each minute's open, high, low, and close for [ticker] on [date]
    """
    logger.info(f'getting intraday data for ticker: {ticker}, date: {date}')
    df = blp.bdib(ticker + " US Equity", dt=date)
    try:
        df = df.droplevel(level=0, axis=1).drop(columns=['volume', 'num_trds', 'value'])
    except ValueError:
        print('bloomberg data doesnt exist')
    if df.empty:
        df = get_intraday(ticker, date, 2, "minute")
        df = df.drop(columns=['volume', 'vwap', 'transactions', 'otc'])
    dt_date = datetime.strptime(date, "%Y-%m-%d")
    df = df[df.index >= dt_date.replace(hour=9, minute=30).strftime("%Y-%m-%d %H:%M:%S")]
    df = df[df.index <= dt_date.replace(hour=16).strftime("%Y-%m-%d %H:%M:%S")]
    return df
def floor_time(time, interval):
    time = pd.Timestamp(time)
    return time.floor(freq='{}min'.format(str(interval)))
def get_datetime(time):
    try:
        return datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.strptime(time, "%m/%d/%Y %H:%M")
def convert_1min_intraday(df, interval):
    converted = df.groupby(pd.Grouper(freq='{}min'.format(str(interval)))).agg({"open": "first",
                                                                                "high": "max",
                                                                                "low": "min",
                                                                                "close": "last"})
    converted.columns = ["open", "high", "low", "close"]
    return converted
def get_rally_dataframe(df, start, side, interval):
    """

    :param df:
    :param start:
    :param side:
    :param interval:
    :return:
    """
    start = (floor_time(get_datetime(start), interval) - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
    df = df[df.index > start]

    # # this code will assume strategy fcks you over if you buy on the bottom/sell on top
    # start_minus_one = floor_time((floor_time(get_datetime(start), interval) - timedelta(minutes=1)), interval)
    # start_minus_one = start_minus_one.strftime("%Y-%m-%d %H:%M:%S")
    # df = df[df.index >= start_minus_one]

    if side == 1:
        df["prev_low"] = df["low"].shift(1)
        df["delaystrat_trend"] = (df["close"] > df["prev_low"])
        df["quickstrat_trend"] = (df['low'] > df["prev_low"])
        df = df[df["prev_low"].notna()]  # ASSUMES YOU WAIT FOR CANDLE YOU BROUGHT IN AT TO COMPLETE
    else:
        df["prev_high"] = df['high'].shift(1)
        df["delaystrat_trend"] = (df["close"] < df["prev_high"])
        df["quickstrat_trend"] = (df["high"] < df["prev_high"])
        df = df[df["prev_high"].notna()]  # ASSUMES YOU WAIT FOR CANDLE YOU BROUGHT IN AT TO COMPLETE

    return df
def get_delaystrat_exit(df):
    """

    :param df: processed
    :return:
    """
    countertrends = df.loc[df["delaystrat_trend"] == False]
    if countertrends.empty:
        time = df.index[-1]
        price = df.iloc[-1]["close"]
    else:
        time = countertrends.index[0]
        price = countertrends.iloc[0]["close"]
    return time, round(price, 2)
def get_quickstrat_exit(df, side):
    quickstrat_offset = 0.01
    countertrends = df.loc[df["quickstrat_trend"] == False]
    if countertrends.empty:
        time = df.index[-1]
        price = df.iloc[-1]["close"]
    else:
        time = countertrends.index[0]
        if side == 1:
            price = countertrends.iloc[0]["prev_low"] - quickstrat_offset
        else:
            price = countertrends.iloc[0]["prev_high"] + quickstrat_offset
    return time, round(price, 2)
def run_exitstrategy(ticker, start, side):
    """

    :param ticker: ticker of US Equity
    :param start: string representation of trade start time in "%Y-%m-%d %H:%M:%S" format
    :param side: 1 if long, -1 if short
    :return:
    """

    # get dataframe of intraday data for ticker on start date
    logger.info(f"Processing Symbol: {ticker}")

    try:
        intraday_data = get_ism_intraday(ticker, start.split()[0])
    except AttributeError:
        start = start.strftime("%Y-%m-%d %H:%M:%S")
        intraday_data = get_ism_intraday(ticker, start.split()[0])
    results = []

    # determines exit time and price using exit strategies
    intervals = [2]
    for i in intervals:
        df = convert_1min_intraday(intraday_data, i)
        df = get_rally_dataframe(df, start, side, i)

        delay_exit_data = get_delaystrat_exit(df)
        quick_exit_data = get_quickstrat_exit(df, side)
        exit_data = (delay_exit_data, quick_exit_data)
        results.append(exit_data)
        logger.info(f'symbol: {ticker} results: {exit_data}')
    return results
def calculate_potential_size(price, Side):
    if price <= 10 and Side == 1:
        max_size = small_cap_max_size
    elif price <= 10 and Side == -1:
        max_size = large_cap_max_size
    elif price < 15:
        max_size = math.floor(small_cap_BP / price)
    elif price < 55:
        max_size = large_cap_max_size
    else:
        max_size = single_stock_BP / price
    return max_size

def calculate_metrics(data, ticker, trade_date):
    """
    Calculates 2-minute delayed and quick returns using run_exitstrategy.

    1) We fix start time at 10:00:00 on the given trade_date.
    2) We find the first bar at or after 10:00, determine side:
         side = +1 if bar's close > bar's open
                 -1 otherwise
    3) We run run_exitstrategy(ticker, f"{trade_date} 10:00:00", side).
    4) We parse the delayed- and quick-exit data from run_exitstrategy,
       compute returns (in %), and store them as columns in `data`.
    """

    # If no data, return None
    if data is None or data.empty:
        return None

    # Ensure trade_date is in YYYY-MM-DD if needed
    # If 'trade_date' is already 'YYYY-MM-DD', skip this step
    date_str = pd.to_datetime(trade_date).strftime("%Y-%m-%d")
    start_str = f"{date_str} 10:00:00"
    start = pd.to_datetime(start_str)

    # Filter to rows at or after 10:00
    after_10am = data.loc[data.index >= start_str]
    if after_10am.empty:
        # No bars after 10:00 => no calculations
        data['2m_delayed_return'] = float('nan')
        data['2m_quick_return'] = float('nan')
        return data

    # --- 1) Determine side from the first 2-minute bar after 10:00 ---
    first_bar_idx = after_10am.index[0]
    first_open = data.loc[first_bar_idx, 'open']
    first_close = data.loc[first_bar_idx, 'close']
    side = 1 if (first_close > first_open) else -1

    reference = GetReferencePrice(ticker, date_str, start, side).get_reference_price()

    # --- 2) Run exit strategy to get delayed and quick exits ---
    # run_exitstrategy returns a list of results (one per interval),
    # with each element containing a tuple for (delayed_exit) and (quick_exit).
    # By default, we only have intervals=[2], so there's a single element: results[0]
    results = run_exitstrategy(ticker, start_str, side)
    if not results:
        # In case run_exitstrategy returns an empty list
        data['2m_delayed_return'] = float('nan')
        data['2m_quick_return'] = float('nan')
        return data

    # results[0] => ((delay_exit_time, delay_exit_price), (quick_exit_time, quick_exit_price))
    (delay_exit_time, delay_exit_price), (quick_exit_time, quick_exit_price) = results[0]

    # --- 3) Compute returns from the "entry price" to each exit price ---
    # We define "entry price" as the close of the first bar after 10:00
    entry_price = reference['ref_price']

    # Delayed exit return: side * (exit_price - entry_price)/entry_price * 100
    delayed_return_pct = side * (delay_exit_price - entry_price) / entry_price * 100.0
    # Quick exit return: same logic
    quick_return_pct = side * (quick_exit_price - entry_price) / entry_price * 100.0

    # --- 4) Store results in columns for the entire DataFrame or just for the post-10:00 portion ---
    # Often people store these as single values (e.g., repeated across all rows).
    data['2m_delayed_return'] = delayed_return_pct
    data['2m_quick_return'] = quick_return_pct

    # Optionally, if you also want exit timestamps:
    data['2m_delayed_exit_time'] = str(delay_exit_time)
    data['2m_quick_exit_time'] = str(quick_exit_time)

    return data

def process_tickers(tickers, release_dates):
    results = []

    for ticker in tickers:
        for release in release_dates:
            release_type = release["type"]
            date_str = release["date"]  # e.g., "01/03/2023"
            formatted_date = datetime.strptime(date_str, "%m/%d/%Y").strftime("%Y-%m-%d")

            print(f"Processing {ticker} for {release_type} on {date_str}...")
            # 1) Fetch intraday data
            data = get_intraday(ticker, formatted_date, multiplier=1, timespan='minute')
            if data is None or data.empty:
                continue

            # 2) Convert to 2-minute aggregates if you want (or just keep 1-min data)
            # data = convert_1min_intraday(data, 2)

            # 3) Calculate metrics with run_exitstrategy inside
            processed_data = calculate_metrics(data, ticker, date_str)

            if processed_data is not None and not processed_data.empty:
                processed_data['ticker'] = ticker
                processed_data['date'] = date_str
                processed_data['release_type'] = release_type
                results.append(processed_data)

    if results:
        combined_df = pd.concat(results)
        return combined_df
    else:
        return None

if __name__ == "__main__":
    tickers = ["QQQ", "SPY", "IWM", "TNA", "SPXL", "TQQQ", "SOXL"]

    release_dates = [
        {"type": "Manufacturing", "date": "01/03/2023"},
        {"type": "Services", "date": "01/05/2023"},
        {"type": "Manufacturing", "date": "02/01/2023"},
        {"type": "Services", "date": "02/03/2023"},
        {"type": "Manufacturing", "date": "03/01/2023"},
        {"type": "Services", "date": "03/03/2023"},
        {"type": "Manufacturing", "date": "04/03/2023"},
        {"type": "Services", "date": "04/05/2023"},
        {"type": "Manufacturing", "date": "05/01/2023"},
        {"type": "Services", "date": "05/03/2023"},
        {"type": "Manufacturing", "date": "06/01/2023"},
        {"type": "Services", "date": "06/05/2023"},
        {"type": "Manufacturing", "date": "07/03/2023"},
        {"type": "Services", "date": "07/05/2023"},
        {"type": "Manufacturing", "date": "08/01/2023"},
        {"type": "Services", "date": "08/03/2023"},
        {"type": "Manufacturing", "date": "09/01/2023"},
        {"type": "Services", "date": "09/05/2023"},
        {"type": "Manufacturing", "date": "10/02/2023"},
        {"type": "Services", "date": "10/04/2023"},
        {"type": "Manufacturing", "date": "11/01/2023"},
        {"type": "Services", "date": "11/03/2023"},
        {"type": "Manufacturing", "date": "12/01/2023"},
        {"type": "Services", "date": "12/05/2023"},
        {"type": "Manufacturing", "date": "01/03/2024"},
        {"type": "Services", "date": "01/05/2024"},
        {"type": "Manufacturing", "date": "02/01/2024"},
        {"type": "Services", "date": "02/05/2024"},
        {"type": "Manufacturing", "date": "03/01/2024"},
        {"type": "Services", "date": "03/05/2024"},
        {"type": "Manufacturing", "date": "04/01/2024"},
        {"type": "Services", "date": "04/03/2024"},
        {"type": "Manufacturing", "date": "05/01/2024"},
        {"type": "Services", "date": "05/03/2024"},
        {"type": "Manufacturing", "date": "06/03/2024"},
        {"type": "Services", "date": "06/05/2024"},
        {"type": "Manufacturing", "date": "07/01/2024"},
        {"type": "Services", "date": "07/03/2024"},
        {"type": "Manufacturing", "date": "08/01/2024"},
        {"type": "Services", "date": "08/05/2024"},
        {"type": "Manufacturing", "date": "09/03/2024"},
        {"type": "Services", "date": "09/05/2024"},
        {"type": "Manufacturing", "date": "10/01/2024"},
        {"type": "Services", "date": "10/03/2024"},
        {"type": "Manufacturing", "date": "11/01/2024"},
        {"type": "Services", "date": "11/05/2024"},
        {"type": "Manufacturing", "date": "12/02/2024"},
        {"type": "Services", "date": "12/04/2024"}
    ]

    processed_data = process_tickers(tickers, release_dates)

    if processed_data is not None:
        output_file = "C:\\Users\\zmbur\\PycharmProjects\\backtester\\data\\ism_data.csv"
        processed_data.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    else:
        print("No data processed.")

