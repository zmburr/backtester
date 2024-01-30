import pandas as pd
from datetime import datetime, timedelta
from polygon.rest import RESTClient
import pandas_market_calendars as mcal
import os
from data_collectors.momentum_data_collection import get_intraday
import logging

from tabulate import tabulate

exit_dict = {'one': 1,
             'two': 2,
             'three': 3,
             'four': 4,
             'five': 5}

poly_client = RESTClient(api_key="b_s_dRysgNN_kZF_nzxwSLdvClTyopGgxtJSqX")

df = pd.read_csv("C:\\Users\\zmbur\\PycharmProjects\\InOffice\\data\\breakout_data.csv")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

signal = 'premarket_high_breakout'
exit_bars = 'five'
exit_strategy = 'delayed_exit'
stop_strategy = 'premarket_low'
time_delay = 0
on_close = False


def find_signal_time(full_data, signal):
    if signal == 'premarket_high_breakout':
        premarket_high = full_data.between_time('06:00:00', '09:29:59').high.max()
        data = full_data.between_time('09:30:00', '16:00:00')
        # print(tabulate(data, headers=data.columns))
        breakout_row = data[data['close'] > premarket_high].first_valid_index()
        index_pos = data.index.get_loc(breakout_row)
        return data.iloc[index_pos - 1]
    if signal == 'open_price_breakout':
        open_price = full_data.between_time('09:30:00', '16:00:00').iloc[0].open
        data = full_data.between_time('09:30:00', '16:00:00')
        # print(tabulate(data, headers=data.columns))
        breakout_row = data[data['close'] > open_price].first_valid_index()
        index_pos = data.index.get_loc(breakout_row)
        return data.iloc[index_pos - 1]


def find_stop_price(df, signal_time):
    df = df.between_time('06:00:00', signal_time)
    if stop_strategy == 'low_of_day':
        df = df.between_time('09:30:00', signal_time)
        return df.low.min()
    elif stop_strategy == 'premarket_low':
        return df.between_time('6:00:00', '09:29:59').low.min()
    elif stop_strategy == 'prev_day_low':
        return df.between_time('16:00:00', '23:59:59').low.min()


def backtest_trade(row):
    stopped_out = False
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = wrong_date.strftime('%Y-%m-%d')
    full_data = get_intraday(row['ticker'], date, multiplier=5, timespan='second')
    signal_row = find_signal_time(full_data, signal)
    entry_price = signal_row['close']

    initial_signal_time = datetime.strftime(signal_row.name, '%Y-%m-%d %H:%M:%S')

    delayed_signal_time = signal_row.name + timedelta(minutes=time_delay)
    delayed_signal_time = datetime.strftime(delayed_signal_time, '%Y-%m-%d %H:%M:%S')

    stop_price = find_stop_price(full_data, initial_signal_time[11:])
    full_stop_data = full_data.between_time(initial_signal_time[11:], delayed_signal_time[11:])
    if full_stop_data['low'].min() < stop_price:
        stopped_out = True

    profit_data = get_intraday(row['ticker'], date, multiplier=exit_dict[exit_bars], timespan='minute').between_time(
        delayed_signal_time[11:], '16:00:00')

    trade = {'headline_time': initial_signal_time,
             'ticker': row['ticker'],
             'date': date,
             'ref_price': signal_row['close'],
             'position_size': 1000
             }
    pct_captured = PercentCaptured(trade).compile_pct_captured_dict()

    if on_close:
        exit_price = profit_data.iloc[-1].close if stopped_out is False else stop_price
    else:
        exit_price = pct_captured[exit_bars + '_min'][exit_strategy][1] if stopped_out is False else stop_price

    net_pnl = (stop_price - entry_price) * trade['position_size'] if stopped_out is True else (
                                                                                                      exit_price - entry_price) * \
                                                                                              trade['position_size']

    trade_details = {
        'ticker': row['ticker'],
        'date': date,
        'net_pnl': net_pnl,
        'stop_price': stop_price,
        'exit_price': exit_price,
        'signal_time': initial_signal_time,
        'entry_price': entry_price,
        'high_price': profit_data['high'].max(),
        'stop_strategy': stop_strategy,
        'bar_type': exit_bars,
        'exit_strategy': exit_strategy,
        'signal': signal,
        'stopped_out': stopped_out,
        'time_delay': time_delay
    }

    return trade_details


trades = []

if __name__ == '__main__':
    try:
        for index, row in df.iterrows():
            trade_details = backtest_trade(row)
            print(trade_details)
            trades.append(trade_details)
    except:
        print(f'Error on {row["ticker"]} on {row["date"]}')
    # Create a DataFrame from the collected trade details
    trades_df = pd.DataFrame(trades)
    print(trades_df)
    trades_df.to_csv(f'C:\\Users\\zmbur\\PycharmProjects\\InOffice\\data\\{time_delay}_backtest_results.csv')
