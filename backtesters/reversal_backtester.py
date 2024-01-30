import pandas as pd
from datetime import datetime, timedelta
from data_queries.polygon_queries import get_intraday
import logging

exit_dict = {'one': 1,
             'two': 2,
             'three': 3,
             'four': 4,
             'five': 5}


df = pd.read_csv("C:\\Users\\zmbur\\PycharmProjects\\InOffice\\data\\reversal_data.csv")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

exit_bars = 'two'
exit_strategy = 'quick_exit'
stop_strategy = None
time_delay = 0
on_close = False


def check_premarket_low(full_data, end_time):
    premarket_low = full_data.between_time('06:00:00', '09:29:59').low.min()
    df = full_data.between_time('09:30:00', end_time)
    breakout_row = df[df['close'] < premarket_low].first_valid_index()
    try:
        index_pos = df.index.get_loc(breakout_row)
        return df.iloc[index_pos - 1]
    except KeyError:
        return None


def get_return(full_data, end_time):
    df = full_data.between_time('09:30:00', end_time)
    return df.iloc[-1].close - df.iloc[0].open


def find_signal_time(full_data, signal):
    if signal == 'premarket_low_break':
        low_break_row = check_premarket_low(full_data, '16:00:00')
        return low_break_row
    if signal == 'open_price_break':
        open_price = full_data.between_time('09:30:00', '16:00:00').iloc[0].open
        data = full_data.between_time('09:30:00', '16:00:00')
        # print(tabulate(data, headers=data.columns))
        breakout_row = data[data['close'] < open_price].first_valid_index()
        index_pos = data.index.get_loc(breakout_row)
        return data.iloc[index_pos - 1]
    if signal == 'two_min_break':
        data = full_data.between_time('09:44:00', '16:00:00')
        breakout_row = data[data['close'] < data['low'].shift(1)].first_valid_index()
        # print(tabulate(data, headers=data.columns))
        index_pos = data.index.get_loc(breakout_row)
        return data.iloc[index_pos]


def find_stop_price(df, signal_time):
    global stop_strategy
    df = df.between_time('06:00:00', signal_time)
    high_of_day = df.between_time('09:30:00', signal_time).high.max()
    premarket_high = df.between_time('6:00:00', '09:29:59').high.max()
    if high_of_day > premarket_high:
        stop_strategy = 'high_of_day'
        return high_of_day * 1.001
    else:
        stop_strategy = 'premarket_high'
        return premarket_high * 1.001


def get_time_of_low(df):
    return df['low'].idxmin()


def backtest_trade(row):
    stopped_out = False
    wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
    date = wrong_date.strftime('%Y-%m-%d')
    full_data = get_intraday(row['ticker'], date, multiplier=5, timespan='second')
    time_of_low = get_time_of_low(full_data)
    time_of_low = datetime.strftime(time_of_low, '%Y-%m-%d %H:%M:%S')

    five_min_return = get_return(full_data, '09:35:00')
    fifteen_min_return = get_return(full_data, '09:45:00')

    pre_low_break = check_premarket_low(full_data, '09:45:00')
    all_day_pre_low_break = check_premarket_low(full_data, '16:00:00')
    pre_low_break_time = datetime.strftime(all_day_pre_low_break.name, '%Y-%m-%d %H:%M:%S')

    if fifteen_min_return > 0 and pre_low_break is None:
        signal = 'two_min_break'
        data = get_intraday(row['ticker'], date, multiplier=2, timespan='minute')
        signal_row = find_signal_time(data, 'two_min_break')
        entry_price = signal_row['close']
    else:
        signal = 'premarket_low_break'
        signal_row = find_signal_time(full_data, signal)
        entry_price = signal_row['close']

    initial_signal_time = datetime.strftime(signal_row.name, '%Y-%m-%d %H:%M:%S')
    delayed_signal_time = signal_row.name + timedelta(minutes=2)
    delayed_signal_time = datetime.strftime(delayed_signal_time, '%Y-%m-%d %H:%M:%S')

    stop_price = find_stop_price(full_data, initial_signal_time[11:])
    if signal == 'two_min_break':
        full_stop_data = full_data.between_time(initial_signal_time[11:], pre_low_break_time[11:])
    else:
        full_stop_data = full_data.between_time(initial_signal_time[11:], time_of_low[11:])

    # print(stop_price, full_stop_data.iloc[1:]['close'].max())
    # print(tabulate(full_stop_data, headers=full_stop_data.columns))
    if full_stop_data.iloc[1:]['close'].max() > stop_price:
        stopped_out = True

    profit_data = get_intraday(row['ticker'], date, multiplier=exit_dict[exit_bars], timespan='minute').between_time(
        delayed_signal_time[11:], '16:00:00')

    trade = {'headline_time': delayed_signal_time,
             'ticker': row['ticker'],
             'date': date,
             'ref_price': signal_row['close'],
             'position_size': -1000
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
        'low_price': profit_data['low'].min(),
        'stop_strategy': stop_strategy,
        'bar_type': exit_bars,
        'exit_strategy': exit_strategy,
        'signal': signal,
        'stopped_out': stopped_out,
        'time_delay': time_delay,
        'five_min_return': five_min_return,
        'fifteen_min_return': fifteen_min_return
    }

    return trade_details


trades = []

if __name__ == '__main__':
    for index, row in df.iterrows():
        try:
            # if row['ticker'] == 'AVGO':
            trade_details = backtest_trade(row)
            trades.append(trade_details)
        except:
            print(f'Error on {row["ticker"]} on {row["date"]}')
    # Create a DataFrame from the collected trade details
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(
        f'C:\\Users\\zmbur\\PycharmProjects\\InOffice\\data\\{exit_bars}_{exit_strategy}_mix_backtest_results.csv')
