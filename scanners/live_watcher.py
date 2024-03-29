import threading
from datetime import datetime, timedelta
import warnings
import logging
import gtts
import os
from playsound import playsound
from data_queries.polygon_queries import get_levels_data, get_atr
from data_queries.trillium_queries import adjust_date_to_market
from scipy.stats import percentileofscore
from data_collectors.combined_data_collection import reversal_df, momentum_df
from stock_screener import get_stock_data
from tabulate import tabulate

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import pytz
from data_queries.trillium_queries import get_daily


def play_sounds(text):
    try:
        tts = gtts.gTTS(text)
        tempfile = "C:\\Users\\zmbur\\PycharmProjects\\backtester\\scanners\\temps.mp3"
        tts.save(tempfile)
        playsound(tempfile, block=False)
        os.remove(tempfile)
    except:
        print("could not play sound")


def get_time_elapsed(headline_time, current_time):
    """
    Function to calculate time elapsed since headline time and changes instance variable as time goes on.
    :return: timedelta: difference between headline time and current time
    """
    then_time = datetime.strptime(headline_time, '%Y-%m-%d %H:%M:%S')
    # now_time = datetime.strptime(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    now_time = datetime.strptime(datetime.strftime(current_time, '%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    diff = now_time - then_time
    return diff


def convert_timestamp_gmt_to_est(timestamp):
    gmt_time = datetime.fromtimestamp(timestamp, pytz.timezone('GMT'))
    est_time = gmt_time.astimezone(pytz.timezone('US/Eastern'))
    return est_time


def convert_aggs(agg):
    try:
        required_keys = ['close-time', 'open', 'high', 'low', 'close', 'volume', 'vwap']
        for key in required_keys:
            if key not in agg or not isinstance(agg[key], (int, float)):
                logging.error(f"Missing or invalid data for key: {key}")
                return None

        if not isinstance(agg['close-time'], int):
            logging.error("Invalid data type for close-time")
            return None

        unix_timestamp = int(agg['close-time'] / 1e9)
        timestamp = convert_timestamp_gmt_to_est(unix_timestamp)

        series_data = {
            "open": agg['open'],
            "high": agg['high'],
            "low": agg['low'],
            "close": agg['close'],
            "volume": agg['volume'],
            "vwap": agg['vwap']
        }

        series = pd.Series(series_data, name=timestamp)
        return series

    except Exception as e:
        # Handle any errors that occur during the conversion
        logging.error(f"Error in convert_aggs: {e}")
        return None


def concatenate_trade_df(trade_df, new_series):
    """
    Function to check for new data and returns a dataframe with the new data added
    :param trade_df: Dataframe: to append to
    :param new_series: Pandas Series: to append
    :return: trade_df: with added data
    """
    if new_series.name not in trade_df.index:
        trade_df = pd.concat([trade_df, new_series.to_frame().T])
    return trade_df


class TradeManager:
    """
    Class that focuses on managing data used in watchers instances so that each function is looking at the right,
    most up to trade.date data.
    """

    def __init__(self, trade, profit_strategy):
        self.trade = trade
        self.ticker = self.trade.ticker
        self.stock_data = get_stock_data(self.ticker)
        self.start_time = self.trade.start_time
        self.position_size = self.trade.position_size
        self.closed = False
        self.profit_strategy = profit_strategy
        self.handle = None
        self.adv_data = get_levels_data(self.ticker, self.trade.date, 30, 1, 'day')
        self.adv = self.adv_data['volume'].sum() / len(self.adv_data['volume']) if len(
            self.adv_data['volume']) > 0 else 0
        self.logger = logging.getLogger(self.__class__.__name__)
        self.time_elapsed = timedelta(0)
        self.trade_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions', 'otc'])
        self.trade_df.index = pd.to_datetime(self.trade_df.index)
        self.profit_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'vwap'])
        self.profit_df.index = pd.to_datetime(self.profit_df.index)
        self.current_time = None
        self.percentiles = None
        self.stop_focus = None
        self.prior_bar_low = None
        self.prior_bar_high = None
        self.prior_bar_out = None
        self.acceleration = None
        self.pct_volume = None
        self.open_break = False
        self.premarket_break = False
        self.new_data_event = threading.Event()
        self.count = 0
        self.logger.info(f'managing starting for ticker: {self.ticker}')
        self.run_range_watcher()

    def run(self):
        while self.closed is False:
            # waits for data class to flag new data has arrived
            self.new_data_event.wait()
            # adds new data to current trade_df
            self.trade_df = concatenate_trade_df(self.trade_df, self.stop_focus)
            # calculate the time elapsed since headline time
            self.time_elapsed = get_time_elapsed(self.start_time, self.current_time)
            # Watch stops for first x minutes: watch profit taking after x minutes
            if not self.time_elapsed > timedelta(minutes=60):
                if self.count > 1:
                    try:
                        self.run_stop_watcher(position_size=self.position_size, stop_price=self.trade.stop_price,
                                              last_price=self.stop_focus.close, profit_strategy='stopped_out')
                        self.watch_open_price()
                        self.run_volume_watcher()
                        self.watch_premarket()
                        self.get_percentiles(reversal_df if self.trade.side == -1 else momentum_df, self.stock_data, 'percent_of_vol_on_breakout_day')
                    except TypeError:
                        # if TypeError: check for halt as None Type would be supplied for last price
                        self.check_for_halt(self.trade_df)
            else:
                self.prior_bar_out = self.prior_bar_low if self.position_size > 0 else self.prior_bar_high
                try:
                    if self.profit_strategy[6:] == 'close':
                        self.run_stop_watcher(position_size=self.position_size,
                                              last_price=self.profit_df.iloc[-1].close,
                                              stop_price=self.prior_bar_out, profit_strategy=self.profit_strategy)
                    elif self.profit_strategy[6:] == 'quick':
                        self.run_stop_watcher(position_size=self.position_size,
                                              last_price=self.stop_focus.close,
                                              stop_price=self.prior_bar_out, profit_strategy=self.profit_strategy)
                    else:
                        self.logger.info('Invalid Watcher Type - 1, 2, 3, 5 min and quick, close or hybrid')
                        raise TypeError
                except TypeError:
                    self.check_for_halt(self.trade_df)
                except IndexError:
                    print(self.profit_df)
                    pass
            # reset flag - ready for next set of data
            self.new_data_event.clear()
            self.count += 1
        self.logger.info(f'managing done - trade closed for ticker: {self.ticker}')

    def process_incoming_data_for_profit(self, current_data):
        """
        Every two minutes: this function is updated by data object to update the prior bar highs and lows so the code can adjust its stop.
        :return:
        """
        series = convert_aggs(current_data)
        if series is not None:
            self.profit_df = concatenate_trade_df(self.profit_df, series)
            if len(self.profit_df) >= 2:
                self.prior_bar_low = self.profit_df.iloc[-2].low
                self.prior_bar_high = self.profit_df.iloc[-2].high
            else:
                self.logger.info('Not enough bars to adjust profit focus')
            self.logger.info(
                f'adjusted profit focus - new bars: prior bar high: {self.prior_bar_high}, prior bar low: {self.prior_bar_low}')

    def process_incoming_data_for_stop(self, new_series):
        """
        Market data object uses this function to update manager with new data
        :param new_series:
        :return:
        """
        self.logger.debug(f'Received new data - processing - {new_series}')
        series = convert_aggs(new_series)
        if series is not None:
            self.stop_focus = series
            self.current_time = series.name

    def run_stop_watcher(self, position_size, last_price, stop_price, profit_strategy):
        """
        Combined function for watching profit and stop loss levels.
        :param position_size: Size of the position.
        :param last_price: The last traded price.
        :param stop_price: Prior bar's low/high for profit watcher or stop price for stop watcher.
        :param profit_strategy: Type of watcher ('2_min_close' or 'stopped_out').
        :return: String indicating the result of the operation.
        """
        self.logger.info(
            f'Time Elapsed: {self.time_elapsed} Last Price: {last_price}, Stop Price: {stop_price}, Percent Vol: {self.pct_volume}, Percentiles: {self.percentiles}')

        if position_size > 0 and last_price < stop_price:
            self.logger.info(f'STOP TRIGGERED - mkt sell {self.ticker} @ {last_price} after {self.time_elapsed}')
            play_sounds(f'stopped out on {self.trade.ticker}')
            self.close_trade(last_price, self.time_elapsed, profit_strategy)

        elif position_size < 0 and last_price > stop_price:
            self.logger.info(f'STOP TRIGGERED - mkt buy {self.ticker} @ {last_price} after {self.time_elapsed}')
            play_sounds(f'stopped out on {self.trade.ticker}')
            self.close_trade(last_price, self.time_elapsed, profit_strategy)

    def get_percentiles(self, df, stock_data, columns):
        percentiles = {}
        df = df.dropna(subset=columns)
        stock_data = stock_data[self.ticker]        # Ensure stock_data is a DataFrame for easier handling
        if isinstance(stock_data, dict):
            flat_data = {**stock_data, **stock_data['pct_data'], **stock_data['volume_data']}
            flat_data.pop('pct_data', None)
            flat_data.pop('volume_data', None)
            stock_data_df = pd.DataFrame([flat_data])
        else:
            stock_data_df = stock_data
        value = self.pct_volume
        percentiles['percent_of_vol_on_breakout_day'] = percentileofscore(df['percent_of_vol_on_breakout_day'], value, kind='weak')

        self.percentiles = percentiles

    def check_for_halt(self, df):
        """
        Function that checks for lapses in data is Stop Price is None in run function above.
        :param df:
        :return:0
        """
        if df.iloc[-1].name - df.iloc[-2].name > timedelta(seconds=15):
            self.logger.info('Stock Halted')
            self.prior_bar_out = df.iloc[-2].low if self.position_size > 0 else df.iloc[-2].high
            self.run_stop_watcher(position_size=self.position_size, last_price=self.stop_focus.close,
                                  stop_price=self.prior_bar_out, profit_strategy=self.profit_strategy)

    def close_trade(self, last_price, duration, exit_strategy):
        """
        Closes the trade and updates trade information.
        :param trade: Trade instance to be modified.
        :param last_price: The last price for calculating PnL.
        :param duration: Duration of the trade.
        :param exit_strategy: Profit strategy used for the trade.
        :return: None
        """
        self.closed = True
        self.logger.info('cancelling handle')
        self.handle.cancel()
        self.logger.info(f'data ending for ticker: {self.ticker}')
        # self.measure_acceleration(self.trade_df).to_csv(accel_csv_path+f'\\{self.trade.ticker}.csv')

    def measure_acceleration(self, df):
        """
        Code to take in trade_df and calculate price over time. Filters price over time for large moves.
        :param df: trade_df from instance
        :return: filtered df based on price over time calculation
        """
        df['velocity'] = df['close'].diff() / 5
        # code below finds capitulations in price moves
        std = df['velocity'].std()
        mean = df['velocity'].mean()
        if self.position_size < 0:
            threshold = mean - (3 * std)
            sorted_df = df.sort_values('velocity', ascending=True)
            filtered_df = sorted_df[sorted_df['velocity'] <= threshold]
        else:
            threshold = mean + (3 * std)
            sorted_df = df.sort_values('velocity', ascending=True)
            filtered_df = sorted_df[sorted_df['velocity'] >= threshold]
        return filtered_df

    def watch_open_price(self):
        if self.trade.open_price is not None and self.open_break is False:
            if self.trade.side == 1:
                if self.stop_focus.close > self.trade.open_price:
                    play_sounds('open price break')
                    self.open_break = True
                    self.logger.info(f'Open Price Break - {self.ticker} @ {self.stop_focus.close}')
            else:
                if self.stop_focus.close < self.trade.open_price:
                    play_sounds('open price break')
                    self.open_break = True
                    self.logger.info(f'Open Price Break - {self.ticker} @ {self.stop_focus.close}')

    def watch_premarket(self):
        if self.premarket_break is False:
            if self.trade.side == 1:
                if self.stop_focus.close > self.trade.premarket_high:
                    play_sounds('premarket high break')
                    self.premarket_break = True
                    self.logger.info(f'Premarket High Break - {self.ticker} @ {self.stop_focus.close}')
            else:
                if self.stop_focus.close < self.trade.premarket_low:
                    play_sounds('premarket low break')
                    self.premarket_break = True
                    self.logger.info(f'Premarket Low Break - {self.ticker} @ {self.stop_focus.close}')

    def run_volume_watcher(self):
        """
        Function to watch volume - thinking will act as A,B,C,D grading in terms of MSFT(7/23/23) like volume where
        it is extremely high, consistent, and building vs other trades where dissem is fully realized yet volume
        does not come in. Important factor to exit rules.
        :return:
        """
        if self.trade.daily and self.count < 2:
            cumulative_volume = self.trade.daily['volume']
            self.pct_volume = cumulative_volume / self.adv

        else:
            self.trade.daily = get_daily(self.trade.ticker, self.trade.date)
            cumulative_volume = self.trade.daily['volume']
            self.pct_volume = cumulative_volume / self.adv

    def run_key_level_watcher(self):
        """
        Function will watch for key levels/call(put) walls - place orders infront or take note of key level breaks.
        :return:
        """
        pass

    def run_range_watcher(self):
        """
        Function to watch for range expansion.
        :return:
        """
        atr = get_atr(self.ticker, self.trade.date)
        df = get_levels_data(self.ticker, self.trade.date, 60, 1, 'day')
        df['high-low'] = df['high'] - df['low']
        df['high-previous_close'] = abs(df['high'] - df['close'].shift())
        df['low-previous_close'] = abs(df['low'] - df['close'].shift())
        df['TR'] = df[['high-low', 'high-previous_close', 'low-previous_close']].max(axis=1)
        df['PCT_ATR'] = (df['TR'] / atr) * 100
        print(tabulate(df, headers=df.columns))
        range = df['TR'][-1]
        pct_of_atr = (range / atr) * 100
        self.logger.info(f'Range Expansion - {self.ticker}: Percent of ATR: {pct_of_atr} Range:{range} ATR: {atr}')
