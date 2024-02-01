from data_queries.polygon_queries import get_intraday, timestamp_to_string, add_two_hours
import logging
from backtesting_strategies.stop_strategies import stopStrategies


class entrySignals:
    def __init__(self, trade):
        self.trade = trade
        self.data = get_intraday(self.trade.ticker, self.trade.date, multiplier=5, timespan='second')
        self.two_min_data = get_intraday(self.trade.ticker, self.trade.date, multiplier=2, timespan='minute')
        self.search_time = '09:30:00'

        try:
            self.premarket_low_break()
        except KeyError:
            logging.info('No premarket low break found')
        try:
            self.premarket_high_break()
        except KeyError:
            logging.info('No premarket high break found')
        try:
            self.open_price_break()
        except:
            logging.info('No open price break found')
        self.two_min_break()

    def premarket_low_break(self):
        premarket_low = self.data.between_time('06:00:00', '09:29:59').low.min()
        df = self.data.between_time(self.search_time, add_two_hours(self.search_time))
        # Check if the trade side is 1 and five_min_return is less than 0
        if self.trade.side == 1 and (self.trade.contingency_data['five_min_return'] < 0 or self.trade.contingency_data[
            'two_min_return'] < 0):
            breakout_row = df[df['close'] < premarket_low].first_valid_index()
            first_break_index_pos = df.index.get_loc(breakout_row)
            first_break_row = df.iloc[first_break_index_pos]
            if first_break_row is not None:
                new_start_time = timestamp_to_string(first_break_row.name)[11:]
                new_df = df.between_time(new_start_time, add_two_hours(new_start_time))
                signal_breakout_row = new_df[new_df['close'] > premarket_low].first_valid_index()
                if signal_breakout_row is not None:
                    index_pos = df.index.get_loc(signal_breakout_row)
                    row = new_df.iloc[index_pos]
                    self.trade.premarket_low_break_time = timestamp_to_string(row.name)
                    self.trade.premarket_low_break_price = row.close

                    return

        # Original logic for other cases
        breakout_row = df[df['close'] < premarket_low].first_valid_index()
        if breakout_row is not None:
            index_pos = df.index.get_loc(breakout_row)
            row = df.iloc[index_pos]
            self.trade.premarket_low_break_time = timestamp_to_string(row.name)
            self.trade.premarket_low_break_price = row.close

    def premarket_high_break(self):
        premarket_high = self.data.between_time('06:00:00', '09:29:59').high.max()
        df = self.data.between_time(self.search_time, add_two_hours(self.search_time))

        # Check if the trade side is -1 and five_min_return or two_min_return is greater than 0
        if self.trade.side == -1 and (self.trade.contingency_data['five_min_return'] > 0 or self.trade.contingency_data[
            'two_min_return'] > 0):
            breakout_row = df[df['close'] > premarket_high].first_valid_index()
            first_break_index_pos = df.index.get_loc(breakout_row)
            first_break_row = df.iloc[first_break_index_pos]
            if first_break_row is not None:
                new_start_time = timestamp_to_string(first_break_row.name)[11:]
                new_df = df.between_time(new_start_time, add_two_hours(new_start_time))
                signal_breakout_row = new_df[new_df['close'] < premarket_high].first_valid_index()
                if signal_breakout_row is not None:
                    index_pos = new_df.index.get_loc(signal_breakout_row)
                    row = new_df.iloc[index_pos]
                    self.trade.premarket_high_break_time = timestamp_to_string(row.name)
                    self.trade.premarket_high_break_price = row.close
                    return

        # Original logic for other cases
        breakout_row = df[df['close'] > premarket_high].first_valid_index()
        if breakout_row is not None:
            index_pos = df.index.get_loc(breakout_row)
            row = df.iloc[index_pos]
            self.trade.premarket_high_break_time = timestamp_to_string(row.name)
            self.trade.premarket_high_break_price = row.close

    def open_price_break(self):
        # Get the open price of the trading day
        open_price = self.data.between_time('09:30:00', '09:30:00').iloc[0].open
        df = self.data.between_time(self.search_time, add_two_hours(self.search_time))

        # Determine the direction of the trade and identify the first break
        if self.trade.side == 1 and (self.trade.contingency_data['five_min_return'] < 0 or self.trade.contingency_data[
            'two_min_return'] < 0):
            breakout_row = df[df['close'] < open_price].first_valid_index()
        elif self.trade.side == -1 and (
                self.trade.contingency_data['five_min_return'] > 0 or self.trade.contingency_data[
            'two_min_return'] > 0):
            breakout_row = df[df['close'] > open_price].first_valid_index()
        else:
            # If conditions are not met, exit the function
            self.trade.open_price_break_time = None
            self.trade.open_price_break_price = None
            return

        if breakout_row is not None:
            first_break_index_pos = df.index.get_loc(breakout_row)
            first_break_row = df.iloc[first_break_index_pos]
            new_start_time = timestamp_to_string(first_break_row.name)[11:]
            new_df = df.between_time(new_start_time, add_two_hours(new_start_time))

            # Identify the signal breakout row based on trade side
            if self.trade.side == 1:
                signal_breakout_row = new_df[new_df['close'] > open_price].first_valid_index()
            else:
                signal_breakout_row = new_df[new_df['close'] < open_price].first_valid_index()

            if signal_breakout_row is not None:
                index_pos = new_df.index.get_loc(signal_breakout_row)
                row = new_df.iloc[index_pos]
                self.trade.open_price_break_time = timestamp_to_string(row.name)
                self.trade.open_price_break_price = row.close

    def two_min_break(self):
        df = self.two_min_data.between_time(self.search_time, add_two_hours(self.search_time))
        if self.trade.side == 1:
            breakout_row = df[df['close'] > df['high'].shift(1)].first_valid_index()
        else:
            breakout_row = df[df['close'] < df['low'].shift(1)].first_valid_index()
        index_pos = df.index.get_loc(breakout_row)
        row = df.iloc[index_pos]
        self.trade.two_min_break_time = timestamp_to_string(row.name)
        self.trade.two_min_break_price = row.close
