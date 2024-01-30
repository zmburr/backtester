from data_queries.polygon_queries import get_intraday


class entrySignals:
    def __init__(self, trade):
        self.trade = trade
        self.data = get_intraday(self.trade.ticker, self.trade.date, multiplier=5, timespan='second')
        self.two_min_data = get_intraday(self.trade.ticker, self.trade.date, multiplier=2, timespan='minute')
        self.search_time = '09:30:00'
        self.premarket_low_break()
        self.premarket_high_break()
        self.open_price_break()
        self.two_min_break()

    def premarket_low_break(self):
        try:
            premarket_low = self.data.between_time('06:00:00', '09:29:59').low.min()
            df = self.data.between_time(self.search_time, '11:00:00')
            breakout_row = df[df['close'] < premarket_low].first_valid_index()
            index_pos = df.index.get_loc(breakout_row)
            row = df.iloc[index_pos]
            self.trade.premarket_low_break_time = row.name
            self.trade.premarket_low_break_price = row.close
        except KeyError:
            pass  # Handle exception or leave empty

    def premarket_high_break(self):
        try:
            premarket_high = self.data.between_time('06:00:00', '09:29:59').high.max()
            df = self.data.between_time(self.search_time, '11:00:00')
            breakout_row = df[df['close'] > premarket_high].first_valid_index()
            index_pos = df.index.get_loc(breakout_row)
            row = df.iloc[index_pos]
            self.trade.premarket_high_break_time = row.name
            self.trade.premarket_high_break_price = row.close
        except KeyError:
            pass

    def open_price_break(self):
        try:
            open_price = self.data.between_time('09:30:00', '16:00:00').iloc[0].open
            df = self.data.between_time(self.search_time, '11:00:00')
            if self.trade.side == 1:
                breakout_row = df[df['close'] > open_price].first_valid_index()
            else:
                breakout_row = df[df['close'] < open_price].first_valid_index()
            index_pos = df.index.get_loc(breakout_row)
            row = df.iloc[index_pos]
            self.trade.open_price_break_time = row.name
            self.trade.open_price_break_price = row.close
        except KeyError:
            pass

    def two_min_break(self):
        try:
            df = self.two_min_data.between_time(self.search_time, '11:00:00')
            if self.trade.side == 1:
                breakout_row = df[df['close'] > df['high'].shift(1)].first_valid_index()
            else:
                breakout_row = df[df['close'] < df['low'].shift(1)].first_valid_index()
            index_pos = df.index.get_loc(breakout_row)
            row = df.iloc[index_pos]
            self.trade.two_min_break_time = row.name
            self.trade.two_min_break_price = row.close
        except KeyError:
            pass




