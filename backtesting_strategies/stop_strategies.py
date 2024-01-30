from data_queries.polygon_queries import get_intraday


class stopStrategies:
    def __init__(self, trade):
        self.trade = trade
        self.data = get_intraday(self.trade.ticker, self.trade.date, multiplier=5, timespan='second')
        self.search_time = '06:00:00'

    def high_of_day(self):
        df = self.data.between_time(self.search_time, self.trade.signal_time)
        return df.high.max()

    def low_of_day(self):
        df = self.data.between_time(self.search_time, self.trade.signal_time)
        return df.low.min()

    def premarket_high(self):
        df = self.data.between_time(self.search_time, self.trade.signal_time)
        return df.between_time('06:00:00', '09:29:59').high.max()

    def premarket_low(self):
        df = self.data.between_time(self.search_time, self.trade.signal_time)
        return df.between_time('06:00:00', '09:29:59').low.min()

    def open_price(self):
        return self.data.between_time('09:30:00', '09:32:00').iloc[0].open
