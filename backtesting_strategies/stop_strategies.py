from data_queries.polygon_queries import get_intraday, add_two_hours


class stopStrategies:
    def __init__(self, trade):
        self.trade = trade
        self.data = get_intraday(self.trade.ticker, self.trade.date, multiplier=5, timespan='second')
        self.search_time = self.trade.signal_time[11:]

    def high_of_day(self):
        df = self.data.between_time('06:00:00', self.search_time)
        return df.high.max()

    def low_of_day(self):
        df = self.data.between_time('06:00:00', self.search_time)
        return df.low.min()

    def premarket_high(self):
        df = self.data.between_time('06:00:00', '09:29:59')
        return df.high.max()

    def premarket_low(self):
        df = self.data.between_time('06:00:00', '09:29:59')
        return df.low.min()

    def open_price(self):
        return self.data.between_time('09:30:00', '09:32:00').iloc[0].open

    def set_stop_price(self):
        strategy_method = getattr(self, self.trade.stop_strategy, None)
        if callable(strategy_method):
            self.trade.stop_price = strategy_method()
        else:
            print(f"Unknown strategy: {self.trade.stop_strategy}")

    def drawdown_to_stop(self):
        df = self.data.between_time(self.search_time, add_two_hours(self.search_time))
        if self.trade.side == 1:
            low = df.low.min()
            if self.trade.stop_price:
                drawdown = (self.trade.signal_price - low) / (self.trade.signal_price - self.trade.stop_price)
                return drawdown
        else:
            high = df.high.max()
            if self.trade.stop_price:
                drawdown = (high - self.trade.signal_price) / (self.trade.stop_price - self.trade.signal_price)
                return drawdown

