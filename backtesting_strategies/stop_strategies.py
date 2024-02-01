from data_queries.polygon_queries import get_intraday, add_two_hours
from tabulate import tabulate

class stopStrategies:
    def __init__(self, trade):
        self.trade = trade
        self.entry_signals = ['premarket_low_break', 'premarket_high_break', 'open_price_break', 'two_min_break']
        self.data = get_intraday(self.trade.ticker, self.trade.date, multiplier=5, timespan='second')
        for each in self.entry_signals:
            time = getattr(self.trade, each + '_time')
            price = getattr(self.trade, each + '_price')
            if getattr(self.trade, each + '_time'):
                self.set_stop_price(each, time)
                self.get_stop_stats(each, price, time)

    def high_of_day(self, time):
        df = self.data.between_time('06:00:00', time[11:])
        return df.high.max()

    def low_of_day(self, time):
        df = self.data.between_time('06:00:00', time[11:])
        return df.low.min()

    def market_hours_low(self, time):
        df = self.data.between_time('09:30:00', time[11:])
        return df.low.min()

    def premarket_high(self, time):
        df = self.data.between_time('06:00:00', '09:29:59')
        return df.high.max()

    def premarket_low(self, time):
        df = self.data.between_time('06:00:00', '09:29:59')
        return df.low.min()

    def open_price(self, time):
        return self.data.between_time('09:30:00', '09:32:00').iloc[0].open

    def set_stop_price(self, signal, time):
        strategy_method = getattr(self, self.trade.stop_strategy, None)
        if callable(strategy_method):
            setattr(self.trade, f'{signal}_stop_price', strategy_method(time))
        else:
            print(f"Unknown strategy: {self.trade.stop_strategy}")

    def get_stop_stats(self, signal, signal_price, time):
        df = self.data.between_time(time[11:], add_two_hours(time[11:]))
        stop_price = getattr(self.trade, signal + '_stop_price')
        if self.trade.side == 1:
            low = df.low.min()
            if stop_price:
                setattr(self.trade, f'{signal}_stopped_out', True if df['low'].min() < stop_price else False)
                setattr(self.trade, f'{signal}_drawdown', (signal_price - low) / (signal_price - stop_price))
        else:
            high = df.high.max()
            if stop_price:
                setattr(self.trade, f'{signal}_stopped_out', True if df['high'].max() > stop_price else False)
                setattr(self.trade, f'{signal}_drawdown', (high - signal_price) / (stop_price - signal_price))

