from data_queries.trillium_queries import get_intraday, get_daily
from datetime import datetime
import logging


class Trade:
    def __init__(self, headline_time, ticker, recommendation):
        # initial trade info
        self.date = datetime.strftime(datetime.strptime(headline_time, '%Y-%m-%d %H:%M:%S'), '%Y-%m-%d')
        self.ticker = ticker
        self.start_time = datetime.strftime(datetime.strptime(headline_time, '%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
        self.current_time = datetime.now().strftime('%H:%M:%S')
        self.data = get_intraday(self.ticker, self.date, 'bar-5s')
        self.recommendation = recommendation
        self.position_size = 1000 if self.recommendation == 'BUY' else -1000
        self.side = 1 if self.position_size > 0 else -1
        self.premarket_low = self.data.between_time('06:00:00', self.current_time).low.min()
        self.premarket_high = self.data.between_time('06:00:00', self.current_time).high.max()
        self.low_of_day = self.data.low.min()
        self.high_of_day = self.data.high.max()
        self.open_price = None
        self.profit_strategy = None
        self.stop_price = None

    def set_low_high_of_day(self):
        self.low_of_day = self.data.between_time('09:30:00', self.current_time).low.min()
        self.high_of_day = self.data.between_time('09:30:00', self.current_time).high.max()

    def set_stop(self):
        if self.side == 1:
            if self.premarket_low < self.low_of_day:
                self.stop_price = self.low_of_day
            else:
                self.stop_price = self.premarket_low
        else:
            if self.premarket_high > self.high_of_day:
                self.stop_price = self.high_of_day
            else:
                self.stop_price = self.premarket_high

    def set_open(self):
        daily = get_daily(self.ticker, self.date)
        self.open_price = daily['open']
        logging.info(f'set open price {self.open_price}')
