from  data_collectors.contingency_data import contingencyData

class backtestTrade:
    def __init__(self, date, ticker, recommendation):
        # initial trade info
        self.date = date
        self.ticker = ticker
        self.recommendation = recommendation
        self.position_size = 1000 if self.recommendation == 'BUY' else -1000
        self.side = 1 if self.position_size > 0 else -1
        # signal info
        self.premarket_low_break_price = None
        self.premarket_low_break_time = None
        self.premarket_high_break_price = None
        self.premarket_high_break_time = None
        self.open_price_break_price = None
        self.open_price_break_time = None
        self.two_min_break_price = None
        self.two_min_break_time = None
        # best signal based on options
        self.best_signal = None
        self.signal_time = None
        self.signal_price = None
        # stop info
        self.stop_price = None
        self.stop_strategy = 'high_of_day' if self.side == -1 else 'low_of_day'
        self.stopped_out = False
        # exit info
        self._2_minute_quick_exit_time = None
        self._2_minute_quick_exit_price = None
        self._2_minute_delayed_exit_time = None
        self._2_minute_delayed_exit_price = None
        self._3_minute_quick_exit_time = None
        self._3_minute_quick_exit_price = None
        self._3_minute_delayed_exit_time = None
        self._3_minute_delayed_exit_price = None
        self._4_minute_quick_exit_time = None
        self._4_minute_quick_exit_price = None
        self._4_minute_delayed_exit_time = None
        self._4_minute_delayed_exit_price = None
        self._5_minute_quick_exit_time = None
        self._5_minute_quick_exit_price = None
        self._5_minute_delayed_exit_time = None
        self._5_minute_delayed_exit_price = None
        self.contingency_data = contingencyData(self.ticker, self.date).get_data()

    def determine_signal(self):
        # Mapping signal names to their (price, time) tuples
        signal_map = {
            'premarket_low_break': (self.premarket_low_break_price, self.premarket_low_break_time),
            'premarket_high_break': (self.premarket_high_break_price, self.premarket_high_break_time),
            'open_price_break': (self.open_price_break_price, self.open_price_break_time),
            'two_min_break': (self.two_min_break_price, self.two_min_break_time)
        }

        # Filter out signals where the price or time is None
        valid_signals = {name: (price, time) for name, (price, time) in signal_map.items() if price is not None and time is not None}

        if not valid_signals:
            # No valid signals, you might want to handle this case
            return

        if self.side == -1:
            # For short positions, take the max price and its corresponding time and name
            self.best_signal, (self.signal_price, self.signal_time) = max(valid_signals.items(), key=lambda item: item[1][0])
        else:
            # For long positions, take the min price and its corresponding time and name
            self.best_signal, (self.signal_price, self.signal_time) = min(valid_signals.items(), key=lambda item: item[1][0])


    def parse_exit_dicts(self, delayed_exit_info, quick_exit_info):
        self._2_minute_quick_exit_time = quick_exit_info['2-minute_quick']['exit_time']
        self._2_minute_quick_exit_price = quick_exit_info['2-minute_quick']['exit_price']
        self._3_minute_quick_exit_time = quick_exit_info['3-minute_quick']['exit_time']
        self._3_minute_quick_exit_price = quick_exit_info['3-minute_quick']['exit_price']
        self._4_minute_quick_exit_time = quick_exit_info['4-minute_quick']['exit_time']
        self._4_minute_quick_exit_price = quick_exit_info['4-minute_quick']['exit_price']
        self._5_minute_quick_exit_time = quick_exit_info['5-minute_quick']['exit_time']
        self._5_minute_quick_exit_price = quick_exit_info['5-minute_quick']['exit_price']

        self._2_minute_delayed_exit_time = delayed_exit_info['2-minute_delayed']['exit_time']
        self._2_minute_delayed_exit_price = delayed_exit_info['2-minute_delayed']['exit_price']
        self._3_minute_delayed_exit_time = delayed_exit_info['3-minute_delayed']['exit_time']
        self._3_minute_delayed_exit_price = delayed_exit_info['3-minute_delayed']['exit_price']
        self._4_minute_delayed_exit_time = delayed_exit_info['4-minute_delayed']['exit_time']
        self._4_minute_delayed_exit_price = delayed_exit_info['4-minute_delayed']['exit_price']
        self._5_minute_delayed_exit_time = delayed_exit_info['5-minute_delayed']['exit_time']
        self._5_minute_delayed_exit_price = delayed_exit_info['5-minute_delayed']['exit_price']

    def to_dict(self):
        return self.__dict__
