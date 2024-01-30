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
        self.signal_time = None
        self.signal_price = None
        # stop info
        self.stop_price = None
        self.stop_strategy = 'high_of_day'
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

    def determine_signal(self):
        # Create a list of (price, time) tuples
        price_time_pairs = [
            (self.premarket_low_break_price, self.premarket_low_break_time),
            (self.premarket_high_break_price, self.premarket_high_break_time),
            (self.open_price_break_price, self.open_price_break_time),
            (self.two_min_break_price, self.two_min_break_time)
        ]

        # Filter out pairs where the price or time is None
        valid_pairs = [(price, time) for price, time in price_time_pairs if price is not None and time is not None]

        if not valid_pairs:
            # No valid signals, you might want to handle this case
            return

        if self.side == -1:
            # Take the max price and its corresponding time
            self.signal_price, self.signal_time = max(valid_pairs, key=lambda pair: pair[0])
        else:
            # Take the min price and its corresponding time
            self.signal_price, self.signal_time = min(valid_pairs, key=lambda pair: pair[0])

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
