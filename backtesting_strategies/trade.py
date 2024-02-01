from data_collectors.contingency_data import contingencyData


class backtestTrade:
    def __init__(self, date, ticker, recommendation):
        # initial trade info
        self.date = date
        self.ticker = ticker
        self.recommendation = recommendation
        self.position_size = 1000 if self.recommendation == 'BUY' else -1000
        self.side = 1 if self.position_size > 0 else -1
        # Risk metrics
        self.max_loss = None
        self.max_profit = None
        self.max_close_profit = None
        self.risk_reward_ratio = None
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
        self.stop_strategy = 'high_of_day' if self.side == -1 else 'market_hours_low'
        self.best_stop = None
        self.best_stop_price = None

        # Premarket Low Break variables
        self.premarket_low_break_stop_price = None
        self.premarket_low_break_stopped_out = False
        self.premarket_low_break_drawdown = None

        # Premarket High Break variables
        self.premarket_high_break_stop_price = None
        self.premarket_high_break_stopped_out = False
        self.premarket_high_break_drawdown = None

        # Open Price Break variables
        self.open_price_break_stop_price = None
        self.open_price_break_stopped_out = False
        self.open_price_break_drawdown = None

        # Two Minute Break variables
        self.two_min_break_stop_price = None
        self.two_min_break_stopped_out = False
        self.two_min_break_drawdown = None

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
        self.close_price = None
        self.high_price = None
        self.low_price = None
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
        valid_signals = {name: (price, time) for name, (price, time) in signal_map.items() if
                         price is not None and time is not None}

        if not valid_signals:
            # No valid signals, you might want to handle this case
            return

        if self.side == -1:
            # For short positions, take the max price and its corresponding time and name
            self.best_signal, (self.signal_price, self.signal_time) = max(valid_signals.items(),
                                                                          key=lambda item: item[1][0])
        else:
            # For long positions, take the min price and its corresponding time and name
            self.best_signal, (self.signal_price, self.signal_time) = min(valid_signals.items(),
                                                                          key=lambda item: item[1][0])

    def determine_stop(self):
        entry_signals = ['premarket_low_break', 'premarket_high_break', 'open_price_break', 'two_min_break']
        highest_stop = None
        highest_stop_price = None

        # Loop through each entry signal to check corresponding stop prices and stopped_out status
        for signal in entry_signals:
            stop_price_attr = f"{signal}_stop_price"
            stopped_out_attr = f"{signal}_stopped_out"

            # Retrieve the stop price and stopped_out status using getattr
            stop_price = getattr(self, stop_price_attr, None)
            stopped_out = getattr(self, stopped_out_attr, True)  # Default to True if attribute doesn't exist

            # Check if the signal has not been stopped out and the stop price is higher than the current highest
            if not stopped_out and stop_price is not None and (
                    highest_stop_price is None or stop_price > highest_stop_price):
                highest_stop = signal
                highest_stop_price = stop_price

        # If a highest stop is found that hasn't been stopped out, you might want to handle it here
        if highest_stop:
            # You can handle the highest stop found here, for example, setting it as the current stop
            self.best_stop = highest_stop
            self.best_stop_price = highest_stop_price
            # Additional handling can be added here as needed

        return highest_stop, highest_stop_price

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

    def calculate_risk_metrics(self):
        if self.signal_price is None or self.signal_time is None:
            return
        if self.side == 1:
            self.max_loss = self.signal_price - self.best_stop_price
            self.max_profit = self.high_price - self.signal_price
            self.max_close_profit = self.close_price - self.signal_price
        else:
            self.max_loss = self.best_stop_price - self.signal_price
            self.max_profit = self.signal_price - self.low_price
            self.max_close_profit = self.signal_price - self.close_price
        self.risk_reward_ratio = self.max_close_profit / self.max_loss
