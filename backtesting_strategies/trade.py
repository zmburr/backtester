class backtestTrade:
    def __init__(self, headline_time, ticker, recommendation):
        # initial trade info
        self.headline_time = headline_time
        self.ticker = ticker
        self.recommendation = recommendation
        self.position_size = None
        self.date = self.headline_time.split(' ')[0]
        # signal info
        self.signal_price = None
        self.signal_time = None
        self.signal_strategy = None
        # stop info
        self.stop_price = None
        self.stop_strategy = None
        self.stopped_out = False
        # exit info
        self.exit_price = None
        self.profit_strategy = None
        self.exit_time = None

