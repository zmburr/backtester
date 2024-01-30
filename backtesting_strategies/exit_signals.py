from data_queries.polygon_queries import get_intraday


def remove_halts(df):
    return df[df["open"].notna()]


def get_move_df(df, side):
    """
    Function that takes minutes as input for bar types then gets percent captured of total move of a self.trade.
    :return: Dict {minute_bar : pct_captured}
    """
    df = remove_halts(df)
    if side == 1:
        df["prev_low"] = df["low"].shift(1)
        df["delaystrat_trend"] = (df["close"] > df["prev_low"])
        df["quickstrat_trend"] = (df['low'] > df["prev_low"])
        df = df[df["prev_low"].notna()]
    else:
        df["prev_high"] = df['high'].shift(1)
        df["delaystrat_trend"] = (df["close"] < df["prev_high"])
        df["quickstrat_trend"] = (df["high"] < df["prev_high"])
        df = df[df["prev_high"].notna()]
    return df


def get_delaystrat_exit(df, side, bar_type):
    df = get_move_df(df, side)
    countertrends = df.loc[df["delaystrat_trend"] == False]
    if countertrends.empty:
        time = df.index[-1]
        exit_price = df.iloc[-1]["close"]
    else:
        time = countertrends.index[0]
        exit_price = countertrends.iloc[0]["close"]
    return {'exit_time': time, 'exit_price': round(exit_price, 2), 'bar_type': bar_type}


def get_quickstrat_exit(df, side, bar_type):
    df = get_move_df(df, side)
    countertrends = df.loc[df["quickstrat_trend"] == False]
    if countertrends.empty:
        time = df.index[-1]
        exit_price = df.iloc[-1]["close"]
    else:
        time = countertrends.index[0]
        if side == 1:
            exit_price = countertrends.iloc[0]["prev_low"] - .01
        else:
            exit_price = countertrends.iloc[0]["prev_high"] + .01
    return {'exit_time': time, 'exit_price': round(exit_price, 2), 'bar_type': bar_type}


class exitSignals:
    def __init__(self, trade):
        self.trade = trade
        self.data = get_intraday(self.trade.ticker, self.trade.date, multiplier=5, timespan='second')
        self.search_time = '09:30:00'

    def multi_bar_exit(self, exit_strategy):
        results = {}
        for bar_type in [2, 3, 4, 5]:
            self.data = get_intraday(self.trade.ticker, self.trade.date, multiplier=bar_type, timespan='minute')
            df = self.data.between_time(self.trade.signal_time, '16:00:00')
            if exit_strategy == 'delayed':
                results[f'{bar_type}-minute'] = get_delaystrat_exit(df, self.trade.side, bar_type)
            elif exit_strategy == 'quick':
                results[f'{bar_type}-minute'] = get_quickstrat_exit(df, self.trade.side, bar_type)
        return results

    def on_close(self):
        return {'exit_time': '16:00:00', 'exit_price': self.data.iloc[-1].close}


# from backtesting_strategies.trade import backtestTrade
#
# trade = backtestTrade('2024-01-29', 'AAPL', 'SELL')
# trade.position_size = -1000
# trade.signal_time = '09:30:00'
# e = exitSignals(trade)
# print(e.multi_bar_exit('quick'))
# print(e.multi_bar_exit('delayed'))
