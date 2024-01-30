from data_queries.polygon_queries import get_intraday, get_levels_data


class contingencyData:
    def __init__(self, ticker, date):
        self.ticker = ticker
        self.date = date
        # volume as % of adv
        self.premarket_vol_pct = None
        self.five_min_vol_pct = None
        self.ten_min_vol_pct = None
        self.fifteen_min_vol_pct = None
        self.twenty_min_vol_pct = None
        self.twenty_five_min_vol_pct = None
        self.thirty_min_vol_pct = None
        # price return off open
        self.premarket_return = None
        self.five_min_return = None
        self.ten_min_return = None
        self.fifteen_min_return = None
        self.twenty_min_return = None
        self.twenty_five_min_return = None
        self.thirty_min_return = None
        # breaks pre high or low _mins
        self.break_pre_low_5 = None
        self.break_pre_high_5 = None
        self.break_pre_low_10 = None
        self.break_pre_high_10 = None
        self.break_pre_low_15 = None
        self.break_pre_high_15 = None
        self.break_pre_low_20 = None
        self.break_pre_high_20 = None
        self.break_pre_low_25 = None
        self.break_pre_high_25 = None
        self.break_pre_low_30 = None
        self.break_pre_high_30 = None
        # data
        self.adv = get_levels_data(self.ticker, self.date, 30, 1, 'day')
        self.avg_daily_vol = self.adv['volume'].sum() / len(self.adv['volume'])
        self.data = get_intraday(self.ticker, self.date, multiplier=5, timespan='second')
        self.premarket_data = self.data.between_time('06:00:00', '09:29:59')
        self.five_min_data = self.data.between_time('09:30:00', '09:35:00')
        self.ten_min_data = self.data.between_time('09:30:00', '09:40:00')
        self.fifteen_min_data = self.data.between_time('09:30:00', '09:45:00')
        self.twenty_min_data = self.data.between_time('09:30:00', '09:50:00')
        self.twenty_five_min_data = self.data.between_time('09:30:00', '09:55:00')
        self.thirty_min_data = self.data.between_time('09:30:00', '10:00:00')
        self.premarket_high = self.premarket_data['high'].max()
        self.premarket_low = self.premarket_data['low'].min()
        self.premarket_vwap = self.premarket_data['vwap'].sum() / len(self.premarket_data['vwap'])

    def get_vol(self):
        self.premarket_vol_pct = self.premarket_data['volume'].sum() / self.avg_daily_vol
        self.five_min_vol_pct = self.five_min_data['volume'].sum() / self.avg_daily_vol
        self.ten_min_vol_pct = self.ten_min_data['volume'].sum() / self.avg_daily_vol
        self.fifteen_min_vol_pct = self.fifteen_min_data['volume'].sum() / self.avg_daily_vol
        self.twenty_min_vol_pct = self.twenty_min_data['volume'].sum() / self.avg_daily_vol
        self.twenty_five_min_vol_pct = self.twenty_five_min_data['volume'].sum() / self.avg_daily_vol
        self.thirty_min_vol_pct = self.thirty_min_data['volume'].sum() / self.avg_daily_vol

    def get_return(self):
        self.premarket_return = self.premarket_data['close'].iloc[-1] / self.premarket_data['open'].iloc[0] - 1
        self.five_min_return = self.five_min_data['close'].iloc[-1] / self.five_min_data['open'].iloc[0] - 1
        self.ten_min_return = self.ten_min_data['close'].iloc[-1] / self.ten_min_data['open'].iloc[0] - 1
        self.fifteen_min_return = self.fifteen_min_data['close'].iloc[-1] / self.fifteen_min_data['open'].iloc[0] - 1
        self.twenty_min_return = self.twenty_min_data['close'].iloc[-1] / self.twenty_min_data['open'].iloc[0] - 1
        self.twenty_five_min_return = self.twenty_five_min_data['close'].iloc[-1] / self.twenty_five_min_data['open'].iloc[0] - 1
        self.thirty_min_return = self.thirty_min_data['close'].iloc[-1] / self.thirty_min_data['open'].iloc[0] - 1

    def get_breaks(self):
        self.break_pre_low_5 = self.five_min_data['low'].min() < self.premarket_low
        self.break_pre_high_5 = self.five_min_data['high'].max() > self.premarket_high
        self.break_pre_low_10 = self.ten_min_data['low'].min() < self.premarket_low
        self.break_pre_high_10 = self.ten_min_data['high'].max() > self.premarket_high
        self.break_pre_low_15 = self.fifteen_min_data['low'].min() < self.premarket_low
        self.break_pre_high_15 = self.fifteen_min_data['high'].max() > self.premarket_high
        self.break_pre_high_20 = self.twenty_min_data['high'].max() > self.premarket_high
        self.break_pre_low_20 = self.twenty_min_data['low'].min() < self.premarket_low
        self.break_pre_low_25 = self.twenty_five_min_data['low'].min() < self.premarket_low
        self.break_pre_high_25 = self.twenty_five_min_data['high'].max() > self.premarket_high
        self.break_pre_low_30 = self.thirty_min_data['low'].min() < self.premarket_low
        self.break_pre_high_30 = self.thirty_min_data['high'].max() > self.premarket_high

    def get_data(self):

        self.get_vol()
        self.get_return()
        self.get_breaks()
        initially_none_fields = [
            'premarket_vol_pct', 'five_min_vol_pct', 'ten_min_vol_pct', 'fifteen_min_vol_pct',
            'twenty_min_vol_pct', 'twenty_five_min_vol_pct', 'thirty_min_vol_pct',
            'premarket_return', 'five_min_return', 'ten_min_return', 'fifteen_min_return',
            'twenty_min_return', 'twenty_five_min_return', 'thirty_min_return',
            'break_pre_low_5', 'break_pre_high_5', 'break_pre_low_10', 'break_pre_high_10',
            'break_pre_low_15', 'break_pre_high_15', 'break_pre_low_20', 'break_pre_high_20',
            'break_pre_low_30', 'break_pre_high_30'
        ]

        # Create a dictionary for these fields with updated values
        data = {field: getattr(self, field) for field in initially_none_fields}

        return data
