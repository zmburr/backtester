class contingencyData:
    def __init__(self):
        # volume as % of adv
        self.five_min_vol = None
        self.ten_min_vol = None
        self.fifteen_min_vol = None
        self.twenty_min_vol = None
        self.twenty_five_min_vol = None
        self.thirty_min_vol = None
        # price return off open
        self.five_min_return = None
        self.ten_min_return = None
        self.fifteen_min_return = None
        self.twenty_min_return = None
        self.twenty_five_min_return = None
        self.thirty_min_return = None
        # breaks pre high or low _mins
        self.break_pre_low_5 = None
        self.break_pre_high_5 = None
        self.break_pre_low_15 = None
        self.break_pre_high_15 = None
        self.break_pre_low_30 = None
        self.break_pre_high_30 = None