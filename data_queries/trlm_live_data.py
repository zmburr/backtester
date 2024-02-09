import journal
import ctxcapmd
from datetime import date, datetime
import logging
import datetime

bar_data_dict = {
    '1_min': 'bar-1min',
    '2_min': 'bar-2min',
    '3_min': 'bar-3min',
    '5_min': 'bar-5min'
}


def convert_str_date(date: str):
    return datetime.strptime(date, '%Y-%m-%d').date()


# with ctxcapmd.Session('10.195.0.102', 65500, journal.any_decompress) as session:
#     def print_object(obj):
#         print(obj)
#
#
#     handle = session.request_data_stream(print_object, 'DWAC', datetime.date(2024, 2, 5), ['bar-5s'])
#
#     handle.wait()
#
#     handle.raise_on_error()


class TrlmData:
    def __init__(self, trade, manager):
        self.trade = trade
        self.ticker = self.trade.ticker
        self.manager = manager
        self.profit_bars = bar_data_dict[self.manager.profit_strategy[:5]]
        self.handle = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f'live data starting - ticker: {self.ticker}')

    def start_data_stream(self):
        with ctxcapmd.Session('10.195.0.102', 65500, journal.any_decompress) as session:
            def print_object(obj):
                if obj['type'] == 'bar-5s':
                    # if not obj.get('preview', False):
                    self.update_manager(obj)
                elif obj['type'] == self.profit_bars:
                    if not obj.get('preview', False):
                        self.update_profit_manager(obj)

            self.handle = session.request_stream(print_object, self.ticker,
                                                      ['bar-5s', self.profit_bars])
            self.manager.handle = self.handle
            self.handle.wait()
            try:
                self.handle.raise_on_error()
            except:
                pass

    def update_manager(self, current_data):
        logging.debug(f'Sending new data to stop manager - {current_data}')
        self.manager.process_incoming_data_for_stop(current_data)
        self.manager.new_data_event.set()

    def update_profit_manager(self, current_data):
        logging.debug(f'Sending new data to profit manager - {current_data}')
        self.manager.process_incoming_data_for_profit(current_data)
