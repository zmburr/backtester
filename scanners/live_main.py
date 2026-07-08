import logging
from scanners.live_watcher import TradeManager
from scanners.live_trade import Trade
from datetime import datetime
import threading
from data_queries.trlm_live_data import TrlmData

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
now_time = datetime.now()
time = now_time.strftime('%Y-%m-%d %H:%M:%S')

live_test = {
    'headline_time': time,
    'ticker': 'GME',
    'recommendation': 'SELL',
    'profit_strategy': '5_min_close'
}


def live_main(headline_time, ticker, recommendation, profit_strategy):
    trade = Trade(headline_time, ticker, recommendation)
    # trade.liquidity = get_liquidity(trade.ticker)
    trade.profit_strategy = profit_strategy
    trade.set_low_high_of_day()
    trade.set_stop()
    trade.set_open()
    manager = TradeManager(trade, trade.profit_strategy)
    data = TrlmData(trade, manager)
    data_thread = threading.Thread(target=data.start_data_stream, daemon=True)
    data_thread.start()
    manager_thread = threading.Thread(target=manager.run, daemon=True)
    manager_thread.start()
    # trade.levels = GetLevels(trade).get_key_levels()
    # logging.info(f'got levels: {trade.levels}')
    manager_thread.join()
    data_thread.join()
    # trade.positioning_score = GetPositioningScore(trade).positioning_score()
    # create_chart(trade)
    # trade.print_trade_info()
    # trade.save(mongo_client)


if __name__ == '__main__':
    live_main(**live_test)
