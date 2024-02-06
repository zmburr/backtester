import pandas as pd
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
from analyzers.charter import create_chart
from backtesting_strategies.trade import backtestTrade
from backtesting_strategies.entry_signals import entrySignals
from backtesting_strategies.stop_strategies import stopStrategies
from backtesting_strategies.exit_signals import exitSignals
import logging
from data_collectors.combined_data_collection import momentum_df
from data_collectors.combined_data_collection import reversal_df
from tabulate import tabulate


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

trade_type = 'reversal'

trades = []

if __name__ == '__main__':
    df = reversal_df if trade_type == 'reversal' else momentum_df
    for index, row in df.iterrows():
        ticker = row['ticker']
        wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
        date = wrong_date.strftime('%Y-%m-%d')
        logging.info(f'Processing trade for {ticker} on {date}')
        trade = backtestTrade(date, ticker, 'SELL') if trade_type == 'reversal' else backtestTrade(date, ticker, 'BUY')
        entry = entrySignals(trade)
        trade.determine_signal()
        stop = stopStrategies(trade)
        trade.determine_stop()
        exit = exitSignals(trade)
        quick_dict = exit.multi_bar_exit('quick')
        delayed_dict = exit.multi_bar_exit('delayed')
        trade.parse_exit_dicts(delayed_dict, quick_dict)
        trade.calculate_risk_metrics()
        trades.append(trade.__dict__)
        # create_chart(trade)

    # Create a DataFrame from the collected trade details
    trades_df = pd.DataFrame(trades)
    trades_df.drop('recommendation', axis=1, inplace=True)
    trades_df.drop('side', axis=1, inplace=True)
    print(tabulate(trades_df, headers=trades_df.columns))
    trades_df.to_csv(f'C:\\Users\\zmbur\\PycharmProjects\\backtester\\data\\{trade_type}_backtest_results.csv')
