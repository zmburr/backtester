from data_queries.polygon_queries import get_actual_current_price, get_levels_data, get_atr, fetch_and_calculate_volumes, get_ticker_pct_move
from datetime import datetime
from tabulate import tabulate
import pandas as pd
from data_collectors.combined_data_collection import reversal_df, momentum_df
from scipy.stats import percentileofscore
from data_queries.bloomberg_screener import cleaned_tickers
import matplotlib.pyplot as plt

columns_to_compare = [
    'pct_change_120', 'pct_change_90', 'pct_change_30', 'pct_change_15',
    'pct_change_3', 'percent_of_premarket_vol', 'percent_of_vol_one_day_before',
    'percent_of_vol_two_day_before',  'percent_of_vol_three_day_before','day_of_range_pct',
    'one_day_before_range_pct', 'two_day_before_range_pct', 'three_day_before_range_pct'
]# watchlist = ['NVDA','ROOT','AMD','MSTR','SMR','BITO' ,'ANET', 'SYM','SMCI', 'GOOG', 'PLTR', 'MSFT', 'META','VRT', 'AVGO', 'ARM', 'COIN','SNOW', 'RXRX', 'DELL']
# my_watchlist = ['PLTR', 'IONQ','MSTR','APP','OKLO','SMR','RKLB','TSLA','COIN','LMND','IBIT']
# tickers_to_remove = ['SILJ',"ARIS", 'XMTR', 'SMST','AGFY','MDXG','TEO','DAVE','GBTC','BITI','BITO','CERO','NPWR','MDXG','IBIT','BITO','SILJ','BITU','CTOS','CERO','PLAG','MVST','ATEC']  # Replace with actual tickers
# watchlist = list(set(my_watchlist + cleaned_tickers))
# watchlist = [ticker for ticker in watchlist if ticker not in tickers_to_remove]
watchlist = ['ACHR']
print(watchlist)
date = datetime.now().strftime('%Y-%m-%d')


def add_range_data(ticker):
    """
    Function to watch for range expansion across a list of stocks.
    :param watchlist: List of stock tickers to scan for range expansion.
    :param date: The date for which to perform the analysis.
    """
    try:
        # Get the ATR and levels data for the stock
        df = get_levels_data(ticker, date, 60, 1, 'day')

        # Calculate True Range (TR) components
        df['high-low'] = df['high'] - df['low']
        df['high-previous_close'] = abs(df['high'] - df['close'].shift())
        df['low-previous_close'] = abs(df['low'] - df['close'].shift())

        # Calculate the True Range (TR)
        df['TR'] = df[['high-low', 'high-previous_close', 'low-previous_close']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()

        # Calculate the percentage of ATR
        df['PCT_ATR'] = (df['TR'] / df['ATR'])

        # Calculate the 30-day average daily volume and current day's volume
        df['20Day_Avg_Volume'] = df['volume'].rolling(window=20).mean()
        df['pct_avg_volume'] = (df['volume'] / df['20Day_Avg_Volume'])
        # Get the latest range and percentage of ATR
        latest_range = df['TR'].iloc[-1]
        latest_atr = df['ATR'].iloc[-1]
        day_before_range = df['TR'].iloc[-2]
        pct_of_atr = df['PCT_ATR'].iloc[-1]
        day_before_pct_of_atr = df['PCT_ATR'].iloc[-2]
        two_day_before_pct_of_atr = df['PCT_ATR'].iloc[-3]
        three_day_before_pct_of_atr = df['PCT_ATR'].iloc[-4]
        day_before_pct_adv = df['pct_avg_volume'].iloc[-2]
        two_day_before_pct_adv = df['pct_avg_volume'].iloc[-3]
        three_day_before_pct_adv = df['pct_avg_volume'].iloc[-4]

        # Log the range expansion information
        result = {
            'percent_of_vol_one_day_before': day_before_pct_adv,
            'percent_of_vol_two_day_before': two_day_before_pct_adv,
            'percent_of_vol_three_day_before': three_day_before_pct_adv,
            'day_of_range_pct': pct_of_atr,
            'one_day_before_range_pct': day_before_pct_of_atr,
            'two_day_before_range_pct': two_day_before_pct_of_atr,
            'three_day_before_range_pct': three_day_before_pct_of_atr
        }
        return result
        # Print the table for each stock (optional)
        # if result['Ticker'] == 'MSTR':
        #     print(f"\nRange Expansion Data for {ticker}:")
        #     print(tabulate(df, headers=df.columns, tablefmt='grid'))

    except Exception as e:
        print(f"Error processing {ticker}: {e}")




def add_percent_of_adv_columns(volume_data):
    # List of volume columns to compare with avg_daily_vol
    volume_columns = ['premarket_vol', 'vol_in_first_5_min', 'vol_in_first_15_min', 'vol_in_first_10_min',
                      'vol_in_first_30_min', 'vol_on_breakout_day']

    # Ensure 'avg_daily_vol' is not 0 to avoid division by zero error
    if volume_data.get('avg_daily_vol', 0) > 0:
        avg_daily_vol = volume_data['avg_daily_vol']

        # Add new keys representing percent of avg_daily_vol
        for col in volume_columns:
            if col in volume_data:
                percent_col_name = f'percent_of_{col}'
                volume_data[percent_col_name] = (volume_data[col] / avg_daily_vol)
    else:
        # Handle case where avg_daily_vol is 0 or not present
        for col in volume_columns:
            percent_col_name = f'percent_of_{col}'
            volume_data[percent_col_name] = None  # Assign a default value indicating calculation is not possible

    return volume_data


def get_stock_data(ticker):
    current_price = get_actual_current_price(ticker)
    pct_data = get_ticker_pct_move(ticker, date, current_price)
    volume_data = fetch_and_calculate_volumes(ticker, date)
    volume_data = add_percent_of_adv_columns(volume_data)
    range_data = add_range_data(ticker)
    # if ticker =='ROOT':
    #     print(pct_data)
    return {ticker: {'pct_data': pct_data, 'volume_data': volume_data,'range_data':range_data}}


def get_all_stocks_data(watchlist):
    all_data = {}
    for ticker in watchlist:
        ticker_data = get_stock_data(ticker)
        all_data.update(ticker_data)
    return all_data


def calculate_percentiles(df, stock_data, columns):
    """
    Calculate the percentile ranks for specified columns in stock_data compared to historical data in df.

    :param df: DataFrame containing historical data.
    :param stock_data: Dictionary or DataFrame with the screened stock's data.
    :param columns: List of column names to compare.
    :return: Dictionary with column names as keys and percentile ranks as values.
    """
    percentiles = {}
    df = df.dropna(subset=columns)
    # Ensure stock_data is a DataFrame for easier handling
    if isinstance(stock_data, dict):
        flat_data = {**stock_data, **stock_data['pct_data'], **stock_data['volume_data'], **stock_data['range_data']}
        flat_data.pop('pct_data', None)
        flat_data.pop('volume_data', None)
        flat_data.pop('range_data', None)
        stock_data_df = pd.DataFrame([flat_data])
    else:
        stock_data_df = stock_data

    for column in columns:
        if column in df.columns and column in stock_data_df.columns:
            value = stock_data_df.iloc[0][column]
            percentiles[column] = percentileofscore(df[column], value, kind='weak')

    return percentiles


def convert_dict_to_df(filtered_stocks):
    data_for_df = []
    for ticker, values in filtered_stocks.items():
        for key, val in values.items():
            if key == 'pct_data' or key == 'volume_data':
                for metric, metric_val in val.items():
                    data_for_df.append({
                        'Ticker': ticker,
                        'Data Type': key,
                        'Metric': metric,
                        'Value': metric_val
                    })

    df = pd.DataFrame(data_for_df)
    # for each in drop_columns:
    #     df.drop(each, axis=0, inplace=True)
    return df

#TODO - add distance from MAVS

if __name__ == '__main__':
    all_stock_data = get_all_stocks_data(watchlist)
    reversal_stocks = convert_dict_to_df(all_stock_data)
    # print(tabulate(reversal_stocks, headers=reversal_stocks.columns))
    # print(tabulate(momentum_stocks, headers=momentum_stocks.columns))
    reversal_results = []

    for ticker, data in all_stock_data.items():
        reversal_percentiles = calculate_percentiles(reversal_df, data, columns_to_compare)
        reversal_results.append((ticker, reversal_percentiles))
        # print('reversal', ticker, reversal_percentiles)
        # print('momentum', ticker, momentum_percentiles)

    reversal_sorted = sorted(reversal_results, key=lambda x: x[1]['pct_change_3'], reverse=True)
    print("Sorted Reversal Stock Percentiles:")

    for ticker, percentiles in reversal_sorted:
        # Extract pct_data values for the ticker from all_stock_data
        pct_data = all_stock_data.get(ticker, {}).get('pct_data', {})
        range_data = all_stock_data.get(ticker, {}).get('range_data', {})
        # Format percentiles, pct_data, and range_data as multi-line strings
        percentiles_str = '\n'.join([f"    {k}: {v:.2f}" for k, v in percentiles.items()])
        pct_data_str = '\n'.join([f"    {k}: {v:.2f}" for k, v in pct_data.items()])
        range_data_str = '\n'.join([f"    {k}: {v:.2f}" for k, v in range_data.items()])

        # Print formatted output
        print(f"Reversal: {ticker}")
        print("  Percentiles:")
        print(percentiles_str if percentiles else "    None")
        print("  Absolute PCT Changes (in hundreds):")
        print(pct_data_str if pct_data else "    None")
        print("  Range Data:")
        print(range_data_str if range_data else "    None")
        print("-" * 50)  # Separator for better readability

        # Bar chart visualization for percentiles
        plt.figure(figsize=(10, 6))
        plt.bar(percentiles.keys(), percentiles.values())
        plt.title(f"Reversal Stock Percentiles for {ticker}")
        plt.xlabel("Percentile Metrics")
        plt.ylabel("Percentile Values")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()