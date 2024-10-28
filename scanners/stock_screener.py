from data_queries.polygon_queries import get_actual_current_price, get_levels_data, get_atr, fetch_and_calculate_volumes, get_ticker_pct_move
from datetime import datetime
from tabulate import tabulate
import pandas as pd
from data_collectors.combined_data_collection import reversal_df, momentum_df
from scipy.stats import percentileofscore

columns_to_compare = ['pct_change_120', 'pct_change_90', 'pct_change_30', 'pct_change_15', 'pct_change_3','percent_of_premarket_vol']
# watchlist = ['NVDA','ROOT','AMD','MSTR','SMR','BITO' ,'ANET', 'SYM','SMCI', 'GOOG', 'PLTR', 'MSFT', 'META','VRT', 'AVGO', 'ARM', 'COIN','SNOW', 'RXRX', 'DELL']
watchlist = ['PLTR', 'IONQ','MSTR','APP','OKLO','SMR','DJT','TSLA']
date = datetime.now().strftime('%Y-%m-%d')


def range_expansion_watcher(watchlist, date):
    """
    Function to watch for range expansion across a list of stocks.
    :param watchlist: List of stock tickers to scan for range expansion.
    :param date: The date for which to perform the analysis.
    """
    results = []

    for ticker in watchlist:
        try:
            # Get the ATR and levels data for the stock
            atr = get_atr(ticker, date)
            df = get_levels_data(ticker, date, 60, 1, 'day')

            # Calculate True Range (TR) components
            df['high-low'] = df['high'] - df['low']
            df['high-previous_close'] = abs(df['high'] - df['close'].shift())
            df['low-previous_close'] = abs(df['low'] - df['close'].shift())

            # Calculate the True Range (TR)
            df['TR'] = df[['high-low', 'high-previous_close', 'low-previous_close']].max(axis=1)

            # Calculate the percentage of ATR
            df['PCT_ATR'] = (df['TR'] / atr) * 100

            # Get the latest range and percentage of ATR
            latest_range = df['TR'].iloc[-1]
            day_before_range = df['TR'].iloc[-2]
            pct_of_atr = (latest_range / atr) * 100
            day_before_pct_of_atr = (day_before_range/atr)  * 100

            # Log the range expansion information
            result = {
                'Ticker': ticker,
                'Range': latest_range,
                'ATR': atr,
                'Percent of ATR': pct_of_atr,
                'Day Before % ATR': day_before_pct_of_atr
            }
            results.append(result)

            # Print the table for each stock (optional)
            # if result['Percent of ATR'] > 100:
            #     print(f"\nRange Expansion Data for {ticker}:")
            #     print(tabulate(df, headers=df.columns, tablefmt='grid'))

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # Convert results to DataFrame for better readability
    results_df = pd.DataFrame(results)
    return results_df

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
    # if ticker =='ROOT':
    #     print(pct_data)
    return {ticker: {'pct_data': pct_data, 'volume_data': volume_data}}


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
        flat_data = {**stock_data, **stock_data['pct_data'], **stock_data['volume_data']}
        flat_data.pop('pct_data', None)
        flat_data.pop('volume_data', None)
        stock_data_df = pd.DataFrame([flat_data])
    else:
        stock_data_df = stock_data

    for column in columns:
        if column in df.columns and column in stock_data_df.columns:
            value = stock_data_df.iloc[0][column]
            percentiles[column] = percentileofscore(df[column], value, kind='weak')

    return percentiles


def does_stock_meet_criteria(data, pct_criteria, volume_criteria):
    # Check if percentage change data meets the criteria
    if not meets_criteria(data['pct_data'], pct_criteria):
        return False

    # Check if volume data meets the criteria
    for vol_metric, threshold in volume_criteria.items():
        if vol_metric not in data['volume_data'] or data['volume_data'][vol_metric] < threshold:
            return False

    return True


def meets_criteria(pct_data, criteria):
    """
    Check if the percentage change data meets the specified criteria.

    :param pct_data: A dictionary containing percentage change data for different periods.
    :param criteria: A dictionary containing criteria thresholds for each period.
    :return: True if all criteria are met, False otherwise.
    """
    return all(pct_data[period] > threshold for period, threshold in criteria.items())


def filter_stocks(all_stocks_data, stock_type):
    filtered_stocks = {}

    # Define distinct criteria for momentum and reversal
    criteria = {
        'momentum': {
            'pct_criteria': {
                'pct_change_30': 0.15,
                'pct_change_15': 0.05

            },
            'volume_criteria': {
                'percent_of_premarket_vol': .01
            }
        },
        'reversal': {
            'pct_criteria': {
                'pct_change_120': 0.3,  # Example criteria; adjust as necessary
                'pct_change_90': 0.25,  # Negative values for reversal
                'pct_change_30': 0.2,
                'pct_change_15': 0.14,
                'pct_change_3': 0.01
            },
            'volume_criteria': {
                'percent_of_premarket_vol': .001  # Example criterion; adjust as necessary
            }
        }
    }

    # Check if the stock type is valid
    if stock_type in criteria:
        for ticker, data in all_stocks_data.items():
            pct_criteria = criteria[stock_type]['pct_criteria']
            volume_criteria = criteria[stock_type]['volume_criteria']

            if does_stock_meet_criteria(data, pct_criteria, volume_criteria):
                filtered_stocks[ticker] = data
                percentiles = calculate_percentiles(reversal_df, data, columns_to_compare)
                # print(ticker, percentiles)
    else:
        print(f"Unknown stock type: {stock_type}")

    return filtered_stocks


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


if __name__ == '__main__':
    all_stock_data = get_all_stocks_data(watchlist)
    reversal_stocks = convert_dict_to_df(filter_stocks(all_stock_data, 'reversal'))
    momentum_stocks = convert_dict_to_df(filter_stocks(all_stock_data, 'momentum'))
    # print(tabulate(reversal_stocks, headers=reversal_stocks.columns))
    # print(tabulate(momentum_stocks, headers=momentum_stocks.columns))
    reversal_results = []
    momentum_results = []
    for ticker, data in all_stock_data.items():
        reversal_percentiles = calculate_percentiles(reversal_df, data, columns_to_compare)
        momentum_percentiles = calculate_percentiles(momentum_df, data, columns_to_compare)
        reversal_results.append((ticker, reversal_percentiles))
        momentum_results.append((ticker, momentum_percentiles))
        # print('reversal', ticker, reversal_percentiles)
        # print('momentum', ticker, momentum_percentiles)
    reversal_sorted = sorted(reversal_results, key=lambda x: x[1]['pct_change_3'], reverse=True)
    momentum_sorted = sorted(momentum_results, key=lambda x: x[1]['pct_change_3'], reverse=True)
    print("Sorted Reversal Stock Percentiles:")
    for ticker, percentiles in reversal_sorted:
        # Extract pct_data values for the ticker from all_stock_data
        pct_data = all_stock_data.get(ticker, {}).get('pct_data', {})
        pct_data_str = ', '.join([f"{k}: {v}" for k, v in pct_data.items()])
        print(f'reversal {ticker}, percentiles: {percentiles}, Absolute PCT Changes (in hundreds): {pct_data_str}')

    range_expansion_results = range_expansion_watcher(watchlist, date)
    range_expansion_results_sorted = range_expansion_results.sort_values(by='Percent of ATR', ascending=False)
    print("\nRange Expansion Results:")
    print(tabulate(range_expansion_results_sorted, headers=range_expansion_results_sorted.columns, tablefmt='grid'))

    # print("\nSorted Momentum Stocks Percentiles:")
    # for ticker, percentiles in momentum_sorted:
    #     print('momentum', ticker, percentiles)