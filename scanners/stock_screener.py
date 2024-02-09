from data_queries.polygon_queries import get_ticker_pct_move, get_actual_current_price, fetch_and_calculate_volumes
from datetime import datetime
from tabulate import tabulate
import pandas as pd
from data_collectors.combined_data_collection import reversal_df, momentum_df
from scipy.stats import percentileofscore

columns_to_compare = ['pct_change_120', 'pct_change_90', 'pct_change_30', 'pct_change_15', 'percent_of_premarket_vol']
ai_stocks = ['NVDA', 'AMD', 'ANET', 'SMCI', 'GOOG', 'PLTR', 'MSFT', 'META','VRT', 'AVGO', 'ARM', 'NTNX','SNOW', 'RXRX', 'CFLT', 'MDB', 'COHR', 'DELL']
# ai_stocks = ['NVDA']
date = datetime.now().strftime('%Y-%m-%d')


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
    return {ticker: {'pct_data': pct_data, 'volume_data': volume_data}}


def get_all_stocks_data(ai_stocks):
    all_data = {}
    for ticker in ai_stocks:
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
                'pct_change_15': 0.14
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
    all_stock_data = get_all_stocks_data(ai_stocks)
    reversal_stocks = convert_dict_to_df(filter_stocks(all_stock_data, 'reversal'))
    momentum_stocks = convert_dict_to_df(filter_stocks(all_stock_data, 'momentum'))
    # print(tabulate(reversal_stocks, headers=reversal_stocks.columns))
    # print(tabulate(momentum_stocks, headers=momentum_stocks.columns))
    for ticker, data in all_stock_data.items():
        reversal_percentiles = calculate_percentiles(reversal_df, data, columns_to_compare)
        momentum_percentiles = calculate_percentiles(momentum_df, data, columns_to_compare)
        print('reversal', ticker, reversal_percentiles)
        print('momentum', ticker, momentum_percentiles)