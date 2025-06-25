from data_queries.polygon_queries import get_actual_current_price, get_levels_data, get_atr, fetch_and_calculate_volumes, get_ticker_pct_move
from datetime import datetime, timedelta
from tabulate import tabulate
import pandas as pd
from data_collectors.combined_data_collection import reversal_df, momentum_df
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
import os

# Adjust the date to the last market day if today is Saturday or Sunday
today = datetime.now()
if today.weekday() == 5:  # Saturday
    adjusted_date = today - timedelta(days=1)
elif today.weekday() == 6:  # Sunday
    adjusted_date = today - timedelta(days=2)
else:
    adjusted_date = today
date = adjusted_date.strftime('%Y-%m-%d')

columns_to_compare = [
    'pct_change_120', 'pct_change_90', 'pct_change_30', 'pct_change_15',
    'pct_change_3', 'percent_of_premarket_vol', 'percent_of_vol_one_day_before',
    'percent_of_vol_two_day_before', 'percent_of_vol_three_day_before', 'day_of_range_pct',
    'one_day_before_range_pct', 'two_day_before_range_pct', 'three_day_before_range_pct'
]
# Example watchlist
watchlist = ['CRCL','ORCL','CRWV','AAPL','GOOGL','NVDA','MP','USAR','OKLO','SMR','NBIS','TEM','CEP']

print(watchlist)

def add_range_data(ticker):
    """
    Function to watch for range expansion across a list of stocks.
    This implementation is IPO-friendly: if there are fewer trading days than the
    standard rolling windows (ATR 14 or ADV 20) the windows are reduced automatically
    and look-back calculations gracefully degrade to the data that is actually
    available.
    :param ticker: Stock ticker to scan for range expansion.
    """
    try:
        # Retrieve historical price/volume data. A large look-back is requested, but
        # get_levels_data will simply return what exists (e.g. only 10 days for a new IPO).
        df = get_levels_data(ticker, date, 60, 1, 'day')
        if df is None or df.empty:
            return {}

        # -----------------------------
        # Calculations
        # -----------------------------
        df['high-low'] = df['high'] - df['low']
        df['high-previous_close'] = abs(df['high'] - df['close'].shift())
        df['low-previous_close'] = abs(df['low'] - df['close'].shift())

        df['TR'] = df[['high-low', 'high-previous_close', 'low-previous_close']].max(axis=1)

        # Dynamically choose ATR window. 14 for mature stocks, otherwise whatever is available (min 1).
        atr_window = 14 if len(df) >= 14 else max(1, len(df))
        df['ATR'] = df['TR'].rolling(window=atr_window, min_periods=1).mean()

        # Percentage of ATR for each day
        df['PCT_ATR'] = df['TR'] / df['ATR']

        # Dynamically choose ADV window (20 by default)
        adv_window = 20 if len(df) >= 20 else max(1, len(df))
        df['20Day_Avg_Volume'] = df['volume'].rolling(window=adv_window, min_periods=1).mean()
        df['pct_avg_volume'] = df['volume'] / df['20Day_Avg_Volume']

        # Helper to fetch a value "n" rows back if it exists, else None
        def safe_iloc(series, n_back):
            return series.iloc[-n_back] if len(series) >= n_back else None

        # Prepare ATR percentages (today and up to three prior days)
        pct_of_atr = safe_iloc(df['PCT_ATR'], 1)
        day_before_pct_of_atr = safe_iloc(df['PCT_ATR'], 2)
        two_day_before_pct_of_atr = safe_iloc(df['PCT_ATR'], 3)
        three_day_before_pct_of_atr = safe_iloc(df['PCT_ATR'], 4)

        # Volume comparisons
        day_before_volume = safe_iloc(df['volume'], 2)
        two_day_before_volume = safe_iloc(df['volume'], 3)
        three_day_before_volume = safe_iloc(df['volume'], 4)

        day_before_adv = safe_iloc(df['20Day_Avg_Volume'], 2)
        two_day_before_adv = safe_iloc(df['20Day_Avg_Volume'], 3)
        three_day_before_adv = safe_iloc(df['20Day_Avg_Volume'], 4)

        def _pct_str(num, den):
            if num is None or den in (None, 0):
                return "N/A"
            pct = num / den
            return f"({num:,} / {den:,}) = {pct:.2f}"

        # Build result dict with graceful fall-backs
        result = {
            'percent_of_vol_one_day_before': _pct_str(day_before_volume, day_before_adv),
            'percent_of_vol_two_day_before': _pct_str(two_day_before_volume, two_day_before_adv),
            'percent_of_vol_three_day_before': _pct_str(three_day_before_volume, three_day_before_adv),
            'day_of_range_pct': _pct_str(safe_iloc(df['TR'],1), safe_iloc(df['ATR'],1)),
            'one_day_before_range_pct': _pct_str(safe_iloc(df['TR'],2), safe_iloc(df['ATR'],2)),
            'two_day_before_range_pct': _pct_str(safe_iloc(df['TR'],3), safe_iloc(df['ATR'],3)),
            'three_day_before_range_pct': _pct_str(safe_iloc(df['TR'],4), safe_iloc(df['ATR'],4)),
        }
        return result

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return {}

def add_percent_of_adv_columns(volume_data):
    # List of volume columns to compare with avg_daily_vol
    volume_columns = [
        'premarket_vol', 'vol_in_first_5_min', 'vol_in_first_15_min',
        'vol_in_first_10_min', 'vol_in_first_30_min', 'vol_on_breakout_day'
    ]

    # Ensure 'avg_daily_vol' is not 0 to avoid division by zero error
    if volume_data.get('avg_daily_vol', 0) > 0:
        avg_daily_vol = volume_data['avg_daily_vol']
        # Add new keys representing percent of avg_daily_vol
        for col in volume_columns:
            if col in volume_data:
                percent_col_name = f'percent_of_{col}'
                volume_data[percent_col_name] = volume_data[col] / avg_daily_vol
    else:
        # Handle case where avg_daily_vol is 0 or not present
        for col in volume_columns:
            percent_col_name = f'percent_of_{col}'
            volume_data[percent_col_name] = None

    return volume_data

def get_stock_data(ticker):
    # Pass the date argument to all query functions
    current_price = get_actual_current_price(ticker, date)
    pct_data = get_ticker_pct_move(ticker, date, current_price)
    volume_data = fetch_and_calculate_volumes(ticker, date)
    volume_data = add_percent_of_adv_columns(volume_data)
    range_data = add_range_data(ticker)
    return {ticker: {'pct_data': pct_data, 'volume_data': volume_data, 'range_data': range_data}}

def get_all_stocks_data(watchlist):
    all_data = {}
    for ticker in watchlist:
        ticker_data = get_stock_data(ticker)
        all_data.update(ticker_data)
    return all_data

def calculate_percentiles(df, stock_data, columns):
    """
    Calculate the percentile ranks for specified columns in stock_data compared to historical data in df.
    """
    percentiles = {}
    # Only consider columns that actually exist in df
    available_cols = [c for c in columns if c in df.columns]
    if available_cols:
        df = df.dropna(subset=available_cols)
    # Flatten stock_data for easier handling
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
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                # Skip strings or non-convertible values
                continue
            percentiles[column] = percentileofscore(df[column], numeric_value, kind='weak')

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
    return df

if __name__ == '__main__':
    all_stock_data = get_all_stocks_data(watchlist)
    reversal_stocks = convert_dict_to_df(all_stock_data)
    reversal_results = []

    for ticker, data in all_stock_data.items():
        reversal_percentiles = calculate_percentiles(reversal_df, data, columns_to_compare)
        reversal_results.append((ticker, reversal_percentiles))

    reversal_sorted = sorted(reversal_results, key=lambda x: x[1]['pct_change_3'], reverse=True)
    print("Sorted Reversal Stock Percentiles:")

    for ticker, percentiles in reversal_sorted:
        pct_data = all_stock_data.get(ticker, {}).get('pct_data', {})
        range_data = all_stock_data.get(ticker, {}).get('range_data', {})
        percentiles_str = '\n'.join([f"    {k}: {v:.2f}" for k, v in percentiles.items()])
        pct_data_str = '\n'.join([f"    {k}: {v:.2f}" for k, v in pct_data.items()])
        def _fmt(val):
            return f"{val:.2f}" if isinstance(val, (int, float)) else str(val)

        range_data_str = '\n'.join([f"    {k}: {_fmt(v)}" for k, v in range_data.items()])

        print(f"Reversal: {ticker}")
        print("  Percentiles:")
        print(percentiles_str if percentiles else "    None")
        print("  Absolute PCT Changes (in hundreds):")
        print(pct_data_str if pct_data else "    None")
        print("  Range Data:")
        print(range_data_str if range_data else "    None")
        print("-" * 50)
