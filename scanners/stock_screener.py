from data_queries.ticker_cache import TickerCache, is_today_finalized
from data_queries.local_metrics import compute_screener_metrics
import data_queries.polygon_queries as _pq
from data_queries.polygon_queries import get_intraday
from datetime import datetime, timedelta
from tabulate import tabulate
import pandas as pd
from data_collectors.combined_data_collection import reversal_df, momentum_df
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
import os

# Shared ticker cache instance (persistent across calls, thread-safe)
_cache = TickerCache()

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
watchlist = ['AMD','AAPL','GOOGL','NVDA',"CRWD","EWY",'GLW','AVGO','PLTR','ORCL','LITE','MSFT','MU','IONQ','WDC','STX','BITF','IREN','HYMC','HL','PAAS','SLV','GLD','MP','GDXJ','BE','OKLO','SMR','QS','RKLB','GWRE','APP','OPEN','CRML','FIGR','SNDK','PL','BETR','RGTI','CRWV','NBIS','CRDO','U','TSLA','HUBS','DOCU','DUOL','FIG','IBIT','ETHE','TEAM','MSTR']
# watchlist = ['TEAM','NOW','CRM','FIG','SAP','ADBE','PANW','ZS', 'CRWD' ,'INTU','HUBS','DOCU','DUOL','U','DDOG','TYL','MNDY','ASAN','NICE','GWRE','NET','GTLB']

print(watchlist)

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
                # Check if volume value is not None before dividing
                if volume_data[col] is not None:
                    volume_data[percent_col_name] = volume_data[col] / avg_daily_vol
                else:
                    volume_data[percent_col_name] = None
    else:
        # Handle case where avg_daily_vol is 0 or not present
        for col in volume_columns:
            percent_col_name = f'percent_of_{col}'
            volume_data[percent_col_name] = None

    return volume_data

def get_stock_data(ticker):
    """Fetch all screening metrics for *ticker* using cached daily bars + 1 intraday call.

    Reduces API calls from ~15/ticker to ~2/ticker (1 daily bars from cache,
    1 intraday for premarket/first-N-minute volume and current price).

    When today's session has closed, also tries the live-price API so that
    percentile and scoring calculations use the most current price available.
    """
    empty = {ticker: {'pct_data': {}, 'volume_data': {}, 'range_data': {}, 'mav_data': {}}}
    try:
        # 0-1 API calls (0 on cache hit, 1 on cache miss / incremental update)
        daily_bars = _cache.get_daily_bars(ticker, date, window=310)
        if daily_bars is None or daily_bars.empty:
            print(f"[Cache] No daily data for {ticker}")
            return empty

        # 1 API call for intraday data (volume metrics + current price)
        intraday = None
        current_price = None
        try:
            intraday = get_intraday(ticker, date, 1, 'minute')
            if intraday is not None and not intraday.empty:
                current_price = intraday.iloc[-1].close
                print(f"[Intraday] Current price for {ticker}: {current_price}")
        except Exception as e:
            print(f"[Intraday] Failed for {ticker}: {e}")

        # Try live-price API for the most up-to-date quote (e.g. after-hours).
        # Uses _pq module reference so the ReportCache monkey-patch is honoured.
        today_date = pd.to_datetime(date).date()
        if is_today_finalized(today_date):
            try:
                live = _pq.get_actual_current_price(ticker, date)
                if live is not None:
                    print(f"[Live] Current price for {ticker}: {live}")
                    current_price = live
            except Exception as e:
                print(f"[Live] Failed for {ticker}: {e}")

        # Fall back to last daily close if no intraday price
        if current_price is None:
            current_price = daily_bars.iloc[-1]['close']
            print(f"[Cache] Using last bar close for {ticker}: {current_price}")

        metrics = compute_screener_metrics(
            daily_bars, date, current_price=current_price, intraday=intraday,
        )
        return {ticker: metrics}

    except Exception as e:
        print(f"Error in get_stock_data for {ticker}: {e}")
        return empty

def get_all_stocks_data(watchlist, max_workers=8):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    all_data = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_stock_data, t): t for t in watchlist}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                all_data.update(future.result())
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
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
        flat_data = {
            **stock_data,
            **stock_data.get('pct_data', {}),
            **stock_data.get('volume_data', {}),
            **stock_data.get('range_data', {}),
            **stock_data.get('mav_data', {}),
        }
        # Remove nested dicts so DataFrame contains only scalars
        for nested_key in ('pct_data', 'volume_data', 'range_data', 'mav_data'):
            flat_data.pop(nested_key, None)
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
            if key in ('pct_data', 'volume_data', 'mav_data'):
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
        mav_data = all_stock_data.get(ticker, {}).get('mav_data', {})  # <-- add this

        percentiles_str = '\n'.join([f"    {k}: {v:.2f}" for k, v in percentiles.items()])
        pct_data_str = '\n'.join([f"    {k}: {v:.2f}" for k, v in pct_data.items()])


        def _fmt(val):
            return f"{val:.2f}" if isinstance(val, (int, float)) else str(val)


        range_data_str = '\n'.join([f"    {k}: {_fmt(v)}" for k, v in range_data.items()])
        mav_data_str = '\n'.join([f"    {k}: {_fmt(v)}" for k, v in mav_data.items()])  # <-- add this

        print(f"Reversal: {ticker}")
        print("  Percentiles:")
        print(percentiles_str if percentiles else "    None")
        print("  Absolute PCT Changes (in hundreds):")
        print(pct_data_str if pct_data else "    None")
        print("  Range Data:")
        print(range_data_str if range_data else "    None")
        print("  Mav Data (pct from MAs):")  # <-- add this
        print(mav_data_str if mav_data else "    None")  # <-- add this
        print("-" * 50)
