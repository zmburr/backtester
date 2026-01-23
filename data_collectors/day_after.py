from data_queries.polygon_queries import get_daily, adjust_date_forward, get_levels_data, get_price_with_fallback, \
    adjust_date_to_market, get_intraday, check_pct_move, fetch_and_calculate_volumes
import pandas as pd
import logging
from tabulate import tabulate
from datetime import datetime, timedelta

momentum_df = pd.read_csv("C:\\Users\\zmbur\\PycharmProjects\\backtester\\data\\breakout_data.csv")
momentum_df = momentum_df.dropna(subset=['ticker'])
momentum_df = momentum_df.dropna(subset=['date'])


def get_day_after(df):
    """
    Get the day after a breakout for each stock in the DataFrame.

    :param df: DataFrame containing breakout data.
    :return: DataFrame with the day after breakout data.
    """
    day_after_data = []
    for index, row in df.iterrows():
        try:
            wrong_date = datetime.strptime(row['date'], '%m/%d/%Y')
            date = datetime.strftime(wrong_date, '%Y-%m-%d')
            day_of = get_daily(row['ticker'], date)
            day_of_high = day_of.high
            day_of_low = day_of.low
            day_of_open = day_of.open
            day_of_close = day_of.close
            day_of_pct_of_high_close = 1 - (day_of_close/day_of.high)
            day_after = get_daily(row['ticker'], adjust_date_forward(row['date'], 1))
            day_high = day_after.high
            day_low = day_after.low
            day_open = day_after.open
            day_close = day_after.close
            day_move = (day_of_high - day_of_open) / day_of_open
            day_move_pct = (day_close - day_open) / day_open
            print(f'{row["ticker"]} on {row["date"]} - The previous day move was {day_move} and led to a {day_move_pct} percent move the next day.')
        except Exception as e:
            logging.error(f"Error getting day after data for {row['ticker']} on {row['date']}: {e}")
            day_after_data.append(None)
    return day_after_data


new_df = get_day_after(momentum_df)

