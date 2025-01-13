from datetime import datetime, timedelta
from data_queries.polygon_queries import poly_client
import pandas as pd
from pytz import timezone
import pytz
from dateutil import tz

pd.options.mode.chained_assignment = None
import logging
import numpy as np

def get_data(self):
    aggs = []
    try:
        for a in self.client.list_aggs(
                ticker=self.ticker,
                multiplier=1,
                timespan="minute",
                from_=self.start_date,
                to=self.end_date,
                limit=50000
        ):
            aggs.append(a)
    except KeyError:
        print('Data does not exist for', self.ticker, self.start_date)
        return None
    except self.client.exceptions.BadResponse as e:
        print(f"API returned a bad response: {e}")
        return None

    df = pd.DataFrame([vars(a) for a in aggs])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(timezone('US/Eastern'))
    df.set_index('timestamp', inplace=True)

    # Convert self.updated to a time string
    if isinstance(self.updated, pd.Timestamp):
        self.updated = self.updated.time().strftime('%H:%M:%S')

    # Restrict to everything up to self.updated
    df = df.between_time(start_time='00:00', end_time=self.updated)
    return df

class GetReferencePrice:
    def __init__(self, ticker, start_date, updated, side):
        self.ticker = ticker
        self.client = poly_client
        self.side = side
        self.start_date = start_date
        self.end_date = self.start_date

        # If 'updated' might be a full datetime or Timestamp, convert it here
        # to a time-only string, e.g. "HH:MM:SS"
        if isinstance(updated, (pd.Timestamp, datetime)):
            updated = updated.strftime('%H:%M:%S')
        elif isinstance(updated, str):
            # If it's a string with a date, parse and keep only the time
            try:
                dt_obj = pd.to_datetime(updated)
                updated = dt_obj.strftime('%H:%M:%S')
            except ValueError:
                # fallback if updated is already "HH:MM" or "HH:MM:SS"
                pass

        self.updated = updated  # now guaranteed to be "HH:MM:SS"

        # We’ll use these to replicate File 1's logic
        self.window = 17
        self.lookback_minutes = 40

        self.data = self.get_data()

    def get_data(self):
        aggs = []
        try:
            for a in self.client.list_aggs(
                    ticker=self.ticker,
                    multiplier=1,
                    timespan="minute",
                    from_=self.start_date,
                    to=self.end_date,
                    limit=50000
            ):
                aggs.append(a)
        except KeyError:
            print('data doesnt exist', self.ticker, self.start_date)
            return None
        except self.client.exceptions.BadResponse as e:
            print(f"API returned a bad response: {e}")
            return None

        df = pd.DataFrame([vars(a) for a in aggs])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(timezone('US/Eastern'))
        df.set_index('timestamp', inplace=True)

        # Restrict to everything up to self.updated (File 1 logic only looks back from updated time)
        df = df.between_time(start_time='00:00', end_time=self.updated)
        return df

    ######################################################################
    #               Helper Methods From File 1 (Logic Transfer)          #
    ######################################################################

    def adjust_time(self, time_obj, minutes, direction):
        """
        Adjusts a datetime object by 'minutes', either forward or backward.
        If adjusted time is earlier than 09:30:00, sets it to 09:30:00.
        Returns a string in '%H:%M:%S' format.
        """
        # If self.updated is just HH:MM, parse properly
        # (If it already has HH:MM:SS, adjust this logic accordingly)
        if isinstance(time_obj, str):
            # Attempt to parse as %H:%M:%S or fallback to %H:%M
            try:
                time_obj = datetime.strptime(time_obj, '%H:%M:%S')
            except ValueError:
                time_obj = datetime.strptime(time_obj, '%H:%M')

        delta = timedelta(minutes=minutes)
        if direction == 'forward':
            time_obj += delta
        elif direction == 'back':
            time_obj -= delta

        nine_thirty = datetime.strptime('09:30:00', '%H:%M:%S')
        # Force date from time_obj, just keep the time check
        if time_obj.time() < nine_thirty.time():
            time_obj = time_obj.replace(
                hour=nine_thirty.hour,
                minute=nine_thirty.minute,
                second=nine_thirty.second
            )
        return datetime.strftime(time_obj, '%H:%M:%S')

    def check_halt(self, df):
        """
        Checks for a gap of >= 5 minutes between consecutive bars.
        If found, returns the row right before the gap.
        Otherwise, returns None.
        """
        if df is None or df.empty:
            return None

        for i in range(1, len(df)):
            time_diff = df.index[i] - df.index[i - 1]
            if time_diff >= pd.Timedelta(minutes=5):
                return df.iloc[i - 1]
        return None

    def calculate_new_columns(self, df, window):
        """
        Calculates rolling average volume and volume_to_avg_ratio.
        Returns the df with added columns, skipping the 9:30 bar.
        """
        df['rolling_avg_volume'] = df['volume'].rolling(window=window).mean()
        df['volume_to_avg_ratio'] = round(df['volume'] / df['rolling_avg_volume'], 2)
        temp_df = df[~(df.index.time == pd.to_datetime('9:30').time())]
        return temp_df

    def find_volume_spike(self, temp_df):
        """
        Looks for volume spikes exceeding (mean + 2.5 * std_dev)
        in volume_to_avg_ratio. Returns only the rows exceeding
        that threshold, sorted descending.
        """
        if temp_df is None or temp_df.empty:
            return pd.DataFrame()

        mean = temp_df['volume_to_avg_ratio'].mean()
        std_dev = temp_df['volume_to_avg_ratio'].std()
        threshold = mean + (2.5 * std_dev)
        sorted_df = temp_df.sort_values('volume_to_avg_ratio', ascending=False)
        filtered_df = sorted_df[sorted_df['volume_to_avg_ratio'] >= threshold]
        return filtered_df

    ######################################################################
    #            Original File 2 Methods (Left Intact)                   #
    #         (We do not remove them, but we won't use them now)         #
    ######################################################################

    def add_return(self, df, timestamp):
        index = df.index.get_loc(timestamp)
        initial_price = df.iloc[index]['open']
        if index + self.window < len(df):
            future_price = df.iloc[index + self.window]['close']
        else:
            future_price = df.iloc[-1]['close']
        percent_return = ((future_price - initial_price) / initial_price) * 100
        df.loc[timestamp, 'percent_return'] = abs(percent_return)
        return df

    def find_transaction_spike(self, df):
        df['vol_per_trans'] = df['volume'] / df['transactions']
        temp_df = df[~(df.index.time == pd.to_datetime('9:30').time())]
        df = temp_df.sort_values('vol_per_trans', ascending=True)
        return df

    def find_price_spike(self, df):
        df = df.between_time(start_time='6:00', end_time='20:00:00')
        df['price_range'] = df['high'] - df['low']
        df['avg_price_range'] = df['price_range'].rolling(window=self.window).mean()
        df['range_ratio'] = df['price_range'] / df['avg_price_range']
        temp_df = df[~(df.index.time == pd.to_datetime('9:30').time())]
        sorted_df = temp_df.sort_values('range_ratio', ascending=False)
        return sorted_df.head(10)

    def compare_df(self, df1, df2, df3):
        index_df1 = set(df1.index)
        index_df2 = set(df2.index)
        index_df3 = set(df3.index)
        common_indices = index_df1.intersection(index_df2, index_df3)
        common_indices = list(common_indices)
        common_rows_df1 = df1.loc[common_indices, ['volume_to_avg_ratio']]
        common_rows_df2 = df2.loc[common_indices, ['range_ratio']]
        common_rows_df3 = df3.loc[common_indices, ['vol_per_trans']]
        common_df = pd.merge(common_rows_df1, common_rows_df2, left_index=True, right_index=True)
        common_df = pd.merge(common_rows_df, common_rows_df3, left_index=True, right_index=True)
        common_df['vol_multiplier'] = (common_df['volume_to_avg_ratio'] * common_df['range_ratio']) / common_df[
            'vol_per_trans']
        sorted_df = common_df.sort_values('vol_multiplier', ascending=False)
        return sorted_df

    ######################################################################
    #                 Replaced Logic for get_reference_price             #
    #                 (Now Uses File 1 Simpler Logic)                    #
    ######################################################################

    def get_reference_price(self):
        """
        Uses the simpler File 1 logic to compute a reference price.
        Returns a dict: {'ref_price': <float or 0>, 'ref_time': <pd.Timestamp or datetime>}
        """
        logging.info(f'Getting reference price for {self.ticker} on {self.start_date} at {self.updated}')

        # If self.updated doesn't have seconds, ensure we parse it properly
        try:
            updated_time = datetime.strptime(self.updated, '%H:%M:%S')
        except ValueError:
            updated_time = datetime.strptime(self.updated, '%H:%M')

        # Subset the self.data from (updated_time - lookback_minutes) up to updated_time
        headline_start_str = self.adjust_time(updated_time, self.lookback_minutes, 'back')
        headline_end_str = self.adjust_time(updated_time, 0, 'back')

        # If data is None or empty, just return early
        if self.data is None or self.data.empty:
            return {'ref_price': 0, 'ref_time': None}

        try:
            subset_df = self.data.between_time(
                start_time=headline_start_str,
                end_time=headline_end_str
            )
        except Exception as e:
            logging.error(f"Error subsetting dataframe: {e}")
            return {'ref_price': 0, 'ref_time': None}

        if subset_df.empty:
            return {'ref_price': 0, 'ref_time': None}

        # Check for halts
        halt = self.check_halt(subset_df)
        # We won't do anything special with 'halt' in this version, but you could

        # Calculate rolling columns
        new_data = self.calculate_new_columns(subset_df, self.window)

        # Find volume spike
        df1 = self.find_volume_spike(new_data)

        # If we found a spike, reference price is the earliest spike's 'open'
        if not df1.empty:
            spike_time = df1.index.min()
            ref_price = df1.loc[spike_time]['open']
            if not isinstance(ref_price, (float, int, np.float64)):
                # If we somehow got a Series
                ref_price = ref_price.iloc[0]
            logging.info(f'Got reference price: {ref_price}')
            return {'ref_price': ref_price, 'ref_time': spike_time}

        # Otherwise, reference price is the last bar's 'open' in our subset
        else:
            last_time = subset_df.index.max()
            ref_price = subset_df.loc[last_time]['open']
            if not isinstance(ref_price, (float, int, np.float64)):
                ref_price = ref_price.iloc[0]
            logging.info(f'Got reference price: {ref_price}')
            return {'ref_price': ref_price, 'ref_time': last_time}


def get_reference_price(df):
    for index, row in df.iterrows():
        tag = row['Tags']
        if not any(tag.strip() == 'news' for tag in tag.split(',')):
            continue

        ticker = row['Symbol']
        try:
            trade_date = datetime.strptime(row['Date'], "%m/%d/%Y")
        except AttributeError:
            print(row['Date'])
            trade_date = row['Date']
        formatted_date = trade_date.strftime("%Y-%m-%d")

        start = row['Start']
        est_tz = tz.gettz('America/New_York')
        start = start.replace(tzinfo=est_tz)
        side = row['Side']

        ref = GetReferencePrice(ticker, formatted_date, start.time().strftime('%H:%M:%S'), side)
        ref = ref.get_reference_price()
        ref_price = ref['ref_price']

        try:
            if row['Avg Price at Max'] < ref_price and row['Side'] == 1:
                ref_price = row['Avg Price at Max'] - .05
            elif row['Avg Price at Max'] > ref_price and row['Side'] == -1:
                ref_price = row['Avg Price at Max'] + .05
        except Exception as e:
            print('error in ref price in processor:', e)

        reftime = ref['ref_time']
        if reftime is None:
            df.loc[index, 'ref_time'] = start
            df.loc[index, 'ref_price'] = 0
            continue

        timedif = start - reftime
        if timedif.total_seconds() / 60 > 10 or timedif.total_seconds() / 60 < -10:
            df.loc[index, 'ref_time'] = start
            data = get_data(ticker, poly_client, formatted_date, start.time())['open'].iloc[0]
            df.loc[index, 'ref_price'] = data
            print(ticker, get_data(ticker, poly_client, formatted_date, start.time())['open'].iloc[0])
        else:
            df.loc[index, 'ref_time'] = reftime
            df.loc[index, 'ref_price'] = ref_price - .05
            logging.info(f'got reference price: {ref_price}')
    return df

if __name__ == "__main__":
    analysis = GetReferencePrice('AMD', '2024-12-26', datetime.strptime('09:50:04', '%H:%M:%S').strftime('%H:%M'), -1)
    result = analysis.get_reference_price()
    print(result)  # -> {'ref_price': some_float, 'ref_time': some_timestamp}
