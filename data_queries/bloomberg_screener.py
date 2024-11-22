from xbbg import blp
import pandas as pd
from datetime import datetime

# Get today's date in the format 'YYYY-MM-DD'
today_date = datetime.today().strftime('%Y-%m-%d')

# Define a list of screen names to fetch tickers from
screen_names = ['Medium Cap Reversal Scanner', 'micro cap offering stocks',
                'large/mega reversal screen']  # Add as many screens as needed

# Initialize an empty list to store cleaned tickers
cleaned_tickers = []

# Iterate through each screen name and collect tickers
for screen in screen_names:
    # Run the equity screen
    screen_results = blp.beqs(screen=screen, asof=today_date)

    # Convert to DataFrame
    df = pd.DataFrame(screen_results)

    # Remove "US" from ticker symbols and add them to the cleaned tickers list
    if 'ticker' in df.columns:
        df['ticker_cleaned'] = df['ticker'].str.replace(" US", "")
        cleaned_tickers.extend(df['ticker_cleaned'].tolist())

# Remove any duplicates by converting to a set, then back to a list if necessary
cleaned_tickers = list(set(cleaned_tickers))


