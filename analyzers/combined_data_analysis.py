import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
from pytz import timezone
import logging
from data_collectors.combined_data_collection import momentum_df
from data_collectors.combined_data_collection import reversal_df

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

pct_change_columns = ['pct_change_120', 'pct_change_90', 'pct_change_30', 'pct_change_15']
reversal_columns = ['reversal_open_low_pct', 'reversal_open_close_pct', 'reversal_open_post_low_pct',
                    'reversal_open_to_day_after_open_pct']


def clean_df(df, analysis_type):
    """
    Cleans the DataFrame by converting columns to their appropriate data types,
    tailored for either breakout or reversal analysis.

    Parameters:
    - df: The DataFrame to be cleaned.
    - analysis_type: A string specifying the type of analysis ('breakout' or 'reversal').

    Returns:
    - The cleaned DataFrame.
    """

    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Convert time_of_event (either breakout or reversal) to datetime and adjust timezone
    time_col = 'time_of_breakout' if analysis_type == 'breakout' else 'time_of_reversal'
    df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_convert('America/New_York')

    # Convert duration to timedelta
    duration_col = 'breakout_duration' if analysis_type == 'breakout' else 'reversal_duration'
    if duration_col in df.columns:
        df[duration_col] = pd.to_timedelta(df[duration_col])

    # Convert specific columns to 'category' data type
    categorical_columns = ['ticker', 'trade_grade'] + (['news_type'] if analysis_type == 'breakout' else ['cap'])
    for col in categorical_columns:
        df[col] = df[col].astype('category')

    # Convert numeric columns to float, excluding specific non-numeric columns
    exclude_cols = categorical_columns + ['date', time_col, duration_col]
    numeric_cols = [col for col in df.columns if col not in exclude_cols]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def analyze_volume_data_plotly(df):
    # Filter columns that represent percent of ADV
    percent_columns = [col for col in df.columns if col.startswith('percent_of_adv_')]

    # Plot scatter charts for percent columns
    for col in percent_columns:
        fig = px.scatter(df, x='date', y=col, hover_data=['ticker', 'date'], title=f"{col} over Time")
        fig.show()

    # Histograms and Density Plots
    for col in percent_columns:
        fig_hist = px.histogram(df, x=col, marginal='box', nbins=30, title=f"Histogram of {col}")
        fig_hist.show()

    # Box Plots
    fig_box = go.Figure()
    for col in percent_columns:
        fig_box.add_trace(go.Box(y=df[col], name=col))
    fig_box.update_layout(title="Box Plot of Volume Data", xaxis_title="Volume Types", yaxis_title="Volume")
    fig_box.show()


def analyze_percent_columns(df):
    # Filter columns that represent percent of ADV
    percent_columns = [col for col in df.columns if col.startswith('percent_of_adv_')]

    # Create a dictionary to store the analysis results
    analysis_results = {}

    # Calculate min, max, mean, median, standard deviation, and percentiles for each percent column
    for col in percent_columns:
        analysis_results[col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            '25th_percentile': df[col].quantile(0.25),
            'median': df[col].median(),
            '75th_percentile': df[col].quantile(0.75),
            'std_deviation': df[col].std()
        }

    # Convert the results dictionary to a DataFrame for better readability
    analysis_df = pd.DataFrame(analysis_results).T

    return analysis_df


def generate_heatmap(df, analysis_type):
    """
    Generates a correlation heatmap for either breakout or reversal analysis.

    Parameters:
    - df: DataFrame containing the data.
    - analysis_type: A string specifying the type of analysis ('breakout' or 'reversal').

    Returns:
    - Displays a correlation heatmap.
    """
    # Select additional columns based on analysis type
    if analysis_type == 'breakout':
        additional_columns = ['breakout_open_high_pct', 'breakout_open_post_high_pct',
                              'breakout_open_to_day_after_open_pct']
    elif analysis_type == 'reversal':
        additional_columns = ['reversal_open_low_pct', 'reversal_open_close_pct', 'reversal_open_post_low_pct',
                              'reversal_open_to_day_after_open_pct']
    else:
        raise ValueError("Invalid analysis type. Please specify 'breakout' or 'reversal'.")

    # Combine the columns for correlation analysis
    correlation_df = df[pct_change_columns + additional_columns]

    # Calculate correlation matrix
    correlation_matrix = correlation_df.corr()

    # Generate and display the heatmap
    fig = px.imshow(correlation_matrix, aspect='auto', title=f"Correlation Heatmap - {analysis_type.capitalize()}")
    fig.show()


def pct_change_breakout_scatters(df, analysis_type):
    comparison_column = 'breakout_open_high_pct' if analysis_type == 'breakout' else 'reversal_open_low_pct'
    for col in pct_change_columns:
        fig = px.scatter(df, x=col, y=comparison_column, hover_data=['ticker', 'date'],
                         title=f"{col} vs {comparison_column}")
        fig.show()


def pct_change_box(df):
    fig_box = go.Figure()
    for col in pct_change_columns:
        fig_box.add_trace(go.Box(y=df[col], name=col))
    fig_box.update_layout(title="Box Plot of Percent Change Data", xaxis_title="Pct Change Types",
                          yaxis_title="Pct Change")
    fig_box.show()


def time_of_event(df, analysis_type):
    """
    Analyzes the distribution of breakout or reversal times and displays it as a percentage of the total.

    Parameters:
    - df: DataFrame containing the data.
    - analysis_type: A string specifying the type of analysis ('breakout' or 'reversal').

    Returns:
    - Displays a bar chart visualizing the distribution of event times.
    """
    time_col = 'time_of_breakout' if analysis_type == 'breakout' else 'time_of_reversal'
    event_time_col = 'breakout_time' if analysis_type == 'breakout' else 'reversal_time'

    # Convert time column to datetime, drop NA, and extract time
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df.dropna(subset=[time_col], inplace=True)
    df[event_time_col] = df[time_col].dt.time

    # Frequency analysis
    total_events = len(df)
    event_frequency = df[event_time_col].value_counts().sort_index() / total_events * 100

    # Generate and display the bar chart
    fig = px.bar(x=event_frequency.index, y=event_frequency.values,
                 labels={'x': 'Time of Day', 'y': f'Percentage of Total {analysis_type.capitalize()}s (%)'})
    title = f"Distribution of {analysis_type.capitalize()} Times as Percentage of Total (EST)"
    fig.update_layout(title=title, xaxis_title="Time of Day",
                      yaxis_title=f"Percentage of Total {analysis_type.capitalize()}s (%)")
    fig.show()


def boolean(df):
    # Merge the lists of boolean columns from both versions, ensuring no duplicates
    boolean_columns = [
        't', 'move_together', 'close_at_highs', 'breaks_ath', 'breaks_fifty_two_wk', 'close_at_lows',
         'close_green_red', 'hit_green_red', 'hit_prior_day_hilo']

    # Calculate the percentage of True values for each boolean column
    percent_true = {col: (df[col].sum() / len(df) * 100) for col in boolean_columns if col in df.columns}
    percent_true_df = pd.DataFrame(list(percent_true.items()), columns=['Column', 'Percentage of True Values'])

    # Generate and display the bar chart
    fig = px.bar(percent_true_df, x='Column', y='Percentage of True Values', text='Percentage of True Values')
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',
                      title="Percentage of True Values in Boolean Columns")
    fig.show()


def duration_analysis(df, analysis_type):
    if analysis_type == 'breakout':
        df['breakout_duration'] = pd.to_timedelta(df['breakout_duration'])
        average_duration = df['breakout_duration'].mean()
        median_duration = df['breakout_duration'].median()
        print(f"Average Breakout Duration: {average_duration}")
        print(f"Median Breakout Duration: {median_duration}")
    else:
        df['reversal_duration'] = pd.to_timedelta(df['reversal_duration'])
        average_duration = df['reversal_duration'].mean()
        median_duration = df['reversal_duration'].median()
        print(f"Average Reversal Duration: {average_duration}")
        print(f"Median Reversal Duration: {median_duration}")


def news_type_pie(df= momentum_df):
    news_type_distribution = df['news_type'].value_counts()
    fig = px.pie(names=news_type_distribution.index, values=news_type_distribution.values,
                 title='Distribution of News Types')
    fig.show()


def correlate_volume(df):
    correlation_day = df[['percent_of_adv_vol_on_breakout_day', 'breakout_open_high_pct']].corr()
    correlation_30 = df[['percent_of_adv_vol_in_first_30_min', 'breakout_open_high_pct']].corr()
    correlation_15 = df[['percent_of_adv_vol_in_first_15_min', 'breakout_open_high_pct']].corr()
    correlation_5 = df[['percent_of_adv_vol_in_first_5_min', 'breakout_open_high_pct']].corr()
    print(correlation_day)
    print(correlation_30)
    print(correlation_15)
    print(correlation_5)


def pct_change_analysis(df, analysis_type):
    """
    Generates a box plot and descriptive statistics for percent change data,
    tailored for either breakout or reversal analysis.

    Parameters:
    - df: DataFrame containing the data.
    - analysis_type: A string specifying the type of analysis ('breakout' or 'reversal').

    Returns:
    - Displays a box plot of percent change data and prints descriptive statistics.
    """
    if analysis_type == 'breakout':
        pct_change_columns = ['breakout_open_high_pct', 'breakout_open_close_pct',
                              'breakout_open_post_high_pct', 'breakout_open_to_day_after_open_pct']
    elif analysis_type == 'reversal':
        pct_change_columns = ['reversal_open_low_pct', 'reversal_open_close_pct',
                              'reversal_open_post_low_pct', 'reversal_open_to_day_after_open_pct']
    else:
        raise ValueError("Invalid analysis type. Please specify 'breakout' or 'reversal'.")

    # Generate box plots for the specified percent change columns
    fig_box = go.Figure()
    for col in pct_change_columns:
        fig_box.add_trace(go.Box(y=df[col], name=col))
    fig_box.update_layout(title=f"Box Plot of Percent Change Data - {analysis_type.capitalize()}",
                          xaxis_title="Pct Change Types", yaxis_title="Pct Change")
    fig_box.show()

    # Calculate and print descriptive statistics for the specified columns
    stats = df[pct_change_columns].describe(percentiles=[.25, .5, .75])
    print(stats)


if __name__ == '__main__':
    reversal_df = clean_df(reversal_df, 'reversal')
    breakout_df = clean_df(momentum_df, 'breakout')
    pct_change_analysis(reversal_df, 'reversal')
    pct_change_analysis(breakout_df, 'breakout')
    boolean(reversal_df)
    boolean(breakout_df)
    duration_analysis(reversal_df, 'reversal')
    duration_analysis(breakout_df, 'breakout')
    time_of_event(reversal_df, 'reversal')
    time_of_event(breakout_df, 'breakout')