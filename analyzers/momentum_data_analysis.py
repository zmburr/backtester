import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
from pytz import timezone
import logging
from data_collectors.momentum_data_collection import df


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

pct_change_columns = ['pct_change_120', 'pct_change_90', 'pct_change_30', 'pct_change_15']


def clean_df(df):
    # Convert date fields to datetime
    df['date'] = pd.to_datetime(df['date'])
    df['time_of_breakout'] = pd.to_datetime(df['time_of_breakout'], utc=True)
    df['time_of_breakout'] = df['time_of_breakout'].dt.tz_convert('America/New_York')

    # Convert 'breakout_duration' to timedelta
    df['breakout_duration'] = pd.to_timedelta(df['breakout_duration'])

    # Ensure categorical data are treated as category dtype
    categorical_columns = ['ticker', 'trade_grade', 'news_type']
    for col in categorical_columns:
        df[col] = df[col].astype('category')


    return df


def add_percent_of_adv_columns(df):
    # List of volume columns to compare with avg_daily_vol
    volume_columns = ['premarket_vol', 'vol_in_first_5_min', 'vol_in_first_15_min', 'vol_in_first_10_min',
                      'vol_in_first_30_min', 'vol_on_breakout_day']

    # Add new columns representing percent of avg_daily_vol
    for col in volume_columns:
        percent_col_name = f'percent_of_adv_{col}'
        df[percent_col_name] = (df[col] / df['avg_daily_vol']) * 100

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


def generate_heatmap(df):
    additional_columns = ['breakout_open_high_pct', 'breakout_open_post_high_pct',
                          'breakout_open_to_day_after_open_pct']
    correlation_df = df[pct_change_columns + additional_columns]
    correlation_matrix = correlation_df.corr()
    fig = px.imshow(correlation_matrix, aspect='auto', title="Correlation Heatmap")
    fig.show()


def pct_change_breakout_scatters(df):
    comparison_column = 'breakout_open_high_pct'
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


def time_of_breakout(df):
    df['time_of_breakout'] = pd.to_datetime(df['time_of_breakout'], errors='coerce')
    df.dropna(subset=['time_of_breakout'], inplace=True)  # Drop rows where conversion failed
    df['breakout_time'] = df['time_of_breakout'].dt.time

    # Frequency analysis: Calculate the percentage of total breakouts per time
    total_breakouts = len(df)
    breakout_frequency = df['breakout_time'].value_counts().sort_index() / total_breakouts * 100

    fig = px.bar(x=breakout_frequency.index, y=breakout_frequency.values,
                 labels={'x': 'Breakout Time', 'y': 'Percentage of Total Breakouts (%)'})
    fig.update_layout(title="Distribution of Breakout Times as Percentage of Total (EST)", xaxis_title="Time of Day",
                      yaxis_title="Percentage of Total Breakouts (%)")
    fig.show()


def boolean(df):
    boolean_columns = ['t', 'move_together', 'close_at_highs', 'breaks_ath', 'breaks_fifty_two_wk']
    percent_true = {col: (df[col].sum() / len(df) * 100) for col in boolean_columns}
    percent_true_df = pd.DataFrame(list(percent_true.items()), columns=['Column', 'Percentage of True Values'])
    fig = px.bar(percent_true_df, x='Column', y='Percentage of True Values', text='Percentage of True Values')
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',
                      title="Percentage of True Values in Boolean Columns")
    fig.show()


def duration_analysis(df):
    df['breakout_duration'] = pd.to_timedelta(df['breakout_duration'])

    average_duration = df['breakout_duration'].mean()
    median_duration = df['breakout_duration'].median()
    print(f"Average Breakout Duration: {average_duration}")
    print(f"Median Breakout Duration: {median_duration}")


def news_type_pie(df):
    news_type_distribution = df['news_type'].value_counts()
    fig = px.pie(names=news_type_distribution.index, values=news_type_distribution.values,
                 title='Distribution of News Types')
    fig.show()


def breakout_pct_change(df):
    breakout_columns = ['breakout_open_high_pct', 'breakout_open_close_pct', 'breakout_open_post_high_pct',
                        'breakout_open_to_day_after_open_pct']
    fig_box = go.Figure()
    for col in breakout_columns:
        fig_box.add_trace(go.Box(y=df[col], name=col))
    fig_box.update_layout(title="Box Plot of Percent Change Data", xaxis_title="Pct Change Types",
                          yaxis_title="Pct Change")
    fig_box.show()
    breakout_stats = df[breakout_columns].describe(percentiles=[.25, .5, .75])
    print(breakout_stats)


def correlate_volume(df):
    correlation_day = df[['percent_of_adv_vol_on_breakout_day', 'breakout_open_high_pct']].corr()
    correlation_30 = df[['percent_of_adv_vol_in_first_30_min', 'breakout_open_high_pct']].corr()
    correlation_15 = df[['percent_of_adv_vol_in_first_15_min', 'breakout_open_high_pct']].corr()
    correlation_5 = df[['percent_of_adv_vol_in_first_5_min', 'breakout_open_high_pct']].corr()
    print(correlation_day)
    print(correlation_30)
    print(correlation_15)
    print(correlation_5)


if __name__ == '__main__':
    df = clean_df(df)
    df = add_percent_of_adv_columns(df)
    correlate_volume(df)
    analyze_volume_data_plotly(df)
    generate_heatmap(df)
    pct_change_breakout_scatters(df)
    time_of_breakout(df)
    pct_change_box(df)
    boolean(df)
    duration_analysis(df)
    breakout_pct_change(df)
    percent_analysis = analyze_percent_columns(df)
    analysis_df = df[pct_change_columns].describe()
    print(percent_analysis)
