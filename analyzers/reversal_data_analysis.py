import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analyzers.momentum_data_analysis import analyze_volume_data_plotly, add_percent_of_adv_columns, analyze_percent_columns
import logging
from data_collectors.reversal_data_collection import df

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

pct_change_columns = ['pct_change_120', 'pct_change_90', 'pct_change_30', 'pct_change_15']
reversal_columns = ['reversal_open_low_pct', 'reversal_open_close_pct', 'reversal_open_post_low_pct',
                    'reversal_open_to_day_after_open_pct']


def clean_df(df):
    """
    Cleans the DataFrame by converting each column to its appropriate data type.

    Parameters:
    df (DataFrame): The DataFrame to be cleaned.

    Returns:
    DataFrame: The cleaned DataFrame.
    """

    # Convert date columns to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['time_of_reversal'] = pd.to_datetime(df['time_of_reversal'], utc=True)
    df['time_of_reversal'] = df['time_of_reversal'].dt.tz_convert('America/New_York')

    # Convert string/object columns to categorical
    categorical_cols = ['ticker', 'reversal_duration', 'trade_grade', 'cap']
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    # Convert numeric columns to float
    numeric_cols = [col for col in df.columns if col not in categorical_cols + ['date', 'time_of_reversal']]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def generate_heatmap(df):
    additional_columns = ['reversal_open_low_pct', 'reversal_open_close_pct', 'reversal_open_post_low_pct',
                          'reversal_open_to_day_after_open_pct']
    correlation_df = df[pct_change_columns + additional_columns]
    correlation_matrix = correlation_df.corr()
    fig = px.imshow(correlation_matrix, aspect='auto', title="Correlation Heatmap")
    fig.show()


def pct_change_reversal_scatters(df):
    comparison_column = 'reversal_open_low_pct'
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


def time_of_reversal(df):
    df['time_of_reversal'] = pd.to_datetime(df['time_of_reversal'], errors='coerce')
    df.dropna(subset=['time_of_reversal'], inplace=True)  # Drop rows where conversion failed
    df['reversal_time'] = df['time_of_reversal'].dt.time

    # Frequency analysis: Calculate the percentage of total reversals per time
    total_reversals = len(df)
    reversal_frequency = df['reversal_time'].value_counts().sort_index() / total_reversals * 100

    fig = px.bar(x=reversal_frequency.index, y=reversal_frequency.values,
                 labels={'x': 'Reversal Time', 'y': 'Percentage of Total Reversals (%)'})
    fig.update_layout(title="Distribution of Reversal Times as Percentage of Total (EST)", xaxis_title="Time of Day",
                      yaxis_title="Percentage of Total Reversals (%)")
    fig.show()


def boolean(df):
    boolean_columns = ['close_at_lows', 'move_together', 'close_green_red', 'hit_green_red', 'hit_prior_day_hilo',
                       'breaks_fifty_two_wk', 'breaks_ath']

    percent_true = {col: (df[col].sum() / len(df) * 100) for col in boolean_columns}
    percent_true_df = pd.DataFrame(list(percent_true.items()), columns=['Column', 'Percentage of True Values'])
    fig = px.bar(percent_true_df, x='Column', y='Percentage of True Values', text='Percentage of True Values')
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',
                      title="Percentage of True Values in Boolean Columns")
    fig.show()


def duration_analysis(df):
    df['reversal_duration'] = pd.to_timedelta(df['reversal_duration'])

    average_duration = df['reversal_duration'].mean()
    median_duration = df['reversal_duration'].median()
    print(f"Average Reversal Duration: {average_duration}")
    print(f"Median Reversal Duration: {median_duration}")


def reversal_pct_change(df):
    fig_box = go.Figure()
    for col in reversal_columns:
        fig_box.add_trace(go.Box(y=df[col], name=col))
    fig_box.update_layout(title="Box Plot of Percent Change Data", xaxis_title="Pct Change Types",
                          yaxis_title="Pct Change")
    fig_box.show()
    reversal_stats = df[reversal_columns].describe(percentiles=[.25, .5, .75])
    print(reversal_stats)


def analyze_by_cap_reversal(df, reversal_columns):
    """
    Analyzes reversal-related columns by the category of cap.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    reversal_columns (list): List of column names to analyze.

    Returns:
    DataFrame: A DataFrame with analysis results.
    """
    results = df.groupby('cap')[reversal_columns].agg(['mean', 'median', 'std', 'min', 'max', 'count'])
    return results


def analyze_by_cap_pct_return(df, pct_change_columns):
    """
    Analyzes pct_return columns by the category of cap.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    pct_change_columns (list): List of pct_return column names to analyze.

    Returns:
    DataFrame: A DataFrame with analysis results.
    """
    results = df.groupby('cap')[pct_change_columns].agg(['mean', 'median', 'std', 'min', 'max', 'count'])
    return results


def correlate_volume(df):
    correlation_day = df[['percent_of_adv_vol_on_breakout_day', 'reversal_open_low_pct']].corr()
    correlation_30 = df[['percent_of_adv_vol_in_first_30_min', 'reversal_open_low_pct']].corr()
    correlation_15 = df[['percent_of_adv_vol_in_first_15_min', 'reversal_open_low_pct']].corr()
    correlation_5 = df[['percent_of_adv_vol_in_first_5_min', 'reversal_open_low_pct']].corr()
    print(correlation_day)
    print(correlation_30)
    print(correlation_15)
    print(correlation_5)


if __name__ == '__main__':
    df = clean_df(df)
    df = add_percent_of_adv_columns(df)
    # correlate_volume(df)
    # analyze_volume_data_plotly(df)
    # generate_heatmap(df)
    # pct_change_reversal_scatters(df)
    # time_of_reversal(df)
    pct_change_box(df)
    boolean(df)
    # duration_analysis(df)
    reversal_pct_change(df)
    percent_analysis = analyze_percent_columns(df)
    analysis_df = df[pct_change_columns].describe()
    print(percent_analysis)
