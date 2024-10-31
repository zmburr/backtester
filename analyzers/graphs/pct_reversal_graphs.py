import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
import numpy as np

def plot_reversal_pct_connection_plotly(data):
    # Convert date column to datetime if it's not already
    data['time_of_reversal'] = pd.to_datetime(data['time_of_reversal'])

    # Initialize a list for all traces
    traces = []

    # Loop through each row to plot points and connecting lines
    for _, row in data.iterrows():
        x = row['time_of_reversal']
        y1 = row['reversal_open_close_pct']
        y2 = row['reversal_open_low_pct']
        ticker = row['ticker']

        # Add line trace between points
        line_trace = go.Scatter(
            x=[x, x],
            y=[y1, y2],
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False  # No legend for lines
        )

        # Add scatter point for Open to Close with hover info
        open_close_trace = go.Scatter(
            x=[x],
            y=[y1],  # Adding jitter to avoid overlap
            mode='markers',
            marker=dict(color='red', size=8),
            name='Open to Close' if _ == 0 else "",  # Only show legend once
            hovertemplate=f"Ticker: {ticker}<br>Date: {x}<br>Open to Close: {y1}"
        )

        # Add scatter point for Open to Low with hover info
        open_low_trace = go.Scatter(
            x=[x],
            y=[y2],  # Adding jitter to avoid overlap
            mode='markers',
            marker=dict(color='blue', size=8),
            name='Open to Low' if _ == 0 else "",  # Only show legend once
            hovertemplate=f"Ticker: {ticker}<br>Date: {x}<br>Open to Low: {y2}"
        )

        # Append traces to the list
        traces.extend([line_trace, open_close_trace, open_low_trace])

    # Define layout
    layout = go.Layout(
        title="Reversal Open to Close and Open to Low Percentage",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Reversal Percentage"),
        showlegend=True
    )

    # Create figure
    fig = go.Figure(data=traces, layout=layout)

    # Show figure
    return fig


