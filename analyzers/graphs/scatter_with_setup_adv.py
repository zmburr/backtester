import pandas as pd
from data_collectors.combined_data_collection import reversal_df
import plotly.express as px
from dash import dcc, html, Dash
from dash.dependencies import Input, Output


def create_interactive_scatter_dropdown(data):
    # Convert date column to datetime if it's not already
    data['date'] = pd.to_datetime(data['date'])

    # Filter out rows with missing values in 'setup' or 'reversal_open_low_pct'
    data = data.dropna(subset=['setup', 'reversal_open_low_pct'])

    # Identify columns related to 'percent_of_vol' for the dropdown
    percent_vol_columns = [col for col in data.columns if 'percent_of' in col]

    # Initialize the Dash app
    app = Dash(__name__)

    # Layout of the app
    app.layout = html.Div([
        html.H1("Scatter Plot of Reversal Open Low Percentage with Volume Percentage Selection"),
        dcc.Dropdown(
            id='percent-vol-dropdown',
            options=[{'label': col, 'value': col} for col in percent_vol_columns],
            value=percent_vol_columns[0],  # Default to the first column
            clearable=False
        ),
        dcc.Graph(id="scatter-chart")
    ])

    # Callback to update the scatter plot based on dropdown selection
    @app.callback(
        Output("scatter-chart", "figure"),
        Input("percent-vol-dropdown", "value")
    )
    def update_scatter(selected_column):
        # Create the scatter plot with dynamic sizing based on selected column
        fig = px.scatter(
            data,
            x='date',
            y='reversal_open_low_pct',
            size=selected_column,  # Size of points based on selected volume column
            color='setup',  # Color by setup type
            hover_data={'ticker': True, 'date': True, 'setup': True, selected_column: True},
            title="Scatter Plot of Reversal Open Low Percentage by Date with Setup and Volume Percentage",
            labels={'reversal_open_low_pct': 'Open to Low Reversal %', selected_column: 'Volume %'}
        )

        # Customize layout
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Reversal Open Low Percentage",
            showlegend=True,
            legend_title_text="Setup Type"
        )

        return fig

    # Run the Dash app (commented out as it cannot be run in this environment)
    app.run_server(debug=True)


# Call the function to initialize the app (This would be run locally)
create_interactive_scatter_dropdown(reversal_df)


