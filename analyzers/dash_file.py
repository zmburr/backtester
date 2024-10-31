import pandas as pd
import dash
from dash import dcc, html
from data_collectors.combined_data_collection import reversal_df
from graphs.pct_reversal_graphs import plot_reversal_pct_connection_plotly
from graphs.scatter_with_setup_adv import create_interactive_scatter_dropdown

pct_rev_graph = plot_reversal_pct_connection_plotly(reversal_df)
vol_pct_scatter = create_interactive_scatter_dropdown(reversal_df)

app = dash.Dash(__name__)

app.layout = html.Div([
        html.H1("Open to Low Reversal Percentage Analysis"),
        dcc.Graph(id="scatter-low-chart", figure=pct_rev_graph),
        html.H1("Open to Close Reversal Percentage Analysis"),
        dcc.Graph(id="scatter-close-chart", figure=plot_reversal_pct_connection_plotly),
        html.H1("Volume to Reversal Pct Scatter"),
        dcc.Graph(id="scatter-vol-chart", figure=vol_pct_scatter)
    ])

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)