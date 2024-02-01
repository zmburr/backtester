import plotly.graph_objects as go
from data_queries.polygon_queries import get_intraday
import logging


def create_chart(trade):
    logging.info(f'Creating chart for {trade.ticker} on {trade.date}')
    lt_df = get_intraday(trade.ticker, trade.date, 1, 'minute')

    # Create the candlestick figure
    fig = go.Figure(data=[go.Candlestick(x=lt_df.index,
                                         open=lt_df['open'],
                                         high=lt_df['high'],
                                         low=lt_df['low'],
                                         close=lt_df['close'])])

    # Define a function to add signals to the chart
    def add_signal_to_chart(time, price, color, label):
        if time and price:
            fig.add_trace(go.Scatter(x=[time], y=[price],
                                     mode='markers', marker=dict(color=color, size=10), name=label))
            fig.add_annotation(x=time, y=price, text=label, showarrow=True, arrowhead=1, yshift=10, bgcolor=color)

    # Add scatter plot for each signal type
    add_signal_to_chart(trade.premarket_low_break_time, trade.premarket_low_break_price, 'orange', 'Premarket Low Break')
    add_signal_to_chart(trade.premarket_high_break_time, trade.premarket_high_break_price, 'purple', 'Premarket High Break')
    add_signal_to_chart(trade.open_price_break_time, trade.open_price_break_price, 'pink', 'Open Price Break')
    add_signal_to_chart(trade.two_min_break_time, trade.two_min_break_price, 'cyan', '2 Min Break')
    add_signal_to_chart(trade.signal_time, trade.signal_price, 'yellow', 'Signal')
    add_signal_to_chart(f'{trade.signal_time[:10]} 15:59:59', trade.close_price, 'green', 'Close Price')

    # Add stop price line
    fig.add_shape(type='line', x0=min(lt_df.index), x1=max(lt_df.index), y0=trade.stop_price,
                  y1=trade.stop_price, line=dict(color="red", dash="dash"))
    fig.add_annotation(x=trade.signal_time, y=trade.stop_price, text="Stop Price",
                       showarrow=False, yshift=10, bgcolor="red")

    # Chart details
    fig.add_annotation(x=0, y=1, xref="paper", yref="paper",
                       text=f"Ticker: {trade.ticker}", showarrow=False, xanchor="left", yanchor="top",
                       bgcolor="lightgrey", bordercolor="black", borderwidth=2)

    # Update layout
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()