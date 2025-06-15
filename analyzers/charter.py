# Standard logging
import logging
# Plotly is still used for intraday trade charts, keep the import.
import plotly.graph_objects as go

from data_queries.polygon_queries import get_intraday

# New dependency for faster static charts
import mplfinance as mpf

# Matplotlib for figure management
import matplotlib.pyplot as plt


def create_trade_chart(trade, multiplier: int = 1, timespan: str = 'minute', show: bool = True):
    """
    Generate a candlestick chart for the given Trade object.

    Parameters
    ----------
    trade : Trade
        The trade instance containing ticker and date information.
    multiplier : int, optional
        The size of the aggregate window. For example, ``multiplier=5`` combined with
        ``timespan='minute'`` will produce 5-minute candles. Defaults to ``1``.
    timespan : str, optional
        The granularity of the data: ``'second'``, ``'minute'``, ``'hour'``, ``'day'`` …
        Defaults to ``'minute'``.
    show : bool, optional
        Whether to display the chart after creating it. Defaults to ``True``.
    """

    logging.info(
        f'Creating chart for {trade.ticker} on {trade.date} | multiplier={multiplier}, timespan={timespan}'
    )
    # Fetch intraday/aggregated data using the provided timeframe arguments
    lt_df = get_intraday(trade.ticker, trade.date, multiplier, timespan)

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
    fig.add_shape(type='line', x0=min(lt_df.index), x1=max(lt_df.index), y0=trade.best_stop_price,
                  y1=trade.best_stop_price, line=dict(color="red", dash="dash"))
    fig.add_annotation(x=trade.signal_time, y=trade.best_stop_price, text="Stop Price",
                       showarrow=False, yshift=10, bgcolor="red")

    # Chart details
    fig.add_annotation(x=0, y=1, xref="paper", yref="paper",
                       text=f"Ticker: {trade.ticker}", showarrow=False, xanchor="left", yanchor="top",
                       bgcolor="lightgrey", bordercolor="black", borderwidth=2)

    # Update layout
    fig.update_layout(xaxis_rangeslider_visible=False)

    # Display the chart only if requested
    if show:
        fig.show()

    return fig


def save_chart(
    df,
    ticker: str,
    output_dir: str = "charts",
    label: str | None = None,
    style: str | dict = "charles",
    dpi: int = 150,
    sma_windows: tuple[int, ...] = (200, 100, 50, 10),
    ema_windows: tuple[int, ...] = (9,),
) -> str:
    """Save a DataFrame *df* as a candlestick PNG using **mplfinance**.

    Parameters
    ----------
    df : pandas.DataFrame
        Indexed by datetime with ``open, high, low, close, volume`` columns.
    ticker : str
        Stock symbol (used in filename).
    output_dir : str, optional
        Directory where the file is saved. Created if it doesn't exist.
    label : str, optional
        Extra tag appended to the filename (e.g., ``'1y_daily'``).
    style : str or dict, optional
        mplfinance style. Defaults to "charles".
    dpi : int, optional
        Resolution of saved image.
    sma_windows : tuple of int, optional
        Windows for simple moving averages.
    ema_windows : tuple of int, optional
        Windows for exponential moving averages.

    Returns
    -------
    str
        Path to the PNG file.
    """

    import os

    os.makedirs(output_dir, exist_ok=True)

    file_base = ticker.upper()
    if label:
        file_base += f"_{label}"

    png_path = os.path.join(output_dir, f"{file_base}.png")

    # Compute EMA addplots and maintain list for legend
    addplots = []
    ema_colors = []
    for window in ema_windows:
        ema_col = f"EMA_{window}"
        if ema_col not in df.columns:
            df[ema_col] = df['close'].ewm(span=window, adjust=False).mean()
        # choose orange for first, then darker shades
        color = 'orange' if window == 9 else 'brown'
        ema_colors.append(color)
        addplots.append(mpf.make_addplot(df[ema_col], color=color, width=1, linestyle='dashed'))

    # Colors for SMAs
    sma_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"][: len(sma_windows)]

    # Build figure
    fig, axes = mpf.plot(
        df,
        type="candle",
        style=style,
        title=f"{ticker.upper()} – Daily Candles (Last 1 Year)",
        mav=sma_windows if sma_windows else None,
        mavcolors=sma_colors,
        addplot=addplots if addplots else None,
        returnfig=True,
    )

    # Build legend handles
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=sma_colors[i], label=f"SMA{sma_windows[i]}") for i in range(len(sma_windows))]
    handles += [Line2D([], [], color=ema_colors[i], linestyle='dashed', label=f"EMA{ema_windows[i]}") for i in range(len(ema_windows))]

    axes[0].legend(handles=handles, loc='upper left')

    # Save figure
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    logging.info(f"Chart saved to {png_path}")

    return png_path


def cleanup_charts(output_dir: str = "charts") -> None:
    """Remove all files in *output_dir* and delete the directory itself.

    Call this **after** the report/email has been generated and sent so the chart
    folder does not grow without bound.

    Parameters
    ----------
    output_dir : str, optional
        Directory created by :pyfunc:`save_chart`. Defaults to ``'charts'``.
    """

    import os
    import shutil

    if not os.path.exists(output_dir):
        logging.warning(f"cleanup_charts: directory '{output_dir}' does not exist – nothing to clean up.")
        return

    # Delete the entire directory tree
    shutil.rmtree(output_dir)
    logging.info(f"Removed chart directory '{output_dir}'.")


# Backward-compatibility alias
create_chart = create_trade_chart  # noqa: E305


def create_daily_chart(ticker: str, output_dir: str = "charts") -> str:
    """Create a 1-year daily candlestick chart for *ticker* and save it.

    Parameters
    ----------
    ticker : str
        Stock symbol.
    output_dir : str, optional
        Where the PNG is written.

    Returns
    -------
    str
        Path to the saved PNG file.
    """

    from datetime import datetime, timedelta
    from data_queries.polygon_queries import get_levels_data

    # Determine date range: today (or last trading day) back 365 calendar days
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_window = 205  # days back

    # Fetch last year's daily data (trading days only handled by get_levels_data)
    df = get_levels_data(ticker, end_date, window=start_window, multiplier=1, timespan="day")
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Quick on-screen preview with legend
    preview_fig, preview_axes = mpf.plot(
        df,
        type="candle",
        style="charles",
        title=f"{ticker.upper()} – Daily (1Y)",
        mav=(200, 100, 50, 10),
        mavcolors=["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"],
        addplot=[mpf.make_addplot(df['close'].ewm(span=9, adjust=False).mean(), color='orange', width=1, linestyle='dashed')],
        returnfig=True,
    )
    from matplotlib.lines import Line2D
    prev_handles = [
        Line2D([], [], color=c, label=lbl)
        for c, lbl in zip(["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"], ["SMA200", "SMA100", "SMA50", "SMA10"])
    ] + [Line2D([], [], color='orange', linestyle='dashed', label='EMA9')]
    preview_axes[0].legend(handles=prev_handles, loc='upper left')
    preview_fig.show()
    plt.close(preview_fig)

    # Save PNG and return path using mplfinance backend with MAs
    return save_chart(
        df,
        ticker=ticker,
        output_dir=output_dir,
        label="1y_daily",
        sma_windows=(200, 100, 50, 10),
        ema_windows=(9,),
    )

if __name__ == "__main__":
    # Quick test
    path = create_daily_chart("AAPL")
    print("Chart saved to", path)