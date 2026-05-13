# Standard logging
import logging
# Plotly is still used for intraday trade charts, keep the import.
import plotly.graph_objects as go

from data_queries.polygon_queries import get_intraday

# New dependency for faster static charts
import mplfinance as mpf

# Force non-interactive backend before importing pyplot (thread-safe, headless)
import matplotlib
matplotlib.use('Agg')
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
    hlines: list[tuple[float, str, str]] | None = None,
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
    hlines : list of tuple, optional
        List of tuples (y, color, label) for horizontal lines.

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
        volume=True,
        returnfig=True,
    )

    # Build legend handles
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=sma_colors[i], label=f"SMA{sma_windows[i]}") for i in range(len(sma_windows))]
    handles += [Line2D([], [], color=ema_colors[i], linestyle='dashed', label=f"EMA{ema_windows[i]}") for i in range(len(ema_windows))]

    # Horizontal lines (e.g., 1-year high)
    if hlines:
        for y, color, label in hlines:
            axes[0].axhline(y=y, color=color, linestyle='--', linewidth=1)
            handles.append(Line2D([], [], color=color, linestyle='--', label=label))

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


def _build_today_intraday_bar(ticker: str):
    """Roll today's minute aggs into a single daily-style OHLCV row.

    Returns a one-row DataFrame indexed by today (US/Eastern midnight), or None
    if no intraday data is available yet (pre-premarket, weekend, holiday).
    Includes premarket/afterhours since Polygon minute aggs span all sessions.
    """
    import pandas as pd
    from datetime import datetime
    from pytz import timezone as _tz
    from data_queries.polygon_queries import get_intraday

    today_str = datetime.now(_tz('US/Eastern')).strftime('%Y-%m-%d')
    minute_df = get_intraday(ticker, today_str, 1, 'minute')
    if minute_df is None or minute_df.empty:
        return None

    # Match the rest of the chart's daily bars (regular session OHLC).
    # If we're before 9:30 ET (regular session empty), fall back to whatever's
    # available so the chart still reflects today's premarket print.
    regular = minute_df.between_time('09:30', '15:59')
    session_df = regular if not regular.empty else minute_df

    bar = pd.DataFrame(
        {
            'open':   [float(session_df['open'].iloc[0])],
            'high':   [float(session_df['high'].max())],
            'low':    [float(session_df['low'].min())],
            'close':  [float(session_df['close'].iloc[-1])],
            'volume': [float(session_df['volume'].sum())],
        },
        index=[pd.Timestamp(today_str, tz='US/Eastern')],
    )
    return bar


def create_daily_chart(ticker: str, output_dir: str = "charts", extra_hlines: list = None,
                       end_date: str = None, label: str = "1y_daily") -> str:
    """Create a 1-year daily candlestick chart for *ticker* and save it.

    Parameters
    ----------
    ticker : str
        Stock symbol.
    output_dir : str, optional
        Where the PNG is written.
    extra_hlines : list of tuple, optional
        Additional (y, color, label) horizontal lines to draw on the chart
        (e.g., ATR target levels for reversal setups).
    end_date : str, optional
        End date for the 1-year window in 'YYYY-MM-DD' format. Defaults to
        today. Use to render historical comp charts (e.g., the year leading
        up to a past trade date).
    label : str, optional
        Filename suffix to disambiguate when generating multiple charts for
        the same ticker (e.g., today's chart + multiple historical comps).

    Returns
    -------
    str
        Path to the saved PNG file.
    """

    from datetime import datetime, timedelta
    import pandas as pd
    from data_queries.polygon_queries import get_levels_data

    # Determine date range: end_date (or today) back ~1 year of trading days
    is_live = end_date is None
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    start_window = 365  # calendar days back — ~1 year of trading data

    # Fetch last year's daily data (trading days only handled by get_levels_data)
    df = get_levels_data(ticker, end_date, window=start_window, multiplier=1, timespan="day")
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Hard cap: keep at most the last 252 trading days (one trading year).
    # Defense in depth in case the API returns more than requested.
    if len(df) > 252:
        df = df.iloc[-252:]

    # For live charts, splice today's intraday OHLC onto the daily series so the
    # chart, MAs, and 1Y-high hline reflect price action through *now*, not the
    # last fully-closed daily bar (which Polygon may not finalize until after-hours).
    if is_live:
        try:
            today_bar = _build_today_intraday_bar(ticker)
        except Exception as e:
            logging.warning(f"create_daily_chart: today bar fetch failed for {ticker}: {e}")
            today_bar = None
        if today_bar is not None and not today_bar.empty:
            today_date = today_bar.index[0].date()
            mask_today = df.index.map(lambda ts: ts.date() == today_date)
            if mask_today.any():
                # Polygon already returned a (possibly stale) bar for today — overwrite it
                df = df.loc[~mask_today]
            df = pd.concat([df, today_bar]).sort_index()

    # Save PNG and return path using mplfinance backend with MAs
    hlines = [(df['high'].max(), 'grey', '1Y High')]
    if extra_hlines:
        hlines.extend(extra_hlines)

    return save_chart(
        df,
        ticker=ticker,
        output_dir=output_dir,
        label=label,
        sma_windows=(200, 100, 50, 10),
        ema_windows=(9,),
        hlines=hlines,
    )

if __name__ == "__main__":
    # Quick test
    path = create_daily_chart("AAPL")
    print("Chart saved to", path)