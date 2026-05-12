"""SNDK 6-month dollar volume chart.

Dollar volume = (high+low)/2 * volume per day. Charts dollar volume bars
with the price line overlaid on a secondary y-axis.

Run:
    venv/Scripts/python.exe scripts/sndk_dollar_volume.py
"""
import sys
from pathlib import Path
sys.path.insert(0, '.')

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_queries.polygon_queries import get_levels_data

TICKER = 'SNDK'
END_DATE = '2026-05-06'
LOOKBACK_DAYS = 200  # ~6 months of calendar days = ~130 trading days


def main():
    levels = get_levels_data(TICKER, END_DATE, LOOKBACK_DAYS, 1, 'day')
    if levels is None or levels.empty:
        print(f'{TICKER}: no data')
        return

    df = levels.copy()
    df['mid'] = (df['high'] + df['low']) / 2.0
    df['dollar_volume'] = df['mid'] * df['volume']
    df['dollar_volume_b'] = df['dollar_volume'] / 1e9
    df['dollar_volume_20d_avg'] = df['dollar_volume'].rolling(20, min_periods=1).mean()
    df['dollar_volume_20d_avg_b'] = df['dollar_volume_20d_avg'] / 1e9

    print(f'{TICKER} dollar-volume summary ({df.index[0].date()} -> {df.index[-1].date()}):')
    print(f'  Sessions:                     {len(df)}')
    print(f'  Mean daily $-vol:             ${df["dollar_volume"].mean()/1e9:.2f}B')
    print(f'  Median daily $-vol:           ${df["dollar_volume"].median()/1e9:.2f}B')
    print(f'  Max daily $-vol:              ${df["dollar_volume"].max()/1e9:.2f}B  on {df["dollar_volume"].idxmax().date()}')
    print(f'  Last 5 sessions $-vol:')
    for ts, row in df.tail(5).iterrows():
        print(f'    {ts.date()}  mid=${row["mid"]:>8,.2f}  vol={int(row["volume"]):>14,}  $-vol=${row["dollar_volume"]/1e9:>5.2f}B')

    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
        subplot_titles=(f'{TICKER} Dollar Volume (mid * volume) over Last 6 Months',),
    )

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['dollar_volume_b'],
            name='Daily $-Volume ($B)',
            marker_color='rgba(31,119,180,0.7)',
            hovertemplate='%{x|%Y-%m-%d}<br>$%{y:.2f}B<extra></extra>',
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['dollar_volume_20d_avg_b'],
            name='20d avg $-Volume ($B)',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            hovertemplate='%{x|%Y-%m-%d}<br>20d avg $%{y:.2f}B<extra></extra>',
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['close'],
            name='Close Price ($)',
            line=dict(color='#2ca02c', width=2),
            hovertemplate='%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>',
        ),
        secondary_y=True,
    )

    fig.update_layout(
        template='plotly_dark',
        height=720,
        hovermode='x unified',
        legend=dict(orientation='h', y=1.07, x=0.0),
        margin=dict(l=60, r=60, t=80, b=50),
    )
    fig.update_xaxes(title_text='Date', showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text='Dollar Volume ($B)', secondary_y=False,
                     showgrid=True, gridcolor='rgba(128,128,128,0.15)')
    fig.update_yaxes(title_text='Close Price ($)', secondary_y=True,
                     showgrid=False)

    out_dir = Path('charts')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f'{TICKER}_dollar_volume_6mo.html'
    fig.write_html(str(out_path), include_plotlyjs='cdn')
    print(f'\nChart written to: {out_path.resolve()}')

    # Also save a PNG if kaleido is available
    try:
        png_path = out_dir / f'{TICKER}_dollar_volume_6mo.png'
        fig.write_image(str(png_path), width=1600, height=720, scale=2)
        print(f'PNG written to:   {png_path.resolve()}')
    except Exception as e:
        print(f'(PNG skipped — install kaleido for static images: {e})')


if __name__ == '__main__':
    main()
