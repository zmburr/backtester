"""Static PNG version of dollar-volume chart for a ticker (default SNDK)."""
import sys
from pathlib import Path
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_queries.polygon_queries import get_levels_data

TICKER = sys.argv[1] if len(sys.argv) > 1 else 'SNDK'
END_DATE = '2026-05-06'
LOOKBACK_DAYS = 200

levels = get_levels_data(TICKER, END_DATE, LOOKBACK_DAYS, 1, 'day')
df = levels.copy()
df['mid'] = (df['high'] + df['low']) / 2.0
df['dollar_volume'] = df['mid'] * df['volume']
df['dv_b'] = df['dollar_volume'] / 1e9
df['dv_20d'] = df['dollar_volume'].rolling(20, min_periods=1).mean() / 1e9

# Print summary stats so the caller has them
print(f'{TICKER} dollar-volume summary ({df.index[0].date()} -> {df.index[-1].date()}):')
print(f'  Sessions:               {len(df)}')
print(f'  Mean daily $-vol:       ${df["dollar_volume"].mean()/1e9:.2f}B')
print(f'  Median daily $-vol:     ${df["dollar_volume"].median()/1e9:.2f}B')
print(f'  Max daily $-vol:        ${df["dollar_volume"].max()/1e9:.2f}B  on {df["dollar_volume"].idxmax().date()}')
print(f'  Last 5 sessions $-vol:')
for ts, row in df.tail(5).iterrows():
    print(f'    {ts.date()}  mid=${row["mid"]:>9,.2f}  vol={int(row["volume"]):>14,}  $-vol=${row["dollar_volume"]/1e9:>5.2f}B')

fig, ax1 = plt.subplots(figsize=(16, 8), facecolor='#0e1117')
ax1.set_facecolor('#0e1117')

# Bars: daily dollar volume
ax1.bar(df.index, df['dv_b'], color='#1f77b4', alpha=0.75, label='Daily $-Volume')
ax1.plot(df.index, df['dv_20d'], color='#ff7f0e', linewidth=2.0,
         linestyle='--', label='20d avg $-Volume')

ax1.set_xlabel('Date', color='white')
ax1.set_ylabel('Dollar Volume ($B)', color='#1f77b4')
ax1.tick_params(axis='y', labelcolor='#1f77b4')
ax1.tick_params(axis='x', labelcolor='white')
ax1.grid(True, alpha=0.2, color='gray')

# Secondary axis: close price
ax2 = ax1.twinx()
ax2.plot(df.index, df['close'], color='#2ca02c', linewidth=2.5, label='Close Price ($)')
ax2.set_ylabel('Close Price ($)', color='#2ca02c')
ax2.tick_params(axis='y', labelcolor='#2ca02c')

# Mark the recent extremes
peak_idx = df['dollar_volume'].idxmax()
peak_val = df.loc[peak_idx, 'dv_b']
ax1.annotate(
    f'${peak_val:.1f}B\n{peak_idx.date()}',
    xy=(peak_idx, peak_val),
    xytext=(peak_idx, peak_val * 1.05),
    color='white',
    ha='center',
    fontsize=9,
    arrowprops=dict(arrowstyle='->', color='white', alpha=0.6),
)

# X-axis formatting
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')

# Title
fig.suptitle(
    f'{TICKER} — Dollar Volume Over Last 6 Months  ({df.index[0].date()} to {df.index[-1].date()})',
    color='white', fontsize=14, y=0.97,
)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='upper left', framealpha=0.85, facecolor='#1a1d24',
           edgecolor='#3a3f4b', labelcolor='white')

plt.tight_layout()
out_dir = Path('charts')
out_dir.mkdir(exist_ok=True)
out_path = out_dir / f'{TICKER}_dollar_volume_6mo.png'
plt.savefig(str(out_path), dpi=140, facecolor='#0e1117', bbox_inches='tight')
print(f'PNG: {out_path.resolve()}')
