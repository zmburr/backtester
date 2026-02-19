"""
Setup Tagger - Visual Y/N labeling tool for backscanner results.

Shows 3-month daily candlestick chart + intraday 2-min chart for each unlabeled setup.
Press keys ON THE CHART WINDOW to label (Tinder-style).

Controls (press while chart window is focused):
  Y         = proper 3DGapFade setup
  N         = not a proper setup
  A         = not proper + enter alternate setup type (prompts in terminal)
  S / Right = skip (leave unlabeled)
  Q         = quit and save progress

The trade date is highlighted with a magenta vertical line on the daily chart.

Usage:
    python scripts/setup_tagger.py
    python scripts/setup_tagger.py --csv data/backscanner_2021_A_plus_v2.csv
"""

import os
import sys
import pickle
import time
import re

import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime
from polygon.rest import RESTClient
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# 3 months of trading days
DAILY_LOOKBACK = 65


def normalize_date(date_str):
    """Convert any date format (M/D/YYYY or YYYY-MM-DD) to YYYY-MM-DD."""
    date_str = str(date_str).strip()
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str
    try:
        dt = datetime.strptime(date_str, '%m/%d/%Y')
        return dt.strftime('%Y-%m-%d')
    except ValueError:
        try:
            dt = datetime.strptime(date_str, '%m/%d/%y')
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            return date_str


class SetupTagger:
    def __init__(self, csv_path=None, cap_filter=None):
        if csv_path is None:
            csv_path = os.path.join(DATA_DIR, 'backscanner_2021_A_plus_v2.csv')
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.cap_filter = cap_filter
        self.client = RESTClient(api_key=POLYGON_API_KEY)
        self._ticker_index = None
        self._result = None  # set by key handler
        self._load_daily_cache()

    # ------------------------------------------------------------------
    # Daily data from backscanner cache
    # ------------------------------------------------------------------

    def _load_daily_cache(self):
        """Load the backscanner grouped daily cache and build ticker index."""
        cache_dir = os.path.join(DATA_DIR, 'backscanner_cache')
        if not os.path.exists(cache_dir):
            print("No backscanner cache directory found. Daily charts will be unavailable.")
            return

        cache_files = sorted(
            [f for f in os.listdir(cache_dir) if f.endswith('.pkl')],
            key=lambda f: os.path.getmtime(os.path.join(cache_dir, f)),
            reverse=True,
        )
        if not cache_files:
            print("No cache files found.")
            return

        cache_path = os.path.join(cache_dir, cache_files[0])
        print(f"Loading daily cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            daily_data = pickle.load(f)

        print("Building ticker index...")
        frames = []
        for date_str in sorted(daily_data.keys()):
            df = daily_data[date_str]
            if df is not None and not df.empty:
                df_copy = df.copy()
                df_copy['date'] = date_str
                frames.append(df_copy)

        if not frames:
            return

        all_data = pd.concat(frames, ignore_index=True)
        self._ticker_index = {}
        for ticker, group in all_data.groupby('ticker'):
            self._ticker_index[ticker] = group.sort_values('date').set_index('date')

        print(f"Ticker index ready: {len(self._ticker_index)} tickers")

    def get_daily_data(self, ticker, date_str):
        """Get ~3 months of daily OHLCV from cache, formatted for mplfinance."""
        if self._ticker_index is None or ticker not in self._ticker_index:
            return None, None

        ticker_df = self._ticker_index[ticker]
        history = ticker_df[ticker_df.index <= date_str].tail(DAILY_LOOKBACK).copy()

        if len(history) < 10:
            return None, None

        # Find the index position of the trade date for the vertical line
        trade_date_pos = None
        if date_str in history.index:
            trade_date_pos = list(history.index).index(date_str)

        # Format for mplfinance: DatetimeIndex + capitalized columns
        history.index = pd.to_datetime(history.index)
        history = history[['open', 'high', 'low', 'close', 'volume']].copy()
        history.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return history, trade_date_pos

    # ------------------------------------------------------------------
    # Intraday data from Polygon API
    # ------------------------------------------------------------------

    def get_intraday_data(self, ticker, date_str):
        """Fetch 2-min intraday bars from Polygon API."""
        try:
            aggs = list(self.client.get_aggs(
                ticker=ticker,
                multiplier=2,
                timespan='minute',
                from_=date_str,
                to=date_str,
                limit=50000,
            ))
            if not aggs:
                return None

            rows = []
            for a in aggs:
                rows.append({
                    'datetime': pd.Timestamp(a.timestamp, unit='ms', tz='US/Eastern'),
                    'Open': a.open,
                    'High': a.high,
                    'Low': a.low,
                    'Close': a.close,
                    'Volume': a.volume,
                })

            df = pd.DataFrame(rows).set_index('datetime')
            return df

        except Exception as e:
            print(f"  Warning: Could not fetch intraday for {ticker} {date_str}: {e}")
            return None

    # ------------------------------------------------------------------
    # Key event handler (Tinder-style)
    # ------------------------------------------------------------------

    def _on_key(self, event):
        """Handle keypress on the chart window."""
        key = event.key.lower() if event.key else ''
        if key in ('y', 'n', 'a', 's', 'q', 'right'):
            self._result = key
            plt.close(event.canvas.figure)

    # ------------------------------------------------------------------
    # Chart display
    # ------------------------------------------------------------------

    def show_trade(self, idx, position, total):
        """Show daily + intraday charts, wait for keypress label."""
        row = self.df.iloc[idx]
        ticker = row['ticker']
        date_str = normalize_date(row['date'])
        score = row.get('score', '?')
        grade = row.get('grade', '?')
        cap = row.get('cap', '?')
        pct_9ema = row.get('pct_from_9ema', 0)
        gap = row.get('gap_pct', 0)
        pct_50ma = row.get('pct_from_50mav', 0)
        up_days = row.get('consecutive_up_days', 0)
        rvol = row.get('rvol_score', 0)
        price = row.get('current_price', 0)
        pct_15 = row.get('pct_change_15', 0)
        pct_30 = row.get('pct_change_30', 0)

        print(f"\n  [{position}/{total}] {ticker} ({cap}) — {date_str} | "
              f"{score}/5 ({grade}) | 9EMA +{pct_9ema*100:.0f}% | "
              f"Gap +{gap*100:.1f}% | 50MA +{pct_50ma*100:.0f}%")

        # Fetch data
        daily, trade_date_pos = self.get_daily_data(ticker, date_str)
        intraday = self.get_intraday_data(ticker, date_str)
        time.sleep(0.12)  # rate limit

        if daily is None:
            print(f"  No daily data. Skipping...")
            return 'S'

        has_intraday = intraday is not None and len(intraday) > 0

        # Moving averages
        ema9 = daily['Close'].ewm(span=9, adjust=False).mean()
        sma50 = daily['Close'].rolling(50).mean() if len(daily) >= 50 else None

        # --- Build figure ---
        mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', inherit=True)
        style = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=True,
                                    facecolor='#1e1e1e', figcolor='#1e1e1e',
                                    edgecolor='#333333')

        if has_intraday:
            fig = plt.figure(figsize=(19, 9))
            gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.05, wspace=0.25)
            ax_daily = fig.add_subplot(gs[0, 0])
            ax_daily_vol = fig.add_subplot(gs[1, 0], sharex=ax_daily)
            ax_intra = fig.add_subplot(gs[0, 1])
            ax_intra_vol = fig.add_subplot(gs[1, 1], sharex=ax_intra)
        else:
            fig = plt.figure(figsize=(13, 8))
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
            ax_daily = fig.add_subplot(gs[0])
            ax_daily_vol = fig.add_subplot(gs[1], sharex=ax_daily)
            ax_intra = None
            ax_intra_vol = None

        # --- Daily chart ---
        ap_daily = [mpf.make_addplot(ema9, ax=ax_daily, color='dodgerblue',
                                     width=1, linestyle='--')]
        if sma50 is not None:
            ap_daily.append(mpf.make_addplot(sma50, ax=ax_daily, color='orange', width=1.2))

        mpf.plot(daily, type='candle', ax=ax_daily, volume=ax_daily_vol,
                 addplot=ap_daily, style=style)

        # Highlight trade date with vertical line
        if trade_date_pos is not None:
            ax_daily.axvline(x=trade_date_pos, color='magenta', linewidth=2,
                             alpha=0.8, linestyle='-', zorder=1)
            ax_daily_vol.axvline(x=trade_date_pos, color='magenta', linewidth=2,
                                 alpha=0.8, linestyle='-', zorder=1)
            # Label the line
            y_top = ax_daily.get_ylim()[1]
            ax_daily.text(trade_date_pos, y_top, f' {date_str} ',
                          color='magenta', fontsize=8, fontweight='bold',
                          ha='center', va='top',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='#1e1e1e',
                                    edgecolor='magenta', alpha=0.9))

        ax_daily.set_title(f'{ticker} — 3-Month Daily  (blue=9EMA, orange=50SMA)',
                           fontsize=10, color='white')

        # --- Intraday chart ---
        if has_intraday and ax_intra is not None:
            mpf.plot(intraday, type='candle', ax=ax_intra, volume=ax_intra_vol,
                     style=style)
            ax_intra.set_title(f'{ticker} — Intraday 2-Min ({date_str})',
                               fontsize=10, color='white')

        # --- Title bar with metrics ---
        line1 = (f"[{position}/{total}]  {ticker} ({cap})  —  {date_str}  |  "
                 f"Score {score}/5 ({grade})")
        line2 = (f"9EMA +{pct_9ema*100:.0f}%   |   Gap +{gap*100:.1f}%   |   "
                 f"50MA +{pct_50ma*100:.0f}%   |   Up Days: {int(up_days)}   |   "
                 f"RVOL {rvol:.1f}x   |   15d +{pct_15*100:.0f}%   |   "
                 f"30d +{pct_30*100:.0f}%   |   ${price:.2f}")
        fig.suptitle(f"{line1}\n{line2}", fontsize=11, fontweight='bold',
                     color='white', y=0.99)

        # --- Controls hint at bottom ---
        fig.text(0.5, 0.01,
                 'Y = proper setup  |  N = not proper  |  A = alternate type  |  '
                 'S/\u2192 = skip  |  Q = quit',
                 ha='center', fontsize=10, color='#aaaaaa',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#2a2a2a',
                           edgecolor='#555555'))

        # --- Key handler ---
        self._result = None
        fig.canvas.mpl_connect('key_press_event', self._on_key)
        plt.show(block=True)  # blocks until user presses a key and figure closes

        # --- Process result ---
        result = self._result or 's'

        if result == 'y':
            print(f"  -> Y (proper setup)")
            return 'Y'
        elif result == 'n':
            alt = input("  >> Alternate setup type? (Enter to skip): ").strip()
            return ('N', alt)
        elif result == 'a':
            alt = input("  >> Alternate setup type: ").strip()
            return ('N', alt)
        elif result in ('s', 'right'):
            print(f"  -> skipped")
            return 'S'
        elif result == 'q':
            return 'Q'

        return 'S'

    # ------------------------------------------------------------------
    # Save progress
    # ------------------------------------------------------------------

    def save(self):
        """Save current labels back to CSV."""
        self.df.to_csv(self.csv_path, index=False)
        print("  [Saved]")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Run the interactive tagging loop."""
        # Ensure label columns exist
        if 'proper_setup?' not in self.df.columns:
            self.df['proper_setup?'] = ''
        if 'alternate_setup?' not in self.df.columns:
            self.df['alternate_setup?'] = ''

        # Find unlabeled rows, respecting cap filter
        labeled_mask = self.df['proper_setup?'].astype(str).str.strip().isin(['Y', 'N'])
        pool = self.df[~labeled_mask]
        if self.cap_filter:
            pool = pool[pool['cap'].isin(self.cap_filter)]
        unlabeled_indices = pool.index.tolist()

        total = len(unlabeled_indices)
        already = labeled_mask.sum()
        cap_note = f"  Cap filter: {', '.join(self.cap_filter)}" if self.cap_filter else ""
        print(f"\n{'='*70}")
        print(f"  SETUP TAGGER")
        if cap_note:
            print(cap_note)
        print(f"  {total} unlabeled trades remaining ({already} already tagged)")
        print(f"  Press keys ON THE CHART WINDOW:")
        print(f"    Y=proper  N=not proper  A=alternate  S/Right=skip  Q=quit")
        print(f"{'='*70}")

        tagged_this_session = 0

        for i, idx in enumerate(unlabeled_indices):
            result = self.show_trade(idx, i + 1, total)

            if result == 'Q':
                print(f"\nQuitting. Tagged {tagged_this_session} trades this session.")
                self.save()
                return
            elif result == 'S':
                continue
            elif result == 'Y':
                self.df.at[idx, 'proper_setup?'] = 'Y'
                tagged_this_session += 1
            elif isinstance(result, tuple):
                label, alt = result
                self.df.at[idx, 'proper_setup?'] = label
                if alt:
                    self.df.at[idx, 'alternate_setup?'] = alt
                tagged_this_session += 1

            # Auto-save every 5 labels
            if tagged_this_session > 0 and tagged_this_session % 5 == 0:
                self.save()

        self.save()
        print(f"\nDone! Tagged {tagged_this_session} trades. All trades labeled.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Visual setup tagger for backscanner results')
    parser.add_argument('--csv', default=None,
                        help='Path to CSV (default: backscanner_2021_A_plus_v2.csv)')
    parser.add_argument('--cap', default="Medium,Large",
                        help='Filter to specific cap(s), comma-separated (e.g. "Medium,Large")')
    args = parser.parse_args()

    cap_filter = [c.strip() for c in args.cap.split(',')] if args.cap else None
    tagger = SetupTagger(csv_path=args.csv, cap_filter=cap_filter)
    tagger.run()
