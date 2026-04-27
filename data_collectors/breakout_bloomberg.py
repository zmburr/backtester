"""
Standalone Bloomberg fill for breakout_data.csv float / short-interest columns.

Run separately when the Bloomberg Terminal is up:
    python data_collectors/breakout_bloomberg.py

Fills only NaN cells in:
    - float_shares
    - short_interest_pct (short interest as % of float)
    - days_to_cover     (short_int / avg_daily_vol)

Caveat: blp.bdp returns CURRENT snapshot values. Historical float and SI for
old rows will be approximate (point-in-time-as-of-now, not as-of-trade-date).
For freshly-traded rows this is exactly right; for older rows expect drift.
"""
from pathlib import Path
import logging
import pandas as pd

from support.csv_utils import load_csv, save_csv_atomic

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

_DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
_CSV = _DATA_DIR / 'breakout_data.csv'

_BLP_FIELDS = ['EQY_FLOAT', 'SHORT_INT', 'EQY_FLOAT_PCT_SHORT_INT']


def _bloomberg_ticker(ticker):
    """Convert a Polygon-style ticker to the Bloomberg US-equity form."""
    return f'{ticker.upper().strip()} US Equity'


def get_float_si(ticker):
    """
    Pull current float / SI snapshot for a ticker via xbbg.

    Returns dict with keys: float_shares, short_interest_pct, days_to_cover (None if unavailable).
    days_to_cover requires avg_daily_vol from caller side, so it's left None here
    and computed by the caller when filling rows.
    """
    try:
        from xbbg import blp
    except ImportError:
        logging.error("xbbg not installed. Install via: pip install xbbg")
        return {}

    bbg_tkr = _bloomberg_ticker(ticker)
    try:
        df = blp.bdp(bbg_tkr, _BLP_FIELDS)
    except Exception as e:
        logging.error(f'Bloomberg query failed for {ticker}: {e}')
        return {}

    if df is None or df.empty:
        return {}

    # bdp returns ticker-indexed DataFrame with lowercase field names
    row = df.iloc[0]
    result = {}
    if 'eqy_float' in row.index and pd.notna(row['eqy_float']):
        result['float_shares'] = float(row['eqy_float'])
    if 'eqy_float_pct_short_int' in row.index and pd.notna(row['eqy_float_pct_short_int']):
        result['short_interest_pct'] = float(row['eqy_float_pct_short_int'])
    if 'short_int' in row.index and pd.notna(row['short_int']):
        result['_short_int_raw'] = float(row['short_int'])
    return result


def fill(force=False):
    """
    Fill missing float/SI values in breakout_data.csv.

    Args:
        force: if True, overwrite existing values too (default False — only fill NaN).
    """
    df = load_csv(_CSV, 'breakout')
    for col in ('float_shares', 'short_interest_pct', 'days_to_cover'):
        if col not in df.columns:
            df[col] = pd.NA

    # Cache per-ticker (Bloomberg snapshot is the same regardless of date)
    cache = {}

    for idx, row in df.iterrows():
        ticker = str(row['ticker']).strip()
        needs_fill = (
            force
            or pd.isna(row.get('float_shares'))
            or pd.isna(row.get('short_interest_pct'))
            or pd.isna(row.get('days_to_cover'))
        )
        if not needs_fill:
            continue

        if ticker not in cache:
            cache[ticker] = get_float_si(ticker)
        data = cache[ticker]
        if not data:
            continue

        if 'float_shares' in data:
            df.at[idx, 'float_shares'] = data['float_shares']
        if 'short_interest_pct' in data:
            df.at[idx, 'short_interest_pct'] = data['short_interest_pct']

        # days_to_cover = short_int / avg_daily_vol (caller-row-specific)
        adv = row.get('avg_daily_vol')
        si_raw = data.get('_short_int_raw')
        if si_raw and adv and pd.notna(adv) and float(adv) > 0:
            df.at[idx, 'days_to_cover'] = si_raw / float(adv)

    save_csv_atomic(df, _CSV)
    logging.info(f'Bloomberg float/SI fill complete. {len(df)} rows.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fill breakout_data.csv float/SI columns from Bloomberg.')
    parser.add_argument('--force', action='store_true', help='Overwrite existing values.')
    args = parser.parse_args()
    fill(force=args.force)
