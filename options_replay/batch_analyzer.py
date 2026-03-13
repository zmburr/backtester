"""
Batch analysis — run the options replay pipeline across many trades
and aggregate results to find patterns in delta, DTE, moneyness, IV.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import pandas as pd
import numpy as np

from options_replay.theta_client import (
    check_terminal_running, get_chain_snapshot, get_option_ohlc,
    get_option_quotes, get_option_greeks, ThetaTerminalOfflineError,
)
from options_replay.chain_analyzer import (
    filter_chain, compute_option_returns, score_options,
    contract_key,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "batch_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def analyze_single_trade(
    trade: dict,
    hold_minutes: int = 30,
) -> Optional[pd.DataFrame]:
    """Run the full pipeline for one trade at a single hold window.

    Returns a scored DataFrame with trade metadata columns,
    or None if the trade fails.
    """
    symbol = trade["symbol"]
    date_str = str(trade["date"])
    entry_time = trade["entry_time"]
    if isinstance(entry_time, str):
        entry_time = pd.Timestamp(entry_time)
    time_of_day = entry_time.strftime("%H:%M:%S")
    side = int(trade.get("side", 1))
    underlying_price = float(trade.get("avg_price", 0))

    # Fetch underlying if we don't have a price
    if underlying_price <= 0:
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from data_queries.polygon_queries import get_intraday
            underlying_df = get_intraday(symbol, date_str, 1, "minute")
            if underlying_df is not None and not underlying_df.empty:
                entry_tz = entry_time
                if hasattr(underlying_df.index, 'tz') and underlying_df.index.tz is not None:
                    if entry_tz.tzinfo is None:
                        from pytz import timezone
                        entry_tz = timezone("US/Eastern").localize(entry_tz)
                idx = underlying_df.index.get_indexer([entry_tz], method="nearest")[0]
                underlying_price = float(underlying_df.iloc[idx]["close"])
        except Exception as e:
            logger.warning("Failed to get underlying price for %s: %s", symbol, e)

    if underlying_price <= 0:
        logger.warning("Skipping %s on %s — no underlying price", symbol, date_str)
        return None

    # Chain snapshot
    try:
        chain_df = get_chain_snapshot(symbol, date_str, time_of_day)
    except Exception as e:
        logger.warning("Chain snapshot failed for %s on %s: %s", symbol, date_str, e)
        return None

    if chain_df.empty:
        return None

    # Filter
    filtered = filter_chain(chain_df, underlying_price, side, date_str)
    if filtered.empty:
        return None

    # Per-contract data (parallel)
    ohlc_dict = {}
    quotes_dict = {}
    greeks_dict = {}

    def _fetch(row):
        key = contract_key(row)
        ohlc = get_option_ohlc(symbol, row["expiration"], row["strike"],
                               row["right"], date_str)
        quotes = get_option_quotes(symbol, row["expiration"], row["strike"],
                                   row["right"], date_str)
        greeks = get_option_greeks(symbol, row["expiration"], row["strike"],
                                   row["right"], date_str)
        return key, ohlc, quotes, greeks

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_fetch, row): idx
                   for idx, row in filtered.iterrows()}
        for future in as_completed(futures):
            try:
                key, ohlc, quotes, greeks = future.result()
                ohlc_dict[key] = ohlc
                quotes_dict[key] = quotes
                greeks_dict[key] = greeks
            except Exception as e:
                logger.debug("Contract fetch failed: %s", e)

    # Compute returns
    entry_dt = entry_time.to_pydatetime() if hasattr(entry_time, 'to_pydatetime') else entry_time
    returns_df = compute_option_returns(
        filtered, ohlc_dict, quotes_dict, entry_dt,
        hold_minutes, greeks_dict=greeks_dict,
    )

    # Score
    scored_df = score_options(returns_df)
    if scored_df.empty:
        return None

    # Add trade metadata
    scored_df["symbol"] = symbol
    scored_df["trade_date"] = date_str
    scored_df["entry_time"] = str(entry_time)
    scored_df["side"] = side
    scored_df["net_pnl"] = trade.get("net_pnl", 0)
    scored_df["news_type"] = trade.get("news_type", "")
    scored_df["underlying_price"] = underlying_price
    scored_df["hold_window"] = hold_minutes

    return scored_df


def run_batch(
    trades_df: pd.DataFrame,
    trade_indices: list,
    hold_windows: list = None,
    progress_callback: Optional[Callable] = None,
) -> pd.DataFrame:
    """Process multiple trades, each at multiple hold windows.

    Args:
        trades_df: full DataFrame from load_trades()
        trade_indices: list of integer indices into trades_df
        hold_windows: list of hold_minutes to evaluate (default [5, 15, 30])
        progress_callback: fn(completed, total, symbol) called after each trade

    Returns:
        DataFrame with all per-contract results across all trades and windows.
    """
    if hold_windows is None:
        hold_windows = [5, 15, 30]

    total = len(trade_indices)
    all_results = []

    for i, idx in enumerate(trade_indices):
        trade = trades_df.iloc[idx].to_dict()
        trade["_trade_idx"] = idx
        symbol = trade.get("symbol", "?")

        for hold_min in hold_windows:
            try:
                result = analyze_single_trade(trade, hold_minutes=hold_min)
                if result is not None:
                    result["trade_idx"] = idx
                    all_results.append(result)
            except Exception as e:
                logger.warning("Trade %s on %s failed at %d min: %s",
                              symbol, trade.get("date", "?"), hold_min, e)

        if progress_callback:
            progress_callback(i + 1, total, symbol)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def save_batch_results(results_df: pd.DataFrame, filename: str = None) -> Path:
    """Save batch results to CSV."""
    if filename is None:
        filename = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    path = RESULTS_DIR / filename
    results_df.to_csv(path, index=False)
    logger.info("Saved batch results to %s (%d rows)", path, len(results_df))
    return path


def load_batch_results(filename: str = None) -> Optional[pd.DataFrame]:
    """Load batch results. If no filename, load the most recent."""
    if filename:
        path = RESULTS_DIR / filename
    else:
        csvs = sorted(RESULTS_DIR.glob("batch_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not csvs:
            return None
        path = csvs[0]

    if not path.exists():
        return None

    df = pd.read_csv(path)
    logger.info("Loaded batch results from %s (%d rows)", path, len(df))
    return df
