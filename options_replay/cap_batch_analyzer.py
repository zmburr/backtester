"""
Capitulation batch analysis — run the options replay pipeline across
bounce and reversal trades with ATR-target tracking.
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
    filter_chain, compute_option_returns_with_targets, score_options,
    contract_key,
)

logger = logging.getLogger(__name__)

CAP_RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "batch_results"
CAP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _fetch_underlying_price(symbol: str, date_str: str, time_str: str) -> float:
    """Fetch underlying stock price at a specific time from Polygon."""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data_queries.polygon_queries import get_intraday

        df = get_intraday(symbol, date_str, 1, "minute")
        if df is None or df.empty:
            return 0.0

        entry_ts = pd.Timestamp(f"{date_str} {time_str}")
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            if entry_ts.tzinfo is None:
                from pytz import timezone
                entry_ts = timezone("US/Eastern").localize(entry_ts)

        idx = df.index.get_indexer([entry_ts], method="nearest")[0]
        return float(df.iloc[idx]["close"])
    except Exception as e:
        logger.warning("Failed to get underlying price for %s: %s", symbol, e)
        return 0.0


def _fetch_underlying_bars(symbol: str, date_str: str) -> pd.DataFrame:
    """Fetch full day of 1-min underlying bars from Polygon."""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data_queries.polygon_queries import get_intraday

        df = get_intraday(symbol, date_str, 1, "minute")
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception as e:
        logger.warning("Failed to get underlying bars for %s: %s", symbol, e)
        return pd.DataFrame()


def _compute_targets(entry_price: float, atr_pct: float, side: int,
                     multiples: list = None) -> dict:
    """Compute ATR target price levels.

    Args:
        entry_price: stock price at entry
        atr_pct: ATR as fraction of price (e.g. 0.05 = 5%)
        side: 1 = long (targets above), -1 = short (targets below)
        multiples: list of ATR multiples to track (default [0.5, 1.0, 1.5])

    Returns:
        dict of {name: price_level}
    """
    if multiples is None:
        multiples = [0.5, 1.0, 1.5]

    atr_dollars = entry_price * atr_pct
    targets = {}
    for mult in multiples:
        name = f"{mult}x"
        if side == 1:
            targets[name] = entry_price + (mult * atr_dollars)
        else:
            targets[name] = entry_price - (mult * atr_dollars)
    return targets


def analyze_cap_trade(
    trade: dict,
    hold_minutes: int = 120,
    atr_targets: list = None,
) -> Optional[pd.DataFrame]:
    """Run the full pipeline for one capitulation trade.

    Returns a scored DataFrame with trade metadata and target hit columns,
    or None if the trade fails.
    """
    if atr_targets is None:
        atr_targets = [0.5, 1.0, 1.5]

    symbol = trade["symbol"]
    date_str = str(trade["date"])
    entry_time = trade["entry_time"]
    if isinstance(entry_time, str):
        entry_time = pd.Timestamp(entry_time)
    time_of_day = entry_time.strftime("%H:%M:%S")
    side = int(trade.get("side", 1))
    atr_pct = float(trade.get("atr_pct", 0))

    # Fetch underlying price
    underlying_price = _fetch_underlying_price(symbol, date_str, time_of_day)
    if underlying_price <= 0:
        logger.warning("Skipping %s on %s — no underlying price", symbol, date_str)
        return None

    # Compute ATR target levels
    target_levels = _compute_targets(underlying_price, atr_pct, side, atr_targets) if atr_pct > 0 else {}

    # Fetch underlying bars for target tracking
    underlying_bars = _fetch_underlying_bars(symbol, date_str)

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

    # Compute returns with target tracking
    entry_dt = entry_time.to_pydatetime() if hasattr(entry_time, 'to_pydatetime') else entry_time
    returns_df = compute_option_returns_with_targets(
        filtered, ohlc_dict, quotes_dict, entry_dt,
        hold_minutes=hold_minutes,
        greeks_dict=greeks_dict,
        target_levels=target_levels,
        underlying_bars=underlying_bars,
        side=side,
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
    scored_df["source"] = trade.get("source", "")
    scored_df["setup_type"] = trade.get("setup_type", "")
    scored_df["trade_grade"] = trade.get("trade_grade", "")
    scored_df["cap"] = trade.get("cap", "")
    scored_df["atr_pct"] = atr_pct
    scored_df["entry_offset"] = trade.get("entry_offset", "")
    scored_df["underlying_price"] = underlying_price
    scored_df["hold_window"] = hold_minutes

    # Add target level values for reference
    for name, level in target_levels.items():
        scored_df[f"target_{name}_level"] = level

    return scored_df


def run_cap_batch(
    trades_df: pd.DataFrame,
    trade_indices: list,
    hold_windows: list = None,
    atr_targets: list = None,
    progress_callback: Optional[Callable] = None,
) -> pd.DataFrame:
    """Process multiple capitulation trades.

    Args:
        trades_df: DataFrame from load_bounce_trades() or load_all_cap_trades()
        trade_indices: list of integer indices into trades_df
        hold_windows: list of hold_minutes to evaluate (default [30, 60, 120])
        atr_targets: ATR multiples to track (default [0.5, 1.0, 1.5])
        progress_callback: fn(completed, total, symbol) called after each trade

    Returns:
        DataFrame with all results across trades and windows.
    """
    if hold_windows is None:
        hold_windows = [30, 60, 120]
    if atr_targets is None:
        atr_targets = [0.5, 1.0, 1.5]

    total = len(trade_indices)
    all_results = []

    for i, idx in enumerate(trade_indices):
        trade = trades_df.iloc[idx].to_dict()
        symbol = trade.get("symbol", "?")

        for hold_min in hold_windows:
            try:
                result = analyze_cap_trade(
                    trade, hold_minutes=hold_min, atr_targets=atr_targets,
                )
                if result is not None:
                    result["trade_idx"] = idx
                    all_results.append(result)
            except Exception as e:
                logger.warning("Cap trade %s on %s failed at %d min: %s",
                               symbol, trade.get("date", "?"), hold_min, e)

        if progress_callback:
            progress_callback(i + 1, total, symbol)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def save_cap_results(results_df: pd.DataFrame, filename: str = None) -> Path:
    """Save capitulation batch results to CSV."""
    if filename is None:
        filename = f"cap_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    path = CAP_RESULTS_DIR / filename
    results_df.to_csv(path, index=False)
    logger.info("Saved cap batch results to %s (%d rows)", path, len(results_df))
    return path


def load_cap_results(filename: str = None) -> Optional[pd.DataFrame]:
    """Load capitulation batch results. If no filename, load the most recent."""
    if filename:
        path = CAP_RESULTS_DIR / filename
    else:
        csvs = sorted(CAP_RESULTS_DIR.glob("cap_batch_*.csv"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not csvs:
            return None
        path = csvs[0]

    if not path.exists():
        return None

    df = pd.read_csv(path)
    logger.info("Loaded cap batch results from %s (%d rows)", path, len(df))
    return df
