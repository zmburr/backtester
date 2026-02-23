"""Persistent per-ticker daily OHLCV cache.

Stores daily bars per ticker as parquet files in ``data/ticker_cache/``.
On each call: load cached file, find last cached date, fetch only new bars
from Polygon, append, save back.

**Never caches today's bar** (partial during market hours) but includes it
in the returned DataFrame when available from a fresh fetch.

Thread-safe with per-ticker locks for use with ``ThreadPoolExecutor``.
"""

import os
import threading
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import pandas_market_calendars as mcal

# Import the original Polygon function *before* any monkey-patching happens.
# This module is imported at the top of generate_report.py, which saves
# originals before calling cache.install(), so this reference is safe.
from data_queries.polygon_queries import get_levels_data as _polygon_get_levels_data

_nyse = mcal.get_calendar('NYSE')


class TickerCache:
    """Persistent per-ticker daily OHLCV cache backed by parquet files."""

    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data', 'ticker_cache',
            )
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict = {}
        self._global_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lock helpers
    # ------------------------------------------------------------------

    def _get_ticker_lock(self, ticker: str) -> threading.Lock:
        with self._global_lock:
            if ticker not in self._locks:
                self._locks[ticker] = threading.Lock()
            return self._locks[ticker]

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def _cache_path(self, ticker: str) -> Path:
        return self._cache_dir / f"{ticker}.parquet"

    def _load_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(ticker)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            print(f"[TickerCache] Failed to load cache for {ticker}: {e}")
            return None

    def _save_cache(self, ticker: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        path = self._cache_path(ticker)
        try:
            df.to_parquet(path)
        except Exception as e:
            print(f"[TickerCache] Failed to save cache for {ticker}: {e}")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get_daily_bars(
        self, ticker: str, date: str, window: int = 310,
    ) -> Optional[pd.DataFrame]:
        """Fetch daily bars, using disk cache to minimise Polygon API calls.

        Args:
            ticker: Stock ticker symbol.
            date: Reference date in ``YYYY-MM-DD`` format.
            window: Calendar-day lookback (default 310 ≈ 200+ trading days).

        Returns:
            DataFrame matching ``polygon_queries.get_levels_data()`` format
            (OHLCV indexed by tz-aware timestamp), or *None* if no data.
        """
        lock = self._get_ticker_lock(ticker)
        with lock:
            return self._get_daily_bars_locked(ticker, date, window)

    def _get_daily_bars_locked(
        self, ticker: str, date: str, window: int,
    ) -> Optional[pd.DataFrame]:
        today_date = pd.to_datetime(date).date()
        cached_df = self._load_cache(ticker)

        if cached_df is not None and not cached_df.empty:
            cache_start = cached_df.index[0].date()
            cache_end = cached_df.index[-1].date()
            earliest_needed = (
                pd.to_datetime(date) - pd.Timedelta(days=window)
            ).date()

            has_enough_history = cache_start <= earliest_needed + timedelta(days=7)
            missing_days = self._count_missing_trading_days(cache_end, today_date)

            if has_enough_history and missing_days == 0:
                # Cache is fully up-to-date — zero API calls.
                return cached_df

            if has_enough_history and missing_days > 0:
                # Incremental fetch: only the gap between cache end and today.
                gap_calendar = (today_date - cache_end).days + 3  # small buffer
                new_data = _polygon_get_levels_data(
                    ticker, date, gap_calendar, 1, 'day',
                )
                if new_data is not None and not new_data.empty:
                    combined = pd.concat([cached_df, new_data]).sort_index()
                    combined = combined[~combined.index.duplicated(keep='last')]
                    self._persist_completed(combined, today_date, ticker)
                    return combined
                # Fetch returned nothing — return what we have.
                return cached_df

        # Full fetch (no usable cache).
        result = _polygon_get_levels_data(ticker, date, window, 1, 'day')
        if result is not None and not result.empty:
            self._persist_completed(result, today_date, ticker)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_missing_trading_days(last_cached_date, today_date) -> int:
        """Return the number of *completed* trading days between cache end
        and today (exclusive on both ends of today).

        A return value of 0 means the cache already contains data for every
        completed trading day up to (but not including) today.
        """
        start = last_cached_date + timedelta(days=1)
        end = today_date - timedelta(days=1)  # exclude today (possibly partial)
        if start > end:
            return 0
        return len(_nyse.valid_days(start_date=start, end_date=end))

    def _persist_completed(
        self, df: pd.DataFrame, today_date, ticker: str,
    ) -> None:
        """Save only *completed* daily bars (everything before today)."""
        mask = df.index.map(lambda ts: ts.date() < today_date)
        to_cache = df.loc[mask]
        if not to_cache.empty:
            self._save_cache(ticker, to_cache)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear_ticker(self, ticker: str) -> None:
        path = self._cache_path(ticker)
        if path.exists():
            path.unlink()

    def clear_all(self) -> None:
        for path in self._cache_dir.glob("*.parquet"):
            path.unlink()
