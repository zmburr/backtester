"""
trillium_queries.py   – SHEL DataGateway edition
------------------------------------------------
Drop‑in compatible with code that used the original ctxcapmd helper.

Key changes
-----------
* uses `sheldatagateway.Session` (Prod env) instead of ctxcapmd sockets
* identical public API: get_intraday, get_daily, get_vwap, get_levels_data…
* retains helper utilities for volume, VWAP, %‑move, etc.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, date as dt_date, timedelta
from typing import Dict, List, Union

import pandas as pd
import pandas_market_calendars as mcal
import pytz
from pytz import timezone
HAS_SHEL = False
try:
    import sheldatagateway
    from sheldatagateway import environments
    HAS_SHEL = True
except ImportError:
    sheldatagateway = None  # type: ignore[assignment]
    environments = None      # type: ignore[assignment]

from dotenv import load_dotenv

# -------------------------------------------------------------------------- #
#  Configuration                                                             #
# -------------------------------------------------------------------------- #

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

logger = logging.getLogger(__name__)

# Credentials – *strongly* recommend exporting these as env vars instead of
# hard‑coding (shown here for parity with your previous examples).
USER = os.getenv("SHEL_USER", "zburr")
PWD  = os.getenv("SHEL_API_PWD")

EASTERN = pytz.timezone("US/Eastern")

# -------------------------------------------------------------------------- #
#  Core data access helpers (all SHEL)                                       #
# -------------------------------------------------------------------------- #

def _ensure_date_str(d: Union[str, dt_date, datetime]) -> str:
    """Return *d* formatted as 'YYYY‑MM‑DD'."""
    if isinstance(d, datetime):
        d = d.date()
    if isinstance(d, dt_date):
        return d.strftime("%Y-%m-%d")
    return d  # already a str

def _shel_request(
    ticker: str,
    start_date: str,
    end_date: str,
    subscriptions: List[str]
) -> pd.DataFrame:
    """
    Generic one‑shot request to SHEL DataGateway; blocks until all data arrive.
    Returns a DataFrame of the raw message objects.
    """
    if not HAS_SHEL:
        raise ImportError(
            "sheldatagateway is not installed. "
            "Install it or use polygon_queries as an alternative."
        )

    aggs: List[Dict] = []

    with sheldatagateway.Session(environments.env_defs.Prod, USER, PWD) as session:

        def append(obj):
            aggs.append(obj)

        handle = session.request_data(
            callback=append,
            symbol=ticker,
            start_date=datetime.strptime(start_date, "%Y-%m-%d").date(),
            end_date=datetime.strptime(end_date,   "%Y-%m-%d").date(),
            subscriptions=subscriptions,
        )

        handle.wait()
        handle.raise_on_error()

    return pd.DataFrame(aggs)


def get_intraday(
    ticker: str,
    date: Union[str, dt_date, datetime],
    bar_type: str
) -> pd.DataFrame:
    """
    Intraday OHLCV bars for *ticker* on *date*.
    Example *bar_type*: 'bar-1min', 'bar-5s', 'bar-3min', etc.
    """
    date_str = _ensure_date_str(date)
    df = _shel_request(ticker, date_str, date_str, [bar_type])

    if "close-time" in df.columns:
        df["close-time"] = (
            pd.to_datetime(df["close-time"], unit="ns")
              .dt.tz_localize("UTC")
              .dt.tz_convert(EASTERN)
        )
        df.set_index("close-time", inplace=True)

    return df


def get_daily(ticker: str, date: str) -> Dict[str, float]:
    """
    Daily OHLCV snapshot for *ticker* on *date*.
    """
    df = get_intraday(ticker, date, "bar-1day")
    if df.empty:
        raise ValueError(f"No daily bar for {ticker} on {date}")

    # DataGateway returns a single row for bar‑1day.
    bar = df.iloc[0]
    return {
        "open":   bar["open"],
        "close":  bar["close"],
        "high":   bar["high"],
        "low":    bar["low"],
        "volume": bar.get("volume"),
    }


def get_vwap(ticker: str, date: str) -> float | None:
    """
    VWAP (volume‑weighted average price) for *ticker* on *date*.
    Uses 'vwma-1h' bars and returns the final reading of the day.
    """
    df = _shel_request(ticker, date, date, ["vwma-1h"])
    if df.empty:
        return None

    df["time"] = (
        pd.to_datetime(df["time"], unit="ns")
          .dt.tz_localize("UTC")
          .dt.tz_convert(EASTERN)
    )
    df.set_index("time", inplace=True)
    return df.iloc[-1]["value"]


# -------------------------------------------------------------------------- #
#  Utility functions preserved from the legacy helper                        #
# -------------------------------------------------------------------------- #

def _adjust_date(original_date: datetime, days_to_subtract: int) -> str:
    """Find the N‑th prior NYSE trading day."""
    nyse = mcal.get_calendar("NYSE")
    look_back_start = original_date - timedelta(days=365)
    trading_days = nyse.valid_days(start_date=look_back_start, end_date=original_date)
    if trading_days.empty:
        raise ValueError("No valid trading days in look‑back window")

    adjusted = trading_days[-days_to_subtract - 1].date()
    if adjusted == original_date.date():
        return _adjust_date(original_date, days_to_subtract + 1)
    return adjusted.strftime("%Y-%m-%d")


def adjust_date_to_market(date_string: str, days_to_subtract: int) -> str:
    """Public wrapper that keeps the original name used by callers."""
    return _adjust_date(pd.to_datetime(date_string), days_to_subtract)


def get_levels_data(
    ticker: str,
    date: str,
    window: int,
    bar_type: str
) -> pd.DataFrame:
    """
    Return a DataFrame of *bar_type* bars from *(date - window)* through *date*.
    """
    end_date = _ensure_date_str(date)
    start_date = adjust_date_to_market(end_date, window)

    df = _shel_request(ticker, start_date, end_date, [bar_type])

    if "close-time" in df.columns:
        df["close-time"] = (
            pd.to_datetime(df["close-time"], unit="ns")
              .dt.tz_localize("UTC")
              .dt.tz_convert(EASTERN)
        )
        df.set_index("close-time", inplace=True)

    return df


def fetch_and_calculate_volumes(ticker: str, trade_date: str) -> Dict[str, float]:
    """
    Re‑implements the old ctxcapmd routine entirely with SHEL helpers.
    """
    intraday = get_intraday(ticker, trade_date, "bar-1min")
    adv      = get_levels_data(ticker, trade_date, 30, "bar-1day")

    metrics = {
        "avg_daily_vol": adv["volume"].sum() / len(adv) if not adv.empty else 0,
        "vol_on_breakout_day": intraday["volume"].sum(),
        "premarket_vol": intraday.between_time("06:00", "09:30")["volume"].sum(),
        "vol_in_first_5_min": intraday.between_time("09:30", "09:35")["volume"].sum(),
        "vol_in_first_10_min": intraday.between_time("09:30", "09:40")["volume"].sum(),
        "vol_in_first_15_min": intraday.between_time("09:30", "09:45")["volume"].sum(),
        "vol_in_first_30_min": intraday.between_time("09:30", "10:00")["volume"].sum(),
    }
    return metrics


def get_actual_current_price_trill(ticker: str) -> float:
    today = datetime.now(EASTERN).strftime("%Y-%m-%d")
    df = get_intraday(ticker, today, "bar-1s")
    if df.empty:
        raise ValueError(f"No 1‑second data for {ticker} today")
    return df.iloc[-1]["close"]


def get_price_with_fallback(ticker: str, base_date: str, days_ago: int) -> float | None:
    """
    Walk back up to *days_ago* trading days looking for a valid daily open.
    """
    while days_ago > 0:
        try:
            candidate_date = adjust_date_to_market(base_date, days_ago)
            price = get_daily(ticker, candidate_date)["open"]
            if price is not None:
                return price
        except Exception:
            pass
        days_ago -= 1
    return None


def get_ticker_pct_move(ticker: str, date: str, current_price: float) -> Dict[str, float | None]:
    pct = {}
    for lookback in [120, 90, 30, 15, 3]:
        past_price = get_price_with_fallback(ticker, date, lookback)
        pct[f"pct_change_{lookback}"] = (
            (current_price - past_price) / past_price if past_price else None
        )
    return pct


def get_mav_data(df: pd.DataFrame) -> Dict[str, float | None]:
    """
    Simplified SMA extractor – assumes *df* already contains the needed windows.
    """
    mav = {}
    for window in ["10D", "20D", "50D", "200D"]:
        series = df.get(window)
        mav[f"price_{window.lower()}mav"] = series.get("SMA") if series else None
    return mav


# -------------------------------------------------------------------------- #
#  Self‑test (run `python trillium_queries.py` to smoke‑test the functions)  #
# -------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample_ticker = "GLD"
    print(get_actual_current_price_trill(sample_ticker))
    sample_date   = "2026-01-28"

    print("VWAP:", get_vwap(sample_ticker, sample_date))
    daily = get_daily(sample_ticker, sample_date)
    print("Daily OHLC:", daily)

    df_min = get_intraday(sample_ticker, sample_date, "bar-1min")
    print("First 3 minutes")
    print(df_min.head(3))
