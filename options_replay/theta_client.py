"""
Theta Data REST client for historical options data.

Theta Terminal must be running locally (Java process).
All responses are cached via cache.py so repeated lookups are free.
"""

import os
import logging
from datetime import datetime

import pandas as pd
import requests
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log,
)
from dotenv import load_dotenv

from options_replay.cache import load_cached, save_to_cache

logger = logging.getLogger(__name__)

# Load .env from backtester root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

BASE_URL = os.getenv("THETA_BASE_URL", "http://localhost:25503")
V3 = f"{BASE_URL}/v3"

_session = requests.Session()
_session.timeout = 30


class ThetaTerminalOfflineError(Exception):
    """Raised when Theta Terminal is not reachable."""
    pass


_theta_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
    )),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


def check_terminal_running() -> bool:
    """Verify Theta Terminal is alive."""
    try:
        resp = _session.get(f"{V3}/option/list/expirations", params={"symbol": "AAPL"}, timeout=5)
        return resp.status_code == 200
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False


def _fmt_date(date_str: str) -> str:
    """Normalize date to YYYYMMDD for Theta API."""
    d = pd.Timestamp(date_str)
    return d.strftime("%Y%m%d")


def _fmt_time(time_str: str) -> str:
    """Normalize time to HH:MM:SS for Theta API."""
    # Handle various formats: "14:30", "14:30:00", "2:30 PM"
    try:
        t = pd.Timestamp(f"2000-01-01 {time_str}")
        return t.strftime("%H:%M:%S")
    except Exception:
        return time_str


def _paginated_get(url: str, params: dict) -> list[dict]:
    """Fetch all pages from a Theta Data endpoint."""
    params = {**params, "format": "json"}
    all_rows = []

    while True:
        resp = _session.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        if "response" in data:
            all_rows.extend(data["response"])
        elif isinstance(data, list):
            all_rows.extend(data)

        # Check for pagination
        next_page = resp.headers.get("Next-Page")
        if next_page:
            params["next_page"] = next_page
        else:
            break

    return all_rows


def _flatten_timeseries(rows: list) -> list:
    """Flatten Theta's nested {contract, data: [...]} format into flat rows.

    OHLC/quote endpoints return 1 row per contract with all bars in data[].
    """
    flat = []
    for row in rows:
        if isinstance(row, dict) and "contract" in row and "data" in row:
            data_list = row["data"]
            if isinstance(data_list, list):
                for bar in data_list:
                    flat.append(bar)
        else:
            flat.append(row)
    return flat


def _parse_timestamps(df: pd.DataFrame, date: str) -> pd.DataFrame:
    """Parse timestamp column and set as Eastern-time index."""
    ts_col = None
    for candidate in ["datetime", "timestamp", "time", "date", "ms_of_day"]:
        if candidate in df.columns:
            ts_col = candidate
            break

    if ts_col and ts_col == "ms_of_day":
        base = pd.Timestamp(date)
        df["timestamp"] = base + pd.to_timedelta(df[ts_col].astype(int), unit="ms")
        df["timestamp"] = df["timestamp"].dt.tz_localize("US/Eastern")
    elif ts_col:
        col_data = df[ts_col]
        if col_data.dtype in ["int64", "float64"]:
            df["timestamp"] = pd.to_datetime(col_data, unit="ms")
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
        else:
            # ISO string like "2026-03-05T12:28:00.000"
            df["timestamp"] = pd.to_datetime(col_data)
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("US/Eastern")
            else:
                df["timestamp"] = df["timestamp"].dt.tz_convert("US/Eastern")

    if "timestamp" in df.columns:
        df = df.set_index("timestamp").sort_index()

    return df


@_theta_retry
def get_chain_snapshot(symbol: str, date: str, time_of_day: str,
                       max_dte: int = 7, n_expirations: int = 1) -> pd.DataFrame:
    """Options chain NBBO at a specific point in time.

    Instead of fetching ALL expirations (slow), fetches only the nearest
    n_expirations within max_dte days. ~18x faster than wildcard query.

    Returns DataFrame with columns:
        expiration, strike, right, bid, ask, bid_size, ask_size,
        mid, spread, spread_pct
    """
    date_fmt = _fmt_date(date)
    time_fmt = _fmt_time(time_of_day)

    # Check cache
    cached = load_cached(symbol, date_fmt, "chain_snapshot", time=time_fmt.replace(":", ""))
    if cached is not None:
        return cached

    # Step 1: Get available expirations and pick nearest within DTE window
    try:
        all_expirations = get_expirations(symbol)
    except Exception as e:
        logger.warning("Failed to list expirations for %s: %s", symbol, e)
        all_expirations = []

    trade_date = pd.Timestamp(date)
    target_exps = []
    for exp_str in all_expirations:
        try:
            exp_date = pd.Timestamp(exp_str)
            dte = (exp_date - trade_date).days
            if 0 <= dte <= max_dte:
                target_exps.append(exp_str)
        except Exception:
            continue
    target_exps = sorted(target_exps)[:n_expirations]

    if not target_exps:
        logger.warning("No expirations within %d DTE for %s on %s", max_dte, symbol, date)
        return pd.DataFrame()

    logger.info("Fetching chain for %s: %d expirations %s", symbol, len(target_exps), target_exps)

    # Step 2: Fetch chain per expiration (much faster than wildcard)
    rows = []
    for exp in target_exps:
        exp_fmt = _fmt_date(exp)
        try:
            exp_rows = _paginated_get(f"{V3}/option/at_time/quote", {
                "symbol": symbol.upper(),
                "expiration": exp_fmt,
                "strike": "*",
                "right": "both",
                "start_date": date_fmt,
                "end_date": date_fmt,
                "time_of_day": time_fmt,
            })
            rows.extend(exp_rows)
        except requests.exceptions.ConnectionError:
            raise ThetaTerminalOfflineError(
                "Cannot connect to Theta Terminal. Is it running on localhost:25503?"
            )

    if not rows:
        logger.warning("Empty chain snapshot for %s on %s at %s", symbol, date, time_of_day)
        return pd.DataFrame()

    # Theta v3 at_time/quote returns nested format:
    #   {"contract": {expiration, symbol, strike, right}, "data": [{bid, ask, ...}]}
    # Flatten into one row per contract.
    flat_rows = []
    for row in rows:
        if isinstance(row, dict) and "contract" in row and "data" in row:
            contract_info = row["contract"]
            data_list = row["data"]
            if isinstance(data_list, list) and data_list:
                merged = {**contract_info, **data_list[0]}
                flat_rows.append(merged)
        else:
            # Already flat
            flat_rows.append(row)

    if not flat_rows:
        return pd.DataFrame()

    df = pd.DataFrame(flat_rows)

    # Normalize column names
    col_map = {c: c.lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=col_map)

    # Convert types
    for col in ["bid", "ask", "strike"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["bid_size", "ask_size", "volume", "open_interest"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Format expiration as YYYY-MM-DD string
    if "expiration" in df.columns:
        df["expiration"] = pd.to_datetime(df["expiration"].astype(str), format="mixed").dt.strftime("%Y-%m-%d")

    # Compute derived columns
    if "bid" in df.columns and "ask" in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2
        df["spread"] = df["ask"] - df["bid"]
        df["spread_pct"] = df["spread"] / df["mid"].replace(0, float("nan"))

    # Normalize right to lowercase
    if "right" in df.columns:
        df["right"] = df["right"].astype(str).str.lower().str.strip()
        df["right"] = df["right"].replace({"c": "call", "p": "put", "C": "call", "P": "put"})

    save_to_cache(df, symbol, date_fmt, "chain_snapshot", time=time_fmt.replace(":", ""))
    return df


@_theta_retry
def get_option_ohlc(symbol: str, expiration: str, strike: float, right: str,
                    date: str, interval: str = "1m") -> pd.DataFrame:
    """1-min OHLC bars for a specific option contract.

    Returns DataFrame indexed by timestamp (Eastern):
        open, high, low, close, volume
    """
    date_fmt = _fmt_date(date)
    exp_fmt = _fmt_date(expiration)
    right_short = right[0].upper() if right else "C"
    strike_str = f"{strike:.2f}"

    cached = load_cached(symbol, date_fmt, "ohlc",
                         exp=exp_fmt, strike=strike_str, right=right_short, interval=interval)
    if cached is not None:
        return cached

    try:
        rows = _paginated_get(f"{V3}/option/history/ohlc", {
            "symbol": symbol.upper(),
            "expiration": exp_fmt,
            "strike": strike,
            "right": right_short,
            "start_date": date_fmt,
            "end_date": date_fmt,
            "interval": interval,
        })
    except requests.exceptions.ConnectionError:
        raise ThetaTerminalOfflineError("Theta Terminal not reachable.")

    if not rows:
        return pd.DataFrame()

    # Flatten nested {contract, data} format — data has the 1-min bars
    flat_rows = _flatten_timeseries(rows)
    if not flat_rows:
        return pd.DataFrame()

    df = pd.DataFrame(flat_rows)
    col_map = {c: c.lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=col_map)

    # Parse timestamp
    df = _parse_timestamps(df, date)

    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

    save_to_cache(df, symbol, date_fmt, "ohlc",
                  exp=exp_fmt, strike=strike_str, right=right_short, interval=interval)
    return df


@_theta_retry
def get_option_quotes(symbol: str, expiration: str, strike: float, right: str,
                      date: str, interval: str = "1m") -> pd.DataFrame:
    """1-min bid/ask quote bars for a specific option contract.

    Returns DataFrame indexed by timestamp (Eastern):
        bid, ask, mid, spread
    """
    date_fmt = _fmt_date(date)
    exp_fmt = _fmt_date(expiration)
    right_short = right[0].upper() if right else "C"
    strike_str = f"{strike:.2f}"

    cached = load_cached(symbol, date_fmt, "quotes",
                         exp=exp_fmt, strike=strike_str, right=right_short, interval=interval)
    if cached is not None:
        return cached

    try:
        rows = _paginated_get(f"{V3}/option/history/quote", {
            "symbol": symbol.upper(),
            "expiration": exp_fmt,
            "strike": strike,
            "right": right_short,
            "start_date": date_fmt,
            "end_date": date_fmt,
            "interval": interval,
        })
    except requests.exceptions.ConnectionError:
        raise ThetaTerminalOfflineError("Theta Terminal not reachable.")

    if not rows:
        return pd.DataFrame()

    # Flatten nested {contract, data} format
    flat_rows = _flatten_timeseries(rows)
    if not flat_rows:
        return pd.DataFrame()

    df = pd.DataFrame(flat_rows)
    col_map = {c: c.lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=col_map)

    df = _parse_timestamps(df, date)

    for col in ["bid", "ask"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "bid" in df.columns and "ask" in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2
        df["spread"] = df["ask"] - df["bid"]

    save_to_cache(df, symbol, date_fmt, "quotes",
                  exp=exp_fmt, strike=strike_str, right=right_short, interval=interval)
    return df


@_theta_retry
def get_option_greeks(symbol: str, expiration: str, strike: float, right: str,
                      date: str, interval: str = "1m") -> pd.DataFrame:
    """1-min first-order greeks for a specific option contract.

    Returns DataFrame indexed by timestamp (Eastern):
        delta, theta, vega, rho, implied_vol, underlying_price
    """
    date_fmt = _fmt_date(date)
    exp_fmt = _fmt_date(expiration)
    right_short = right[0].upper() if right else "C"
    strike_str = f"{strike:.2f}"

    cached = load_cached(symbol, date_fmt, "greeks",
                         exp=exp_fmt, strike=strike_str, right=right_short, interval=interval)
    if cached is not None:
        return cached

    try:
        rows = _paginated_get(f"{V3}/option/history/greeks/first_order", {
            "symbol": symbol.upper(),
            "expiration": exp_fmt,
            "strike": strike,
            "right": right_short,
            "start_date": date_fmt,
            "end_date": date_fmt,
            "interval": interval,
        })
    except requests.exceptions.ConnectionError:
        raise ThetaTerminalOfflineError("Theta Terminal not reachable.")

    if not rows:
        return pd.DataFrame()

    flat_rows = _flatten_timeseries(rows)
    if not flat_rows:
        return pd.DataFrame()

    df = pd.DataFrame(flat_rows)
    col_map = {c: c.lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=col_map)

    df = _parse_timestamps(df, date)

    for col in ["delta", "theta", "vega", "rho", "epsilon", "lambda", "implied_vol", "underlying_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    save_to_cache(df, symbol, date_fmt, "greeks",
                  exp=exp_fmt, strike=strike_str, right=right_short, interval=interval)
    return df


@_theta_retry
def get_expirations(symbol: str) -> list:
    """Available expiration dates for a symbol."""
    try:
        rows = _paginated_get(f"{V3}/option/list/expirations", {
            "symbol": symbol.upper(),
        })
    except requests.exceptions.ConnectionError:
        raise ThetaTerminalOfflineError("Theta Terminal not reachable.")

    # Response format varies — could be list of strings or list of dicts
    expirations = []
    for r in rows:
        if isinstance(r, dict):
            val = r.get("expiration", r.get("exp", ""))
            expirations.append(str(val))
        else:
            expirations.append(str(r))
    return sorted(expirations)


@_theta_retry
def get_strikes(symbol: str, expiration: str) -> list:
    """Available strikes for a symbol + expiration."""
    exp_fmt = _fmt_date(expiration)
    try:
        rows = _paginated_get(f"{V3}/option/list/strikes", {
            "symbol": symbol.upper(),
            "expiration": exp_fmt,
        })
    except requests.exceptions.ConnectionError:
        raise ThetaTerminalOfflineError("Theta Terminal not reachable.")

    strikes = []
    for r in rows:
        if isinstance(r, dict):
            val = r.get("strike", r.get("strike_price", 0))
            strikes.append(float(val))
        else:
            strikes.append(float(r))
    return sorted(strikes)
