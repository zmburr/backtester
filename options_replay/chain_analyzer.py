"""
Options chain filtering, scoring, and ideal play computation.

Takes a raw chain snapshot and produces the top 8-10 options
ranked by liquidity-adjusted return.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def filter_chain(chain_df: pd.DataFrame, underlying_price: float,
                 side: int, date: str) -> pd.DataFrame:
    """Filter contracts to the most relevant near-term, near-ATM options.

    Optimized for news/headline trades — tight strikes, nearest expiration.

    Filters:
        1. Right: calls for longs, puts for shorts (include opposite flagged)
        2. Expiration: nearest 1 expiration only (news trades need immediate gamma)
        3. Strike: ATM +/- 3% of underlying (tight — focused on liquid near-money)
        4. Liquidity: bid > 0, ask > 0, and (volume > 0 or OI > 10)
        5. Premium: mid >= $0.05
    """
    if chain_df.empty:
        return chain_df

    df = chain_df.copy()

    # Ensure needed columns
    if "mid" not in df.columns:
        if "bid" in df.columns and "ask" in df.columns:
            df["mid"] = (df["bid"] + df["ask"]) / 2
        else:
            return pd.DataFrame()

    # 1. Right: primary direction + opposite flagged
    primary_right = "call" if side == 1 else "put"
    df["is_primary"] = df["right"] == primary_right

    # 2. Expiration: 0-14 DTE
    trade_date = pd.Timestamp(date).normalize()
    if "expiration" in df.columns:
        df["exp_date"] = pd.to_datetime(df["expiration"], format="mixed")
        df["dte"] = (df["exp_date"] - trade_date).dt.days
        df = df[(df["dte"] >= 0) & (df["dte"] <= 14)].copy()

        if df.empty:
            return df

        # Keep nearest 1 expiration (news trades need maximum gamma)
        unique_exp = sorted(df["dte"].unique())[:1]
        df = df[df["dte"].isin(unique_exp)].copy()

    # 3. Strike range: ATM +/- 3% (tight for news trades)
    if "strike" in df.columns:
        lower = underlying_price * 0.97
        upper = underlying_price * 1.03
        df = df[(df["strike"] >= lower) & (df["strike"] <= upper)].copy()

    # 4. Minimum liquidity
    if "bid" in df.columns:
        df = df[df["bid"] > 0].copy()
    if "ask" in df.columns:
        df = df[df["ask"] > 0].copy()

    vol_ok = pd.Series(True, index=df.index)
    if "volume" in df.columns:
        vol_ok = df["volume"] > 0
    oi_ok = pd.Series(False, index=df.index)
    if "open_interest" in df.columns:
        oi_ok = df["open_interest"] > 10

    df = df[vol_ok | oi_ok].copy()

    # 5. Minimum premium
    df = df[df["mid"] >= 0.10].copy()

    # Compute moneyness
    if "strike" in df.columns:
        df["moneyness"] = (df["strike"] - underlying_price) / underlying_price
        df["moneyness_label"] = df.apply(_moneyness_label, axis=1)

    # Sort: primary right first, then by distance from ATM
    if "strike" in df.columns:
        df["atm_dist"] = abs(df["strike"] - underlying_price)
        df = df.sort_values(["is_primary", "atm_dist"], ascending=[False, True])

    logger.info("Filtered chain: %d candidates from %d total", len(df), len(chain_df))
    return df.reset_index(drop=True)


def _moneyness_label(row) -> str:
    m = row.get("moneyness", 0)
    right = row.get("right", "call")
    if right == "call":
        if m < -0.02:
            return "ITM"
        elif m > 0.02:
            return "OTM"
        else:
            return "ATM"
    else:  # put
        if m > 0.02:
            return "ITM"
        elif m < -0.02:
            return "OTM"
        else:
            return "ATM"


_GREEK_COLS = ["delta", "theta", "vega", "rho", "implied_vol"]


def _extract_entry_greeks(greeks_df: pd.DataFrame, snapshot_time: datetime) -> dict:
    """Pull the greeks row closest to snapshot_time from a greeks DataFrame."""
    defaults = {col: None for col in _GREEK_COLS}
    if greeks_df.empty:
        return defaults

    try:
        snap = snapshot_time
        if hasattr(greeks_df.index, 'tz') and greeks_df.index.tz is not None and snap.tzinfo is None:
            from pytz import timezone
            snap = timezone("US/Eastern").localize(snap)

        idx = greeks_df.index.get_indexer([snap], method="nearest")[0]
        if idx < 0:
            return defaults
        row = greeks_df.iloc[idx]
        return {col: row.get(col, None) for col in _GREEK_COLS}
    except Exception:
        return defaults


def compute_option_returns(filtered_df: pd.DataFrame, ohlc_dict: dict,
                           quotes_dict: dict, snapshot_time: datetime,
                           hold_minutes: int = 30,
                           greeks_dict: dict = None) -> pd.DataFrame:
    """For each filtered contract, compute return metrics over the hold window.

    Args:
        filtered_df: output of filter_chain()
        ohlc_dict: {contract_key: DataFrame} from get_option_ohlc()
        quotes_dict: {contract_key: DataFrame} from get_option_quotes()
        snapshot_time: headline/entry timestamp
        hold_minutes: analysis window in minutes
        greeks_dict: {contract_key: DataFrame} from get_option_greeks()

    Returns DataFrame with added columns:
        entry_mid, max_mid, raw_return_pct, time_to_max_min,
        volume_during_window, avg_spread_pct_window,
        delta, theta, vega, rho, implied_vol
    """
    if filtered_df.empty:
        return filtered_df

    df = filtered_df.copy()
    results = []

    for idx, row in df.iterrows():
        key = _contract_key(row)
        ohlc = ohlc_dict.get(key, pd.DataFrame())
        quotes = quotes_dict.get(key, pd.DataFrame())

        entry_mid = row.get("mid", 0)
        if entry_mid <= 0:
            results.append(_empty_return_row())
            continue

        # Get data within the hold window
        if not ohlc.empty and hasattr(ohlc.index, 'tz'):
            window_start = snapshot_time
            window_end = snapshot_time + timedelta(minutes=hold_minutes)

            # Handle timezone
            if ohlc.index.tz is not None and window_start.tzinfo is None:
                from pytz import timezone
                window_start = timezone("US/Eastern").localize(window_start)
                window_end = timezone("US/Eastern").localize(window_end)

            window_ohlc = ohlc[(ohlc.index >= window_start) & (ohlc.index <= window_end)]
        else:
            window_ohlc = pd.DataFrame()

        if not quotes.empty and hasattr(quotes.index, 'tz'):
            window_start_q = snapshot_time
            window_end_q = snapshot_time + timedelta(minutes=hold_minutes)
            if quotes.index.tz is not None and window_start_q.tzinfo is None:
                from pytz import timezone
                window_start_q = timezone("US/Eastern").localize(window_start_q)
                window_end_q = timezone("US/Eastern").localize(window_end_q)
            window_quotes = quotes[(quotes.index >= window_start_q) & (quotes.index <= window_end_q)]
        else:
            window_quotes = pd.DataFrame()

        # Max favorable excursion
        if not window_ohlc.empty and "high" in window_ohlc.columns:
            if row.get("right") == "call" and row.get("is_primary", True):
                max_mid = window_ohlc["high"].max()
            elif row.get("right") == "put" and row.get("is_primary", True):
                max_mid = window_ohlc["high"].max()
            else:
                max_mid = window_ohlc["high"].max()

            raw_return_pct = (max_mid - entry_mid) / entry_mid if entry_mid > 0 else 0

            # Time to max
            max_idx = window_ohlc["high"].idxmax()
            if hasattr(max_idx, 'timestamp') or hasattr(max_idx, 'minute'):
                time_to_max = (max_idx - window_ohlc.index[0]).total_seconds() / 60
            else:
                time_to_max = 0
        else:
            max_mid = entry_mid
            raw_return_pct = 0
            time_to_max = 0

        # Volume during window
        vol_window = 0
        if not window_ohlc.empty and "volume" in window_ohlc.columns:
            vol_window = int(window_ohlc["volume"].sum())

        # Average spread % during window
        avg_spread_pct = row.get("spread_pct", 0)
        if not window_quotes.empty and "spread" in window_quotes.columns and "mid" in window_quotes.columns:
            valid = window_quotes[window_quotes["mid"] > 0]
            if not valid.empty:
                avg_spread_pct = (valid["spread"] / valid["mid"]).mean()

        # Realistic returns: buy at ask, sell at bid
        entry_ask = row.get("ask", 0)
        if entry_ask <= 0:
            entry_ask = entry_mid  # fallback

        max_bid = 0
        time_to_max_bid = 0
        if not window_quotes.empty and "bid" in window_quotes.columns:
            valid_bids = window_quotes[window_quotes["bid"] > 0]["bid"]
            if not valid_bids.empty:
                max_bid = float(valid_bids.max())
                max_bid_idx = valid_bids.idxmax()
                time_to_max_bid = (max_bid_idx - window_quotes.index[0]).total_seconds() / 60

        realistic_return_pct = (max_bid - entry_ask) / entry_ask if entry_ask > 0 and max_bid > 0 else 0
        spread_cost_pct = raw_return_pct - realistic_return_pct

        # Extract greeks at entry time
        entry_greeks = _extract_entry_greeks(
            greeks_dict.get(key, pd.DataFrame()) if greeks_dict else pd.DataFrame(),
            snapshot_time
        )

        results.append({
            "entry_mid": entry_mid,
            "entry_ask": entry_ask,
            "max_mid": max_mid,
            "max_bid": max_bid,
            "raw_return_pct": raw_return_pct,
            "realistic_return_pct": realistic_return_pct,
            "spread_cost_pct": spread_cost_pct,
            "time_to_max_min": time_to_max,
            "time_to_max_bid_min": time_to_max_bid,
            "volume_during_window": vol_window,
            "avg_spread_pct_window": avg_spread_pct,
            **entry_greeks,
        })

    result_df = pd.DataFrame(results, index=df.index)
    for col in result_df.columns:
        df[col] = result_df[col]

    return df


def _empty_return_row() -> dict:
    return {
        "entry_mid": 0,
        "entry_ask": 0,
        "max_mid": 0,
        "max_bid": 0,
        "raw_return_pct": 0,
        "realistic_return_pct": 0,
        "spread_cost_pct": 0,
        "time_to_max_min": 0,
        "time_to_max_bid_min": 0,
        "volume_during_window": 0,
        "avg_spread_pct_window": 0,
        **{col: None for col in _GREEK_COLS},
    }


def score_options(returns_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Score and rank options by liquidity-adjusted return.

    Composite score (0-100):
        40% raw return
        25% spread cost (lower = better)
        20% volume during window
        15% open interest
    """
    if returns_df.empty:
        return returns_df

    df = returns_df.copy()

    # Only score primary-direction options
    if "is_primary" in df.columns:
        df = df[df["is_primary"]].copy()

    if df.empty:
        return df

    def _pctrank(series):
        """Percentile rank 0-1."""
        if series.nunique() <= 1:
            return pd.Series(0.5, index=series.index)
        return series.rank(pct=True, method="average")

    # Return score — use realistic return (accounts for spread crossing)
    return_col = "realistic_return_pct" if "realistic_return_pct" in df.columns else "raw_return_pct"
    df["return_score"] = _pctrank(df[return_col]) * 40

    # Spread score (lower spread = better, so invert)
    spread = df["avg_spread_pct_window"].fillna(1.0)
    df["spread_score"] = _pctrank(1 - spread) * 25

    # Volume score (higher = better)
    vol = df["volume_during_window"].fillna(0)
    df["volume_score"] = _pctrank(vol) * 20

    # OI score (higher = better)
    oi = df.get("open_interest", pd.Series(0, index=df.index))
    if isinstance(oi, pd.Series):
        oi = oi.fillna(0)
    else:
        oi = pd.Series(0, index=df.index)
    df["oi_score"] = _pctrank(oi) * 15

    # Composite
    df["composite_score"] = (
        df["return_score"] + df["spread_score"] +
        df["volume_score"] + df["oi_score"]
    )

    # Rank
    df = df.sort_values("composite_score", ascending=False).head(top_n)
    df["rank"] = range(1, len(df) + 1)

    return df.reset_index(drop=True)


def contract_key(row) -> str:
    """Public version of _contract_key for external use."""
    return _contract_key(row)


def _contract_key(row) -> str:
    """Build a unique key for a contract from a chain row."""
    exp = row.get("expiration", "")
    strike = row.get("strike", 0)
    right = row.get("right", "call")
    return f"{exp}_{strike:.2f}_{right}"


def contract_label(row) -> str:
    """Human-readable contract label like 'Sep-26 $120 Call'."""
    exp = row.get("expiration", "")
    try:
        exp_dt = pd.Timestamp(exp)
        exp_str = exp_dt.strftime("%b-%d")
    except Exception:
        exp_str = str(exp)

    strike = row.get("strike", 0)
    right = row.get("right", "call").capitalize()
    return f"{exp_str} ${strike:.0f} {right}"


def compute_ideal_play_summary(top_option: pd.Series, underlying_entry: float,
                                underlying_max: float, side: int) -> dict:
    """Generate the ideal play narrative for the #1 ranked option."""
    entry_mid = top_option.get("entry_mid", 0)
    entry_ask = top_option.get("entry_ask", entry_mid)
    max_mid = top_option.get("max_mid", 0)
    max_bid = top_option.get("max_bid", 0)
    raw_return_pct = top_option.get("raw_return_pct", 0)
    realistic_return_pct = top_option.get("realistic_return_pct", 0)
    spread_cost_pct = top_option.get("spread_cost_pct", 0)
    spread_pct = top_option.get("avg_spread_pct_window", 0)
    volume = top_option.get("volume_during_window", 0)
    dte = top_option.get("dte", 0)
    label = contract_label(top_option)
    score = top_option.get("composite_score", 0)

    # Leverage vs stock — use realistic return
    stock_return = abs(underlying_max - underlying_entry) / underlying_entry if underlying_entry > 0 else 0
    leverage = realistic_return_pct / stock_return if stock_return > 0 else 0

    # Return per contract — realistic (buy at ask, sell at max bid)
    return_per_contract = (max_bid - entry_ask) * 100 if max_bid > 0 else 0

    # Fillability assessment
    if volume >= 500 and spread_pct < 0.05:
        fillability = "High"
    elif volume >= 100 and spread_pct < 0.15:
        fillability = "Medium"
    else:
        fillability = "Low"

    # Narrative — show both raw and realistic
    direction = "call" if side == 1 else "put"
    moneyness = top_option.get("moneyness_label", "ATM")
    verdict = (
        f"This was a {fillability.lower()}-liquidity {moneyness} {direction} with "
        f"{realistic_return_pct:.0%} realistic return ({raw_return_pct:.0%} raw) "
        f"vs {stock_return:.1%} stock move — {leverage:.0f}x leverage. "
        f"Spread cost ate {spread_cost_pct:.0%} of theoretical return."
    )
    if volume > 0:
        verdict += f" {volume:,} contracts traded."

    # Greeks at entry
    delta = top_option.get("delta", None)
    theta = top_option.get("theta", None)
    vega = top_option.get("vega", None)
    rho = top_option.get("rho", None)
    iv = top_option.get("implied_vol", None)

    return {
        "label": label,
        "entry_mid": entry_mid,
        "entry_ask": entry_ask,
        "max_mid": max_mid,
        "max_bid": max_bid,
        "return_pct": realistic_return_pct,
        "raw_return_pct": raw_return_pct,
        "spread_cost_pct": spread_cost_pct,
        "return_per_contract": return_per_contract,
        "spread_pct": spread_pct,
        "volume": volume,
        "dte": dte,
        "fillability": fillability,
        "leverage": leverage,
        "score": score,
        "verdict": verdict,
        "delta": delta,
        "theta": theta,
        "vega": vega,
        "rho": rho,
        "implied_vol": iv,
    }
