"""
Capitulation options deep-dive analyzer.

Single-trade analysis answering: "I know this stock is cracking —
what's the optimal option contract to buy at the open?"

Produces: delta return curve, DTE comparison, IV decomposition,
liquidity matrix, and top 3 recommendations.

Usage:
    python -m options_replay.cap_deep_analyzer GLD 2026-01-29 09:30 short
"""

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd

from options_replay.theta_client import (
    check_terminal_running, get_chain_snapshot, get_option_ohlc,
    get_option_quotes, get_option_greeks, ThetaTerminalOfflineError,
)
from options_replay.chain_analyzer import (
    compute_option_returns, _extract_entry_greeks, contract_key, contract_label,
    _GREEK_COLS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CapDeepResult:
    symbol: str
    date: str
    entry_time: datetime
    underlying_price: float
    atr_pct: float
    side: int

    # Core data
    wide_chain: pd.DataFrame = field(default_factory=pd.DataFrame)
    returns_by_window: dict = field(default_factory=dict)
    iv_decomposition: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Analysis products
    delta_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    dte_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    liquidity_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    recommendations: list = field(default_factory=list)

    # Context
    target_levels: dict = field(default_factory=dict)
    underlying_bars: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Wide chain filter (replaces filter_chain for cap deep dives)
# ---------------------------------------------------------------------------

def filter_chain_wide(chain_df: pd.DataFrame, underlying_price: float,
                      side: int, date: str,
                      max_expirations: int = 4,
                      strike_range_pct: float = 0.08) -> pd.DataFrame:
    """Filter chain with wider parameters for capitulation analysis.

    vs filter_chain(): keeps up to 4 expirations (not 1),
    ATM +/-8% strikes (not 3%), same liquidity gates.
    """
    if chain_df.empty:
        return chain_df

    df = chain_df.copy()

    if "mid" not in df.columns:
        if "bid" in df.columns and "ask" in df.columns:
            df["mid"] = (df["bid"] + df["ask"]) / 2
        else:
            return pd.DataFrame()

    # 1. Right: primary direction only for cap trades
    primary_right = "call" if side == 1 else "put"
    df = df[df["right"] == primary_right].copy()
    df["is_primary"] = True

    # 2. Expiration: 0-30 DTE, keep up to max_expirations
    trade_date = pd.Timestamp(date).normalize()
    if "expiration" in df.columns:
        df["exp_date"] = pd.to_datetime(df["expiration"], format="mixed")
        df["dte"] = (df["exp_date"] - trade_date).dt.days
        df = df[(df["dte"] >= 0) & (df["dte"] <= 30)].copy()

        if df.empty:
            return df

        unique_exp = sorted(df["dte"].unique())[:max_expirations]
        df = df[df["dte"].isin(unique_exp)].copy()

    # 3. Strike range: wider for cap analysis
    lower = underlying_price * (1 - strike_range_pct)
    upper = underlying_price * (1 + strike_range_pct)
    if "strike" in df.columns:
        df = df[(df["strike"] >= lower) & (df["strike"] <= upper)].copy()

    # 4. Minimum liquidity
    if "bid" in df.columns:
        df = df[df["bid"] > 0].copy()
    if "ask" in df.columns:
        df = df[df["ask"] > 0].copy()

    vol_ok = df["volume"] > 0 if "volume" in df.columns else pd.Series(True, index=df.index)
    oi_ok = df["open_interest"] > 10 if "open_interest" in df.columns else pd.Series(False, index=df.index)
    df = df[vol_ok | oi_ok].copy()

    # 5. Minimum premium
    df = df[df["mid"] >= 0.05].copy()

    # Compute moneyness
    if "strike" in df.columns:
        df["moneyness"] = (df["strike"] - underlying_price) / underlying_price
        df["pct_otm"] = df["moneyness"].abs() * 100
        df["atm_dist"] = abs(df["strike"] - underlying_price)
        df = df.sort_values(["dte", "atm_dist"], ascending=[True, True])

    logger.info("Wide chain filter: %d candidates from %d total (%d expirations)",
                len(df), len(chain_df), df["dte"].nunique() if "dte" in df.columns else 0)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Underlying data helpers (reuse pattern from cap_batch_analyzer)
# ---------------------------------------------------------------------------

def _fetch_underlying_price(symbol: str, date_str: str, time_str: str) -> float:
    try:
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data_queries.polygon_queries import get_intraday

        df = get_intraday(symbol, date_str, 1, "minute")
        if df is None or df.empty:
            return 0.0

        entry_ts = pd.Timestamp(f"{date_str} {time_str}")
        if hasattr(df.index, 'tz') and df.index.tz is not None and entry_ts.tzinfo is None:
            from pytz import timezone
            entry_ts = timezone("US/Eastern").localize(entry_ts)

        idx = df.index.get_indexer([entry_ts], method="nearest")[0]
        return float(df.iloc[idx]["close"])
    except Exception as e:
        logger.warning("Failed to get underlying price for %s: %s", symbol, e)
        return 0.0


def _fetch_underlying_bars(symbol: str, date_str: str) -> pd.DataFrame:
    try:
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
    if multiples is None:
        multiples = [0.5, 1.0, 1.5, 2.0]

    atr_dollars = entry_price * atr_pct
    targets = {}
    for mult in multiples:
        name = f"{mult}x"
        if side == 1:
            targets[name] = entry_price + (mult * atr_dollars)
        else:
            targets[name] = entry_price - (mult * atr_dollars)
    return targets


# ---------------------------------------------------------------------------
# IV decomposition
# ---------------------------------------------------------------------------

def compute_iv_decomposition(returns_df: pd.DataFrame, greeks_dict: dict,
                             quotes_dict: dict, entry_time: datetime,
                             hold_minutes: int) -> pd.DataFrame:
    """Decompose each contract's P&L into delta / vega / theta / residual.

    Uses greeks timeseries at entry and at max favorable excursion time.
    """
    if returns_df.empty:
        return pd.DataFrame()

    rows = []
    exit_time = entry_time + timedelta(minutes=hold_minutes)

    for idx, r in returns_df.iterrows():
        key = contract_key(r)
        greeks_ts = greeks_dict.get(key, pd.DataFrame())
        entry_ask = r.get("entry_ask", 0)
        max_bid = r.get("max_bid", 0)
        time_to_max = r.get("time_to_max_bid_min", 0)

        if entry_ask <= 0 or greeks_ts.empty:
            rows.append(_empty_decomp_row(r))
            continue

        # Greeks at entry
        entry_g = _extract_entry_greeks(greeks_ts, entry_time)

        # Greeks at max bid time (approximate exit)
        max_time = entry_time + timedelta(minutes=max(time_to_max, 1))
        exit_g = _extract_entry_greeks(greeks_ts, max_time)

        delta_entry = entry_g.get("delta") or 0
        vega_entry = entry_g.get("vega") or 0
        theta_entry = entry_g.get("theta") or 0
        iv_entry = entry_g.get("implied_vol") or 0
        iv_exit = exit_g.get("implied_vol") or 0

        # Underlying price change — get from greeks timeseries
        ul_entry = _get_underlying_from_greeks(greeks_ts, entry_time)
        ul_exit = _get_underlying_from_greeks(greeks_ts, max_time)
        delta_underlying = ul_exit - ul_entry if (ul_entry > 0 and ul_exit > 0) else 0

        # P&L decomposition (per share, not per contract)
        delta_pnl = delta_entry * delta_underlying
        vega_pnl = vega_entry * (iv_exit - iv_entry) if iv_entry > 0 and iv_exit > 0 else 0
        hold_hours = max(time_to_max, 1) / 60.0
        theta_pnl = theta_entry * (hold_hours / 24.0)

        actual_pnl = max_bid - entry_ask
        residual_pnl = actual_pnl - delta_pnl - vega_pnl - theta_pnl

        rows.append({
            "contract": contract_label(r),
            "strike": r.get("strike", 0),
            "dte": r.get("dte", 0),
            "delta_entry": delta_entry,
            "iv_entry": iv_entry,
            "iv_exit": iv_exit,
            "iv_change": iv_exit - iv_entry if iv_entry > 0 and iv_exit > 0 else 0,
            "underlying_move": delta_underlying,
            "actual_pnl": actual_pnl,
            "delta_pnl": delta_pnl,
            "vega_pnl": vega_pnl,
            "theta_pnl": theta_pnl,
            "residual_pnl": residual_pnl,
            "delta_pct": delta_pnl / actual_pnl if actual_pnl != 0 else 0,
            "vega_pct": vega_pnl / actual_pnl if actual_pnl != 0 else 0,
            "theta_pct": theta_pnl / actual_pnl if actual_pnl != 0 else 0,
            "residual_pct": residual_pnl / actual_pnl if actual_pnl != 0 else 0,
            "entry_ask": entry_ask,
            "max_bid": max_bid,
            "realistic_return_pct": r.get("realistic_return_pct", 0),
        })

    return pd.DataFrame(rows)


def _get_underlying_from_greeks(greeks_df: pd.DataFrame, ts: datetime) -> float:
    if greeks_df.empty or "underlying_price" not in greeks_df.columns:
        return 0.0
    try:
        snap = ts
        if hasattr(greeks_df.index, 'tz') and greeks_df.index.tz is not None and snap.tzinfo is None:
            from pytz import timezone
            snap = timezone("US/Eastern").localize(snap)
        i = greeks_df.index.get_indexer([snap], method="nearest")[0]
        if i >= 0:
            return float(greeks_df.iloc[i]["underlying_price"])
    except Exception:
        pass
    return 0.0


def _empty_decomp_row(r) -> dict:
    return {
        "contract": contract_label(r),
        "strike": r.get("strike", 0),
        "dte": r.get("dte", 0),
        "delta_entry": 0, "iv_entry": 0, "iv_exit": 0, "iv_change": 0,
        "underlying_move": 0, "actual_pnl": 0,
        "delta_pnl": 0, "vega_pnl": 0, "theta_pnl": 0, "residual_pnl": 0,
        "delta_pct": 0, "vega_pct": 0, "theta_pct": 0, "residual_pct": 0,
        "entry_ask": r.get("entry_ask", 0), "max_bid": r.get("max_bid", 0),
        "realistic_return_pct": r.get("realistic_return_pct", 0),
    }


# ---------------------------------------------------------------------------
# Analysis builders
# ---------------------------------------------------------------------------

DELTA_BUCKET_EDGES = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 1.0]
DELTA_BUCKET_LABELS = [
    "0-0.05", "0.05-0.10", "0.10-0.15", "0.15-0.20", "0.20-0.25",
    "0.25-0.30", "0.30-0.35", "0.35-0.40", "0.40-0.50", "0.50-0.60", "0.60+",
]


def build_delta_return_curve(returns_by_window: dict) -> pd.DataFrame:
    """Group contracts by delta bucket across hold windows.

    Returns: delta_bucket, delta_mid, hold_window, realistic_return,
             raw_return, spread_cost, dollar_return, count, avg_entry_ask
    """
    rows = []
    for hold_min, df in returns_by_window.items():
        if df.empty or "delta" not in df.columns:
            continue

        df = df[df["is_primary"] == True].copy()
        df["abs_delta"] = df["delta"].abs()
        df["delta_bucket"] = pd.cut(
            df["abs_delta"], bins=DELTA_BUCKET_EDGES,
            labels=DELTA_BUCKET_LABELS, right=False,
        )

        for bucket in DELTA_BUCKET_LABELS:
            sub = df[df["delta_bucket"] == bucket]
            if sub.empty:
                continue

            # Midpoint delta for plotting
            i = DELTA_BUCKET_LABELS.index(bucket)
            delta_mid = (DELTA_BUCKET_EDGES[i] + DELTA_BUCKET_EDGES[i + 1]) / 2

            rows.append({
                "delta_bucket": bucket,
                "delta_mid": delta_mid,
                "hold_window": hold_min,
                "realistic_return": sub["realistic_return_pct"].mean(),
                "raw_return": sub["raw_return_pct"].mean(),
                "spread_cost": sub["spread_cost_pct"].mean(),
                "dollar_return": ((sub["max_bid"] - sub["entry_ask"]) * 100).mean(),
                "count": len(sub),
                "avg_entry_ask": sub["entry_ask"].mean(),
                "avg_time_to_max": sub["time_to_max_bid_min"].mean(),
            })

    return pd.DataFrame(rows)


def build_dte_comparison(returns_by_window: dict, primary_window: int = 60) -> pd.DataFrame:
    """Compare return across DTEs for overlapping strikes.

    Uses the primary hold window (default 60 min).
    """
    df = returns_by_window.get(primary_window, pd.DataFrame())
    if df.empty or "dte" not in df.columns or "strike" not in df.columns:
        return pd.DataFrame()

    df = df[df["is_primary"] == True].copy()

    # Only keep strikes that appear in 2+ expirations
    strike_counts = df.groupby("strike")["dte"].nunique()
    multi_strikes = strike_counts[strike_counts >= 2].index
    df = df[df["strike"].isin(multi_strikes)].copy()

    if df.empty:
        return pd.DataFrame()

    return df[["strike", "dte", "expiration", "entry_ask", "entry_mid", "max_bid",
               "realistic_return_pct", "raw_return_pct", "spread_cost_pct",
               "delta", "implied_vol", "vega", "theta",
               "volume_during_window", "avg_spread_pct_window"]].sort_values(
        ["strike", "dte"]
    ).reset_index(drop=True)


def build_liquidity_matrix(returns_df: pd.DataFrame,
                           underlying_price: float) -> pd.DataFrame:
    """All candidates with liquidity grading."""
    if returns_df.empty:
        return pd.DataFrame()

    df = returns_df[returns_df["is_primary"] == True].copy()
    if df.empty:
        return pd.DataFrame()

    cols = ["strike", "dte", "expiration", "entry_ask", "entry_mid",
            "max_bid", "realistic_return_pct", "spread_cost_pct",
            "delta", "implied_vol", "avg_spread_pct_window",
            "volume_during_window"]

    # Add open_interest if available
    if "open_interest" in df.columns:
        cols.append("open_interest")

    out = df[[c for c in cols if c in df.columns]].copy()

    # Liquidity grade
    vol = out.get("volume_during_window", pd.Series(0, index=out.index)).fillna(0)
    oi = out.get("open_interest", pd.Series(0, index=out.index)).fillna(0)
    spread = out.get("avg_spread_pct_window", pd.Series(1, index=out.index)).fillna(1)

    def _grade(v, o, s):
        if v >= 500 and (v + o) >= 1000 and s < 0.05:
            return "A"
        elif v >= 50 and (v + o) >= 200 and s < 0.15:
            return "B"
        else:
            return "C"

    out["liquidity_grade"] = [_grade(v, o, s) for v, o, s in zip(vol, oi, spread)]
    out["dollar_return"] = (out["max_bid"] - out["entry_ask"]) * 100

    return out.sort_values("delta", key=lambda x: x.abs()).reset_index(drop=True)


def recommend_top_contracts(returns_df: pd.DataFrame,
                            iv_decomp: pd.DataFrame) -> list:
    """Pick 3 contracts with cap-specific scoring.

    Weights: 35% return, 25% spread tightness, 20% liquidity,
             10% delta sweet spot (0.15-0.30), 10% vega contribution.
    """
    if returns_df.empty:
        return []

    df = returns_df[returns_df["is_primary"] == True].copy()
    if df.empty:
        return []

    # Need minimum data
    if "realistic_return_pct" not in df.columns or "delta" not in df.columns:
        return []

    def _pctrank(s):
        if s.nunique() <= 1:
            return pd.Series(0.5, index=s.index)
        return s.rank(pct=True, method="average")

    # Score components
    df["s_return"] = _pctrank(df["realistic_return_pct"]) * 35
    df["s_spread"] = _pctrank(1 - df["avg_spread_pct_window"].fillna(1)) * 25

    vol = df["volume_during_window"].fillna(0)
    oi = df.get("open_interest", pd.Series(0, index=df.index)).fillna(0)
    df["s_liquidity"] = _pctrank(vol + oi * 0.5) * 20

    # Delta sweet spot: 0.15-0.30 is optimal for cap trades
    abs_delta = df["delta"].abs()
    df["s_delta"] = (1 - ((abs_delta - 0.225) / 0.225).abs().clip(0, 1)) * 10

    # Vega contribution — higher vega = more upside from IV expansion
    if "vega" in df.columns:
        df["s_vega"] = _pctrank(df["vega"].fillna(0)) * 10
    else:
        df["s_vega"] = 5.0

    df["cap_score"] = df["s_return"] + df["s_spread"] + df["s_liquidity"] + df["s_delta"] + df["s_vega"]

    recs = []
    used = set()

    # 1. Best liquidity-adjusted return (the main recommendation)
    best = df.sort_values("cap_score", ascending=False).iloc[0]
    recs.append(_build_rec(best, iv_decomp, "OPTIMAL",
                           f"Highest cap score ({best['cap_score']:.0f}/100). "
                           f"Best blend of return, liquidity, and delta."))
    used.add(contract_key(best))

    # 2. Best raw return with acceptable liquidity
    liq_ok = df[~df.apply(contract_key, axis=1).isin(used)]
    if "volume_during_window" in liq_ok.columns:
        liq_ok = liq_ok[liq_ok["volume_during_window"] >= 10]
    if not liq_ok.empty:
        aggressive = liq_ok.sort_values("realistic_return_pct", ascending=False).iloc[0]
        recs.append(_build_rec(aggressive, iv_decomp, "AGGRESSIVE",
                               f"Highest return ({aggressive['realistic_return_pct']:.0%}). "
                               f"More OTM, higher leverage."))
        used.add(contract_key(aggressive))

    # 3. Conservative — higher delta, tighter spread
    remaining = df[~df.apply(contract_key, axis=1).isin(used)]
    if not remaining.empty:
        remaining = remaining.copy()
        remaining["cons_score"] = (
            _pctrank(remaining["delta"].abs()) * 40 +
            _pctrank(1 - remaining["avg_spread_pct_window"].fillna(1)) * 35 +
            _pctrank(remaining["realistic_return_pct"]) * 25
        )
        conservative = remaining.sort_values("cons_score", ascending=False).iloc[0]
        recs.append(_build_rec(conservative, iv_decomp, "CONSERVATIVE",
                               f"Higher delta ({conservative['delta']:.2f}), tighter spread. "
                               f"More directional, less leverage risk."))

    return recs


def _build_rec(row, iv_decomp, label, rationale) -> dict:
    rec = {
        "label": label,
        "contract": contract_label(row),
        "strike": row.get("strike", 0),
        "dte": row.get("dte", 0),
        "entry_ask": row.get("entry_ask", 0),
        "max_bid": row.get("max_bid", 0),
        "realistic_return_pct": row.get("realistic_return_pct", 0),
        "dollar_return": (row.get("max_bid", 0) - row.get("entry_ask", 0)) * 100,
        "spread_cost_pct": row.get("spread_cost_pct", 0),
        "delta": row.get("delta", 0),
        "implied_vol": row.get("implied_vol", 0),
        "vega": row.get("vega", 0),
        "volume": row.get("volume_during_window", 0),
        "cap_score": row.get("cap_score", 0),
        "rationale": rationale,
    }

    # Attach IV decomposition if available
    if not iv_decomp.empty:
        strike = row.get("strike", 0)
        dte = row.get("dte", 0)
        match = iv_decomp[(iv_decomp["strike"] == strike) & (iv_decomp["dte"] == dte)]
        if not match.empty:
            m = match.iloc[0]
            rec["delta_pnl_pct"] = m.get("delta_pct", 0)
            rec["vega_pnl_pct"] = m.get("vega_pct", 0)
            rec["theta_pnl_pct"] = m.get("theta_pct", 0)
            rec["iv_change"] = m.get("iv_change", 0)

    return rec


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def analyze_single_cap_trade(
    symbol: str,
    date: str,
    entry_time_str: str = "09:30",
    side_str: str = "short",
    atr_pct: float = 0.0,
    hold_windows: list = None,
    max_expirations: int = 4,
    strike_range_pct: float = 0.08,
    progress_callback=None,
) -> CapDeepResult:
    """Full deep-dive pipeline for a single capitulation trade.

    Args:
        symbol: ticker (e.g. "GLD")
        date: trade date (e.g. "2026-01-29")
        entry_time_str: entry time HH:MM (default "09:30" = market open)
        side_str: "long" or "short"
        atr_pct: ATR as fraction of price (0 = skip targets)
        hold_windows: list of hold minutes (default [30, 60, 120])
        max_expirations: how many expirations to analyze
        strike_range_pct: how wide around ATM
        progress_callback: fn(step, total, message) for UI progress
    """
    if hold_windows is None:
        hold_windows = [30, 60, 120]

    side = -1 if side_str.lower() == "short" else 1
    entry_dt = pd.Timestamp(f"{date} {entry_time_str}:00")
    total_steps = 6

    def _progress(step, msg):
        if progress_callback:
            progress_callback(step, total_steps, msg)
        logger.info("[%d/%d] %s", step, total_steps, msg)

    # Step 1: Underlying data
    _progress(1, f"Fetching underlying data for {symbol}")
    underlying_price = _fetch_underlying_price(symbol, date, entry_time_str + ":00")
    if underlying_price <= 0:
        raise ValueError(f"Could not get underlying price for {symbol} on {date} at {entry_time_str}")

    underlying_bars = _fetch_underlying_bars(symbol, date)
    target_levels = _compute_targets(underlying_price, atr_pct, side) if atr_pct > 0 else {}

    # Step 2: Wide chain snapshot
    _progress(2, f"Fetching options chain ({max_expirations} expirations, ±{strike_range_pct:.0%} strikes)")
    chain_df = get_chain_snapshot(symbol, date, entry_time_str + ":00",
                                  max_dte=30, n_expirations=max_expirations)
    if chain_df.empty:
        raise ValueError(f"Empty chain snapshot for {symbol} on {date}")

    filtered = filter_chain_wide(chain_df, underlying_price, side, date,
                                  max_expirations=max_expirations,
                                  strike_range_pct=strike_range_pct)
    if filtered.empty:
        raise ValueError(f"No contracts passed liquidity filter for {symbol}")

    logger.info("Analyzing %d contracts across %d expirations",
                len(filtered), filtered["dte"].nunique())

    # Step 3: Fetch per-contract data (parallel)
    _progress(3, f"Fetching OHLC/quotes/greeks for {len(filtered)} contracts")
    ohlc_dict = {}
    quotes_dict = {}
    greeks_dict = {}

    def _fetch(row):
        key = contract_key(row)
        ohlc = get_option_ohlc(symbol, row["expiration"], row["strike"],
                               row["right"], date)
        quotes = get_option_quotes(symbol, row["expiration"], row["strike"],
                                   row["right"], date)
        greeks = get_option_greeks(symbol, row["expiration"], row["strike"],
                                   row["right"], date)
        return key, ohlc, quotes, greeks

    with ThreadPoolExecutor(max_workers=6) as executor:
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

    # Step 4: Compute returns for each hold window
    _progress(4, "Computing returns across hold windows")
    returns_by_window = {}
    entry_datetime = entry_dt.to_pydatetime()

    for hold_min in hold_windows:
        returns_df = compute_option_returns(
            filtered, ohlc_dict, quotes_dict, entry_datetime,
            hold_minutes=hold_min, greeks_dict=greeks_dict,
        )
        returns_by_window[hold_min] = returns_df

    # Step 5: IV decomposition (use middle hold window)
    _progress(5, "Computing IV decomposition")
    primary_window = hold_windows[len(hold_windows) // 2] if hold_windows else 60
    primary_returns = returns_by_window.get(primary_window, pd.DataFrame())
    iv_decomp = compute_iv_decomposition(
        primary_returns, greeks_dict, quotes_dict,
        entry_datetime, primary_window,
    )

    # Step 6: Build analysis products
    _progress(6, "Building analysis products")
    delta_curve = build_delta_return_curve(returns_by_window)
    dte_comp = build_dte_comparison(returns_by_window, primary_window)
    liq_matrix = build_liquidity_matrix(primary_returns, underlying_price)
    recs = recommend_top_contracts(primary_returns, iv_decomp)

    return CapDeepResult(
        symbol=symbol,
        date=date,
        entry_time=entry_datetime,
        underlying_price=underlying_price,
        atr_pct=atr_pct,
        side=side,
        wide_chain=filtered,
        returns_by_window=returns_by_window,
        iv_decomposition=iv_decomp,
        delta_curve=delta_curve,
        dte_comparison=dte_comp,
        liquidity_matrix=liq_matrix,
        recommendations=recs,
        target_levels=target_levels,
        underlying_bars=underlying_bars,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(result: CapDeepResult):
    """Print formatted text report to stdout."""
    print(f"\n{'='*70}")
    print(f"  CAP DEEP DIVE: {result.symbol} on {result.date}")
    print(f"  Entry: {result.entry_time.strftime('%H:%M')} | "
          f"Underlying: ${result.underlying_price:.2f} | "
          f"Side: {'SHORT (puts)' if result.side == -1 else 'LONG (calls)'} | "
          f"ATR: {result.atr_pct:.2%}")
    print(f"{'='*70}")

    # Chain summary
    chain = result.wide_chain
    if not chain.empty:
        exps = sorted(chain["dte"].unique())
        print(f"\n  Chain: {len(chain)} contracts across {len(exps)} expirations "
              f"(DTE: {', '.join(str(int(e)) for e in exps)})")

    # Recommendations
    print(f"\n  {'-'*66}")
    print("  TOP RECOMMENDATIONS")
    print(f"  {'-'*66}")
    for rec in result.recommendations:
        print(f"\n  [{rec['label']}] {rec['contract']}")
        print(f"    Entry: ${rec['entry_ask']:.2f}  ->  Max Bid: ${rec['max_bid']:.2f}  "
              f"({rec['realistic_return_pct']:.0%} return, ${rec['dollar_return']:.0f}/contract)")
        print(f"    Delta: {rec['delta']:.3f}  |  IV: {rec.get('implied_vol', 0):.1%}  |  "
              f"Spread cost: {rec['spread_cost_pct']:.1%}  |  Volume: {rec.get('volume', 0):.0f}")
        if "iv_change" in rec:
            print(f"    IV decomp: delta={rec.get('delta_pnl_pct', 0):.0%}  "
                  f"vega={rec.get('vega_pnl_pct', 0):.0%}  "
                  f"theta={rec.get('theta_pnl_pct', 0):.0%}  "
                  f"(IV moved {rec['iv_change']:+.3f})")
        print(f"    {rec['rationale']}")

    # Delta curve summary
    if not result.delta_curve.empty:
        print(f"\n  {'-'*66}")
        print("  DELTA RETURN CURVE")
        print(f"  {'-'*66}")
        for hw in sorted(result.delta_curve["hold_window"].unique()):
            sub = result.delta_curve[result.delta_curve["hold_window"] == hw]
            if sub.empty:
                continue
            best = sub.sort_values("realistic_return", ascending=False).iloc[0]
            print(f"\n  {hw}min hold — Peak at delta {best['delta_mid']:.2f} "
                  f"({best['realistic_return']:.0%} return, "
                  f"n={int(best['count'])})")

            # Show the curve
            for _, row in sub.iterrows():
                bar_len = max(0, int(row["realistic_return"] * 50))
                bar = "#" * bar_len
                print(f"    d {row['delta_mid']:.2f}  {row['realistic_return']:>6.0%}  "
                      f"(n={int(row['count'])})  {bar}")

    # DTE comparison
    if not result.dte_comparison.empty:
        print(f"\n  {'-'*66}")
        print("  DTE COMPARISON (same strikes across expirations)")
        print(f"  {'-'*66}")
        for strike, group in result.dte_comparison.groupby("strike"):
            print(f"\n  ${strike:.0f} strike:")
            for _, r in group.iterrows():
                print(f"    {int(r['dte'])}DTE: {r['realistic_return_pct']:.0%} return  "
                      f"(d={r['delta']:.3f}, IV={r.get('implied_vol', 0):.1%}, "
                      f"spread={r.get('avg_spread_pct_window', 0):.1%})")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if len(sys.argv) < 4:
        print("Usage: python -m options_replay.cap_deep_analyzer SYMBOL DATE TIME [SIDE] [ATR_PCT]")
        print("  e.g. python -m options_replay.cap_deep_analyzer GLD 2026-01-29 09:30 short 0.0148")
        sys.exit(1)

    sym = sys.argv[1].upper()
    dt = sys.argv[2]
    tm = sys.argv[3]
    sd = sys.argv[4] if len(sys.argv) > 4 else "short"
    atr = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0

    try:
        check_terminal_running()
    except ThetaTerminalOfflineError:
        print("ERROR: Theta Terminal is not running. Start it first.")
        sys.exit(1)

    result = analyze_single_cap_trade(sym, dt, tm, sd, atr)
    _print_report(result)
