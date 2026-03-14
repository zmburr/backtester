"""
Live Contract Picker — real-time options recommendation engine.

Given a ticker, date, time, and direction, fetches the current options chain
from Theta Terminal and recommends the ideal contract using snapshot-available
data (greeks, spread, volume, OI) plus historical batch stats.

CLI usage:
    python -m options_replay.live_picker NVDA 2026-03-14 14:36:00 long 120.50 [500]
    Outputs JSON to stdout.
"""

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd

from options_replay.theta_client import (
    check_terminal_running,
    get_chain_snapshot,
    get_option_greeks,
    ThetaTerminalOfflineError,
)
from options_replay.chain_analyzer import (
    filter_chain,
    contract_key,
    contract_label,
)
from options_replay.contract_picker import (
    pick_contracts,
    load_batch_stats,
    PickerResult,
    ContractPick,
    QualityFilters,
)

logger = logging.getLogger(__name__)

# Historical delta-bucket stats from batch analysis (4,260 news contracts)
# Fallback if batch CSV not available
DELTA_HIST = {
    "0.10-0.20": {"win_rate": 0.87, "avg_return": 3.68},
    "0.20-0.30": {"win_rate": 0.90, "avg_return": 1.67},
    "0.30-0.40": {"win_rate": 0.86, "avg_return": 0.86},
    "0.40-0.50": {"win_rate": 0.90, "avg_return": 0.67},
}


def _delta_bucket(delta: float) -> str:
    """Map absolute delta to a bucket string."""
    d = abs(delta)
    if d < 0.10:
        return "0.00-0.10"
    elif d < 0.20:
        return "0.10-0.20"
    elif d < 0.30:
        return "0.20-0.30"
    elif d < 0.40:
        return "0.30-0.40"
    elif d < 0.50:
        return "0.40-0.50"
    else:
        return "0.50+"


def score_live(
    filtered_df: pd.DataFrame,
    greeks_dict: dict,
    risk_budget: float = 500.0,
) -> pd.DataFrame:
    """Score filtered options using snapshot-available data only.

    Unlike score_options() which uses realized returns, this scores on:
        30% spread tightness
        25% delta sweet spot (closeness to 0.25-0.35)
        20% volume + OI
        15% historical win rate by delta bucket
        10% premium affordability (can afford 2+ contracts)

    Returns DataFrame with composite_score and columns expected by pick_contracts().
    """
    if filtered_df.empty:
        return filtered_df

    df = filtered_df.copy()

    # Only score primary direction
    if "is_primary" in df.columns:
        df = df[df["is_primary"]].copy()
    if df.empty:
        return df

    # Attach greeks from snapshot
    snapshot_time = None  # We use the first available timestamp
    for idx, row in df.iterrows():
        key = contract_key(row)
        gdf = greeks_dict.get(key, pd.DataFrame())
        if gdf.empty:
            df.loc[idx, "delta"] = np.nan
            df.loc[idx, "theta"] = np.nan
            df.loc[idx, "vega"] = np.nan
            df.loc[idx, "implied_vol"] = np.nan
            continue
        # Use the last available row (most recent data point)
        last = gdf.iloc[-1]
        df.loc[idx, "delta"] = last.get("delta", np.nan)
        df.loc[idx, "theta"] = last.get("theta", np.nan)
        df.loc[idx, "vega"] = last.get("vega", np.nan)
        df.loc[idx, "implied_vol"] = last.get("implied_vol", np.nan)

    def _pctrank(series):
        if series.nunique() <= 1:
            return pd.Series(0.5, index=series.index)
        return series.rank(pct=True, method="average")

    # --- Spread tightness (30%) ---
    spread = df["spread_pct"].fillna(1.0) if "spread_pct" in df.columns else pd.Series(0.5, index=df.index)
    df["spread_score"] = _pctrank(1 - spread) * 30

    # --- Delta sweet spot (25%) ---
    # Optimal range is 0.20-0.35 based on batch analysis
    abs_delta = df["delta"].abs().fillna(0.3)
    # Distance from ideal center (0.275)
    delta_dist = (abs_delta - 0.275).abs()
    df["delta_score"] = _pctrank(1 - delta_dist) * 25

    # --- Volume + OI (20%) ---
    vol = df["volume"].fillna(0) if "volume" in df.columns else pd.Series(0, index=df.index)
    oi = df["open_interest"].fillna(0) if "open_interest" in df.columns else pd.Series(0, index=df.index)
    liquidity = vol + oi * 0.5  # OI is less actionable than volume
    df["liquidity_score"] = _pctrank(liquidity) * 20

    # --- Historical win rate by delta bucket (15%) ---
    def _hist_wr(delta_val):
        if pd.isna(delta_val):
            return 0.5
        bucket = _delta_bucket(delta_val)
        stats = DELTA_HIST.get(bucket, {})
        return stats.get("win_rate", 0.5)

    hist_wr = df["delta"].apply(_hist_wr)
    df["hist_score"] = _pctrank(hist_wr) * 15

    # --- Premium affordability (10%) ---
    ask = df["ask"].fillna(df["mid"]) if "ask" in df.columns else df["mid"]
    cost_per = ask * 100
    # How many contracts can we afford? More = better
    n_contracts = (risk_budget / cost_per.replace(0, np.inf)).clip(upper=20)
    df["afford_score"] = _pctrank(n_contracts) * 10

    # --- Composite ---
    df["composite_score"] = (
        df["spread_score"] + df["delta_score"] +
        df["liquidity_score"] + df["hist_score"] + df["afford_score"]
    )

    # Populate columns that pick_contracts() expects
    df["entry_ask"] = df["ask"].fillna(df["mid"]) if "ask" in df.columns else df["mid"]
    df["entry_mid"] = df["mid"]
    df["max_bid"] = df["bid"].fillna(0)
    df["avg_spread_pct_window"] = df["spread_pct"].fillna(0.5) if "spread_pct" in df.columns else 0.5
    df["volume_during_window"] = df["volume"].fillna(0) if "volume" in df.columns else 0

    # Set return fields using historical averages by delta bucket
    def _hist_return(delta_val):
        if pd.isna(delta_val):
            return 0.0
        bucket = _delta_bucket(delta_val)
        stats = DELTA_HIST.get(bucket, {})
        return stats.get("avg_return", 0.0)

    df["realistic_return_pct"] = df["delta"].apply(_hist_return)
    df["raw_return_pct"] = df["realistic_return_pct"]
    df["spread_cost_pct"] = 0.0

    # Score columns expected by pick_contracts
    df["return_score"] = _pctrank(df["realistic_return_pct"]) * 40
    df["oi_score"] = _pctrank(df.get("open_interest", pd.Series(0, index=df.index)).fillna(0)) * 15

    # Sort and rank
    df = df.sort_values("composite_score", ascending=False).head(10)
    df["rank"] = range(1, len(df) + 1)

    return df.reset_index(drop=True)


def pick_live(
    ticker: str,
    date: str,
    time_of_day: str,
    side: str,
    underlying_price: float,
    risk_budget: float = 500.0,
) -> dict:
    """Run the full live options picking pipeline.

    Args:
        ticker: stock symbol (e.g. "NVDA")
        date: trade date (YYYY-MM-DD)
        time_of_day: entry time (HH:MM:SS)
        side: "long" or "short" (or "buy"/"sell" or 1/-1)
        underlying_price: stock price at entry
        risk_budget: max dollar premium outlay

    Returns:
        dict with keys: success, pick, conservative, aggressive, meta
        Each pick is a dict with contract details and sizing.
    """
    # Normalize side
    if isinstance(side, str):
        side_int = 1 if side.lower() in ("long", "buy", "1") else -1
    else:
        side_int = int(side)

    # Check Theta Terminal
    if not check_terminal_running():
        return {"success": False, "error": "Theta Terminal offline"}

    # Step 1: Chain snapshot
    try:
        chain_df = get_chain_snapshot(ticker, date, time_of_day)
    except ThetaTerminalOfflineError:
        return {"success": False, "error": "Theta Terminal offline"}
    except Exception as e:
        return {"success": False, "error": f"Chain snapshot failed: {e}"}

    if chain_df.empty:
        return {"success": False, "error": f"No options chain for {ticker} on {date}"}

    # Step 2: Filter
    filtered = filter_chain(chain_df, underlying_price, side_int, date)
    if filtered.empty:
        return {"success": False, "error": f"No contracts pass filters for {ticker}"}

    # Step 3: Fetch greeks (parallel)
    greeks_dict = {}

    def _fetch_greeks(row):
        key = contract_key(row)
        gdf = get_option_greeks(
            ticker, row["expiration"], row["strike"], row["right"], date
        )
        return key, gdf

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_fetch_greeks, row): idx for idx, row in filtered.iterrows()}
        for future in as_completed(futures):
            try:
                key, gdf = future.result()
                greeks_dict[key] = gdf
            except Exception:
                pass

    # Step 4: Score
    scored = score_live(filtered, greeks_dict, risk_budget)
    if scored.empty:
        return {"success": False, "error": f"No scoreable contracts for {ticker}"}

    # Step 5: Pick contracts
    batch_stats = load_batch_stats()
    # Use relaxed filters for live (we have fewer candidates)
    live_filters = QualityFilters(
        min_entry_mid=0.10,
        max_spread_pct=0.25,
        min_delta=0.10,
        max_delta=0.50,
        min_composite_score=30.0,
        max_dte=7,
    )
    result = pick_contracts(scored, risk_budget=risk_budget,
                            filters=live_filters, batch_stats=batch_stats)

    # Serialize to dict
    return _serialize_result(result, ticker)


def _pick_to_dict(pick: Optional[ContractPick]) -> Optional[dict]:
    """Convert a ContractPick to a JSON-serializable dict."""
    if pick is None:
        return None
    return {
        "label": pick.label,
        "strike": pick.strike,
        "right": pick.right,
        "expiration": pick.expiration,
        "dte": pick.dte,
        "entry_ask": pick.entry_ask,
        "entry_mid": pick.entry_mid,
        "delta": pick.delta,
        "implied_vol": pick.implied_vol,
        "spread_pct": pick.avg_spread_pct,
        "contracts": pick.contracts,
        "total_risk": pick.total_risk,
        "composite_score": pick.composite_score,
        "moneyness": pick.moneyness_label,
        "rationale": pick.rationale,
        "hist_win_rate": pick.hist_win_rate,
        "hist_avg_return": pick.hist_avg_return,
    }


def _serialize_result(result: PickerResult, ticker: str) -> dict:
    """Convert PickerResult to a JSON-serializable dict."""
    return {
        "success": result.top_pick is not None,
        "ticker": ticker,
        "pick": _pick_to_dict(result.top_pick),
        "conservative": _pick_to_dict(result.conservative_pick),
        "aggressive": _pick_to_dict(result.aggressive_pick),
        "meta": {
            "risk_budget": result.risk_budget,
            "quality_gate_passed": result.quality_gate_passed,
            "total_candidates": result.total_candidates,
            "risk_guidance": result.risk_guidance,
        },
    }


# ── CLI entry point ─────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    if len(sys.argv) < 6:
        print(json.dumps({
            "success": False,
            "error": f"Usage: python -m options_replay.live_picker TICKER DATE TIME SIDE PRICE [RISK_BUDGET]"
        }))
        sys.exit(1)

    ticker = sys.argv[1].upper()
    date = sys.argv[2]
    time_of_day = sys.argv[3]
    side = sys.argv[4]
    price = float(sys.argv[5])
    budget = float(sys.argv[6]) if len(sys.argv) > 6 else 500.0

    result = pick_live(ticker, date, time_of_day, side, price, budget)
    print(json.dumps(result, indent=2, default=str))
