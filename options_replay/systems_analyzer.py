"""
Systems analyzer — compute hotkey optimization stats from batch results.

Analyzes price × moneyness bands to find optimal options hotkey parameters.
"""

import logging
from itertools import product

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

PRICE_BINS = [0, 0.50, 1.00, 2.00, 3.00, 5.00, float("inf")]
PRICE_LABELS = ["$0-0.50", "$0.50-1", "$1-2", "$2-3", "$3-5", "$5+"]

OTM_BINS = [0, 1, 2, 3, 5, float("inf")]
OTM_LABELS = ["0-1%", "1-2%", "2-3%", "3-5%", "5%+"]

MAX_PRICES = [0.50, 1.00, 1.50, 2.00, 3.00, 5.00]
MAX_OTM_PCTS = [1, 2, 3, 5, 10]


def prepare_systems_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and enrich batch results for systems analysis.

    Filters to hold_window=30, is_primary=True, OTM/ATM only.
    Adds pct_otm, price_band, otm_band columns.
    """
    out = df.copy()

    # Filter to best hold window and primary direction
    if "hold_window" in out.columns:
        out = out[out["hold_window"] == 30].copy()
    if "is_primary" in out.columns:
        out = out[out["is_primary"] == True].copy()

    if out.empty:
        return out

    # Compute % OTM
    # For calls: positive moneyness = OTM. For puts: negative moneyness = OTM (flip)
    out["pct_otm"] = out.apply(
        lambda r: r["moneyness"] * 100 if r["right"] == "call" else -r["moneyness"] * 100,
        axis=1,
    )

    # Keep ATM or OTM only
    out = out[out["pct_otm"] >= 0].copy()

    if out.empty:
        return out

    # Add bands
    out["price_band"] = pd.cut(
        out["entry_ask"], bins=PRICE_BINS, labels=PRICE_LABELS, right=False
    )
    out["otm_band"] = pd.cut(
        out["pct_otm"], bins=OTM_BINS, labels=OTM_LABELS, right=False
    )
    out["is_win"] = out["realistic_return_pct"] > 0

    logger.info("Systems data prepared: %d OTM/ATM contracts", len(out))
    return out


def compute_hotkey_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Compute stats for every (max_price, max_otm_pct) hotkey combo.

    Returns DataFrame with columns:
        max_price, max_otm, count, avg_return, win_rate,
        avg_spread_cost, edge, avg_delta, median_entry, avg_pct_otm
    """
    if df.empty:
        return pd.DataFrame()

    results = []
    for max_price, max_otm in product(MAX_PRICES, MAX_OTM_PCTS):
        mask = (df["entry_ask"] < max_price) & (df["pct_otm"] < max_otm)
        subset = df[mask]
        n = len(subset)

        if n == 0:
            results.append({
                "max_price": max_price, "max_otm": max_otm,
                "count": 0, "avg_return": 0, "win_rate": 0,
                "avg_spread_cost": 0, "edge": 0, "avg_delta": 0,
                "median_entry": 0, "avg_pct_otm": 0,
            })
        else:
            avg_ret = float(subset["realistic_return_pct"].mean())
            wr = float(subset["is_win"].mean())
            results.append({
                "max_price": max_price,
                "max_otm": max_otm,
                "count": n,
                "avg_return": avg_ret,
                "win_rate": wr,
                "avg_spread_cost": float(subset["spread_cost_pct"].mean()),
                "edge": wr * avg_ret,
                "avg_delta": float(subset["delta"].abs().mean()) if "delta" in subset.columns else 0,
                "median_entry": float(subset["entry_ask"].median()),
                "avg_pct_otm": float(subset["pct_otm"].mean()),
            })

    return pd.DataFrame(results)


def compute_price_otm_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab of price_band × otm_band with per-cell stats.

    Returns DataFrame with columns:
        price_band, otm_band, count, avg_return, win_rate,
        avg_spread_cost, median_entry, avg_delta
    """
    if df.empty or "price_band" not in df.columns:
        return pd.DataFrame()

    grid = (
        df.groupby(["price_band", "otm_band"], observed=False)
        .agg(
            count=("realistic_return_pct", "size"),
            avg_return=("realistic_return_pct", "mean"),
            win_rate=("is_win", "mean"),
            avg_spread_cost=("spread_cost_pct", "mean"),
            median_entry=("entry_ask", "median"),
            avg_delta=("delta", lambda x: float(x.abs().mean()) if len(x) > 0 else 0),
        )
        .reset_index()
    )
    return grid


def recommend_hotkeys(grid_df: pd.DataFrame, min_count: int = 10) -> list:
    """Find top 3 recommended hotkey configurations.

    Returns list of dicts, each with:
        max_price, max_otm, count, avg_return, win_rate, edge,
        avg_spread_cost, avg_delta, median_entry, label, rationale
    """
    if grid_df.empty:
        return []

    valid = grid_df[grid_df["count"] >= min_count].copy()
    if valid.empty:
        # Fall back to lower threshold
        valid = grid_df[grid_df["count"] >= 5].copy()
    if valid.empty:
        return []

    valid = valid[valid["edge"] > 0].copy()
    if valid.empty:
        return []

    recs = []
    seen = set()

    def _add(row, label, rationale):
        key = (row["max_price"], row["max_otm"])
        if key not in seen:
            seen.add(key)
            rec = row.to_dict()
            rec["label"] = label
            rec["rationale"] = rationale
            recs.append(rec)

    # 1. Best edge
    best_edge = valid.sort_values("edge", ascending=False).iloc[0]
    _add(best_edge, "AGGRESSIVE",
         f"Highest edge ({best_edge['edge']:.1f}). Cheap OTM contracts with max leverage.")

    # 2. Best win rate
    best_wr = valid.sort_values("win_rate", ascending=False).iloc[0]
    _add(best_wr, "CONSERVATIVE",
         f"Highest win rate ({best_wr['win_rate']:.0%}). Reliable fills, lower variance.")

    # 3. Best balanced (edge × count)
    pe = valid.copy()
    edge_range = pe["edge"].max() - pe["edge"].min()
    count_range = pe["count"].max() - pe["count"].min()
    pe["norm_edge"] = (pe["edge"] - pe["edge"].min()) / (edge_range + 1e-9)
    pe["norm_count"] = (pe["count"] - pe["count"].min()) / (count_range + 1e-9)
    pe["balanced"] = pe["norm_edge"] * 0.6 + pe["norm_count"] * 0.4
    best_bal = pe.sort_values("balanced", ascending=False).iloc[0]
    _add(best_bal, "BALANCED",
         f"Best blend of edge ({best_bal['edge']:.1f}) and opportunity count ({int(best_bal['count'])}).")

    return recs


def compute_delta_profile(df: pd.DataFrame, max_price: float, max_otm: float) -> dict:
    """Delta and return distribution for a specific hotkey filter."""
    mask = (df["entry_ask"] < max_price) & (df["pct_otm"] < max_otm)
    subset = df[mask]

    if subset.empty:
        return {"count": 0}

    deltas = subset["delta"].dropna().abs()
    rets = subset["realistic_return_pct"]

    return {
        "count": len(subset),
        "delta_min": float(deltas.min()) if len(deltas) > 0 else 0,
        "delta_max": float(deltas.max()) if len(deltas) > 0 else 0,
        "delta_mean": float(deltas.mean()) if len(deltas) > 0 else 0,
        "delta_median": float(deltas.median()) if len(deltas) > 0 else 0,
        "delta_q25": float(deltas.quantile(0.25)) if len(deltas) > 0 else 0,
        "delta_q75": float(deltas.quantile(0.75)) if len(deltas) > 0 else 0,
        "return_min": float(rets.min()),
        "return_max": float(rets.max()),
        "return_median": float(rets.median()),
        "return_q25": float(rets.quantile(0.25)),
        "return_q75": float(rets.quantile(0.75)),
    }
