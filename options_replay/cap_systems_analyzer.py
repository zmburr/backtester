"""
Capitulation systems analyzer — compute analytics from bounce/reversal batch results.

Focuses on ATR-target hit rates, entry offset comparison, and cap-specific
hotkey optimization. Parallel to systems_analyzer.py but for capitulation trades.
"""

import logging
from itertools import product

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Reuse price/OTM bands from news systems
PRICE_BINS = [0, 0.50, 1.00, 2.00, 3.00, 5.00, float("inf")]
PRICE_LABELS = ["$0-0.50", "$0.50-1", "$1-2", "$2-3", "$3-5", "$5+"]
MAX_PRICES = [0.50, 1.00, 1.50, 2.00, 3.00, 5.00]
MAX_OTM_PCTS = [1, 2, 3, 5, 10]

DELTA_BINS = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 1.0]
DELTA_LABELS = ["0-0.10", "0.10-0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50", "0.50+"]


def prepare_cap_data(
    df: pd.DataFrame,
    trade_type: str = None,
    entry_offset: str = None,
    hold_window: int = None,
) -> pd.DataFrame:
    """Filter and enrich cap batch results for analysis.

    Args:
        df: raw batch results from cap_batch_analyzer
        trade_type: "bounce", "reversal", or None for both
        entry_offset: e.g. "low+0", "high+0", or None for all
        hold_window: specific hold window to filter, or None for longest
    """
    out = df.copy()

    if trade_type and "source" in out.columns:
        out = out[out["source"] == trade_type].copy()

    if entry_offset and "entry_offset" in out.columns:
        out = out[out["entry_offset"] == entry_offset].copy()

    # Default to longest hold window if not specified
    if hold_window and "hold_window" in out.columns:
        out = out[out["hold_window"] == hold_window].copy()
    elif "hold_window" in out.columns:
        max_hw = out["hold_window"].max()
        out = out[out["hold_window"] == max_hw].copy()

    if "is_primary" in out.columns:
        out = out[out["is_primary"] == True].copy()

    if out.empty:
        return out

    # Compute % OTM
    if "moneyness" in out.columns and "right" in out.columns:
        out["pct_otm"] = out.apply(
            lambda r: r["moneyness"] * 100 if r["right"] == "call" else -r["moneyness"] * 100,
            axis=1,
        )
        out = out[out["pct_otm"] >= 0].copy()

    if out.empty:
        return out

    # Add bands
    if "entry_ask" in out.columns:
        out["price_band"] = pd.cut(
            out["entry_ask"], bins=PRICE_BINS, labels=PRICE_LABELS, right=False
        )

    if "delta" in out.columns:
        out["delta_bucket"] = pd.cut(
            out["delta"].abs(), bins=DELTA_BINS, labels=DELTA_LABELS, right=False
        )

    out["is_win"] = out["realistic_return_pct"] > 0

    logger.info("Cap systems data: %d contracts (%s, offset=%s)",
                len(out), trade_type or "all", entry_offset or "all")
    return out


def compute_target_hit_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute target hit rates by delta bucket.

    Returns DataFrame: delta_bucket, target_name, hit_rate, avg_return_at_target,
                        avg_time_to_target, count
    """
    if df.empty:
        return pd.DataFrame()

    target_cols = [c for c in df.columns if c.endswith("_hit") and c.startswith("target_")]
    if not target_cols:
        return pd.DataFrame()

    results = []
    for hit_col in target_cols:
        target_name = hit_col.replace("target_", "").replace("_hit", "")
        time_col = f"target_{target_name}_time_min"
        ret_col = f"target_{target_name}_option_return"

        if "delta_bucket" in df.columns:
            for bucket in DELTA_LABELS:
                sub = df[df["delta_bucket"] == bucket]
                if sub.empty:
                    continue
                hit_rate = sub[hit_col].mean() if hit_col in sub.columns else 0
                avg_time = sub[time_col].dropna().mean() if time_col in sub.columns else np.nan
                avg_ret = sub[ret_col].dropna().mean() if ret_col in sub.columns else np.nan

                results.append({
                    "delta_bucket": bucket,
                    "target": target_name,
                    "hit_rate": hit_rate,
                    "avg_return_at_target": avg_ret,
                    "avg_time_to_target_min": avg_time,
                    "count": len(sub),
                })
        else:
            # No delta buckets, compute overall
            hit_rate = df[hit_col].mean()
            avg_time = df[time_col].dropna().mean() if time_col in df.columns else np.nan
            avg_ret = df[ret_col].dropna().mean() if ret_col in df.columns else np.nan
            results.append({
                "delta_bucket": "all",
                "target": target_name,
                "hit_rate": hit_rate,
                "avg_return_at_target": avg_ret,
                "avg_time_to_target_min": avg_time,
                "count": len(df),
            })

    return pd.DataFrame(results)


def compute_entry_offset_comparison(df_all: pd.DataFrame) -> pd.DataFrame:
    """Compare metrics across entry offsets.

    Takes the FULL batch results (all offsets) and groups by entry_offset.

    Returns DataFrame: entry_offset, count, avg_return, win_rate, edge,
                        target_0.5x_hit_rate, target_1.0x_hit_rate
    """
    if df_all.empty or "entry_offset" not in df_all.columns:
        return pd.DataFrame()

    # Filter to primary direction only
    if "is_primary" in df_all.columns:
        df_all = df_all[df_all["is_primary"] == True].copy()

    results = []
    for offset, group in df_all.groupby("entry_offset"):
        row = {
            "entry_offset": offset,
            "count": len(group),
            "avg_return": group["realistic_return_pct"].mean(),
            "win_rate": (group["realistic_return_pct"] > 0).mean(),
        }
        row["edge"] = row["win_rate"] * row["avg_return"]

        # Target hit rates
        for target in ["0.5x", "1.0x", "1.5x"]:
            hit_col = f"target_{target}_hit"
            ret_col = f"target_{target}_option_return"
            if hit_col in group.columns:
                row[f"target_{target}_hit_rate"] = group[hit_col].mean()
            if ret_col in group.columns:
                row[f"target_{target}_avg_return"] = group[ret_col].dropna().mean()

        results.append(row)

    return pd.DataFrame(results).sort_values("edge", ascending=False).reset_index(drop=True)


def compute_cap_hotkey_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Compute hotkey grid for capitulation trades (same structure as news).

    Returns DataFrame with: max_price, max_otm, count, avg_return, win_rate,
                             edge, avg_delta, target hit rates
    """
    if df.empty or "pct_otm" not in df.columns:
        return pd.DataFrame()

    results = []
    for max_price, max_otm in product(MAX_PRICES, MAX_OTM_PCTS):
        mask = (df["entry_ask"] < max_price) & (df["pct_otm"] < max_otm)
        sub = df[mask]
        n = len(sub)

        row = {
            "max_price": max_price,
            "max_otm": max_otm,
            "count": n,
        }

        if n == 0:
            row.update({"avg_return": 0, "win_rate": 0, "edge": 0,
                         "avg_delta": 0, "avg_spread_cost": 0})
        else:
            avg_ret = float(sub["realistic_return_pct"].mean())
            wr = float(sub["is_win"].mean())
            row.update({
                "avg_return": avg_ret,
                "win_rate": wr,
                "edge": wr * avg_ret,
                "avg_delta": float(sub["delta"].abs().mean()) if "delta" in sub.columns else 0,
                "avg_spread_cost": float(sub["spread_cost_pct"].mean()) if "spread_cost_pct" in sub.columns else 0,
            })

            # Add target hit rates
            for target in ["0.5x", "1.0x", "1.5x"]:
                hit_col = f"target_{target}_hit"
                if hit_col in sub.columns:
                    row[f"target_{target}_hit_rate"] = float(sub[hit_col].mean())

        results.append(row)

    return pd.DataFrame(results)


def recommend_cap_hotkeys(grid_df: pd.DataFrame, min_count: int = 5) -> list:
    """Find top 3 recommended hotkey configurations for cap trades."""
    if grid_df.empty:
        return []

    valid = grid_df[grid_df["count"] >= min_count].copy()
    if valid.empty:
        valid = grid_df[grid_df["count"] >= 2].copy()
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
         f"Highest edge ({best_edge['edge']:.2f}). Max leverage on capitulation bounce/fade.")

    # 2. Best target hit rate (0.5x ATR)
    if "target_0.5x_hit_rate" in valid.columns:
        target_valid = valid[valid["target_0.5x_hit_rate"].notna()].copy()
        if not target_valid.empty:
            best_target = target_valid.sort_values("target_0.5x_hit_rate", ascending=False).iloc[0]
            hr = best_target.get("target_0.5x_hit_rate", 0)
            _add(best_target, "TARGET-OPTIMIZED",
                 f"Highest 0.5x ATR hit rate ({hr:.0%}). Contracts that reach first target most reliably.")

    # 3. Best balanced
    pe = valid.copy()
    edge_range = pe["edge"].max() - pe["edge"].min()
    count_range = pe["count"].max() - pe["count"].min()
    pe["norm_edge"] = (pe["edge"] - pe["edge"].min()) / (edge_range + 1e-9)
    pe["norm_count"] = (pe["count"] - pe["count"].min()) / (count_range + 1e-9)
    pe["balanced"] = pe["norm_edge"] * 0.6 + pe["norm_count"] * 0.4
    best_bal = pe.sort_values("balanced", ascending=False).iloc[0]
    _add(best_bal, "BALANCED",
         f"Best blend of edge ({best_bal['edge']:.2f}) and opportunity ({int(best_bal['count'])} contracts).")

    return recs


def compute_iv_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze IV at entry by delta bucket — key for capitulation trades.

    Returns DataFrame: delta_bucket, avg_iv, avg_return, avg_spread_cost, count
    """
    if df.empty or "implied_vol" not in df.columns or "delta_bucket" not in df.columns:
        return pd.DataFrame()

    results = []
    for bucket in DELTA_LABELS:
        sub = df[df["delta_bucket"] == bucket]
        if sub.empty:
            continue
        iv = sub["implied_vol"].dropna()
        results.append({
            "delta_bucket": bucket,
            "avg_iv": float(iv.mean()) if len(iv) > 0 else np.nan,
            "median_iv": float(iv.median()) if len(iv) > 0 else np.nan,
            "avg_return": float(sub["realistic_return_pct"].mean()),
            "avg_spread_cost": float(sub["spread_cost_pct"].mean()) if "spread_cost_pct" in sub.columns else 0,
            "count": len(sub),
        })

    return pd.DataFrame(results)


def compute_cap_summary(df: pd.DataFrame) -> dict:
    """Compute high-level summary stats for the cap batch."""
    if df.empty:
        return {}

    summary = {
        "total_contracts": len(df),
        "unique_trades": df["trade_date"].nunique() if "trade_date" in df.columns else 0,
        "unique_symbols": df["symbol"].nunique() if "symbol" in df.columns else 0,
        "avg_return": float(df["realistic_return_pct"].mean()),
        "median_return": float(df["realistic_return_pct"].median()),
        "win_rate": float((df["realistic_return_pct"] > 0).mean()),
        "avg_spread_cost": float(df["spread_cost_pct"].mean()) if "spread_cost_pct" in df.columns else 0,
    }

    # Source breakdown
    if "source" in df.columns:
        for source in ["bounce", "reversal"]:
            sub = df[df["source"] == source]
            if not sub.empty:
                summary[f"{source}_count"] = len(sub)
                summary[f"{source}_avg_return"] = float(sub["realistic_return_pct"].mean())
                summary[f"{source}_win_rate"] = float((sub["realistic_return_pct"] > 0).mean())

    # Target hit rates overall
    for target in ["0.5x", "1.0x", "1.5x"]:
        hit_col = f"target_{target}_hit"
        if hit_col in df.columns:
            summary[f"target_{target}_hit_rate"] = float(df[hit_col].mean())

    return summary
