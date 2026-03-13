"""
Aggregate batch results into statistical summaries by category.

Buckets: delta range, DTE range, moneyness (5-level).
Computes per-category mean/median/std of realistic returns, win rates, spread costs.
"""

import pandas as pd
import numpy as np


DELTA_BUCKETS = [
    (0, 0.20, "0.00-0.20"),
    (0.20, 0.40, "0.20-0.40"),
    (0.40, 0.60, "0.40-0.60"),
    (0.60, 0.80, "0.60-0.80"),
    (0.80, 1.01, "0.80-1.00"),
]

DTE_BUCKETS = [
    (0, 1, "0-1d"),
    (2, 3, "2-3d"),
    (4, 7, "4-7d"),
    (8, 14, "8-14d"),
]

MONEYNESS_ORDER = ["Deep ITM", "ITM", "ATM", "OTM", "Deep OTM"]


def _moneyness_label_5(moneyness: float, right: str) -> str:
    """5-level moneyness classification."""
    # For calls: negative moneyness = ITM, positive = OTM
    # For puts: flip sign
    m = moneyness if right == "call" else -moneyness
    if m < -0.04:
        return "Deep ITM"
    elif m < -0.01:
        return "ITM"
    elif m <= 0.01:
        return "ATM"
    elif m <= 0.04:
        return "OTM"
    else:
        return "Deep OTM"


def bucket_delta(delta: float) -> str:
    """Assign an absolute delta value to its bucket label."""
    d = abs(delta) if pd.notna(delta) else 0
    for lo, hi, label in DELTA_BUCKETS:
        if lo <= d < hi:
            return label
    return "0.80-1.00"


def bucket_dte(dte) -> str:
    """Assign DTE to its bucket label."""
    d = int(dte) if pd.notna(dte) else 0
    for lo, hi, label in DTE_BUCKETS:
        if lo <= d <= hi:
            return label
    return "8-14d"


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Add delta_bucket, dte_bucket, and 5-level moneyness columns."""
    out = df.copy()

    if "delta" in out.columns:
        out["delta_bucket"] = out["delta"].apply(bucket_delta)
    else:
        out["delta_bucket"] = "unknown"

    if "dte" in out.columns:
        out["dte_bucket"] = out["dte"].apply(bucket_dte)
    else:
        out["dte_bucket"] = "unknown"

    if "moneyness" in out.columns and "right" in out.columns:
        out["moneyness_5"] = out.apply(
            lambda r: _moneyness_label_5(r["moneyness"], r["right"]), axis=1
        )
    elif "moneyness_label" in out.columns:
        out["moneyness_5"] = out["moneyness_label"]
    else:
        out["moneyness_5"] = "ATM"

    return out


def compute_category_stats(
    df: pd.DataFrame,
    group_col: str,
    hold_window: int = None,
) -> pd.DataFrame:
    """Group by a category and compute aggregate stats.

    Returns one row per category with:
        count, mean, median, std of realistic_return_pct,
        win_rate, avg spread_cost, avg volume, avg IV.
    """
    d = df.copy()
    if hold_window is not None:
        d = d[d["hold_window"] == hold_window]

    if d.empty or group_col not in d.columns:
        return pd.DataFrame()

    return_col = "realistic_return_pct" if "realistic_return_pct" in d.columns else "raw_return_pct"

    agg = d.groupby(group_col).agg(
        count=(return_col, "size"),
        mean_return=(return_col, "mean"),
        median_return=(return_col, "median"),
        std_return=(return_col, "std"),
        win_rate=(return_col, lambda x: (x > 0).mean()),
        avg_spread_cost=("spread_cost_pct", "mean") if "spread_cost_pct" in d.columns else (return_col, lambda x: 0),
        avg_volume=("volume_during_window", "mean") if "volume_during_window" in d.columns else (return_col, lambda x: 0),
        avg_iv=("implied_vol", "mean") if "implied_vol" in d.columns else (return_col, lambda x: None),
    ).reset_index()

    agg["std_return"] = agg["std_return"].fillna(0)
    return agg


def compute_cross_stats(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    hold_window: int = None,
) -> tuple:
    """Pivot table for heatmap.

    Returns (values_pivot, count_pivot) where:
        values = mean realistic_return_pct
        count = sample count
    """
    d = df.copy()
    if hold_window is not None:
        d = d[d["hold_window"] == hold_window]

    return_col = "realistic_return_pct" if "realistic_return_pct" in d.columns else "raw_return_pct"

    values = d.pivot_table(
        values=return_col, index=row_col, columns=col_col, aggfunc="mean"
    )
    counts = d.pivot_table(
        values=return_col, index=row_col, columns=col_col, aggfunc="count"
    )

    return values, counts


def compute_hold_window_stats(df: pd.DataFrame, group_col: str = "moneyness_5") -> pd.DataFrame:
    """For each hold_window × category, compute mean realistic return.

    Returns DataFrame with columns: hold_window, category, mean_return, count.
    """
    return_col = "realistic_return_pct" if "realistic_return_pct" in df.columns else "raw_return_pct"

    agg = df.groupby(["hold_window", group_col]).agg(
        mean_return=(return_col, "mean"),
        count=(return_col, "size"),
    ).reset_index()

    agg = agg.rename(columns={group_col: "category"})
    return agg


def compute_summary(df: pd.DataFrame) -> dict:
    """Overall summary statistics across all batch results."""
    return_col = "realistic_return_pct" if "realistic_return_pct" in df.columns else "raw_return_pct"

    n_trades = df["trade_idx"].nunique() if "trade_idx" in df.columns else 0
    n_contracts = len(df)

    returns = df[return_col].dropna()
    win_rate = (returns > 0).mean() if len(returns) > 0 else 0
    avg_return = returns.mean() if len(returns) > 0 else 0
    median_return = returns.median() if len(returns) > 0 else 0

    avg_spread_cost = df["spread_cost_pct"].mean() if "spread_cost_pct" in df.columns else 0

    # Best/worst category (by delta bucket at 30-min window)
    best_cat = "N/A"
    worst_cat = "N/A"
    if "delta_bucket" in df.columns:
        d30 = df[df["hold_window"] == 30] if "hold_window" in df.columns else df
        if not d30.empty:
            cat_means = d30.groupby("delta_bucket")[return_col].mean()
            if not cat_means.empty:
                best_cat = cat_means.idxmax()
                worst_cat = cat_means.idxmin()

    return {
        "total_trades": n_trades,
        "total_contracts": n_contracts,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "median_return": median_return,
        "avg_spread_cost": avg_spread_cost,
        "best_delta_bucket": best_cat,
        "worst_delta_bucket": worst_cat,
    }
