"""
Contract Picker — recommendation engine for headline options trades.

Takes scored options from score_options() and produces 1-3 actionable picks
with position sizing and historical context from batch analysis.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

from options_replay.chain_analyzer import contract_label

logger = logging.getLogger(__name__)


@dataclass
class ContractPick:
    pick_type: str              # "top", "conservative", "aggressive"
    label: str                  # "Mar-13 $120 Call"
    strike: float
    right: str
    expiration: str
    dte: int
    entry_ask: float
    entry_mid: float
    max_bid: float
    realistic_return_pct: float
    raw_return_pct: float
    spread_cost_pct: float
    composite_score: float
    delta: Optional[float]
    implied_vol: Optional[float]
    avg_spread_pct: float
    volume: int
    moneyness_label: str
    contracts: int
    total_risk: float
    hist_win_rate: Optional[float] = None
    hist_avg_return: Optional[float] = None
    hist_sample_count: Optional[int] = None
    rationale: str = ""


@dataclass
class PickerResult:
    top_pick: Optional[ContractPick]
    conservative_pick: Optional[ContractPick]
    aggressive_pick: Optional[ContractPick]
    risk_budget: float
    quality_gate_passed: int
    total_candidates: int
    risk_guidance: dict = field(default_factory=dict)


@dataclass
class BatchStats:
    delta_stats: pd.DataFrame
    moneyness_stats: pd.DataFrame
    overall_win_rate: float
    total_contracts: int
    loaded_from: str


@dataclass
class QualityFilters:
    min_entry_mid: float = 0.50
    max_spread_pct: float = 0.15
    min_delta: float = 0.15
    max_delta: float = 0.45
    min_composite_score: float = 50.0
    max_dte: int = 3


DEFAULT_FILTERS = QualityFilters()

_cached_batch_stats: Optional[BatchStats] = None


def load_batch_stats(force_reload: bool = False) -> Optional[BatchStats]:
    """Load the most recent batch CSV and compute per-category stats.

    Uses batch_aggregator infrastructure to avoid duplicating bucketing logic.
    Results are cached at module level.
    """
    global _cached_batch_stats
    if _cached_batch_stats is not None and not force_reload:
        return _cached_batch_stats

    try:
        from options_replay.batch_analyzer import load_batch_results
        from options_replay.batch_aggregator import add_buckets, compute_category_stats

        df = load_batch_results()
        if df is None or df.empty:
            return None

        # Use 30-min window (best from batch findings)
        if "hold_window" in df.columns:
            df30 = df[df["hold_window"] == 30]
            if df30.empty:
                df30 = df
        else:
            df30 = df

        df30 = add_buckets(df30)

        delta_stats = compute_category_stats(df30, "delta_bucket")
        moneyness_stats = compute_category_stats(df30, "moneyness_5")

        return_col = "realistic_return_pct" if "realistic_return_pct" in df30.columns else "raw_return_pct"
        returns = df30[return_col].dropna()
        overall_wr = float((returns > 0).mean()) if len(returns) > 0 else 0.0

        _cached_batch_stats = BatchStats(
            delta_stats=delta_stats,
            moneyness_stats=moneyness_stats,
            overall_win_rate=overall_wr,
            total_contracts=len(df30),
            loaded_from="batch_results",
        )
        logger.info("Loaded batch stats: %d contracts, %.0f%% overall WR",
                     len(df30), overall_wr * 100)
        return _cached_batch_stats

    except Exception as e:
        logger.warning("Could not load batch stats: %s", e)
        return None


def _lookup_hist_stats(delta: Optional[float], batch_stats: Optional[BatchStats]) -> dict:
    """Look up historical win rate and avg return for a delta bucket."""
    result = {"hist_win_rate": None, "hist_avg_return": None, "hist_sample_count": None}
    if batch_stats is None or delta is None:
        return result

    try:
        from options_replay.batch_aggregator import bucket_delta
        bucket = bucket_delta(delta)
        ds = batch_stats.delta_stats
        if ds.empty or "delta_bucket" not in ds.columns:
            return result
        match = ds[ds["delta_bucket"] == bucket]
        if not match.empty:
            row = match.iloc[0]
            result["hist_win_rate"] = float(row.get("win_rate", 0))
            result["hist_avg_return"] = float(row.get("mean_return", 0))
            result["hist_sample_count"] = int(row.get("count", 0))
    except Exception:
        pass
    return result


def apply_quality_filters(
    scored_df: pd.DataFrame,
    filters: QualityFilters = DEFAULT_FILTERS,
) -> pd.DataFrame:
    """Apply quality gates to the scored DataFrame.

    Progressive fallback if zero results survive.
    Returns filtered copy sorted by composite_score descending.
    """
    if scored_df.empty:
        return scored_df

    df = scored_df.copy()

    # Build mask
    entry_col = "entry_mid" if "entry_mid" in df.columns else "mid"
    mask = pd.Series(True, index=df.index)

    if entry_col in df.columns:
        mask &= df[entry_col] >= filters.min_entry_mid

    if "avg_spread_pct_window" in df.columns:
        mask &= df["avg_spread_pct_window"] <= filters.max_spread_pct

    if "delta" in df.columns:
        delta_valid = df["delta"].notna()
        delta_range = (df["delta"].abs() >= filters.min_delta) & (df["delta"].abs() <= filters.max_delta)
        mask &= (~delta_valid) | delta_range  # pass if NaN, otherwise must be in range

    if "composite_score" in df.columns:
        mask &= df["composite_score"] >= filters.min_composite_score

    if "dte" in df.columns:
        mask &= df["dte"] <= filters.max_dte

    filtered = df[mask].copy()

    # Progressive fallback
    if filtered.empty:
        # Relax score to 35
        relaxed_mask = mask.copy()
        if "composite_score" in df.columns:
            relaxed_mask = pd.Series(True, index=df.index)
            if entry_col in df.columns:
                relaxed_mask &= df[entry_col] >= filters.min_entry_mid
            if "avg_spread_pct_window" in df.columns:
                relaxed_mask &= df["avg_spread_pct_window"] <= filters.max_spread_pct
            if "composite_score" in df.columns:
                relaxed_mask &= df["composite_score"] >= 35
        filtered = df[relaxed_mask].copy()

    if filtered.empty:
        # Return top 3 unfiltered
        filtered = df.nlargest(3, "composite_score").copy()

    return filtered.sort_values("composite_score", ascending=False).reset_index(drop=True)


def _select_top_pick(filtered_df: pd.DataFrame) -> Optional[pd.Series]:
    """Highest composite score."""
    if filtered_df.empty:
        return None
    return filtered_df.iloc[0]


def _select_conservative_pick(
    filtered_df: pd.DataFrame,
    exclude_idx: Optional[int],
) -> Optional[pd.Series]:
    """Higher delta, tighter spread, closer to ATM."""
    candidates = filtered_df.copy()
    if exclude_idx is not None and exclude_idx in candidates.index:
        candidates = candidates.drop(exclude_idx)
    if candidates.empty:
        return None

    def _pctrank(s):
        if s.nunique() <= 1:
            return pd.Series(0.5, index=s.index)
        return s.rank(pct=True, method="average")

    # Higher |delta| = more conservative (weight 0.4)
    delta_score = _pctrank(candidates["delta"].abs().fillna(0.3)) * 0.4
    # Tighter spread = better (weight 0.3)
    spread_score = _pctrank(1 - candidates["avg_spread_pct_window"].fillna(0.5)) * 0.3
    # Closer to ATM (weight 0.3)
    atm_dist = candidates["moneyness"].abs() if "moneyness" in candidates.columns else pd.Series(0, index=candidates.index)
    atm_score = _pctrank(1 - atm_dist.fillna(0)) * 0.3

    candidates = candidates.copy()
    candidates["_cons_score"] = delta_score + spread_score + atm_score
    best_idx = candidates["_cons_score"].idxmax()
    return candidates.loc[best_idx]


def _select_aggressive_pick(
    filtered_df: pd.DataFrame,
    exclude_idxs: list,
) -> Optional[pd.Series]:
    """Lower delta, more OTM, higher leverage."""
    candidates = filtered_df.copy()
    for idx in exclude_idxs:
        if idx in candidates.index:
            candidates = candidates.drop(idx)
    if candidates.empty:
        return None

    def _pctrank(s):
        if s.nunique() <= 1:
            return pd.Series(0.5, index=s.index)
        return s.rank(pct=True, method="average")

    # Lower |delta| = more aggressive (weight 0.4)
    delta_score = _pctrank(1 - candidates["delta"].abs().fillna(0.3)) * 0.4
    # Higher return (weight 0.4)
    return_col = "realistic_return_pct" if "realistic_return_pct" in candidates.columns else "raw_return_pct"
    return_score = _pctrank(candidates[return_col].fillna(0)) * 0.4
    # Still want quality (weight 0.2)
    score_val = _pctrank(candidates["composite_score"].fillna(50)) * 0.2

    candidates = candidates.copy()
    candidates["_agg_score"] = delta_score + return_score + score_val
    best_idx = candidates["_agg_score"].idxmax()
    return candidates.loc[best_idx]


def compute_position_size(entry_ask: float, risk_budget: float) -> dict:
    """Compute number of contracts and total risk."""
    cost_per_contract = entry_ask * 100
    if cost_per_contract <= 0:
        return {"contracts": 0, "total_risk": 0, "cost_per_contract": 0}

    contracts = max(1, math.floor(risk_budget / cost_per_contract))
    total_risk = contracts * cost_per_contract

    return {
        "contracts": contracts,
        "total_risk": total_risk,
        "cost_per_contract": cost_per_contract,
    }


def compute_risk_guidance(
    risk_budget: float,
    composite_score: float,
    entry_ask: float,
    avg_spread_pct: float = 0,
    batch_stats: Optional[BatchStats] = None,
) -> dict:
    """Generate risk allocation guidance."""
    warnings = []

    # Scaling tier
    if composite_score >= 70:
        tier = "elevated"
        summary = f"Score {composite_score:.0f} — elevated confidence, full size OK."
    elif composite_score >= 50:
        tier = "standard"
        summary = f"Score {composite_score:.0f} — standard size."
    else:
        tier = "reduced"
        summary = f"Score {composite_score:.0f} — below quality gate, consider half size."

    # Warnings
    cost = entry_ask * 100
    if cost > risk_budget:
        warnings.append(f"Single contract costs ${cost:.0f}, exceeds ${risk_budget:.0f} budget.")

    if avg_spread_pct > 0.12:
        warnings.append(f"Wide spread ({avg_spread_pct:.0%}) — use limit order.")
    elif avg_spread_pct > 0.08:
        summary += f" Spread {avg_spread_pct:.0%} — market order OK."
    else:
        summary += f" Spread {avg_spread_pct:.0%} — tight, market order fine."

    if warnings:
        summary = " | ".join([summary] + warnings)

    return {
        "summary": summary,
        "tier": tier,
        "warnings": warnings,
    }


def _build_rationale(pick_type: str, row: pd.Series, hist: dict) -> str:
    """One-liner explaining why this contract was selected."""
    delta_str = f"{abs(row.get('delta', 0)):.2f} delta" if pd.notna(row.get("delta")) else ""
    moneyness = row.get("moneyness_label", "")
    score = row.get("composite_score", 0)
    spread = row.get("avg_spread_pct_window", 0)

    if pick_type == "top":
        parts = [f"Highest composite ({score:.1f})"]
        if delta_str:
            parts.append(f"{delta_str} {moneyness}")
        wr = hist.get("hist_win_rate")
        if wr is not None:
            parts.append(f"{wr:.0%} hist WR in this delta bucket")
        return ", ".join(parts)

    elif pick_type == "conservative":
        parts = ["Higher delta"]
        if spread < 0.08:
            parts.append("tight spread")
        if moneyness in ("ATM", "ITM"):
            parts.append("closer to ATM")
        parts.append("lower leverage")
        return ", ".join(parts)

    else:  # aggressive
        parts = ["More OTM"]
        ret = row.get("realistic_return_pct", 0)
        if ret > 0:
            parts.append(f"{ret:.0%} return potential")
        parts.append("higher leverage but wider spread")
        return ", ".join(parts)


def _row_to_pick(
    pick_type: str,
    row: pd.Series,
    risk_budget: float,
    batch_stats: Optional[BatchStats],
) -> ContractPick:
    """Convert a scored DataFrame row into a ContractPick."""
    entry_ask = float(row.get("entry_ask", row.get("ask", 0)))
    sizing = compute_position_size(entry_ask, risk_budget)
    hist = _lookup_hist_stats(row.get("delta"), batch_stats)
    rationale = _build_rationale(pick_type, row, hist)

    return ContractPick(
        pick_type=pick_type,
        label=contract_label(row),
        strike=float(row.get("strike", 0)),
        right=str(row.get("right", "call")),
        expiration=str(row.get("expiration", "")),
        dte=int(row.get("dte", 0)),
        entry_ask=entry_ask,
        entry_mid=float(row.get("entry_mid", row.get("mid", 0))),
        max_bid=float(row.get("max_bid", 0)),
        realistic_return_pct=float(row.get("realistic_return_pct", 0)),
        raw_return_pct=float(row.get("raw_return_pct", 0)),
        spread_cost_pct=float(row.get("spread_cost_pct", 0)),
        composite_score=float(row.get("composite_score", 0)),
        delta=float(row["delta"]) if pd.notna(row.get("delta")) else None,
        implied_vol=float(row["implied_vol"]) if pd.notna(row.get("implied_vol")) else None,
        avg_spread_pct=float(row.get("avg_spread_pct_window", 0)),
        volume=int(row.get("volume_during_window", 0)),
        moneyness_label=str(row.get("moneyness_label", "ATM")),
        contracts=sizing["contracts"],
        total_risk=sizing["total_risk"],
        hist_win_rate=hist["hist_win_rate"],
        hist_avg_return=hist["hist_avg_return"],
        hist_sample_count=hist["hist_sample_count"],
        rationale=rationale,
    )


def pick_contracts(
    scored_df: pd.DataFrame,
    risk_budget: float = 500.0,
    filters: QualityFilters = DEFAULT_FILTERS,
    batch_stats: Optional[BatchStats] = None,
) -> PickerResult:
    """Main entry point: recommend 1-3 contracts with position sizing.

    Args:
        scored_df: DataFrame from score_options()
        risk_budget: max dollar premium outlay (premium = max loss)
        filters: quality gates to apply
        batch_stats: optional historical context (auto-loaded if None)

    Returns:
        PickerResult with up to 3 picks and risk guidance.
    """
    if batch_stats is None:
        batch_stats = load_batch_stats()

    total_candidates = len(scored_df)

    if scored_df.empty:
        return PickerResult(
            top_pick=None, conservative_pick=None, aggressive_pick=None,
            risk_budget=risk_budget, quality_gate_passed=0,
            total_candidates=0, risk_guidance={"summary": "No candidates.", "tier": "reduced", "warnings": []},
        )

    # Filter
    filtered = apply_quality_filters(scored_df, filters)
    quality_passed = len(filtered)

    # Select picks
    top_row = _select_top_pick(filtered)
    top_pick = None
    conservative_pick = None
    aggressive_pick = None
    exclude_idxs = []

    if top_row is not None:
        top_pick = _row_to_pick("top", top_row, risk_budget, batch_stats)
        top_idx = top_row.name if hasattr(top_row, "name") else 0
        exclude_idxs.append(top_idx)

        cons_row = _select_conservative_pick(filtered, top_idx)
        if cons_row is not None:
            # Only include if materially different from top pick
            cons_delta = abs(cons_row.get("delta", 0)) if pd.notna(cons_row.get("delta")) else 0
            top_delta = abs(top_row.get("delta", 0)) if pd.notna(top_row.get("delta")) else 0
            if abs(cons_delta - top_delta) > 0.05 or cons_row.get("strike") != top_row.get("strike"):
                conservative_pick = _row_to_pick("conservative", cons_row, risk_budget, batch_stats)
                cons_idx = cons_row.name if hasattr(cons_row, "name") else None
                if cons_idx is not None:
                    exclude_idxs.append(cons_idx)

        agg_row = _select_aggressive_pick(filtered, exclude_idxs)
        if agg_row is not None:
            agg_delta = abs(agg_row.get("delta", 0)) if pd.notna(agg_row.get("delta")) else 0
            if abs(agg_delta - top_delta) > 0.05 or agg_row.get("strike") != top_row.get("strike"):
                aggressive_pick = _row_to_pick("aggressive", agg_row, risk_budget, batch_stats)

    # Risk guidance
    guidance = {"summary": "No picks available.", "tier": "reduced", "warnings": []}
    if top_pick is not None:
        guidance = compute_risk_guidance(
            risk_budget, top_pick.composite_score,
            top_pick.entry_ask, top_pick.avg_spread_pct,
            batch_stats,
        )

    return PickerResult(
        top_pick=top_pick,
        conservative_pick=conservative_pick,
        aggressive_pick=aggressive_pick,
        risk_budget=risk_budget,
        quality_gate_passed=quality_passed,
        total_candidates=total_candidates,
        risk_guidance=guidance,
    )
