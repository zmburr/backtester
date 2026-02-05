"""
Backtester Data Dashboard (Streamlit)

Goal: explore `bounce_data.csv` + `reversal_data.csv` and simulate the *same* checklist
logic used in `scripts/generate_report.py`, so you can refine thresholds with real stats.

Run (from repo root):
  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Must be the first Streamlit command in the script.
st.set_page_config(page_title="Backtester Dashboard", layout="wide")


# -----------------------------------------------------------------------------
# Path setup (Streamlit runs with script dir on sys.path; add repo root)
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Local imports (after sys.path adjustment)
from analyzers.bounce_scorer import (  # noqa: E402
    SETUP_PROFILES,
    BouncePretrade,
    classify_from_setup_column,
)


# -----------------------------------------------------------------------------
# Reversal pre-trade thresholds (mirrors `scripts/generate_report.py`)
# NOTE: This is intentionally duplicated so the dashboard can run without
# importing the full report script (which has heavier dependencies/side-effects).
# -----------------------------------------------------------------------------
REVERSAL_PRETRADE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "Micro": {
        "pct_from_9ema": 0.80,
        "prior_day_range_atr": 3.0,
        "prior_day_rvol": 2.0,
        "premarket_rvol": 0.05,
        "consecutive_up_days": 3,
        "gap_pct": 0.15,
    },
    "Small": {
        "pct_from_9ema": 0.40,
        "prior_day_range_atr": 2.0,
        "prior_day_rvol": 2.0,
        "premarket_rvol": 0.05,
        "consecutive_up_days": 2,
        "gap_pct": 0.10,
    },
    "Medium": {
        "pct_from_9ema": 0.15,
        "prior_day_range_atr": 1.0,
        "prior_day_rvol": 1.5,
        "premarket_rvol": 0.05,
        "consecutive_up_days": 2,
        "gap_pct": 0.05,
    },
    "Large": {
        "pct_from_9ema": 0.08,
        "prior_day_range_atr": 0.8,
        "prior_day_rvol": 1.0,
        "premarket_rvol": 0.05,
        "consecutive_up_days": 1,
        "gap_pct": 0.00,
    },
    "ETF": {
        "pct_from_9ema": 0.04,
        "prior_day_range_atr": 1.0,
        "prior_day_rvol": 1.5,
        "premarket_rvol": 0.05,
        "consecutive_up_days": 1,
        "gap_pct": 0.00,
    },
}


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _safe_to_numeric(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _ensure_rvol_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns so checklist code can rely on:
      - prior_day_rvol: prior day volume / ADV (x)
      - premarket_rvol: premarket volume / ADV (x)

    Both CSVs already contain `percent_of_vol_one_day_before` and
    `percent_of_premarket_vol` most of the time; we use those as primary source.
    """
    out = df.copy()

    # Prior day RVOL
    if "prior_day_rvol" not in out.columns:
        out["prior_day_rvol"] = np.nan
    out["prior_day_rvol"] = _safe_to_numeric(out["prior_day_rvol"])
    if "percent_of_vol_one_day_before" in out.columns:
        out["prior_day_rvol"] = out["prior_day_rvol"].fillna(
            _safe_to_numeric(out["percent_of_vol_one_day_before"])
        )
    if {"vol_one_day_before", "avg_daily_vol"}.issubset(out.columns):
        out["prior_day_rvol"] = out["prior_day_rvol"].fillna(
            _safe_to_numeric(out["vol_one_day_before"]) / _safe_to_numeric(out["avg_daily_vol"])
        )

    # Premarket RVOL
    if "premarket_rvol" not in out.columns:
        out["premarket_rvol"] = np.nan
    out["premarket_rvol"] = _safe_to_numeric(out["premarket_rvol"])
    if "percent_of_premarket_vol" in out.columns:
        out["premarket_rvol"] = out["premarket_rvol"].fillna(
            _safe_to_numeric(out["percent_of_premarket_vol"])
        )
    if {"premarket_vol", "avg_daily_vol"}.issubset(out.columns):
        out["premarket_rvol"] = out["premarket_rvol"].fillna(
            _safe_to_numeric(out["premarket_vol"]) / _safe_to_numeric(out["avg_daily_vol"])
        )

    return out


def _percentile_rank_rankkind(ref_vals: np.ndarray, x: float) -> Optional[float]:
    """
    SciPy-free equivalent of `scipy.stats.percentileofscore(..., kind="rank")`.
    Returns 0-100 percentile or None.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    ref = np.asarray(ref_vals, dtype=float)
    ref = ref[~np.isnan(ref)]
    if ref.size == 0:
        return None
    less = float(np.sum(ref < float(x)))
    equal = float(np.sum(ref == float(x)))
    return 100.0 * (less + 0.5 * equal) / float(ref.size)


def _compute_bounce_intensity_row(
    row: Mapping[str, Any],
    ref_df: pd.DataFrame,
    weights: Dict[str, float],
) -> float:
    """
    Implements the same Bounce Intensity idea as `scripts/generate_report.py`.

    Higher = more extreme (better bounce candidate).
    """
    spec: List[Tuple[str, bool]] = [
        ("selloff_total_pct", False),  # deeper selloff = better => invert
        ("consecutive_down_days", True),
        ("prior_day_rvol", True),
        ("pct_off_30d_high", False),  # further off high (more negative) = better => invert
        ("gap_pct", False),  # larger gap down (more negative) = better => invert
    ]

    weighted_sum = 0.0
    total_weight = 0.0

    for col, higher_is_better in spec:
        w = float(weights.get(col, 0.0))
        if w <= 0:
            continue

        x = row.get(col)
        ref_vals = ref_df[col].to_numpy(dtype=float, copy=False) if col in ref_df.columns else np.array([])
        raw = _percentile_rank_rankkind(ref_vals, float(x)) if x is not None else None
        if raw is None:
            continue

        pctile = raw if higher_is_better else (100.0 - raw)
        weighted_sum += pctile * w
        total_weight += w

    return float(round(weighted_sum / total_weight, 1)) if total_weight > 0 else 0.0


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_bounce_df() -> pd.DataFrame:
    path = REPO_ROOT / "data" / "bounce_data.csv"
    df = pd.read_csv(path).dropna(subset=["ticker", "date"]).copy()

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["cap"] = df.get("cap", "Medium")
    df["cap"] = df["cap"].fillna("Medium").astype(str)

    df["date_dt"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")

    if "Setup" in df.columns:
        df["setup_profile"] = df["Setup"].astype(str).apply(classify_from_setup_column)
    else:
        df["setup_profile"] = "Unknown"

    df = _ensure_rvol_columns(df)

    # Match bounce_scorer.py behavior: pnl = bounce_open_close_pct * 100
    df["pnl"] = _safe_to_numeric(df.get("bounce_open_close_pct")) * 100.0

    return df


@st.cache_data(show_spinner=False)
def load_reversal_df() -> pd.DataFrame:
    path = REPO_ROOT / "data" / "reversal_data.csv"
    df = pd.read_csv(path).dropna(subset=["ticker", "date"]).copy()

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["cap"] = df.get("cap", "Medium")
    df["cap"] = df["cap"].fillna("Medium").astype(str)

    df["date_dt"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")

    df = _ensure_rvol_columns(df)

    # Simple "short P&L proxy": positive when open->close is negative (down day)
    df["pnl"] = -_safe_to_numeric(df.get("reversal_open_close_pct")) * 100.0

    return df


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------
def score_reversal_pretrade_row(
    row: Mapping[str, Any],
    thresholds: Dict[str, float],
) -> Tuple[int, str, List[str]]:
    failed: List[str] = []
    score = 0

    def _get_float(key: str) -> Optional[float]:
        v = row.get(key)
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            return float(v)
        except Exception:
            return None

    # 4 direct criteria
    checks: List[Tuple[str, str, str]] = [
        ("pct_from_9ema", "pct_from_9ema", ">="),
        ("prior_day_range_atr", "prior_day_range_atr", ">="),
        ("consecutive_up_days", "consecutive_up_days", ">="),
        ("gap_pct", "gap_pct", ">="),
    ]

    for name, key, op in checks:
        actual = _get_float(key)
        thresh = float(thresholds[name])
        passed = actual is not None and (actual >= thresh if op == ">=" else actual <= thresh)
        if passed:
            score += 1
        else:
            failed.append(name)

    # Vol signal: prior OR premarket meets threshold
    prior_rvol = _get_float("prior_day_rvol")
    pm_rvol = _get_float("premarket_rvol")
    prior_pass = prior_rvol is not None and prior_rvol >= float(thresholds["prior_day_rvol"])
    pm_pass = pm_rvol is not None and pm_rvol >= float(thresholds["premarket_rvol"])
    vol_pass = prior_pass or pm_pass
    if vol_pass:
        score += 1
    else:
        failed.append("vol_signal")

    if score >= 4:
        rec = "GO"
    elif score == 3:
        rec = "CAUTION"
    else:
        rec = "NO-GO"

    return score, rec, failed


def score_bounce_pretrade_override_row(
    row: Mapping[str, Any],
    required_selloff_pct: int,
    required_down_days: int,
    required_off_30d_high_pct: int,
    required_gap_down_pct: int,
    required_prior_day_rvol: float,
    required_premarket_rvol: float,
) -> Tuple[int, str, List[str]]:
    """
    What-if version of the BouncePretrade checklist (5 criteria).

    Notes:
      - selloff_total_pct, pct_off_30d_high, gap_pct are negative for drawdowns.
      - We accept *required* inputs as positive percentages and convert to negatives.
    """
    failed: List[str] = []
    score = 0

    def _f(key: str) -> Optional[float]:
        v = row.get(key)
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            return float(v)
        except Exception:
            return None

    selloff_thr = -abs(required_selloff_pct) / 100.0
    off30_thr = -abs(required_off_30d_high_pct) / 100.0
    gap_thr = -abs(required_gap_down_pct) / 100.0

    # 4 direct criteria
    criteria: List[Tuple[str, bool]] = [
        ("selloff_total_pct", (_f("selloff_total_pct") is not None and _f("selloff_total_pct") <= selloff_thr)),
        ("consecutive_down_days", (_f("consecutive_down_days") is not None and _f("consecutive_down_days") >= required_down_days)),
        ("pct_off_30d_high", (_f("pct_off_30d_high") is not None and _f("pct_off_30d_high") <= off30_thr)),
        ("gap_pct", (_f("gap_pct") is not None and _f("gap_pct") <= gap_thr)),
    ]

    for name, passed in criteria:
        if passed:
            score += 1
        else:
            failed.append(name)

    # Vol signal
    prior_rvol = _f("prior_day_rvol")
    pm_rvol = _f("premarket_rvol")
    vol_pass = (prior_rvol is not None and prior_rvol >= required_prior_day_rvol) or (
        pm_rvol is not None and pm_rvol >= required_premarket_rvol
    )
    if vol_pass:
        score += 1
    else:
        failed.append("vol_expansion")

    if score >= 4:
        rec = "GO"
    elif score == 3:
        rec = "CAUTION"
    else:
        rec = "NO-GO"

    return score, rec, failed


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------
def _filter_by_multiselect(df: pd.DataFrame, col: str, selected: List[str]) -> pd.DataFrame:
    if not selected or col not in df.columns:
        return df
    return df[df[col].isin(selected)]


def _date_range_filter(df: pd.DataFrame, date_col: str, start, end) -> pd.DataFrame:
    if date_col not in df.columns:
        return df
    out = df.copy()
    out = out[out[date_col].notna()]
    if start is not None:
        out = out[out[date_col].dt.date >= start]
    if end is not None:
        out = out[out[date_col].dt.date <= end]
    return out


def _summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    pnl = _safe_to_numeric(df.get("pnl"))
    pnl = pnl.dropna()
    if len(pnl) == 0:
        return {"trades": len(df), "win_rate": None, "avg_pnl": None, "median_pnl": None}
    return {
        "trades": int(len(df)),
        "win_rate": float(round((pnl > 0).mean() * 100.0, 1)),
        "avg_pnl": float(round(pnl.mean(), 2)),
        "median_pnl": float(round(pnl.median(), 2)),
    }


def _group_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()
    tmp = df.copy()
    tmp["pnl_num"] = _safe_to_numeric(tmp.get("pnl"))
    g = (
        tmp.groupby(group_col, dropna=False)
        .agg(
            trades=("ticker", "count"),
            win_rate=("pnl_num", lambda s: float((s > 0).mean() * 100.0) if len(s.dropna()) else np.nan),
            avg_pnl=("pnl_num", "mean"),
            median_pnl=("pnl_num", "median"),
        )
        .reset_index()
    )
    g["win_rate"] = g["win_rate"].round(1)
    g["avg_pnl"] = g["avg_pnl"].round(2)
    g["median_pnl"] = g["median_pnl"].round(2)
    return g.sort_values(group_col)


def _key_metrics_selector(defaults: List[str], df: pd.DataFrame) -> List[str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    options = [c for c in defaults if c in df.columns] + [c for c in numeric_cols if c not in defaults]
    return options


def _metric_threshold_sweep(
    df: pd.DataFrame,
    metric: str,
    direction: str,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    """
    Sweep a single threshold for a metric and compute win-rate / pnl stats for each cutoff.

    direction:
      - '>=' keeps rows where metric >= threshold
      - '<=' keeps rows where metric <= threshold
    """
    if metric not in df.columns:
        return pd.DataFrame()

    base_n = int(len(df))
    if base_n == 0:
        return pd.DataFrame()

    metric_num = _safe_to_numeric(df[metric])

    rows: List[Dict[str, Any]] = []
    for thr in thresholds:
        try:
            thr_f = float(thr)
        except Exception:
            continue

        if direction == "<=":
            subset = df[metric_num <= thr_f]
        else:
            subset = df[metric_num >= thr_f]

        stats = _summary_stats(subset)
        rows.append(
            {
                "threshold": thr_f,
                "coverage_pct": round((len(subset) / base_n) * 100.0, 1),
                **stats,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("threshold").reset_index(drop=True)
    return out


def render_metric_optimizer(
    df: pd.DataFrame,
    metric_defaults: List[str],
    default_directions: Optional[Dict[str, str]] = None,
    *,
    title: str = "Metric optimizer",
    key_prefix: str,
) -> None:
    """
    Interactive optimizer for a *single* metric threshold.

    It runs inside the current filtered dataset, so you can stack this with
    other filters (cap, grade, setup, etc.).
    """
    default_directions = default_directions or {}

    with st.expander(title, expanded=False):
        if df.empty:
            st.warning("No rows to optimize (empty filter result).")
            return

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        metric_options = _key_metrics_selector(metric_defaults, df)
        metric_options = [m for m in metric_options if m in numeric_cols]
        if not metric_options:
            st.warning("No numeric metrics available in this dataset.")
            return

        default_metric = metric_options[0]
        metric = st.selectbox("Metric", metric_options, index=0, key=f"{key_prefix}_metric")

        default_dir = default_directions.get(metric, ">=")
        direction = st.radio(
            "Keep rows where …",
            [">=", "<="],
            index=0 if default_dir == ">=" else 1,
            horizontal=True,
            key=f"{key_prefix}_dir",
        )

        # Threshold candidates (quantiles over the observed values)
        vals = _safe_to_numeric(df[metric]).dropna()
        if len(vals) == 0:
            st.warning("Selected metric has no numeric values after filtering.")
            return

        c1, c2, c3 = st.columns(3)
        sweep_points = c1.slider("Sweep points", min_value=10, max_value=250, value=60, step=5, key=f"{key_prefix}_points")
        min_trades = c2.slider(
            "Min trades (constraint)",
            min_value=1,
            max_value=max(1, min(500, len(df))),
            value=min(20, len(df)),
            step=1,
            key=f"{key_prefix}_min_trades",
        )
        objective = c3.selectbox(
            "Objective",
            ["win_rate", "avg_pnl", "median_pnl", "trades"],
            index=0,
            key=f"{key_prefix}_obj",
        )

        qs = np.linspace(0.0, 1.0, int(sweep_points))
        thresholds = np.unique(np.quantile(vals.to_numpy(dtype=float), qs))

        sweep = _metric_threshold_sweep(df, metric=metric, direction=direction, thresholds=thresholds)
        if sweep.empty:
            st.warning("No sweep results.")
            return

        feasible = sweep[sweep["trades"] >= int(min_trades)].copy()
        if feasible.empty:
            st.warning("No thresholds meet the minimum trade constraint.")
        else:
            best = feasible.sort_values(objective, ascending=False).iloc[0].to_dict()
            st.markdown(
                f"**Best (by {objective})**: threshold `{best['threshold']:.6g}` "
                f"→ trades={int(best['trades'])}, win={best['win_rate'] if best['win_rate'] is not None else '—'}%, "
                f"avg={best['avg_pnl'] if best['avg_pnl'] is not None else '—'}%"
            )

        # Charts
        st.plotly_chart(px.line(sweep, x="threshold", y="trades", title="Trades vs threshold"), use_container_width=True)
        st.plotly_chart(px.line(sweep, x="threshold", y="win_rate", title="Win rate (%) vs threshold"), use_container_width=True)
        st.plotly_chart(px.line(sweep, x="threshold", y="avg_pnl", title="Avg P&L (%) vs threshold"), use_container_width=True)

        # Table
        show_top = st.checkbox("Show top thresholds table", value=True, key=f"{key_prefix}_show_table")
        if show_top:
            table = feasible.sort_values(objective, ascending=False).head(40) if not feasible.empty else sweep.head(40)
            st.dataframe(table, use_container_width=True, height=420)

        # Download
        st.download_button(
            "Download sweep CSV",
            data=sweep.to_csv(index=False).encode("utf-8"),
            file_name=f"{key_prefix}_metric_sweep.csv",
        )

# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
def render_overview(bounce_df: pd.DataFrame, reversal_df: pd.DataFrame) -> None:
    st.subheader("Overview")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Bounce dataset (`data/bounce_data.csv`)**")
        stats = _summary_stats(bounce_df)
        st.metric("Trades", stats["trades"])
        st.metric("Win rate (pnl>0)", "—" if stats["win_rate"] is None else f"{stats['win_rate']:.1f}%")
        st.metric("Avg P&L (open→close)", "—" if stats["avg_pnl"] is None else f"{stats['avg_pnl']:+.2f}%")
        st.dataframe(
            bounce_df.groupby(["cap", "trade_grade"], dropna=False).size().reset_index(name="trades"),
            use_container_width=True,
            height=220,
        )
    with c2:
        st.markdown("**Reversal dataset (`data/reversal_data.csv`)**")
        stats = _summary_stats(reversal_df)
        st.metric("Trades", stats["trades"])
        st.metric("Win rate (pnl>0)", "—" if stats["win_rate"] is None else f"{stats['win_rate']:.1f}%")
        st.metric("Avg P&L proxy (-(open→close))", "—" if stats["avg_pnl"] is None else f"{stats['avg_pnl']:+.2f}%")
        st.dataframe(
            reversal_df.groupby(["cap", "trade_grade"], dropna=False).size().reset_index(name="trades"),
            use_container_width=True,
            height=220,
        )

    st.markdown(
        """
**Tip:** the Bounce tab uses `analyzers/bounce_scorer.py` profiles (same thresholds as your report).
The Reversal tab mirrors the `score_pretrade_setup()` logic in `scripts/generate_report.py`.
"""
    )


def render_bounce(bounce_df: pd.DataFrame) -> None:
    st.subheader("Bounce dashboard")

    # Sidebar filters
    with st.sidebar:
        st.markdown("### Bounce filters")
        grades = sorted([g for g in bounce_df.get("trade_grade", pd.Series(dtype=str)).dropna().unique()])
        caps = sorted([c for c in bounce_df.get("cap", pd.Series(dtype=str)).dropna().unique()])
        profiles = sorted([p for p in bounce_df.get("setup_profile", pd.Series(dtype=str)).dropna().unique()])

        sel_grades = st.multiselect("Trade grade", grades, default=grades)
        sel_caps = st.multiselect("Cap", caps, default=caps)
        sel_profiles = st.multiselect("Setup profile", profiles, default=profiles)

        dmin = bounce_df["date_dt"].min()
        dmax = bounce_df["date_dt"].max()
        start_end = st.date_input(
            "Date range",
            value=(dmin.date() if pd.notna(dmin) else None, dmax.date() if pd.notna(dmax) else None),
        )
        start_date = start_end[0] if isinstance(start_end, (list, tuple)) and len(start_end) > 0 else None
        end_date = start_end[1] if isinstance(start_end, (list, tuple)) and len(start_end) > 1 else None

    df = bounce_df.copy()
    df = _filter_by_multiselect(df, "trade_grade", sel_grades)
    df = _filter_by_multiselect(df, "cap", sel_caps)
    df = _filter_by_multiselect(df, "setup_profile", sel_profiles)
    df = _date_range_filter(df, "date_dt", start_date, end_date)

    if df.empty:
        st.warning("No bounce trades match the current filters.")
        return

    st.caption("P&L = `bounce_open_close_pct * 100` (open→close %). Win rate = % of rows with P&L > 0.")

    stats = _summary_stats(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades", stats["trades"])
    c2.metric("Win rate", "—" if stats["win_rate"] is None else f"{stats['win_rate']:.1f}%")
    c3.metric("Avg P&L", "—" if stats["avg_pnl"] is None else f"{stats['avg_pnl']:+.2f}%")
    c4.metric("Median P&L", "—" if stats["median_pnl"] is None else f"{stats['median_pnl']:+.2f}%")

    st.markdown("### Sell-off + capitulation stats")
    metric_defaults = [
        "selloff_total_pct",
        "consecutive_down_days",
        "pct_off_30d_high",
        "gap_pct",
        "prior_day_rvol",
        "premarket_rvol",
        "bounce_open_high_pct",
        "bounce_open_close_pct",
    ]
    metric_options = _key_metrics_selector(metric_defaults, df)
    metric = st.selectbox("Metric", metric_options, index=0)

    if metric in df.columns:
        fig = px.histogram(
            df,
            x=metric,
            nbins=35,
            hover_data=["ticker", "date", "cap", "setup_profile", "trade_grade", "pnl"],
            title=f"Distribution: {metric}",
        )
        st.plotly_chart(fig, use_container_width=True)

        if "pnl" in df.columns and pd.api.types.is_numeric_dtype(df["pnl"]):
            scatter = px.scatter(
                df,
                x=metric,
                y="pnl",
                color="setup_profile" if "setup_profile" in df.columns else None,
                hover_data=["ticker", "date", "cap", "trade_grade"],
                title=f"{metric} vs P&L (open→close)",
            )
            st.plotly_chart(scatter, use_container_width=True)

    st.markdown("### Checklist simulator (BouncePretrade)")
    mode = st.radio(
        "Scoring mode",
        ["Current checklist (from analyzers/bounce_scorer.py)", "What-if override (tune thresholds)"],
        horizontal=True,
    )

    if mode.startswith("Current"):
        checker = BouncePretrade()
        rows: List[Dict[str, Any]] = []
        for _, r in df.iterrows():
            cap = str(r.get("cap") or "Medium")
            setup = str(r.get("setup_profile") or "GapFade_strongstock")
            res = checker.validate(
                ticker=str(r.get("ticker", "")),
                metrics=r.to_dict(),
                force_setup=setup,
                cap=cap,
            )
            failed = [i.name for i in res.items if not i.passed]
            rows.append(
                {
                    "checklist_score": int(res.score),
                    "checklist_max": int(res.max_score),
                    "checklist_rec": str(res.recommendation),
                    "checklist_failed": ", ".join(failed) if failed else "PERFECT",
                }
            )
        scored = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
        st.dataframe(_group_stats(scored, "checklist_rec"), use_container_width=True)

        box = px.box(
            scored,
            x="checklist_rec",
            y="pnl",
            points="all",
            hover_data=["ticker", "date", "cap", "setup_profile", "trade_grade", "checklist_score"],
            title="P&L by checklist recommendation",
        )
        st.plotly_chart(box, use_container_width=True)

        st.dataframe(
            scored.sort_values(["checklist_rec", "pnl"], ascending=[True, False]),
            use_container_width=True,
            height=420,
        )

        csv_bytes = scored.to_csv(index=False).encode("utf-8")
        st.download_button("Download scored bounce CSV", data=csv_bytes, file_name="bounce_scored_dashboard.csv")

    else:
        # Pick a tuning context (profile + cap) to show baseline thresholds
        tune_profile = st.selectbox("Tune setup profile", sorted(SETUP_PROFILES.keys()), index=0)
        tune_cap = st.selectbox("Tune cap", sorted(df["cap"].dropna().unique()), index=0)

        profile = SETUP_PROFILES[tune_profile]
        base = {
            "selloff_total_pct": float(profile.get_threshold("selloff_total_pct", tune_cap)),
            "consecutive_down_days": int(profile.get_threshold("consecutive_down_days", tune_cap)),
            "pct_off_30d_high": float(profile.get_threshold("pct_off_30d_high", tune_cap)),
            "gap_pct": float(profile.get_threshold("gap_pct", tune_cap)),
            "prior_day_rvol": float(profile.get_threshold("vol_expansion", tune_cap)),
            "premarket_rvol": float(profile.vol_premarket),
        }

        with st.expander("Current (baseline) thresholds", expanded=True):
            st.json({"setup_profile": tune_profile, "cap": tune_cap, **base})

        # Sliders use positive % for drawdown-style thresholds.
        c1, c2, c3 = st.columns(3)
        required_selloff = c1.slider(
            "Required selloff depth (%)",
            min_value=0,
            max_value=90,
            value=int(round(abs(base["selloff_total_pct"]) * 100)),
            step=1,
        )
        required_off30 = c2.slider(
            "Required discount from 30d high (%)",
            min_value=0,
            max_value=90,
            value=int(round(abs(base["pct_off_30d_high"]) * 100)),
            step=1,
        )
        required_gap = c3.slider(
            "Required gap down (%)",
            min_value=0,
            max_value=50,
            value=int(round(abs(base["gap_pct"]) * 100)),
            step=1,
        )

        c4, c5, c6 = st.columns(3)
        required_down_days = c4.slider(
            "Consecutive down days (min)",
            min_value=0,
            max_value=10,
            value=int(base["consecutive_down_days"]),
            step=1,
        )
        required_prior_rvol = c5.slider(
            "Prior day RVOL (min, x ADV)",
            min_value=0.0,
            max_value=10.0,
            value=float(base["prior_day_rvol"]),
            step=0.1,
        )
        required_pm_rvol = c6.slider(
            "Premarket RVOL (min, x ADV)",
            min_value=0.0,
            max_value=0.50,
            value=float(base["premarket_rvol"]),
            step=0.01,
        )

        only_matching = st.checkbox(
            "Only score rows matching this profile + cap",
            value=True,
            help="Bounce thresholds are profile+cap specific; this keeps the comparison apples-to-apples.",
        )
        to_score = df.copy()
        if only_matching:
            to_score = to_score[(to_score["setup_profile"] == tune_profile) & (to_score["cap"] == tune_cap)]

        rows: List[Dict[str, Any]] = []
        for _, r in to_score.iterrows():
            s, rec, failed = score_bounce_pretrade_override_row(
                r.to_dict(),
                required_selloff_pct=required_selloff,
                required_down_days=required_down_days,
                required_off_30d_high_pct=required_off30,
                required_gap_down_pct=required_gap,
                required_prior_day_rvol=required_prior_rvol,
                required_premarket_rvol=required_pm_rvol,
            )
            rows.append(
                {
                    "checklist_score": int(s),
                    "checklist_max": 5,
                    "checklist_rec": rec,
                    "checklist_failed": ", ".join(failed) if failed else "PERFECT",
                }
            )
        scored = pd.concat([to_score.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

        st.dataframe(_group_stats(scored, "checklist_rec"), use_container_width=True)
        box = px.box(
            scored,
            x="checklist_rec",
            y="pnl",
            points="all",
            hover_data=["ticker", "date", "trade_grade"],
            title="P&L by what-if recommendation",
        )
        st.plotly_chart(box, use_container_width=True)

        st.dataframe(scored.sort_values(["checklist_rec", "pnl"], ascending=[True, False]), use_container_width=True, height=420)

        with st.expander("Export snippet (paste into your checklist config)", expanded=False):
            st.code(
                "\n".join(
                    [
                        f"# Bounce override for {tune_profile} / {tune_cap}",
                        f"# selloff_total_pct <= {-required_selloff/100:.2f}",
                        f"# consecutive_down_days >= {required_down_days}",
                        f"# pct_off_30d_high <= {-required_off30/100:.2f}",
                        f"# gap_pct <= {-required_gap/100:.2f}",
                        f"# prior_day_rvol >= {required_prior_rvol:.2f}",
                        f"# premarket_rvol >= {required_pm_rvol:.2f}",
                    ]
                ),
                language="python",
            )

    render_metric_optimizer(
        df,
        metric_defaults=metric_defaults,
        default_directions={
            "selloff_total_pct": "<=",
            "pct_off_30d_high": "<=",
            "gap_pct": "<=",
            "consecutive_down_days": ">=",
            "prior_day_rvol": ">=",
            "premarket_rvol": ">=",
        },
        title="Metric optimizer (single threshold sweep)",
        key_prefix="bounce_opt",
    )


def render_reversal(reversal_df: pd.DataFrame) -> None:
    st.subheader("Reversal dashboard")

    with st.sidebar:
        st.markdown("### Reversal filters")
        grades = sorted([g for g in reversal_df.get("trade_grade", pd.Series(dtype=str)).dropna().unique()])
        caps = sorted([c for c in reversal_df.get("cap", pd.Series(dtype=str)).dropna().unique()])
        setups = sorted([s for s in reversal_df.get("setup", pd.Series(dtype=str)).dropna().unique()])

        sel_grades = st.multiselect("Trade grade", grades, default=grades)
        sel_caps = st.multiselect("Cap", caps, default=caps)
        sel_setups = st.multiselect("Setup", setups, default=setups)

        dmin = reversal_df["date_dt"].min()
        dmax = reversal_df["date_dt"].max()
        start_end = st.date_input(
            "Date range",
            value=(dmin.date() if pd.notna(dmin) else None, dmax.date() if pd.notna(dmax) else None),
        )
        start_date = start_end[0] if isinstance(start_end, (list, tuple)) and len(start_end) > 0 else None
        end_date = start_end[1] if isinstance(start_end, (list, tuple)) and len(start_end) > 1 else None

    df = reversal_df.copy()
    df = _filter_by_multiselect(df, "trade_grade", sel_grades)
    df = _filter_by_multiselect(df, "cap", sel_caps)
    df = _filter_by_multiselect(df, "setup", sel_setups)
    df = _date_range_filter(df, "date_dt", start_date, end_date)

    if df.empty:
        st.warning("No reversal trades match the current filters.")
        return

    st.caption("P&L proxy = `-reversal_open_close_pct * 100` (short open→close %). Win rate = % of rows with P&L > 0.")

    stats = _summary_stats(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades", stats["trades"])
    c2.metric("Win rate", "—" if stats["win_rate"] is None else f"{stats['win_rate']:.1f}%")
    c3.metric("Avg P&L proxy", "—" if stats["avg_pnl"] is None else f"{stats['avg_pnl']:+.2f}%")
    c4.metric("Median P&L proxy", "—" if stats["median_pnl"] is None else f"{stats['median_pnl']:+.2f}%")

    st.markdown("### Run-up / squeeze metrics")
    metric_defaults = [
        "pct_from_9ema",
        "prior_day_range_atr",
        "consecutive_up_days",
        "gap_pct",
        "prior_day_rvol",
        "premarket_rvol",
        "reversal_open_close_pct",
        "upper_band_distance",
        "bollinger_width",
    ]
    metric_options = _key_metrics_selector(metric_defaults, df)
    metric = st.selectbox("Metric", metric_options, index=0, key="rev_metric")

    if metric in df.columns:
        fig = px.histogram(
            df,
            x=metric,
            nbins=35,
            hover_data=["ticker", "date", "cap", "setup", "trade_grade", "pnl"],
            title=f"Distribution: {metric}",
        )
        st.plotly_chart(fig, use_container_width=True)

        scatter = px.scatter(
            df,
            x=metric,
            y="pnl",
            color="cap" if "cap" in df.columns else None,
            hover_data=["ticker", "date", "setup", "trade_grade"],
            title=f"{metric} vs P&L proxy (-(open→close))",
        )
        st.plotly_chart(scatter, use_container_width=True)

    st.markdown("### Checklist simulator (Reversal pre-trade, 5 criteria)")
    mode = st.radio(
        "Scoring mode",
        ["Current checklist (mirrors generate_report.py)", "What-if override (tune thresholds)"],
        horizontal=True,
        key="rev_mode",
    )

    if mode.startswith("Current"):
        rows: List[Dict[str, Any]] = []
        for _, r in df.iterrows():
            cap = str(r.get("cap") or "Medium")
            thr = REVERSAL_PRETRADE_THRESHOLDS.get(cap, REVERSAL_PRETRADE_THRESHOLDS["Medium"])
            s, rec, failed = score_reversal_pretrade_row(r.to_dict(), thr)
            rows.append(
                {
                    "checklist_score": int(s),
                    "checklist_max": 5,
                    "checklist_rec": rec,
                    "checklist_failed": ", ".join(failed) if failed else "PERFECT",
                }
            )
        scored = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

        st.dataframe(_group_stats(scored, "checklist_rec"), use_container_width=True)
        box = px.box(
            scored,
            x="checklist_rec",
            y="pnl",
            points="all",
            hover_data=["ticker", "date", "cap", "setup", "trade_grade", "checklist_score"],
            title="P&L proxy by checklist recommendation",
        )
        st.plotly_chart(box, use_container_width=True)
        st.dataframe(scored.sort_values(["checklist_rec", "pnl"], ascending=[True, False]), use_container_width=True, height=420)

        csv_bytes = scored.to_csv(index=False).encode("utf-8")
        st.download_button("Download scored reversal CSV", data=csv_bytes, file_name="reversal_scored_dashboard.csv")

    else:
        tune_cap = st.selectbox("Tune cap", sorted(df["cap"].dropna().unique()), index=0, key="rev_tune_cap")
        base = REVERSAL_PRETRADE_THRESHOLDS.get(tune_cap, REVERSAL_PRETRADE_THRESHOLDS["Medium"]).copy()

        with st.expander("Current (baseline) thresholds", expanded=True):
            st.json({"cap": tune_cap, **base})

        c1, c2, c3 = st.columns(3)
        thr_pct9 = c1.slider(
            "% above 9EMA (min)",
            min_value=0.0,
            max_value=2.0,
            value=float(base["pct_from_9ema"]),
            step=0.01,
        )
        thr_range = c2.slider(
            "Prior day range (min, x ATR)",
            min_value=0.0,
            max_value=10.0,
            value=float(base["prior_day_range_atr"]),
            step=0.1,
        )
        thr_up = c3.slider(
            "Consecutive up days (min)",
            min_value=0,
            max_value=10,
            value=int(base["consecutive_up_days"]),
            step=1,
        )

        c4, c5, c6 = st.columns(3)
        thr_gap = c4.slider(
            "Gap up (min, %)",
            min_value=0.0,
            max_value=3.0,
            value=float(base["gap_pct"]) * 100.0,
            step=0.5,
        )
        thr_prior_rvol = c5.slider(
            "Prior day RVOL (min, x ADV)",
            min_value=0.0,
            max_value=10.0,
            value=float(base["prior_day_rvol"]),
            step=0.1,
        )
        thr_pm_rvol = c6.slider(
            "Premarket RVOL (min, x ADV)",
            min_value=0.0,
            max_value=0.50,
            value=float(base["premarket_rvol"]),
            step=0.01,
        )

        base_override = {
            "pct_from_9ema": thr_pct9,
            "prior_day_range_atr": thr_range,
            "consecutive_up_days": thr_up,
            "gap_pct": thr_gap / 100.0,
            "prior_day_rvol": thr_prior_rvol,
            "premarket_rvol": thr_pm_rvol,
        }

        only_matching = st.checkbox("Only score rows matching this cap", value=True, key="rev_only_matching")
        to_score = df.copy()
        if only_matching:
            to_score = to_score[to_score["cap"] == tune_cap]

        rows: List[Dict[str, Any]] = []
        for _, r in to_score.iterrows():
            s, rec, failed = score_reversal_pretrade_row(r.to_dict(), base_override)
            rows.append(
                {
                    "checklist_score": int(s),
                    "checklist_max": 5,
                    "checklist_rec": rec,
                    "checklist_failed": ", ".join(failed) if failed else "PERFECT",
                }
            )
        scored = pd.concat([to_score.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

        st.dataframe(_group_stats(scored, "checklist_rec"), use_container_width=True)
        box = px.box(
            scored,
            x="checklist_rec",
            y="pnl",
            points="all",
            hover_data=["ticker", "date", "setup", "trade_grade"],
            title="P&L proxy by what-if recommendation",
        )
        st.plotly_chart(box, use_container_width=True)
        st.dataframe(scored.sort_values(["checklist_rec", "pnl"], ascending=[True, False]), use_container_width=True, height=420)

        with st.expander("Export snippet (paste into `PRETRADE_THRESHOLDS` in generate_report.py)", expanded=False):
            st.code(
                "\n".join(
                    [
                        f"PRETRADE_THRESHOLDS['{tune_cap}'] = {{",
                        f"    'pct_from_9ema': {thr_pct9:.2f},",
                        f"    'prior_day_range_atr': {thr_range:.1f},",
                        f"    'prior_day_rvol': {thr_prior_rvol:.1f},",
                        f"    'premarket_rvol': {thr_pm_rvol:.2f},",
                        f"    'consecutive_up_days': {thr_up},",
                        f"    'gap_pct': {thr_gap/100.0:.2f},",
                        "}",
                    ]
                ),
                language="python",
            )

    render_metric_optimizer(
        df,
        metric_defaults=metric_defaults,
        default_directions={
            "reversal_open_close_pct": "<=",
            "prior_day_close_vs_high_pct": "<=",
        },
        title="Metric optimizer (single threshold sweep)",
        key_prefix="reversal_opt",
    )


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
st.title("Backtester Checklist Dashboard")

with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("Page", ["Overview", "Bounce", "Reversal"], index=0)

bounce_df = load_bounce_df()
reversal_df = load_reversal_df()

# Optional: add bounce intensity column (based on current bounce dataset)
with st.sidebar:
    add_intensity = st.checkbox("Compute Bounce Intensity (slower)", value=False)
if add_intensity and "bounce_intensity" not in bounce_df.columns:
    # Default weights match `scripts/generate_report.py`
    _weights = {
        "selloff_total_pct": 0.30,
        "consecutive_down_days": 0.10,
        "prior_day_rvol": 0.15,
        "pct_off_30d_high": 0.20,
        "gap_pct": 0.25,
    }
    tmp = bounce_df.copy()
    tmp["bounce_intensity"] = tmp.apply(lambda r: _compute_bounce_intensity_row(r, tmp, _weights), axis=1)
    bounce_df = tmp

if page == "Overview":
    render_overview(bounce_df, reversal_df)
elif page == "Bounce":
    render_bounce(bounce_df)
else:
    render_reversal(reversal_df)

