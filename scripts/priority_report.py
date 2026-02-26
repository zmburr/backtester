"""Priority Report — surfaces only GO & CAUTION trades with deeper analysis.

Filters the full watchlist down to high-conviction setups, then adds:
  1. Historical comps (z-score Euclidean distance)
  2. Upgrade threshold analysis (CAUTION → what flips to GO)
  3. LLM-generated narrative (Cerebras fast_foundation tier)
  4. Charts + exit targets

Sent via email to zmburr@gmail.com morning + evening.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import datetime
import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Reuse from generate_report.py ---
import scripts.generate_report as gr
from scanners import stock_screener as ss
from analyzers.bounce_scorer import (
    BouncePretrade,
    fetch_bounce_metrics,
    classify_stock,
    SETUP_PROFILES,
)
from analyzers.charter import create_daily_chart, cleanup_charts
from analyzers.exit_targets import calculate_exit_targets, format_exit_targets_html
from analyzers.bounce_exit_targets import (
    calculate_bounce_exit_targets,
    format_bounce_exit_targets_html,
)
from support.config import send_email
from support.llm_client import llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s – %(message)s")
log = logging.getLogger(__name__)

_MAX_WORKERS = 8

# ---------------------------------------------------------------------------
# Historical data for comps
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_BOUNCE_DF = pd.read_csv(_DATA_DIR / "bounce_data.csv").dropna(subset=["ticker", "date"])
_REVERSAL_DF = pd.read_csv(_DATA_DIR / "reversal_data.csv").dropna(subset=["ticker", "date"])

# Comp columns by bucket
BOUNCE_COMP_COLUMNS = [
    "selloff_total_pct", "pct_off_30d_high", "gap_pct",
    "prior_day_range_atr", "pct_change_3", "pct_off_52wk_high",
]
REVERSAL_COMP_COLUMNS = [
    "pct_from_9ema", "gap_pct", "pct_change_3", "one_day_before_range_pct",
]


# ---------------------------------------------------------------------------
# 1. Historical Comps
# ---------------------------------------------------------------------------

def find_historical_comps(
    metrics: Dict,
    ref_df: pd.DataFrame,
    comp_columns: List[str],
    cap: str,
    n_comps: int = 5,
) -> pd.DataFrame:
    """Find most similar past trades via z-score standardized Euclidean distance.

    Filters by cap first (relaxes if <5 remain). Returns top-N comp rows with
    a '_distance' column appended.
    """
    # Filter by cap
    if "cap" in ref_df.columns:
        cap_df = ref_df[ref_df["cap"] == cap].copy()
        if len(cap_df) < n_comps:
            cap_df = ref_df.copy()  # relax cap filter
    else:
        cap_df = ref_df.copy()

    # Ensure comp columns exist
    available = [c for c in comp_columns if c in cap_df.columns]
    if not available:
        return pd.DataFrame()

    # Build candidate matrix (only rows with all comp columns present)
    subset = cap_df.dropna(subset=available).copy()
    if subset.empty:
        return pd.DataFrame()

    # Build candidate vector from live metrics
    candidate = []
    for col in available:
        val = metrics.get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            # Can't compute distance without this metric — fall back to mean
            val = subset[col].mean()
        candidate.append(float(val))
    candidate = np.array(candidate)

    # Z-score standardize
    mat = subset[available].values.astype(float)
    means = np.nanmean(mat, axis=0)
    stds = np.nanstd(mat, axis=0)
    stds[stds == 0] = 1.0  # avoid division by zero

    mat_z = (mat - means) / stds
    candidate_z = (candidate - means) / stds

    # Euclidean distance
    distances = np.sqrt(np.nansum((mat_z - candidate_z) ** 2, axis=1))
    subset = subset.copy()
    subset["_distance"] = distances

    return subset.nsmallest(n_comps, "_distance")


# ---------------------------------------------------------------------------
# 2. Upgrade Thresholds (CAUTION → GO)
# ---------------------------------------------------------------------------

def compute_upgrade_thresholds(
    ticker: str,
    bucket: str,
    score_result,
    metrics: Dict,
) -> List[Dict]:
    """For CAUTION tickers, compute what would flip each failed criterion to GO.

    Returns list of dicts: {criterion, current, threshold, price_target, note}
    """
    upgrades = []

    if bucket == "reversal":
        # score_result is a dict with 'criteria' list
        for c in score_result.get("criteria", []):
            if c["passed"]:
                continue
            entry = {
                "criterion": c["name"],
                "key": c["key"],
                "current": c["actual"],
                "threshold": c["threshold"],
                "price_target": None,
                "note": "",
            }
            key = c["key"]
            thresh = c["threshold"]

            if key == "gap_pct" and metrics.get("prior_close"):
                entry["price_target"] = metrics["prior_close"] * (1 + thresh)
                entry["note"] = f"needs open at ${entry['price_target']:.2f}"
            elif key == "pct_from_9ema" and metrics.get("ema_9"):
                entry["price_target"] = metrics["ema_9"] * (1 + thresh)
                entry["note"] = f"needs price at ${entry['price_target']:.2f}"
            elif key == "pct_change_3" and metrics.get("close_3_ago"):
                entry["price_target"] = metrics["close_3_ago"] * (1 + thresh)
                entry["note"] = f"needs price at ${entry['price_target']:.2f}"
            elif key == "prior_day_range_atr":
                entry["note"] = f"needs ≥{thresh:.1f}x ATR range (historical)"
            elif key == "vol_signal":
                entry["note"] = f"needs RVOL ≥{thresh:.1f}x or PM RVOL ≥ threshold"
            else:
                entry["note"] = f"needs ≥{thresh}" if thresh else ""

            upgrades.append(entry)

    elif bucket == "bounce":
        # score_result is a ChecklistResult with .items
        for item in score_result.items:
            if item.passed:
                continue
            entry = {
                "criterion": item.description,
                "key": item.name,
                "current": item.actual,
                "threshold": item.threshold,
                "price_target": None,
                "note": "",
            }

            if item.name == "gap_pct" and metrics.get("prior_close"):
                entry["price_target"] = metrics["prior_close"] * (1 + item.threshold)
                entry["note"] = f"needs open at ${entry['price_target']:.2f}"
            elif item.name == "pct_off_30d_high" and metrics.get("high_30d"):
                entry["price_target"] = metrics["high_30d"] * (1 + item.threshold)
                entry["note"] = f"needs price at ${entry['price_target']:.2f}"
            elif item.name == "pct_off_52wk_high" and metrics.get("high_52wk"):
                entry["price_target"] = metrics["high_52wk"] * (1 + item.threshold)
                entry["note"] = f"needs price at ${entry['price_target']:.2f}"
            elif item.name == "pct_change_3" and metrics.get("close_3_ago"):
                entry["price_target"] = metrics["close_3_ago"] * (1 + item.threshold)
                entry["note"] = f"needs price at ${entry['price_target']:.2f}"
            elif item.name == "prior_day_range_atr":
                entry["note"] = f"needs ≥{item.threshold:.1f}x ATR (historical)"
            else:
                entry["note"] = f"needs value ≥{item.threshold}" if item.threshold else ""

            upgrades.append(entry)

    return upgrades


# ---------------------------------------------------------------------------
# 3. LLM Narrative
# ---------------------------------------------------------------------------

async def generate_llm_narrative(
    ticker: str,
    bucket: str,
    rec: str,
    metrics: Dict,
    comps: pd.DataFrame,
    upgrade_info: List[Dict],
) -> str:
    """Generate a 2-3 sentence actionable narrative via Cerebras fast_foundation."""

    # Summarize comps
    comp_summary = "No historical comps available."
    if comps is not None and not comps.empty:
        outcome_col = "bounce_open_high_pct" if bucket == "bounce" else "reversal_open_low_pct"
        if outcome_col in comps.columns:
            outcomes = comps[outcome_col].dropna()
            if not outcomes.empty:
                comp_summary = (
                    f"{len(comps)} closest comps: median outcome {outcomes.median()*100:.1f}%, "
                    f"range {outcomes.min()*100:.1f}% to {outcomes.max()*100:.1f}%"
                )
                grades = comps["trade_grade"].value_counts().to_dict() if "trade_grade" in comps.columns else {}
                if grades:
                    grade_str = ", ".join(f"{g}: {n}" for g, n in sorted(grades.items()))
                    comp_summary += f". Grades: {grade_str}"

    # Build key metrics summary
    key_metrics = []
    if bucket == "bounce":
        for k in ["selloff_total_pct", "gap_pct", "pct_off_30d_high", "pct_change_3"]:
            v = metrics.get(k)
            if v is not None:
                key_metrics.append(f"{k}: {v*100:.1f}%")
    else:
        for k in ["pct_from_9ema", "gap_pct", "pct_change_3", "prior_day_range_atr"]:
            v = metrics.get(k)
            if v is not None:
                if k == "prior_day_range_atr":
                    key_metrics.append(f"{k}: {v:.1f}x")
                else:
                    key_metrics.append(f"{k}: {v*100:.1f}%")

    # Upgrade gaps
    upgrade_text = ""
    if upgrade_info:
        gaps = [f"{u['criterion']}: {u['note']}" for u in upgrade_info if u.get("note")]
        if gaps:
            upgrade_text = f"Upgrade gaps (CAUTION→GO): {'; '.join(gaps)}"

    system_msg = (
        "You are a concise trading analyst. Write 2-3 sentences of actionable analysis. "
        "Focus on conviction level, key risks, and what to watch for at the open. "
        "No disclaimers, no hedge language."
    )
    user_msg = (
        f"Ticker: {ticker}\n"
        f"Setup: {bucket.upper()} — {rec}\n"
        f"Key metrics: {', '.join(key_metrics) if key_metrics else 'N/A'}\n"
        f"Historical comps: {comp_summary}\n"
        f"{upgrade_text}"
    )

    try:
        result = await llm.chat(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            tier="smart_foundation",
            temperature=0.3,
        )
        return result.strip() if result else _fallback_narrative(ticker, bucket, rec)
    except Exception as e:
        log.warning(f"LLM narrative failed for {ticker}: {e}")
        return _fallback_narrative(ticker, bucket, rec)


def _fallback_narrative(ticker: str, bucket: str, rec: str) -> str:
    return f"{ticker} shows a {bucket} setup rated {rec}. Review checklist criteria and historical comps before entry."


# ---------------------------------------------------------------------------
# 4. HTML Builders
# ---------------------------------------------------------------------------

def _build_upgrade_table_html(upgrades: List[Dict]) -> str:
    """Build 'What Would Flip to GO?' table for CAUTION tickers."""
    if not upgrades:
        return ""
    rows = []
    for u in upgrades:
        current_str = "N/A"
        if u["current"] is not None:
            try:
                if u["key"] in ("prior_day_range_atr",):
                    current_str = f"{u['current']:.1f}x"
                elif u["key"] == "vol_signal":
                    current_str = str(u["current"])
                else:
                    current_str = f"{u['current']*100:.1f}%"
            except (TypeError, ValueError):
                current_str = str(u["current"])

        thresh_str = ""
        if u["threshold"] is not None:
            try:
                if u["key"] in ("prior_day_range_atr",):
                    thresh_str = f"{u['threshold']:.1f}x"
                elif u["key"] == "vol_signal":
                    thresh_str = str(u["threshold"])
                else:
                    thresh_str = f"{u['threshold']*100:.1f}%"
            except (TypeError, ValueError):
                thresh_str = str(u["threshold"])

        price_str = f"${u['price_target']:.2f}" if u.get("price_target") else "—"

        rows.append(
            f'<tr>'
            f'<td style="padding: 4px 8px; border: 1px solid #30363d;">{u["criterion"]}</td>'
            f'<td style="padding: 4px 8px; border: 1px solid #30363d; color: #f85149;">{current_str}</td>'
            f'<td style="padding: 4px 8px; border: 1px solid #30363d; color: #3fb950;">≥{thresh_str}</td>'
            f'<td style="padding: 4px 8px; border: 1px solid #30363d; color: #58a6ff; font-weight: bold;">{price_str}</td>'
            f'<td style="padding: 4px 8px; border: 1px solid #30363d; color: #8b949e; font-size: 0.85em;">{u.get("note", "")}</td>'
            f'</tr>'
        )

    return (
        '<div style="margin: 12px 0;">'
        '<h4 style="color: #e3b341; margin: 0 0 6px 0;">What Would Flip to GO?</h4>'
        '<table style="border-collapse: collapse; font-size: 0.9em; color: #c9d1d9; width: 100%;">'
        '<tr style="background-color: #21262d;">'
        '<th style="padding: 4px 8px; border: 1px solid #30363d; text-align: left;">Criterion</th>'
        '<th style="padding: 4px 8px; border: 1px solid #30363d; text-align: left;">Current</th>'
        '<th style="padding: 4px 8px; border: 1px solid #30363d; text-align: left;">Needed</th>'
        '<th style="padding: 4px 8px; border: 1px solid #30363d; text-align: left;">Price Target</th>'
        '<th style="padding: 4px 8px; border: 1px solid #30363d; text-align: left;">Note</th>'
        '</tr>'
        + "\n".join(rows)
        + '</table></div>'
    )


def _build_comps_table_html(comps: pd.DataFrame, bucket: str) -> str:
    """Build historical comps table."""
    if comps is None or comps.empty:
        return '<div style="color: #8b949e; margin: 8px 0;">No historical comps found.</div>'

    outcome_col = "bounce_open_high_pct" if bucket == "bounce" else "reversal_open_low_pct"
    display_cols = ["ticker", "date", "cap"]
    if "trade_grade" in comps.columns:
        display_cols.append("trade_grade")
    if outcome_col in comps.columns:
        display_cols.append(outcome_col)
    display_cols.append("_distance")

    # Summary stats
    summary = ""
    if outcome_col in comps.columns:
        outcomes = comps[outcome_col].dropna()
        if not outcomes.empty:
            summary = (
                f'<div style="font-size: 0.85em; color: #8b949e; margin: 4px 0;">'
                f'Comp outcomes — median: {outcomes.median()*100:.1f}%, '
                f'mean: {outcomes.mean()*100:.1f}%, '
                f'range: {outcomes.min()*100:.1f}% to {outcomes.max()*100:.1f}%'
                f'</div>'
            )

    header = "".join(
        f'<th style="padding: 4px 8px; border: 1px solid #30363d; text-align: left;">{c}</th>'
        for c in display_cols
    )
    rows = []
    for _, row in comps.iterrows():
        cells = []
        for c in display_cols:
            val = row.get(c, "")
            if c == outcome_col and pd.notna(val):
                try:
                    val = f"{float(val)*100:.1f}%"
                except (TypeError, ValueError):
                    pass
            elif c == "_distance" and pd.notna(val):
                try:
                    val = f"{float(val):.2f}"
                except (TypeError, ValueError):
                    pass
            elif c == "trade_grade":
                color = "#3fb950" if val == "A" else ("#e3b341" if val == "B" else "#f85149")
                val = f'<span style="color: {color}; font-weight: bold;">{val}</span>'
            cells.append(f'<td style="padding: 4px 8px; border: 1px solid #30363d;">{val}</td>')
        rows.append(f'<tr>{"".join(cells)}</tr>')

    return (
        '<div style="margin: 12px 0;">'
        '<h4 style="color: #c9d1d9; margin: 0 0 6px 0;">Historical Comps</h4>'
        + summary
        + '<table style="border-collapse: collapse; font-size: 0.85em; color: #c9d1d9; width: 100%;">'
        f'<tr style="background-color: #21262d;">{header}</tr>'
        + "\n".join(rows)
        + '</table></div>'
    )


def build_priority_ticker_html(item: Dict) -> str:
    """Build full HTML section for one priority ticker."""
    ticker = item["ticker"]
    bucket = item["bucket"]
    rec = item["rec"]
    narrative = item.get("narrative", "")
    score_html = item.get("score_html", "")
    upgrade_html = item.get("upgrade_html", "")
    comps_html = item.get("comps_html", "")
    exit_html = item.get("exit_html", "")
    intensity_html = item.get("intensity_html", "")
    chart_data_uri = item.get("chart_data_uri", "")

    # Rec color
    rec_colors = {"GO": "#3fb950", "CAUTION": "#e3b341", "NO-GO": "#f85149"}
    rec_color = rec_colors.get(rec, "#8b949e")
    bucket_colors = {"reversal": "#58a6ff", "bounce": "#3fb950"}
    bucket_color = bucket_colors.get(bucket, "#8b949e")

    lines = [
        f'<div style="border-top: 3px solid {rec_color}; margin-top: 28px; padding-top: 10px;">',
        f'<h2 style="margin: 0 0 4px 0; color: #f0f6fc;">{ticker} '
        f'<span style="color: {rec_color}; font-size: 0.8em;">{rec}</span> '
        f'<span style="color: {bucket_color}; font-size: 0.65em; font-weight: normal;">{bucket.upper()}</span></h2>',
        f'</div>',
    ]

    # LLM narrative callout
    if narrative:
        lines.append(
            f'<div style="background-color: #1c2333; border-left: 4px solid {rec_color}; '
            f'padding: 10px 14px; margin: 8px 0; border-radius: 0 6px 6px 0; '
            f'font-size: 0.95em; color: #e6edf3; line-height: 1.5;">'
            f'{narrative}</div>'
        )

    # Score checklist
    if score_html:
        lines.append(score_html)

    # Intensity (bounce only)
    if intensity_html:
        lines.append(intensity_html)

    # Upgrade table (CAUTION only)
    if upgrade_html:
        lines.append(upgrade_html)

    # Historical comps
    if comps_html:
        lines.append(comps_html)

    # Exit targets
    if exit_html:
        lines.append(exit_html)

    # Chart
    if chart_data_uri:
        lines.append(
            f'<img src="{chart_data_uri}" alt="{ticker} chart" '
            f'style="max-width:800px; display:block; margin-top:10px; border-radius:4px;">'
        )

    return "\n".join(lines)


def _build_summary_table(priority_list: List[Dict]) -> str:
    """Quick at-a-glance table of all priority tickers."""
    if not priority_list:
        return ""

    rows = []
    for item in priority_list:
        rec = item["rec"]
        color = {"GO": "#3fb950", "CAUTION": "#e3b341"}.get(rec, "#8b949e")
        bucket = item["bucket"].upper()
        score_str = item.get("score_str", "")
        rows.append(
            f'<tr>'
            f'<td style="padding: 4px 10px; border: 1px solid #30363d; font-weight: bold;">{item["ticker"]}</td>'
            f'<td style="padding: 4px 10px; border: 1px solid #30363d; color: {color}; font-weight: bold;">{rec}</td>'
            f'<td style="padding: 4px 10px; border: 1px solid #30363d;">{bucket}</td>'
            f'<td style="padding: 4px 10px; border: 1px solid #30363d;">{score_str}</td>'
            f'</tr>'
        )

    return (
        '<table style="border-collapse: collapse; font-size: 0.9em; color: #c9d1d9; margin: 10px 0;">'
        '<tr style="background-color: #21262d;">'
        '<th style="padding: 4px 10px; border: 1px solid #30363d;">Ticker</th>'
        '<th style="padding: 4px 10px; border: 1px solid #30363d;">Rec</th>'
        '<th style="padding: 4px 10px; border: 1px solid #30363d;">Bucket</th>'
        '<th style="padding: 4px 10px; border: 1px solid #30363d;">Score</th>'
        '</tr>'
        + "\n".join(rows)
        + '</table>'
    )


def build_priority_report_html(priority_list: List[Dict], ticker_html_map: Dict[str, str]) -> str:
    """Assemble the full priority report HTML."""
    now = datetime.datetime.now()
    go_count = sum(1 for p in priority_list if p["rec"] == "GO")
    caution_count = sum(1 for p in priority_list if p["rec"] == "CAUTION")

    header = (
        f'<div style="background-color: #21262d; color: #ffffff; padding: 12px 16px; '
        f'border-radius: 6px 6px 0 0; margin-bottom: 16px; border-bottom: 2px solid #30363d;">'
        f'<h1 style="margin: 0; font-size: 1.4em;">Priority Report</h1>'
        f'<div style="font-size: 0.9em; color: #8b949e; margin-top: 4px;">'
        f'{now.strftime("%A, %B %d %Y  %I:%M %p")} ET &nbsp;|&nbsp; '
        f'<span style="color: #3fb950; font-weight: bold;">{go_count} GO</span>, '
        f'<span style="color: #e3b341; font-weight: bold;">{caution_count} CAUTION</span>'
        f'</div></div>'
    )

    summary = _build_summary_table(priority_list)

    # Ticker sections in priority order (GO first, then CAUTION; within each by score desc)
    sections = []
    for item in priority_list:
        html = ticker_html_map.get(item["ticker"], "")
        if html:
            sections.append(html)

    body = header + summary + "\n".join(sections)

    return (
        '<div style="font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', '
        'Arial, sans-serif; max-width: 860px; margin: 0 auto; color: #c9d1d9; '
        'background-color: #161b22; font-size: 14px; line-height: 1.5; padding: 16px;">'
        + body
        + '</div>'
    )


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def generate_priority_report() -> str:
    """Generate and send the priority report. Returns HTML string."""
    timings = {}
    t_total = time.time()

    watchlist = ss.watchlist
    charts_dir = "charts"
    Path(charts_dir).mkdir(exist_ok=True)

    cache = gr.ReportCache()
    cache.install()

    try:
        # === Phase 1: Screener metrics ===
        t0 = time.time()
        print("Phase 1: Collecting screener metrics...")
        all_data = ss.get_all_stocks_data(watchlist)
        timings["screener_metrics"] = time.time() - t0

        # === Phase 2: Route + Score all tickers ===
        t0 = time.time()
        today = datetime.datetime.now().strftime("%Y-%m-%d")

        print("Phase 2: Fetching pretrade metrics (parallel)...")
        pretrade_metrics_all: Dict[str, Dict] = {}
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futs = {executor.submit(gr.get_pretrade_metrics, t, today): t for t in watchlist}
            for f in as_completed(futs):
                tk = futs[f]
                try:
                    pretrade_metrics_all[tk] = f.result()
                except Exception as e:
                    log.warning(f"{tk}: pretrade metrics failed – {e}")
                    pretrade_metrics_all[tk] = {}

        # Route tickers
        bucket_map: Dict[str, str] = {}
        for ticker in watchlist:
            td = all_data.get(ticker, {})
            bucket, _ = gr.route_playbook(
                td.get("pct_data", {}) or {},
                td.get("mav_data", {}) or {},
            )
            bucket_map[ticker] = bucket

        # Fetch bounce metrics for bounce tickers
        print("Fetching bounce metrics (parallel)...")
        bounce_tickers = [t for t in watchlist if bucket_map.get(t) == "bounce"]
        bounce_metrics_all: Dict[str, Dict] = {}
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futs = {executor.submit(fetch_bounce_metrics, t, today): t for t in bounce_tickers}
            for f in as_completed(futs):
                tk = futs[f]
                try:
                    bounce_metrics_all[tk] = f.result()
                except Exception as e:
                    log.warning(f"{tk}: bounce metrics failed – {e}")

        # Score all tickers and collect recommendations
        print("Scoring all tickers...")
        scored: List[Dict] = []
        for ticker in watchlist:
            bucket = bucket_map[ticker]
            cap = gr.get_ticker_cap(ticker)

            if bucket == "reversal":
                pm = pretrade_metrics_all.get(ticker, {})
                score_result = gr.score_pretrade_setup(ticker, pm, cap=cap)
                rec = score_result["recommendation"]
                scored.append({
                    "ticker": ticker,
                    "bucket": bucket,
                    "cap": cap,
                    "rec": rec,
                    "score_result": score_result,
                    "metrics": pm,
                    "score_str": f"{score_result['score']}/{score_result['max_score']}",
                })
            elif bucket == "bounce":
                bm = bounce_metrics_all.get(ticker, {})
                if bm:
                    td = all_data.get(ticker, {})
                    mav_data = td.get("mav_data", {})
                    if bm.get("pct_from_200mav") is None and mav_data.get("pct_from_200mav") is not None:
                        bm["pct_from_200mav"] = mav_data["pct_from_200mav"]
                    checker = BouncePretrade()
                    bounce_result = checker.validate(ticker, bm, cap=cap)
                    rec = bounce_result.recommendation
                    scored.append({
                        "ticker": ticker,
                        "bucket": bucket,
                        "cap": cap,
                        "rec": rec,
                        "score_result": bounce_result,
                        "metrics": bm,
                        "score_str": f"{bounce_result.score}/{bounce_result.max_score}",
                    })
                else:
                    scored.append({
                        "ticker": ticker,
                        "bucket": bucket,
                        "cap": cap,
                        "rec": "NO-GO",
                        "score_result": None,
                        "metrics": {},
                        "score_str": "N/A",
                    })

        timings["routing_scoring"] = time.time() - t0

        # === Phase 3: Filter → keep only GO + CAUTION ===
        priority = [s for s in scored if s["rec"] in ("GO", "CAUTION")]
        # Sort: GO first, then CAUTION; within each group by score descending
        priority.sort(key=lambda x: (0 if x["rec"] == "GO" else 1, x["score_str"]), reverse=False)

        go_ct = sum(1 for p in priority if p["rec"] == "GO")
        cau_ct = sum(1 for p in priority if p["rec"] == "CAUTION")
        print(f"Phase 3: {len(priority)} priority tickers ({go_ct} GO, {cau_ct} CAUTION) from {len(scored)} total")

        if not priority:
            print("No GO or CAUTION tickers found. Sending empty report.")
            html = build_priority_report_html([], {})
            _send_report(html, 0, 0)
            return html

        # === Phase 4: Deep analysis on priority tickers ===
        t0 = time.time()

        # 4a) Historical comps
        print("Phase 4a: Computing historical comps...")
        comps_map: Dict[str, pd.DataFrame] = {}
        for item in priority:
            ticker = item["ticker"]
            bucket = item["bucket"]
            metrics = item["metrics"]
            cap = item["cap"]

            if bucket == "bounce":
                comps = find_historical_comps(metrics, _BOUNCE_DF, BOUNCE_COMP_COLUMNS, cap)
            else:
                comps = find_historical_comps(metrics, _REVERSAL_DF, REVERSAL_COMP_COLUMNS, cap)
            comps_map[ticker] = comps

        # 4b) Upgrade analysis (CAUTION only)
        print("Phase 4b: Computing upgrade thresholds...")
        upgrade_map: Dict[str, List[Dict]] = {}
        for item in priority:
            if item["rec"] == "CAUTION" and item["score_result"] is not None:
                upgrade_map[item["ticker"]] = compute_upgrade_thresholds(
                    item["ticker"], item["bucket"], item["score_result"], item["metrics"]
                )

        # 4c) LLM narratives (concurrent)
        print("Phase 4c: Generating LLM narratives...")
        narrative_map: Dict[str, str] = {}

        async def _gather_narratives():
            tasks = []
            tickers = []
            for item in priority:
                tickers.append(item["ticker"])
                tasks.append(
                    generate_llm_narrative(
                        item["ticker"],
                        item["bucket"],
                        item["rec"],
                        item["metrics"],
                        comps_map.get(item["ticker"], pd.DataFrame()),
                        upgrade_map.get(item["ticker"], []),
                    )
                )
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for tk, res in zip(tickers, results):
                if isinstance(res, Exception):
                    log.warning(f"Narrative failed for {tk}: {res}")
                    narrative_map[tk] = _fallback_narrative(tk, "", "")
                else:
                    narrative_map[tk] = res

        asyncio.run(_gather_narratives())

        # 4d) Build score HTML, exit targets, charts
        print("Phase 4d: Building HTML sections + charts...")
        ticker_html_map: Dict[str, str] = {}

        for item in priority:
            ticker = item["ticker"]
            bucket = item["bucket"]
            cap = item["cap"]
            metrics = item["metrics"]
            score_result = item["score_result"]

            # Score HTML
            score_html = ""
            intensity_html = ""
            if bucket == "reversal" and score_result:
                score_html = gr.format_pretrade_score_html(score_result)
            elif bucket == "bounce" and score_result:
                score_html = gr.format_bounce_score_html(score_result, bounce_metrics=metrics)
                # Bounce intensity
                setup_type, _ = classify_stock(metrics)
                ref = gr.BOUNCE_DF_WEAK if setup_type == "GapFade_weakstock" else gr.BOUNCE_DF_STRONG
                intensity = gr.compute_bounce_intensity(metrics, ref_df=ref)
                intensity_html = gr.format_bounce_intensity_html(intensity)

            # Exit targets
            exit_html = ""
            exit_data = gr.get_exit_target_data(ticker, today, prefer_open=(bucket == "bounce"))
            if exit_data.get("open_price") and exit_data.get("atr"):
                if bucket == "reversal":
                    targets = calculate_exit_targets(
                        cap=cap,
                        entry_price=exit_data["open_price"],
                        atr=exit_data["atr"],
                        prior_close=exit_data.get("prior_close"),
                        prior_low=exit_data.get("prior_low"),
                        ema_4=exit_data.get("ema_4"),
                    )
                    exit_html = "<strong>Target Price Levels:</strong>" + format_exit_targets_html(targets)
                else:
                    bounce_targets = calculate_bounce_exit_targets(
                        cap=cap,
                        entry_price=exit_data["open_price"],
                        atr=exit_data["atr"],
                        prior_close=exit_data.get("prior_close"),
                        prior_high=exit_data.get("prior_high"),
                    )
                    bounce_targets["entry_price_source"] = exit_data.get("open_price_source")
                    exit_html = "<strong>Bounce Target Levels:</strong>" + format_bounce_exit_targets_html(bounce_targets)

            # Upgrade table
            upgrade_html = ""
            if item["rec"] == "CAUTION" and ticker in upgrade_map:
                upgrade_html = _build_upgrade_table_html(upgrade_map[ticker])

            # Comps table
            comps_html = _build_comps_table_html(comps_map.get(ticker, pd.DataFrame()), bucket)

            # Chart (sequential — mplfinance not thread-safe)
            chart_data_uri = ""
            chart_hlines = []
            if bucket == "reversal" and exit_data.get("open_price") and exit_data.get("atr"):
                op = exit_data["open_price"]
                a = exit_data["atr"]
                chart_hlines = [
                    (op, "blue", f"Open ${op:.2f}"),
                    (op - 1.0 * a, "orange", f"-1 ATR ${op - 1.0*a:.2f}"),
                    (op - 2.0 * a, "red", f"-2 ATR ${op - 2.0*a:.2f}"),
                ]
                pc = exit_data.get("prior_close")
                if pc and pc < op:
                    chart_hlines.append((pc, "green", f"Prior Close ${pc:.2f}"))

            try:
                chart_path = Path(create_daily_chart(ticker, output_dir=charts_dir, extra_hlines=chart_hlines or None))
                chart_data_uri = gr._png_to_data_uri(chart_path)
            except Exception as e:
                log.warning(f"Chart failed for {ticker}: {e}")

            # Assemble ticker HTML
            ticker_html_map[ticker] = build_priority_ticker_html({
                "ticker": ticker,
                "bucket": bucket,
                "rec": item["rec"],
                "narrative": narrative_map.get(ticker, ""),
                "score_html": score_html,
                "intensity_html": intensity_html,
                "upgrade_html": upgrade_html,
                "comps_html": comps_html,
                "exit_html": exit_html,
                "chart_data_uri": chart_data_uri,
            })

        timings["deep_analysis"] = time.time() - t0

        # === Phase 5: Build HTML + send email ===
        t0 = time.time()
        html_report = build_priority_report_html(priority, ticker_html_map)
        _send_report(html_report, go_ct, cau_ct)
        timings["email"] = time.time() - t0

    finally:
        cache.uninstall()

    timings["total"] = time.time() - t_total

    print("\n" + "=" * 60)
    print("PRIORITY REPORT TIMING")
    print("=" * 60)
    for phase, elapsed in timings.items():
        print(f"  {phase:30s} {elapsed:6.1f}s")
    print("=" * 60)

    return html_report


def _send_report(html: str, go_count: int, caution_count: int):
    """Send the priority report email."""
    date_str = datetime.datetime.now().strftime("%m/%d/%Y")
    subject = f"Priority Report — {go_count} GO, {caution_count} CAUTION | {date_str}"
    try:
        send_email(
            to_email="zmburr@gmail.com",
            subject=subject,
            body=html,
            is_html=True,
        )
        print(f"Email sent: {subject}")
    except Exception as e:
        log.error(f"Failed to send email: {e}")


if __name__ == "__main__":
    generate_priority_report()
    cleanup_charts()
    print("Priority report generated and sent.")
