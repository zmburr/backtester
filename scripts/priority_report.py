"""Priority Report — surfaces only GO & CAUTION trades with deeper analysis.

Filters the full watchlist down to high-conviction setups, then adds:
  1. Historical comps (z-score Euclidean distance)
  2. Upgrade threshold analysis (CAUTION → what flips to GO)
  3. Charts + exit targets

Sent via email to zmburr@gmail.com morning + evening.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import json
import datetime
import time
import logging
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

def _compute_comp_intensity(row: pd.Series, bucket: str, cap: str) -> Optional[float]:
    """Score a single historical comp row 0-100 using the same intensity function
    that's applied to live tickers. Returns None if metrics are insufficient."""
    try:
        if bucket == "reversal":
            metrics = {
                'atr_pct':             row.get('atr_pct'),
                'pct_from_9ema':       row.get('pct_from_9ema'),
                'pct_change_3':        row.get('pct_change_3'),
                'gap_pct':             row.get('gap_pct'),
                'prior_day_range_atr': row.get('prior_day_range_atr'),
                'rvol_score':          row.get('rvol_score'),
                'pct_from_50mav':      row.get('pct_from_50mav'),
            }
            if pd.isna(metrics.get('atr_pct')) or metrics.get('atr_pct') in (None, 0):
                return None
            result = gr.compute_reversal_intensity(metrics, cap=cap or row.get('cap'))
            return result.get('composite')
        elif bucket == "bounce":
            row_dict = row.to_dict()
            try:
                setup_type, _ = classify_stock(row_dict)
            except Exception:
                setup_type = 'GapFade_strongstock'
            ref = gr.BOUNCE_DF_WEAK if setup_type == 'GapFade_weakstock' else gr.BOUNCE_DF_STRONG
            result = gr.compute_bounce_intensity(row_dict, ref_df=ref)
            return result.get('composite')
    except Exception as e:
        log.debug(f"Comp intensity compute failed: {e}")
    return None


def _coerce_comp_date(raw_date) -> Optional[str]:
    """Normalize a CSV date (e.g. '4/22/2026' or '2024-03-15') to 'YYYY-MM-DD'."""
    if raw_date is None or (isinstance(raw_date, float) and pd.isna(raw_date)):
        return None
    s = str(raw_date).strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"):
        try:
            return datetime.datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


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
    if "_intensity" in comps.columns:
        display_cols.append("_intensity")
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
            elif c == "_intensity":
                try:
                    s = float(val)
                    color = "#3fb950" if s >= 50 else ("#e3b341" if s >= 25 else "#f85149")
                    val = f'<span style="color: {color}; font-weight: bold;">{s:.0f}</span>'
                except (TypeError, ValueError):
                    val = '<span style="color: #6e7681;">—</span>'
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


def _build_comparables_addendum_html(
    priority_list: List[Dict],
    comps_map: Dict[str, pd.DataFrame],
    comp_chart_cids: Dict[str, str],
    n_per_ticker: int = 3,
) -> str:
    """Render the bottom-of-report addendum: top comps for each priority ticker
    with their intensity score and a 1-year chart ending on the comp's date.

    `comp_chart_cids` is keyed by `(comp_ticker, comp_date_iso)` -> cid string.
    """
    if not priority_list:
        return ""

    blocks = []
    for item in priority_list:
        ticker = item["ticker"]
        bucket = item["bucket"]
        rec = item["rec"]
        comps = comps_map.get(ticker, pd.DataFrame())
        if comps is None or comps.empty:
            continue

        # Top N comps by similarity (smallest distance)
        if "_distance" in comps.columns:
            top = comps.nsmallest(n_per_ticker, "_distance")
        else:
            top = comps.head(n_per_ticker)
        if top.empty:
            continue

        outcome_col = "bounce_open_high_pct" if bucket == "bounce" else "reversal_open_low_pct"
        rec_color = {"GO": "#3fb950", "CAUTION": "#e3b341"}.get(rec, "#8b949e")

        cards = []
        for _, row in top.iterrows():
            comp_ticker = str(row.get("ticker", "?")).upper()
            comp_date_raw = row.get("date", "")
            comp_date_iso = _coerce_comp_date(comp_date_raw)
            grade = row.get("trade_grade", "")
            grade_color = "#3fb950" if grade == "A" else ("#e3b341" if grade == "B" else "#f85149")
            cap_str = row.get("cap", "")
            intensity = row.get("_intensity")
            outcome = row.get(outcome_col)

            # Intensity bar: filled width proportional to score
            if pd.notna(intensity):
                ipct = max(0, min(100, float(intensity)))
                ibar_color = "#3fb950" if ipct >= 50 else ("#e3b341" if ipct >= 25 else "#f85149")
                intensity_bar = (
                    f'<div style="display:inline-block; width:120px; height:8px; background:#21262d; '
                    f'border-radius:4px; vertical-align:middle; margin-left:6px;">'
                    f'<div style="width:{ipct:.0f}%; height:100%; background:{ibar_color}; border-radius:4px;"></div>'
                    f'</div>'
                    f'<span style="margin-left:6px; color:{ibar_color}; font-weight:bold;">{ipct:.0f}/100</span>'
                )
            else:
                intensity_bar = '<span style="color:#6e7681;">— intensity unavailable —</span>'

            outcome_str = ""
            if pd.notna(outcome):
                try:
                    outcome_str = f' <span style="color:#8b949e;">| outcome:</span> <strong>{float(outcome)*100:+.1f}%</strong>'
                except (TypeError, ValueError):
                    pass

            cid_key = f"{comp_ticker}_{comp_date_iso}" if comp_date_iso else f"{comp_ticker}_unknown"
            cid = comp_chart_cids.get(cid_key)
            chart_block = (
                f'<img src="cid:{cid}" alt="{comp_ticker} {comp_date_iso}" '
                f'style="max-width:760px; display:block; margin-top:6px; border-radius:4px;">'
                if cid else
                '<div style="color:#6e7681; font-size:0.85em; padding:6px 0;">[chart unavailable]</div>'
            )

            cards.append(
                '<div style="border-left: 3px solid #30363d; padding: 6px 12px; margin: 10px 0;">'
                f'<div style="font-size:0.95em;">'
                f'<strong style="color:#f0f6fc;">{comp_ticker}</strong> '
                f'<span style="color:#8b949e;">{comp_date_raw}</span> '
                f'<span style="color:#8b949e;">({cap_str})</span> '
                f'<span style="color:{grade_color}; font-weight:bold;">grade {grade}</span>'
                f'{outcome_str}'
                f'</div>'
                f'<div style="margin-top:4px; font-size:0.9em;"><span style="color:#8b949e;">Intensity:</span> {intensity_bar}</div>'
                f'{chart_block}'
                '</div>'
            )

        # Live ticker's own intensity for visual comparison vs comps
        live_intensity = item.get("live_intensity")
        if pd.notna(live_intensity) and live_intensity is not None:
            lpct = max(0, min(100, float(live_intensity)))
            lcolor = "#3fb950" if lpct >= 50 else ("#e3b341" if lpct >= 25 else "#f85149")
            live_intensity_str = f'<span style="color:{lcolor}; font-weight:bold;">{lpct:.0f}/100</span>'
        else:
            live_intensity_str = '<span style="color:#6e7681;">N/A</span>'

        blocks.append(
            f'<div style="margin: 24px 0; border-top: 1px solid #30363d; padding-top: 12px;">'
            f'<h3 style="margin: 0 0 4px 0; color:#f0f6fc;">'
            f'{ticker} <span style="color:{rec_color}; font-size:0.8em;">{rec}</span> '
            f'<span style="color:#8b949e; font-size:0.75em; font-weight:normal;">'
            f'— live intensity {live_intensity_str}, top {len(cards)} comps below'
            f'</span></h3>'
            + "".join(cards)
            + '</div>'
        )

    if not blocks:
        return ""

    return (
        '<div style="margin-top: 40px; padding-top: 16px; border-top: 3px solid #30363d;">'
        '<h2 style="color:#f0f6fc; margin:0 0 6px 0;">Comparables Addendum</h2>'
        '<p style="color:#8b949e; font-size:0.9em; margin: 0 0 16px 0;">'
        'Top similar historical trades for each priority ticker, with that comp\'s '
        'intensity score and a 1-year daily chart ending on the trade date. Use this '
        'to calibrate trust in the live intensity score against trades whose outcome '
        'is already known.'
        '</p>'
        + "\n".join(blocks)
        + '</div>'
    )


def build_priority_ticker_html(item: Dict) -> str:
    """Build full HTML section for one priority ticker."""
    ticker = item["ticker"]
    bucket = item["bucket"]
    rec = item["rec"]
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

    # Off-archetype glyph (reversals only — flag is set by score_pretrade_setup)
    archetype_glyph = ""
    sr = item.get("score_result")
    if bucket == "reversal" and isinstance(sr, dict) and sr.get("archetype_passed") is False:
        archetype_glyph = ' <span title="Off archetype: not near highs" style="color: #e3b341; font-size: 0.75em;">⚠ OFF_ARCHETYPE</span>'

    lines = [
        f'<div style="border-top: 3px solid {rec_color}; margin-top: 28px; padding-top: 10px;">',
        f'<h2 style="margin: 0 0 4px 0; color: #f0f6fc;">{ticker} '
        f'<span style="color: {rec_color}; font-size: 0.8em;">{rec}</span>{archetype_glyph} '
        f'<span style="color: {bucket_color}; font-size: 0.65em; font-weight: normal;">{bucket.upper()}</span></h2>',
        f'</div>',
    ]

    # Score checklist
    if score_html:
        lines.append(score_html)

    # Intensity (bounce + reversal)
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


def build_priority_report_html(priority_list: List[Dict], ticker_html_map: Dict[str, str],
                               addendum_html: str = "") -> str:
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

    body = header + summary + "\n".join(sections) + (addendum_html or "")

    return (
        '<div style="font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', '
        'Arial, sans-serif; max-width: 860px; margin: 0 auto; color: #c9d1d9; '
        'background-color: #161b22; font-size: 14px; line-height: 1.5; padding: 16px;">'
        + body
        + '</div>'
    )


def _numeric_score_parts(item: Dict) -> Tuple[float, float]:
    """Return (score, max_score) for sorting; falls back to score_str."""
    score_result = item.get("score_result")

    if isinstance(score_result, dict):
        try:
            return float(score_result.get("score", -1)), float(score_result.get("max_score", 0))
        except (TypeError, ValueError):
            pass

    if score_result is not None:
        try:
            return float(score_result.score), float(score_result.max_score)
        except (AttributeError, TypeError, ValueError):
            pass

    score_str = str(item.get("score_str", ""))
    if "/" in score_str:
        raw_score, raw_max = score_str.split("/", 1)
        try:
            return float(raw_score), float(raw_max)
        except ValueError:
            pass

    return -1.0, 0.0


def _priority_sort_key(item: Dict) -> Tuple[int, float, float, str]:
    score, max_score = _numeric_score_parts(item)
    score_ratio = score / max_score if max_score else -1.0
    rec_rank = 0 if item.get("rec") == "GO" else 1
    return rec_rank, -score_ratio, -score, item.get("ticker", "")


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

        # Route tickers — uses the new 4-branch route_playbook (bounce / breakout / reversal)
        bucket_map: Dict[str, str] = {}
        ambiguous_map: Dict[str, bool] = {}
        for ticker in watchlist:
            td = all_data.get(ticker, {})
            bucket, _, is_ambig = gr.route_playbook(
                td.get("pct_data", {}) or {},
                td.get("mav_data", {}) or {},
                ticker=ticker,
                pretrade_metrics=pretrade_metrics_all.get(ticker, {}),
                breakout_watchlist=getattr(gr, 'BREAKOUT_WATCHLIST', []),
            )
            bucket_map[ticker] = bucket
            ambiguous_map[ticker] = is_ambig

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

            elif bucket == "breakout":
                # Breakout setup — score against intensity tier (FULL_SIZE / REDUCED_SIZE / AVOID).
                # Map tiers to existing GO/CAUTION/NO-GO so the priority filter still works.
                from analyzers.breakout_scorer import BreakoutPretrade
                pm = pretrade_metrics_all.get(ticker, {})
                td = all_data.get(ticker, {})
                mav_data = td.get("mav_data", {}) or {}
                bm = {
                    't': 1,
                    'pct_from_9ema':            pm.get('pct_from_9ema'),
                    'pct_from_50mav':           mav_data.get('pct_from_50mav'),
                    'pct_to_52wk_high':         pm.get('pct_to_52wk_high'),
                    'atr_pct':                  pm.get('atr_pct'),
                    'gap_pct':                  pm.get('gap_pct'),
                    'percent_of_premarket_vol': pm.get('premarket_rvol'),
                }
                try:
                    checker = BreakoutPretrade()
                    br_result = checker.validate(ticker, bm)
                    # Map intensity tier to GO/CAUTION/NO-GO so existing priority filter works:
                    #   FULL_SIZE -> GO, REDUCED_SIZE -> CAUTION, else -> NO-GO
                    rec_map = {'FULL_SIZE': 'GO', 'REDUCED_SIZE': 'CAUTION'}
                    rec = rec_map.get(br_result.recommendation, 'NO-GO')
                    scored.append({
                        "ticker": ticker,
                        "bucket": bucket,
                        "cap": cap,
                        "rec": rec,
                        "score_result": br_result,
                        "metrics": bm,
                        "score_str": f"{br_result.score}/{br_result.max_score} [{br_result.recommendation}]",
                    })
                except Exception as e:
                    log.warning(f"{ticker}: breakout scoring failed - {e}")
                    scored.append({
                        "ticker": ticker, "bucket": bucket, "cap": cap, "rec": "NO-GO",
                        "score_result": None, "metrics": bm, "score_str": "N/A",
                    })

        timings["routing_scoring"] = time.time() - t0

        # === Phase 3: Filter → keep only GO + CAUTION (excluding breakout bucket) ===
        priority = [s for s in scored if s["rec"] in ("GO", "CAUTION") and s["bucket"] != "breakout"]
        # Sort: GO first, then CAUTION; within each group by score descending
        priority.sort(key=_priority_sort_key)

        excluded_breakouts = sum(1 for s in scored if s["rec"] in ("GO", "CAUTION") and s["bucket"] == "breakout")
        go_ct = sum(1 for p in priority if p["rec"] == "GO")
        cau_ct = sum(1 for p in priority if p["rec"] == "CAUTION")
        print(f"Phase 3: {len(priority)} priority tickers ({go_ct} GO, {cau_ct} CAUTION) from {len(scored)} total"
              f"{f' [{excluded_breakouts} breakout tickers excluded]' if excluded_breakouts else ''}")

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
            elif bucket == "breakout":
                # For now, breakouts borrow the reversal universe (same "above MAs" geometry).
                # When breakout_data accumulates more rows, swap to its own DF.
                comps = find_historical_comps(metrics, _REVERSAL_DF, REVERSAL_COMP_COLUMNS, cap)
            else:
                comps = find_historical_comps(metrics, _REVERSAL_DF, REVERSAL_COMP_COLUMNS, cap)
            # Score each comp with the same intensity function applied to live tickers,
            # so the comps table + addendum can show calibration vs realized P&L.
            if comps is not None and not comps.empty:
                comps = comps.copy()
                comps["_intensity"] = comps.apply(
                    lambda r: _compute_comp_intensity(r, bucket, cap), axis=1
                )
            comps_map[ticker] = comps

        # 4b) Upgrade analysis (CAUTION only)
        print("Phase 4b: Computing upgrade thresholds...")
        upgrade_map: Dict[str, List[Dict]] = {}
        for item in priority:
            if item["rec"] == "CAUTION" and item["score_result"] is not None:
                upgrade_map[item["ticker"]] = compute_upgrade_thresholds(
                    item["ticker"], item["bucket"], item["score_result"], item["metrics"]
                )

        # 4d) Build score HTML, exit targets, charts
        print("Phase 4d: Building HTML sections + charts...")
        ticker_html_map: Dict[str, str] = {}
        chart_images: Dict[str, str] = {}  # cid -> file path

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
                # Reversal intensity (cap-stratified percentile vs grade-A reference)
                rev_mav = all_data.get(ticker, {}).get("mav_data", {}) or {}
                rev_intensity_metrics = {
                    'atr_pct':             metrics.get('atr_pct'),
                    'pct_from_9ema':       metrics.get('pct_from_9ema'),
                    'pct_change_3':        metrics.get('pct_change_3'),
                    'gap_pct':             metrics.get('gap_pct'),
                    'prior_day_range_atr': metrics.get('prior_day_range_atr'),
                    'rvol_score':          metrics.get('prior_day_rvol'),
                    'pct_from_50mav':      rev_mav.get('pct_from_50mav'),
                }
                rev_intensity = gr.compute_reversal_intensity(rev_intensity_metrics, cap=cap)
                intensity_html = gr.format_reversal_intensity_html(rev_intensity)
                item["live_intensity"] = rev_intensity.get("composite")
            elif bucket == "bounce" and score_result:
                score_html = gr.format_bounce_score_html(score_result, bounce_metrics=metrics)
                # Bounce intensity
                setup_type, _ = classify_stock(metrics)
                ref = gr.BOUNCE_DF_WEAK if setup_type == "GapFade_weakstock" else gr.BOUNCE_DF_STRONG
                intensity = gr.compute_bounce_intensity(metrics, ref_df=ref)
                intensity_html = gr.format_bounce_intensity_html(intensity)
                item["live_intensity"] = intensity.get("composite")

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

            chart_cid = f"chart_{ticker}"
            try:
                chart_path = Path(create_daily_chart(ticker, output_dir=charts_dir, extra_hlines=chart_hlines or None))
                chart_images[chart_cid] = str(chart_path)
                chart_data_uri = f"cid:{chart_cid}"
            except Exception as e:
                log.warning(f"Chart failed for {ticker}: {e}")

            # Assemble ticker HTML
            ticker_html_map[ticker] = build_priority_ticker_html({
                "ticker": ticker,
                "bucket": bucket,
                "rec": item["rec"],
                "score_html": score_html,
                "intensity_html": intensity_html,
                "upgrade_html": upgrade_html,
                "comps_html": comps_html,
                "exit_html": exit_html,
                "chart_data_uri": chart_data_uri,
            })

        timings["deep_analysis"] = time.time() - t0

        # === Phase 4.5: Save signals to JSON for Signal Scorecard ===
        _save_signals_to_json(priority, go_ct, cau_ct)

        # === Phase 4.75: Render top-3 comp charts for the addendum ===
        # 1-year daily chart ending on each comp's trade date. Deduped across
        # priority tickers when the same comp shows up twice.
        t0 = time.time()
        comp_chart_cids: Dict[str, str] = {}
        for item in priority:
            ticker = item["ticker"]
            comps = comps_map.get(ticker, pd.DataFrame())
            if comps is None or comps.empty:
                continue
            top = comps.nsmallest(3, "_distance") if "_distance" in comps.columns else comps.head(3)
            for _, row in top.iterrows():
                comp_ticker = str(row.get("ticker", "")).upper()
                comp_date_iso = _coerce_comp_date(row.get("date"))
                if not comp_ticker or not comp_date_iso:
                    continue
                key = f"{comp_ticker}_{comp_date_iso}"
                if key in comp_chart_cids:
                    continue  # already rendered
                cid = f"compchart_{comp_ticker}_{comp_date_iso.replace('-', '')}"
                try:
                    chart_path = create_daily_chart(
                        comp_ticker,
                        output_dir=charts_dir,
                        end_date=comp_date_iso,
                        label=f"comp_{comp_date_iso}",
                    )
                    chart_images[cid] = str(chart_path)
                    comp_chart_cids[key] = cid
                except Exception as e:
                    log.warning(f"Comp chart failed for {comp_ticker} {comp_date_iso}: {e}")
        timings["addendum_charts"] = time.time() - t0
        print(f"Phase 4.75: Rendered {len(comp_chart_cids)} comp charts in {timings['addendum_charts']:.1f}s")

        addendum_html = _build_comparables_addendum_html(priority, comps_map, comp_chart_cids)

        # === Phase 5: Build HTML + send email ===
        t0 = time.time()
        html_report = build_priority_report_html(priority, ticker_html_map, addendum_html=addendum_html)
        _send_report(html_report, go_ct, cau_ct, inline_images=chart_images)
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


_SIGNAL_DIR = _DATA_DIR / "priority_signals"

# Metrics to persist per bucket (safe subset of the full metrics dict)
_REVERSAL_METRIC_KEYS = [
    "pct_from_9ema", "pct_change_3", "gap_pct", "prior_day_range_atr",
    "atr_pct", "prior_day_rvol", "premarket_rvol", "pct_from_50mav",
]
_BOUNCE_METRIC_KEYS = [
    "selloff_total_pct", "pct_off_30d_high", "gap_pct", "pct_change_3",
    "prior_day_range_atr", "pct_off_52wk_high", "atr_pct",
]


def _save_signals_to_json(priority: List[Dict], go_count: int, caution_count: int):
    """Save GO/CAUTION signals to a date-stamped JSON file for the Signal Scorecard."""
    try:
        _SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
        now = datetime.datetime.now()
        session = "morning" if now.hour < 12 else "evening"
        date_str = now.strftime("%Y-%m-%d")

        signals = []
        for item in priority:
            bucket = item["bucket"]
            keys = _REVERSAL_METRIC_KEYS if bucket == "reversal" else _BOUNCE_METRIC_KEYS
            raw = item.get("metrics", {})
            metrics = {}
            for k in keys:
                v = raw.get(k)
                if v is not None:
                    try:
                        metrics[k] = round(float(v), 6)
                    except (TypeError, ValueError):
                        metrics[k] = v

            signal_entry = {
                "ticker": item["ticker"],
                "bucket": bucket,
                "cap": item.get("cap", ""),
                "recommendation": item["rec"],
                "score": item.get("score_str", ""),
                "metrics": metrics,
            }

            # Carry the archetype flag through for scorecard feedback-loop analysis.
            # Only reversals populate this; bounces leave it out.
            if bucket == "reversal":
                sr = item.get("score_result")
                if isinstance(sr, dict):
                    signal_entry["archetype_passed"] = sr.get("archetype_passed")
                    signal_entry["archetype_detail"] = sr.get("archetype_detail")

            signals.append(signal_entry)

        payload = {
            "date": date_str,
            "session": session,
            "generated_at": now.isoformat(),
            "go_count": go_count,
            "caution_count": caution_count,
            "signals": signals,
        }

        out_path = _SIGNAL_DIR / f"{date_str}_{session}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Signals saved: {out_path.name} ({len(signals)} signals)")
    except Exception as e:
        log.warning(f"Failed to save signals JSON (non-fatal): {e}")


def _send_report(html: str, go_count: int, caution_count: int, inline_images=None):
    """Send the priority report email."""
    date_str = datetime.datetime.now().strftime("%m/%d/%Y")
    subject = f"Priority Report — {go_count} GO, {caution_count} CAUTION | {date_str}"
    try:
        send_email(
            to_email="zmburr@gmail.com",
            subject=subject,
            body=html,
            is_html=True,
            inline_images=inline_images,
        )
        print(f"Email sent: {subject}")
    except Exception as e:
        log.error(f"Failed to send email: {e}")


if __name__ == "__main__":
    generate_priority_report()
    cleanup_charts()
    print("Priority report generated and sent.")
