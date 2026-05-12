"""scripts/analyze_945_rule.py — does the 9:45 wait rule help on reversal shorts?

For each trade in data/reversal_data.csv:
  - Fetch 1-min bars for the trade day (premarket + RTH)
  - Build 6 synthetic short-entry strategies and compute realized P&L (entry → close),
    max favorable excursion (best the price went for the short), and max adverse
    (worst it squeezed against the short)
Aggregates the strategies head-to-head and emails an HTML report.

Strategies:
  A. Short at 9:30 open
  B. Short at 9:45 close (current rule)
  C. Short on first PM-low break after open
  D. Short on first negative 5-min close (9:35)
  E. Short on first negative 10-min close (9:40)
  F. Adaptive — earliest of {C, D, E}; fallback to 9:45 if none fire by then
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import datetime as dt
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytz

from data_queries.polygon_queries import get_intraday
from support.config import send_email

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")
MAX_WORKERS = 8


def localize(d: dt.date, t: dt.time) -> dt.datetime:
    return ET.localize(dt.datetime.combine(d, t))


def parse_csv_date(s: str) -> dt.date:
    return dt.datetime.strptime(str(s).strip(), "%m/%d/%Y").date()


def strategy_metrics(entry_price: Optional[float], entry_time: Optional[pd.Timestamp],
                     rth: pd.DataFrame, close_price: float) -> Dict:
    """Return realized short P&L (entry → close), MFE, MAE based on bars after entry_time."""
    if entry_price is None or entry_time is None:
        return {"fired": False, "pnl_pct": None, "mfe_pct": None, "mae_pct": None}
    post = rth[rth.index > entry_time]
    if post.empty:
        return {"fired": False, "pnl_pct": None, "mfe_pct": None, "mae_pct": None}
    post_low = float(post["low"].min())
    post_high = float(post["high"].max())
    return {
        "fired": True,
        "pnl_pct": (entry_price - close_price) / entry_price * 100.0,
        "mfe_pct": (entry_price - post_low) / entry_price * 100.0,
        "mae_pct": (post_high - entry_price) / entry_price * 100.0,
    }


def scale_in_metrics(entries: List, rth: pd.DataFrame, close_price: float) -> Dict:
    """Equal-weighted scale-in: entries is list of (price, time) tuples (None entries dropped).
    MFE/MAE measured vs avg fill price, from FIRST entry onward."""
    fired = [(p, t) for p, t in entries if p is not None and t is not None]
    if not fired:
        return {"fired": False, "pnl_pct": None, "mfe_pct": None, "mae_pct": None, "n_legs": 0}
    avg_entry = float(np.mean([p for p, _ in fired]))
    first_time = min(t for _, t in fired)
    post = rth[rth.index >= first_time]
    if post.empty:
        return {"fired": False, "pnl_pct": None, "mfe_pct": None, "mae_pct": None, "n_legs": 0}
    post_low = float(post["low"].min())
    post_high = float(post["high"].max())
    return {
        "fired": True,
        "pnl_pct": (avg_entry - close_price) / avg_entry * 100.0,
        "mfe_pct": (avg_entry - post_low) / avg_entry * 100.0,
        "mae_pct": (post_high - avg_entry) / avg_entry * 100.0,
        "n_legs": len(fired),
    }


def compute_vwap_series(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP using typical price (H+L+C)/3, weighted by volume.
    Computed from start of df (premarket inclusive) onward."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_pv = (typical * df["volume"]).cumsum()
    cum_v = df["volume"].cumsum().replace(0, np.nan)
    return cum_pv / cum_v


def compute_for_trade(ticker: str, trade_date: dt.date, df: pd.DataFrame,
                       atr_pct: Optional[float] = None) -> Optional[Dict]:
    """Build all strategies for one trade day. Returns None on bad data."""
    if df is None or df.empty:
        return None

    open_dt = localize(trade_date, dt.time(9, 30))
    close_dt = localize(trade_date, dt.time(16, 0))

    pm = df[df.index < open_dt]
    rth = df[(df.index >= open_dt) & (df.index <= close_dt)]
    if rth.empty or len(rth) < 30:
        return None

    open_price = float(rth.iloc[0]["open"])
    close_price = float(rth.iloc[-1]["close"])

    # VWAP using PM+RTH cumulative
    vwap_series = compute_vwap_series(df)
    # VWAP "as of just before the open" — last PM bar's running VWAP
    if not pm.empty:
        vwap_at_open = float(vwap_series.loc[pm.index[-1]])
    else:
        vwap_at_open = float(vwap_series.iloc[0]) if not vwap_series.empty else None

    def bar_close_at(t: dt.time) -> Optional[float]:
        ts = localize(trade_date, t)
        match = rth[rth.index == ts]
        if match.empty:
            return None
        return float(match.iloc[0]["close"])

    price_935 = bar_close_at(dt.time(9, 35))
    price_940 = bar_close_at(dt.time(9, 40))
    price_945 = bar_close_at(dt.time(9, 45))

    # PM low + first regular-session break
    pm_low: Optional[float] = float(pm["low"].min()) if not pm.empty else None
    pm_break_time: Optional[pd.Timestamp] = None
    pm_break_price: Optional[float] = None
    if pm_low is not None:
        below = rth[rth["low"] < pm_low]
        if not below.empty:
            pm_break_time = below.index[0]
            pm_break_price = float(below.iloc[0]["close"])

    first5_red = price_935 is not None and price_935 < open_price
    first10_red = price_940 is not None and price_940 < open_price

    # LOD info (for the where-does-LOD-print analysis)
    lod_idx = rth["low"].idxmin()
    lod_time: pd.Timestamp = lod_idx
    lod_price = float(rth.loc[lod_idx, "low"])

    bar_935_idx = localize(trade_date, dt.time(9, 35)) if first5_red else None
    bar_940_idx = localize(trade_date, dt.time(9, 40)) if first10_red else None
    bar_945_idx = localize(trade_date, dt.time(9, 45)) if price_945 is not None else None

    A = strategy_metrics(open_price, rth.index[0], rth, close_price)
    B = strategy_metrics(price_945, bar_945_idx, rth, close_price)
    C = strategy_metrics(pm_break_price, pm_break_time, rth, close_price)
    D = strategy_metrics(price_935 if first5_red else None, bar_935_idx, rth, close_price)
    E = strategy_metrics(price_940 if first10_red else None, bar_940_idx, rth, close_price)

    # Adaptive: earliest signal among C/D/E; fallback to 9:45
    candidates: List = []
    if pm_break_time is not None:
        candidates.append((pm_break_time, pm_break_price))
    if first5_red and bar_935_idx is not None and price_935 is not None:
        candidates.append((bar_935_idx, price_935))
    if first10_red and bar_940_idx is not None and price_940 is not None:
        candidates.append((bar_940_idx, price_940))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        f_time, f_price = candidates[0]
    else:
        f_time, f_price = bar_945_idx, price_945
    F = strategy_metrics(f_price, f_time, rth, close_price)

    # === Scale-in strategies ===
    open_entry = (open_price, rth.index[0])

    # G. 50% at 9:30 + 50% at 9:45
    G = scale_in_metrics([open_entry, (price_945, bar_945_idx)], rth, close_price)

    # H. 50% at 9:30 + 50% at first signal (PM break / 5-min red / 10-min red), fallback 9:45
    H = scale_in_metrics([open_entry, (f_price, f_time)], rth, close_price)

    # I. 50% at 9:30 + 50% at session HOD (perfect-foresight benchmark — upper bound on scale-in)
    hod_idx = rth["high"].idxmax()
    hod_price = float(rth.loc[hod_idx, "high"])
    hod_time: pd.Timestamp = hod_idx
    I = scale_in_metrics([open_entry, (hod_price, hod_time)], rth, close_price)

    # J. Ladder 33% at 9:30 + 33% at +0.5 ATR + 33% at +1.0 ATR (no fallback for unhit legs)
    J_legs = [open_entry]
    if atr_pct is not None and not np.isnan(atr_pct):
        target_50 = open_price * (1 + 0.5 * atr_pct)
        target_100 = open_price * (1 + 1.0 * atr_pct)
        hits_50 = rth[rth["high"] >= target_50]
        hits_100 = rth[rth["high"] >= target_100]
        if not hits_50.empty:
            J_legs.append((target_50, hits_50.index[0]))
        if not hits_100.empty:
            J_legs.append((target_100, hits_100.index[0]))
    J = scale_in_metrics(J_legs, rth, close_price)

    # K. VWAP entry — if 9:30 open < PM VWAP, enter at open. Else first 9:30+ bar where close < running VWAP.
    opens_below_vwap = (vwap_at_open is not None) and (open_price < vwap_at_open)
    if opens_below_vwap:
        K_entry_price: Optional[float] = open_price
        K_entry_time: Optional[pd.Timestamp] = rth.index[0]
    else:
        rth_vwap = vwap_series.reindex(rth.index)
        below_mask = rth["close"] < rth_vwap
        below_bars = rth[below_mask.fillna(False)]
        if not below_bars.empty:
            K_entry_price = float(below_bars.iloc[0]["close"])
            K_entry_time = below_bars.index[0]
        else:
            K_entry_price = None
            K_entry_time = None
    K = strategy_metrics(K_entry_price, K_entry_time, rth, close_price)

    # L. Confirmation ladder (33/33/33): VWAP break + PM low break + open-fail break
    # Each leg fires only on confirmed downside; legs that don't fire = smaller total size.
    rth_vwap_series = vwap_series.reindex(rth.index)
    v_mask = rth["close"] < rth_vwap_series
    v_bars = rth[v_mask.fillna(False)]
    v_time = v_bars.index[0] if not v_bars.empty else None
    v_price = float(v_bars.iloc[0]["close"]) if not v_bars.empty else None

    if pm_low is not None:
        p_bars = rth[rth["close"] < pm_low]
        p_time = p_bars.index[0] if not p_bars.empty else None
        p_price = float(p_bars.iloc[0]["close"]) if not p_bars.empty else None
    else:
        p_time = None
        p_price = None

    # Open-fail: stock must first close > open, then later close < open
    above_open = rth[rth["close"] > open_price]
    if not above_open.empty:
        rally_time = above_open.index[0]
        after_rally = rth[rth.index > rally_time]
        o_bars = after_rally[after_rally["close"] < open_price]
        o_time = o_bars.index[0] if not o_bars.empty else None
        o_price = float(o_bars.iloc[0]["close"]) if not o_bars.empty else None
    else:
        o_time = None
        o_price = None

    L_legs = []
    if v_price is not None and v_time is not None:
        L_legs.append((v_price, v_time))
    if p_price is not None and p_time is not None:
        L_legs.append((p_price, p_time))
    if o_price is not None and o_time is not None:
        L_legs.append((o_price, o_time))
    L = scale_in_metrics(L_legs, rth, close_price)

    nine_fortyfive = localize(trade_date, dt.time(9, 45))
    time_to_hod_min = (hod_time - rth.index[0]).total_seconds() / 60.0

    return {
        "ticker": ticker,
        "date": trade_date,
        "open_price": open_price,
        "price_945": price_945,
        "pm_low": pm_low,
        "pm_low_broke": pm_break_time is not None,
        "lod_price": lod_price,
        "lod_time": lod_time,
        "lod_before_945": lod_time < nine_fortyfive,
        "lod_minute": lod_time.hour * 60 + lod_time.minute,
        "hod_time": hod_time,
        "time_to_hod_min": time_to_hod_min,
        "close_price": close_price,
        "first5_red": first5_red,
        "first10_red": first10_red,
        "vwap_at_open": vwap_at_open,
        "opens_below_vwap": opens_below_vwap,
        "A_open": A,
        "B_945": B,
        "C_pm_break": C,
        "D_5min_red": D,
        "E_10min_red": E,
        "F_adaptive": F,
        "G_half_945": G,
        "H_half_signal": H,
        "I_half_hod": I,
        "J_ladder_atr": J,
        "K_vwap": K,
        "L_confirm_ladder": L,
    }


def fetch_one(ticker: str, trade_date: dt.date) -> Optional[pd.DataFrame]:
    iso = trade_date.strftime("%Y-%m-%d")
    try:
        return get_intraday(ticker, iso, 1, "minute")
    except Exception as e:
        log.warning(f"fetch failed {ticker} {iso}: {e}")
        return None


def aggregate_strategy(key: str, results: List[Dict]) -> Dict:
    fired = [r[key] for r in results if r[key]["fired"]]
    n_fired = len(fired)
    n_total = len(results)
    if n_fired == 0:
        return {
            "n_fired": 0, "n_total": n_total, "fire_rate": 0.0,
            "avg_pnl": None, "median_pnl": None, "win_rate": None,
            "median_mfe": None, "median_mae": None, "avg_pnl_all": None,
            "mae_p75": None, "mae_p90": None, "mae_p95": None, "mae_max": None,
        }
    pnls = [s["pnl_pct"] for s in fired if s["pnl_pct"] is not None]
    mfes = [s["mfe_pct"] for s in fired if s["mfe_pct"] is not None]
    maes_sorted = sorted([s["mae_pct"] for s in fired if s["mae_pct"] is not None])
    wins = sum(1 for p in pnls if p > 0)

    def pctile(arr: List[float], p: float) -> Optional[float]:
        if not arr:
            return None
        return float(np.percentile(arr, p))

    return {
        "n_fired": n_fired,
        "n_total": n_total,
        "fire_rate": n_fired / n_total * 100.0,
        "avg_pnl": float(np.mean(pnls)) if pnls else None,
        "median_pnl": float(np.median(pnls)) if pnls else None,
        "win_rate": wins / len(pnls) * 100.0 if pnls else None,
        "median_mfe": float(np.median(mfes)) if mfes else None,
        "median_mae": float(np.median(maes_sorted)) if maes_sorted else None,
        "mae_p75": pctile(maes_sorted, 75),
        "mae_p90": pctile(maes_sorted, 90),
        "mae_p95": pctile(maes_sorted, 95),
        "mae_max": maes_sorted[-1] if maes_sorted else None,
        # All-trades attribution: missed = 0 P&L (penalizes signals that don't fire)
        "avg_pnl_all": sum(pnls) / n_total if pnls else None,
    }


def lod_distribution(results: List[Dict]) -> Dict:
    buckets = {
        "9:30-9:35": 0,
        "9:35-9:45": 0,
        "9:45-10:00": 0,
        "10:00-10:30": 0,
        "10:30-11:00": 0,
        "11:00-12:00": 0,
        "12:00-14:00": 0,
        "14:00-16:00": 0,
    }
    for r in results:
        m = r["lod_minute"]
        if m < 9 * 60 + 35:
            buckets["9:30-9:35"] += 1
        elif m < 9 * 60 + 45:
            buckets["9:35-9:45"] += 1
        elif m < 10 * 60:
            buckets["9:45-10:00"] += 1
        elif m < 10 * 60 + 30:
            buckets["10:00-10:30"] += 1
        elif m < 11 * 60:
            buckets["10:30-11:00"] += 1
        elif m < 12 * 60:
            buckets["11:00-12:00"] += 1
        elif m < 14 * 60:
            buckets["12:00-14:00"] += 1
        else:
            buckets["14:00-16:00"] += 1
    return buckets


def build_html(results: List[Dict], agg: Dict) -> str:
    n = len(results)
    lod_buckets = lod_distribution(results)
    lod_before_945 = sum(1 for r in results if r["lod_before_945"])
    pm_low_break = sum(1 for r in results if r["pm_low_broke"])
    first5_red_n = sum(1 for r in results if r["first5_red"])
    first10_red_n = sum(1 for r in results if r["first10_red"])
    opens_below_vwap_n = sum(1 for r in results if r.get("opens_below_vwap"))

    hod_minutes = sorted(r["time_to_hod_min"] for r in results if r.get("time_to_hod_min") is not None)
    hod_p25 = float(np.percentile(hod_minutes, 25)) if hod_minutes else 0.0
    hod_p50 = float(np.percentile(hod_minutes, 50)) if hod_minutes else 0.0
    hod_p75 = float(np.percentile(hod_minutes, 75)) if hod_minutes else 0.0
    hod_p90 = float(np.percentile(hod_minutes, 90)) if hod_minutes else 0.0
    hod_max = float(max(hod_minutes)) if hod_minutes else 0.0

    # Strategy table rows
    rows: List[str] = []
    label_key = [
        ("A", "Short at 9:30 open", "A_open"),
        ("B", "Short at 9:45 close (current rule)", "B_945"),
        ("C", "Short on first PM-low break", "C_pm_break"),
        ("D", "Short on first 5-min red close (9:35)", "D_5min_red"),
        ("E", "Short on first 10-min red close (9:40)", "E_10min_red"),
        ("F", "Adaptive (earliest of C/D/E; 9:45 fallback)", "F_adaptive"),
        ("G", "Scale-in: 50% open + 50% at 9:45", "G_half_945"),
        ("H", "Scale-in: 50% open + 50% at first signal", "H_half_signal"),
        ("I", "Scale-in: 50% open + 50% at HOD (perfect-foresight)", "I_half_hod"),
        ("J", "Ladder: 33% open / 33% +0.5 ATR / 33% +1.0 ATR", "J_ladder_atr"),
        ("K", "VWAP: open if open<PM_VWAP, else first close<VWAP", "K_vwap"),
        ("L", "Confirmation ladder: 33% VWAP-break + 33% PM-low-break + 33% open-fail", "L_confirm_ladder"),
    ]

    def fmt(v, suffix="%", places=2):
        return f"{v:+.{places}f}{suffix}" if v is not None else "n/a"

    def fmt_pos(v, places=0):
        return f"{v:.{places}f}%" if v is not None else "n/a"

    def build_rows(local_agg: Dict, label_key_local: List, highlight_best: bool = True) -> str:
        local_best = max(label_key_local, key=lambda lk: (local_agg[lk[2]]["avg_pnl_all"] or -1e9))[2] if highlight_best else None
        out: List[str] = []
        for letter_l, name_l, key_l in label_key_local:
            a = local_agg[key_l]
            is_best_l = key_l == local_best
            row_bg = "#1c3426" if is_best_l else "transparent"
            pnl_color_l = "#3fb950" if (a["avg_pnl_all"] or 0) > 0 else "#f85149"
            out.append(
                f'<tr style="background-color:{row_bg};">'
                f'<td style="padding:6px 10px; border:1px solid #30363d; font-weight:bold;">{letter_l}</td>'
                f'<td style="padding:6px 10px; border:1px solid #30363d;">{name_l}</td>'
                f'<td style="padding:6px 10px; border:1px solid #30363d;">{a["n_fired"]}/{a["n_total"]} ({a["fire_rate"]:.0f}%)</td>'
                f'<td style="padding:6px 10px; border:1px solid #30363d; color:{pnl_color_l}; font-weight:bold;">{fmt(a["avg_pnl_all"])}</td>'
                f'<td style="padding:6px 10px; border:1px solid #30363d;">{fmt(a["avg_pnl"])}</td>'
                f'<td style="padding:6px 10px; border:1px solid #30363d;">{fmt_pos(a["win_rate"])}</td>'
                f'<td style="padding:6px 10px; border:1px solid #30363d; color:#3fb950;">{fmt(a["median_mfe"])}</td>'
                f'<td style="padding:6px 10px; border:1px solid #30363d; color:#f85149;">{fmt(a["median_mae"])}</td>'
                f'<td style="padding:6px 10px; border:1px solid #30363d; color:#f85149;">{fmt(a["mae_p90"])}</td>'
                f'<td style="padding:6px 10px; border:1px solid #30363d; color:#f85149;">{fmt(a["mae_max"])}</td>'
                f'</tr>'
            )
        return "".join(out)

    # Pick the highest avg_pnl_all for the master table (apples-to-apples)
    best_key = max(label_key, key=lambda lk: (agg[lk[2]]["avg_pnl_all"] or -1e9))[2]
    rows.append(build_rows(agg, label_key, highlight_best=True))

    # === VWAP regime cohorts ===
    strategy_keys_local = [k for _, _, k in label_key]
    below_vwap_results = [r for r in results if r.get("opens_below_vwap")]
    above_vwap_results = [r for r in results if not r.get("opens_below_vwap")]
    agg_below = {k: aggregate_strategy(k, below_vwap_results) for k in strategy_keys_local} if below_vwap_results else {}
    agg_above = {k: aggregate_strategy(k, above_vwap_results) for k in strategy_keys_local} if above_vwap_results else {}
    n_below = len(below_vwap_results)
    n_above = len(above_vwap_results)

    def cohort_table(local_agg: Dict, n_local: int, header_bg: str) -> str:
        if not local_agg or n_local == 0:
            return '<p style="color:#8b949e;">No trades in this cohort.</p>'
        return (
            '<table style="border-collapse:collapse; font-size:0.9em; width:100%;">'
            f'<tr style="background-color:{header_bg};">'
            '<th style="padding:6px 10px; border:1px solid #30363d; text-align:left;">#</th>'
            '<th style="padding:6px 10px; border:1px solid #30363d; text-align:left;">Strategy</th>'
            '<th style="padding:6px 10px; border:1px solid #30363d; text-align:left;">Fired</th>'
            '<th style="padding:6px 10px; border:1px solid #30363d; text-align:left;">Avg P&L (All)</th>'
            '<th style="padding:6px 10px; border:1px solid #30363d; text-align:left;">Avg P&L (Fired)</th>'
            '<th style="padding:6px 10px; border:1px solid #30363d; text-align:left;">Win %</th>'
            '<th style="padding:6px 10px; border:1px solid #30363d; text-align:left;">Median MFE</th>'
            '<th style="padding:6px 10px; border:1px solid #30363d; text-align:left;">Median MAE</th>'
            '<th style="padding:6px 10px; border:1px solid #30363d; text-align:left;">P90 MAE</th>'
            '<th style="padding:6px 10px; border:1px solid #30363d; text-align:left;">Max MAE</th>'
            '</tr>'
            + build_rows(local_agg, label_key, highlight_best=True)
            + '</table>'
        )

    below_table = cohort_table(agg_below, n_below, "#0f3a26")
    above_table = cohort_table(agg_above, n_above, "#3a1f1f")

    # Cohort verdicts
    def cohort_verdict(local_agg: Dict, n_local: int, regime: str) -> str:
        if not local_agg or n_local == 0:
            return f"No {regime} trades to analyze."
        b_pnl_l = local_agg["B_945"]["avg_pnl_all"] or 0
        winner_lk = max(label_key, key=lambda lk: (local_agg[lk[2]]["avg_pnl_all"] or -1e9))
        winner_letter, winner_name, winner_key = winner_lk
        winner_pnl = local_agg[winner_key]["avg_pnl_all"] or 0
        delta_l = winner_pnl - b_pnl_l
        if winner_key == "B_945":
            return (
                f"<strong style='color:#3fb950;'>9:45 rule (B) wins.</strong> "
                f"Avg P&L {b_pnl_l:+.2f}%, {n_local} trades. No earlier-trigger beat it."
            )
        return (
            f"<strong style='color:#58a6ff;'>{winner_letter} ({winner_name}) wins.</strong> "
            f"Avg P&L {winner_pnl:+.2f}% vs 9:45 rule {b_pnl_l:+.2f}% — "
            f"<strong>{delta_l:+.2f}%</strong> better, {n_local} trades."
        )
    verdict_below = cohort_verdict(agg_below, n_below, "below-VWAP")
    verdict_above = cohort_verdict(agg_above, n_above, "above-VWAP")

    # LOD distribution
    max_bucket = max(lod_buckets.values()) or 1
    lod_rows: List[str] = []
    cumulative = 0
    for bucket, count in lod_buckets.items():
        cumulative += count
        bar_w = int(count / max_bucket * 200)
        pct = count / n * 100.0
        cum_pct = cumulative / n * 100.0
        bar_color = "#f85149" if "9:30-9:35" in bucket or "9:35-9:45" in bucket else "#58a6ff"
        lod_rows.append(
            f'<tr>'
            f'<td style="padding:4px 8px; border:1px solid #30363d;">{bucket}</td>'
            f'<td style="padding:4px 8px; border:1px solid #30363d;">{count}</td>'
            f'<td style="padding:4px 8px; border:1px solid #30363d;">{pct:.0f}%</td>'
            f'<td style="padding:4px 8px; border:1px solid #30363d;">{cum_pct:.0f}%</td>'
            f'<td style="padding:4px 8px; border:1px solid #30363d;">'
            f'<div style="width:{bar_w}px; height:10px; background:{bar_color}; border-radius:2px;"></div>'
            f'</td>'
            f'</tr>'
        )

    # Headline: B vs F (or B vs winner)
    b_pnl = agg["B_945"]["avg_pnl_all"]
    best_label = next(name for letter, name, key in label_key if key == best_key)
    best_pnl = agg[best_key]["avg_pnl_all"]
    delta = (best_pnl or 0) - (b_pnl or 0)
    if best_key == "B_945":
        verdict = (
            "<strong style='color:#3fb950;'>The 9:45 rule wins.</strong> "
            "No tested earlier-trigger strategy beats it on average P&L per trade."
        )
    else:
        verdict = (
            f"<strong style='color:#e3b341;'>The 9:45 rule is leaving money on the table.</strong> "
            f"Strategy {best_key.split('_')[0][0]} ({best_label}) beats the 9:45 rule by "
            f"<strong>{delta:+.2f}%</strong> per trade on average."
        )

    html = f"""
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            max-width:920px; margin:0 auto; color:#c9d1d9; background-color:#161b22;
            font-size:14px; line-height:1.5; padding:16px;">

  <div style="background-color:#21262d; padding:14px 18px; border-radius:6px; border-bottom:2px solid #30363d;">
    <h1 style="margin:0; font-size:1.5em;">9:45 Wait-Rule Audit — Reversal Shorts</h1>
    <div style="font-size:0.9em; color:#8b949e; margin-top:4px;">
      n={n} historical reversals | source: <code>data/reversal_data.csv</code>
      | bars: Polygon 1-min (premarket + RTH)
    </div>
  </div>

  <div style="margin-top:20px; padding:12px 16px; border-left:4px solid #58a6ff; background-color:#0d1117;">
    <h2 style="margin:0 0 6px 0; color:#f0f6fc;">Verdict</h2>
    <div style="font-size:1.05em;">{verdict}</div>
    <div style="font-size:0.9em; color:#8b949e; margin-top:8px;">
      Comparison metric: <em>avg P&L per trade across all {n} trades</em> (strategies that
      didn't fire on a given day count as 0 P&L — fair penalty for missing setups). Strategies
      held entry → 4:00 PM close.
    </div>
  </div>

  <h2 style="color:#f0f6fc; margin-top:28px;">Strategy Comparison</h2>
  <p style="color:#8b949e; font-size:0.9em;">
    All strategies hold entry → close. <strong>Avg P&L (All)</strong> counts missed-fire
    days as 0, so it's the apples-to-apples figure for "would I make more money switching."
    <strong>Avg P&L (Fired)</strong> is conditional on firing — measures signal quality.
  </p>
  <table style="border-collapse:collapse; font-size:0.9em; width:100%;">
    <tr style="background-color:#21262d;">
      <th style="padding:6px 10px; border:1px solid #30363d; text-align:left;">#</th>
      <th style="padding:6px 10px; border:1px solid #30363d; text-align:left;">Strategy</th>
      <th style="padding:6px 10px; border:1px solid #30363d; text-align:left;">Fired</th>
      <th style="padding:6px 10px; border:1px solid #30363d; text-align:left;">Avg P&L (All)</th>
      <th style="padding:6px 10px; border:1px solid #30363d; text-align:left;">Avg P&L (Fired)</th>
      <th style="padding:6px 10px; border:1px solid #30363d; text-align:left;">Win %</th>
      <th style="padding:6px 10px; border:1px solid #30363d; text-align:left;">Median MFE</th>
      <th style="padding:6px 10px; border:1px solid #30363d; text-align:left;">Median MAE</th>
      <th style="padding:6px 10px; border:1px solid #30363d; text-align:left;">P90 MAE</th>
      <th style="padding:6px 10px; border:1px solid #30363d; text-align:left;">Max MAE</th>
    </tr>
    {"".join(rows)}
  </table>

  <h2 style="color:#f0f6fc; margin-top:32px;">By VWAP Regime — Conditional Strategy Comparison</h2>
  <p style="color:#8b949e; font-size:0.9em;">
    Splits all {n} trades into two cohorts based on whether the 9:30 open was below
    or above PM VWAP. The hypothesis: ladder dominates 9:45 cleanly when the stock
    is already weak at the open (below VWAP), while 9:45's "let it confirm" logic
    earns its keep on days the stock opens strong (above VWAP).
  </p>

  <h3 style="color:#3fb950; margin-top:18px;">Cohort 1: Opens BELOW PM VWAP (n={n_below})</h3>
  <div style="margin-bottom:8px;">{verdict_below}</div>
  {below_table}

  <h3 style="color:#f85149; margin-top:24px;">Cohort 2: Opens ABOVE PM VWAP (n={n_above})</h3>
  <div style="margin-bottom:8px;">{verdict_above}</div>
  {above_table}

  <h2 style="color:#f0f6fc; margin-top:32px;">Where does the LOD print?</h2>
  <p style="color:#8b949e; font-size:0.9em;">
    The 9:45 rule's biggest cost is when the stock cracks <em>before</em> 9:45 — by the time
    you enter, you're shorting a price that's already lower than where it would've cracked.
    <strong>{lod_before_945}/{n} ({lod_before_945/n*100:.0f}%)</strong> of trades printed
    their LOD before 9:45.
  </p>
  <table style="border-collapse:collapse; font-size:0.9em;">
    <tr style="background-color:#21262d;">
      <th style="padding:4px 8px; border:1px solid #30363d; text-align:left;">Time bucket</th>
      <th style="padding:4px 8px; border:1px solid #30363d;">N</th>
      <th style="padding:4px 8px; border:1px solid #30363d;">% of trades</th>
      <th style="padding:4px 8px; border:1px solid #30363d;">Cumulative</th>
      <th style="padding:4px 8px; border:1px solid #30363d;">Distribution</th>
    </tr>
    {"".join(lod_rows)}
  </table>

  <h2 style="color:#f0f6fc; margin-top:32px;">Signal Frequency</h2>
  <ul style="color:#c9d1d9;">
    <li><strong>{pm_low_break}/{n} ({pm_low_break/n*100:.0f}%)</strong> trades broke the premarket low at some point</li>
    <li><strong>{first5_red_n}/{n} ({first5_red_n/n*100:.0f}%)</strong> closed the first 5-min bar (9:35) red vs open</li>
    <li><strong>{first10_red_n}/{n} ({first10_red_n/n*100:.0f}%)</strong> closed the first 10-min bar (9:40) red vs open</li>
    <li><strong>{opens_below_vwap_n}/{n} ({opens_below_vwap_n/n*100:.0f}%)</strong> opened <em>below</em> PM VWAP (these get instant entry under strategy K)</li>
  </ul>

  <h2 style="color:#f0f6fc; margin-top:32px;">Time from 9:30 to HOD</h2>
  <p style="color:#8b949e; font-size:0.9em;">
    How long do you sit through the squeeze before it tops? Short = scale-in works
    (the high comes fast). Long = need to size smaller and respect levels.
  </p>
  <table style="border-collapse:collapse; font-size:0.9em;">
    <tr style="background-color:#21262d;">
      <th style="padding:4px 10px; border:1px solid #30363d;">Percentile</th>
      <th style="padding:4px 10px; border:1px solid #30363d;">Minutes after 9:30</th>
    </tr>
    <tr><td style="padding:4px 10px; border:1px solid #30363d;">P25</td><td style="padding:4px 10px; border:1px solid #30363d;">{hod_p25:.0f}m</td></tr>
    <tr><td style="padding:4px 10px; border:1px solid #30363d;">Median</td><td style="padding:4px 10px; border:1px solid #30363d;">{hod_p50:.0f}m</td></tr>
    <tr><td style="padding:4px 10px; border:1px solid #30363d;">P75</td><td style="padding:4px 10px; border:1px solid #30363d;">{hod_p75:.0f}m</td></tr>
    <tr><td style="padding:4px 10px; border:1px solid #30363d;">P90</td><td style="padding:4px 10px; border:1px solid #30363d;">{hod_p90:.0f}m</td></tr>
    <tr><td style="padding:4px 10px; border:1px solid #30363d;">Max</td><td style="padding:4px 10px; border:1px solid #30363d;">{hod_max:.0f}m</td></tr>
  </table>

  <h2 style="color:#f0f6fc; margin-top:32px;">Caveats</h2>
  <ul style="color:#8b949e; font-size:0.9em;">
    <li>This dataset contains trades that <em>did</em> reverse (survivorship: bad ideas you skipped aren't here). Conclusion is about <em>timing</em> entry on these setups, not about whether the rule helps you avoid bad trades.</li>
    <li>"Avg P&L (All)" treats not-firing as 0 P&L. In reality you might fall back to 9:45 anyway — that's what strategy F simulates.</li>
    <li>P&L is measured entry → 4 PM close. Real exits use stops + targets, so absolute P&L will differ; the comparison between strategies is what matters.</li>
    <li>No transaction cost / borrow fee modeled.</li>
  </ul>

</div>
"""
    return html


def main():
    # Load trades
    csv_path = PROJECT_ROOT / "data" / "reversal_data.csv"
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["ticker", "date"]).copy()
    df["parsed_date"] = df["date"].apply(parse_csv_date)
    log.info(f"Loaded {len(df)} reversal trades from {csv_path.name}")

    trades = list(df[["ticker", "parsed_date", "atr_pct"]].itertuples(index=False, name=None))

    # Fetch + compute in parallel
    log.info(f"Fetching 1-min bars + computing strategies (parallel, max_workers={MAX_WORKERS})...")
    results: List[Dict] = []
    failures: List[str] = []

    def work(t: tuple):
        ticker, trade_date, atr_pct = t
        bars = fetch_one(ticker, trade_date)
        if bars is None:
            return None, f"{ticker} {trade_date}: no bars"
        out = compute_for_trade(ticker, trade_date, bars, atr_pct=float(atr_pct) if pd.notna(atr_pct) else None)
        if out is None:
            return None, f"{ticker} {trade_date}: insufficient bars"
        return out, None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(work, t): t for t in trades}
        for i, fut in enumerate(as_completed(futs), 1):
            out, err = fut.result()
            if out is not None:
                results.append(out)
            else:
                failures.append(err or "unknown")
            if i % 20 == 0:
                log.info(f"  {i}/{len(trades)} processed (ok={len(results)}, fail={len(failures)})")

    log.info(f"Done. ok={len(results)}, fail={len(failures)}")
    if failures:
        for f in failures[:5]:
            log.warning(f"  example failure: {f}")

    if not results:
        log.error("No results — aborting report.")
        return

    # Aggregate per strategy
    strategy_keys = ["A_open", "B_945", "C_pm_break", "D_5min_red", "E_10min_red", "F_adaptive",
                     "G_half_945", "H_half_signal", "I_half_hod", "J_ladder_atr", "K_vwap",
                     "L_confirm_ladder"]
    agg = {k: aggregate_strategy(k, results) for k in strategy_keys}

    # Build report
    html = build_html(results, agg)

    # Save a local copy
    out_path = PROJECT_ROOT / "scripts" / "reports" / "analyze_945_rule_report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    log.info(f"Saved report: {out_path}")

    # Email
    today_str = dt.date.today().strftime("%m/%d/%Y")
    subject = f"9:45 Wait-Rule Audit — Reversal Shorts | n={len(results)} | {today_str}"
    try:
        send_email(
            to_email="zmburr@gmail.com",
            subject=subject,
            body=html,
            is_html=True,
        )
        log.info(f"Email sent: {subject}")
    except Exception as e:
        log.error(f"Failed to send email: {e}")


if __name__ == "__main__":
    main()
