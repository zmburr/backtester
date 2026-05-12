"""scripts/analyze_day_early_3dgapfade.py

What if I'd applied the ladder short ONE DAY EARLY on every 3DGapFade trade?
Simulates 33% at 9:30 / 33% at +0.5 ATR / 33% at +1.0 ATR on T-1, then measures:
  - T-1 close P&L (intraday outcome if you covered EOD)
  - Overnight gap from T-1 close → T open (the killer scenario)
  - Held-through P&L (T-1 entry → T close, riding through the trap)
  - Worst-point MAE across the entire hold
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
import pandas_market_calendars as mcal
import pytz

from data_queries.polygon_queries import get_intraday
from support.config import send_email

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")
NYSE = mcal.get_calendar("NYSE")
MAX_WORKERS = 8


def localize(d: dt.date, t: dt.time) -> dt.datetime:
    return ET.localize(dt.datetime.combine(d, t))


def parse_csv_date(s) -> dt.date:
    return dt.datetime.strptime(str(s).strip(), "%m/%d/%Y").date()


def prior_trading_day(d: dt.date) -> Optional[dt.date]:
    sched = NYSE.valid_days(start_date=(pd.Timestamp(d) - pd.Timedelta(days=10)).date(),
                            end_date=pd.Timestamp(d).date())
    if sched.empty:
        return None
    valid = [x.date() for x in sched if x.date() < d]
    return valid[-1] if valid else None


def rth_summary(df: pd.DataFrame, day: dt.date) -> Optional[Dict]:
    if df is None or df.empty:
        return None
    open_dt = localize(day, dt.time(9, 30))
    close_dt = localize(day, dt.time(16, 0))
    rth = df[(df.index >= open_dt) & (df.index <= close_dt)]
    if rth.empty or len(rth) < 30:
        return None
    pm = df[df.index < open_dt]
    pm_vwap = None
    if not pm.empty:
        typical = (pm["high"] + pm["low"] + pm["close"]) / 3.0
        cum_pv = float((typical * pm["volume"]).sum())
        cum_v = float(pm["volume"].sum())
        pm_vwap = cum_pv / cum_v if cum_v > 0 else None
    return {
        "open": float(rth.iloc[0]["open"]),
        "close": float(rth.iloc[-1]["close"]),
        "high": float(rth["high"].max()),
        "low": float(rth["low"].min()),
        "rth": rth,
        "first_idx": rth.index[0],
        "pm_high": float(pm["high"].max()) if not pm.empty else None,
        "pm_vwap": pm_vwap,
    }


def simulate_ladder(t1_sum: Dict, atr_pct: float) -> Dict:
    rth = t1_sum["rth"]
    open_p = t1_sum["open"]
    close_p = t1_sum["close"]

    target_50 = open_p * (1 + 0.5 * atr_pct)
    target_100 = open_p * (1 + 1.0 * atr_pct)

    legs = [(open_p, t1_sum["first_idx"])]
    hits_50 = rth[rth["high"] >= target_50]
    hits_100 = rth[rth["high"] >= target_100]
    leg_50_hit = not hits_50.empty
    leg_100_hit = not hits_100.empty
    if leg_50_hit:
        legs.append((target_50, hits_50.index[0]))
    if leg_100_hit:
        legs.append((target_100, hits_100.index[0]))

    avg_entry = float(np.mean([p for p, _ in legs]))
    first_time = min(t for _, t in legs)
    post = rth[rth.index >= first_time]
    post_high = float(post["high"].max())
    return {
        "avg_entry": avg_entry,
        "n_legs": len(legs),
        "leg_50_hit": leg_50_hit,
        "leg_100_hit": leg_100_hit,
        "t1_close_pnl_pct": (avg_entry - close_p) / avg_entry * 100.0,
        "t1_intraday_mae_pct": (post_high - avg_entry) / avg_entry * 100.0,
        "first_time": first_time,
    }


def simulate_confirmation_ladder(t1_sum: Dict, df_t1_full: pd.DataFrame) -> Dict:
    """L. 33%/33%/33% on (VWAP break, PM low break, open-fail break).
    Legs that don't fire = no entry for that slice. Returns avg_entry of fired legs,
    or None if no legs fired."""
    rth = t1_sum["rth"]
    open_p = t1_sum["open"]
    close_p = t1_sum["close"]
    pm_low = t1_sum.get("pm_low") if isinstance(t1_sum, dict) else None  # may not be present

    # Build VWAP series across full df (PM+RTH)
    typical = (df_t1_full["high"] + df_t1_full["low"] + df_t1_full["close"]) / 3.0
    cum_pv = (typical * df_t1_full["volume"]).cumsum()
    cum_v = df_t1_full["volume"].cumsum().replace(0, np.nan)
    vwap_series = cum_pv / cum_v
    rth_vwap = vwap_series.reindex(rth.index)

    # Leg V: first close < VWAP
    v_mask = rth["close"] < rth_vwap
    v_bars = rth[v_mask.fillna(False)]
    v_time = v_bars.index[0] if not v_bars.empty else None
    v_price = float(v_bars.iloc[0]["close"]) if not v_bars.empty else None

    # PM low (if not in t1_sum, derive from df)
    pm_part = df_t1_full[df_t1_full.index < rth.index[0]]
    pm_low_val = float(pm_part["low"].min()) if not pm_part.empty else None

    # Leg P: first close < PM low
    if pm_low_val is not None:
        p_bars = rth[rth["close"] < pm_low_val]
        p_time = p_bars.index[0] if not p_bars.empty else None
        p_price = float(p_bars.iloc[0]["close"]) if not p_bars.empty else None
    else:
        p_time, p_price = None, None

    # Leg O: drive then fail — first close > open, then later close < open
    above_open = rth[rth["close"] > open_p]
    if not above_open.empty:
        rally_time = above_open.index[0]
        after_rally = rth[rth.index > rally_time]
        o_bars = after_rally[after_rally["close"] < open_p]
        o_time = o_bars.index[0] if not o_bars.empty else None
        o_price = float(o_bars.iloc[0]["close"]) if not o_bars.empty else None
    else:
        o_time, o_price = None, None

    legs = []
    leg_v_hit = v_price is not None
    leg_p_hit = p_price is not None
    leg_o_hit = o_price is not None
    if leg_v_hit:
        legs.append((v_price, v_time))
    if leg_p_hit:
        legs.append((p_price, p_time))
    if leg_o_hit:
        legs.append((o_price, o_time))

    if not legs:
        return {
            "fired": False,
            "avg_entry": None,
            "n_legs": 0,
            "leg_v_hit": False,
            "leg_p_hit": False,
            "leg_o_hit": False,
            "t1_close_pnl_pct": None,
            "t1_intraday_mae_pct": None,
            "first_time": None,
        }

    avg_entry = float(np.mean([p for p, _ in legs]))
    first_time = min(t for _, t in legs)
    post = rth[rth.index >= first_time]
    post_high = float(post["high"].max())
    return {
        "fired": True,
        "avg_entry": avg_entry,
        "n_legs": len(legs),
        "leg_v_hit": leg_v_hit,
        "leg_p_hit": leg_p_hit,
        "leg_o_hit": leg_o_hit,
        "t1_close_pnl_pct": (avg_entry - close_p) / avg_entry * 100.0,
        "t1_intraday_mae_pct": (post_high - avg_entry) / avg_entry * 100.0,
        "first_time": first_time,
    }


def work(row) -> tuple:
    ticker = row.ticker
    t = row.parsed_date
    atr_pct = float(row.atr_pct) if pd.notna(row.atr_pct) else None
    trade_grade = str(row.trade_grade) if pd.notna(row.trade_grade) else "N/A"
    pct_from_9ema = float(row.pct_from_9ema) if pd.notna(row.pct_from_9ema) else None
    if atr_pct is None or atr_pct <= 0:
        return None, f"{ticker} {t}: no atr_pct"

    t_minus_1 = prior_trading_day(t)
    t_minus_2 = prior_trading_day(t_minus_1) if t_minus_1 is not None else None
    if t_minus_1 is None or t_minus_2 is None:
        return None, f"{ticker} {t}: no prior 2 trading days"

    try:
        df_t1 = get_intraday(ticker, t_minus_1.strftime("%Y-%m-%d"), 1, "minute")
        df_t = get_intraday(ticker, t.strftime("%Y-%m-%d"), 1, "minute")
        df_t2_daily = get_intraday(ticker, t_minus_2.strftime("%Y-%m-%d"), 1, "day")
    except Exception as e:
        return None, f"{ticker} {t}: fetch failed - {e}"

    t1 = rth_summary(df_t1, t_minus_1)
    tt = rth_summary(df_t, t)
    if t1 is None or tt is None:
        return None, f"{ticker} {t}: insufficient bars"

    t2_high = None
    if df_t2_daily is not None and not df_t2_daily.empty:
        t2_high = float(df_t2_daily["high"].iloc[0])

    ladder = simulate_ladder(t1, atr_pct)
    avg_entry = ladder["avg_entry"]

    overnight_gap_pct = (tt["open"] - t1["close"]) / t1["close"] * 100.0
    held_pnl_pct = (avg_entry - tt["close"]) / avg_entry * 100.0

    # Worst-point MAE: max of (T-1 post-entry high, T premarket high, T RTH high)
    t_pm_high = tt.get("pm_high") or 0.0
    post_t1 = t1["rth"][t1["rth"].index >= ladder["first_time"]]
    t1_post_high = float(post_t1["high"].max())
    worst_high = max(t1_post_high, t_pm_high, tt["high"])
    held_mae_pct = (worst_high - avg_entry) / avg_entry * 100.0

    # Strategy L: confirmation ladder on T-1
    L_sim = simulate_confirmation_ladder(t1, df_t1)
    if L_sim["fired"]:
        L_avg = L_sim["avg_entry"]
        L_post_t1 = t1["rth"][t1["rth"].index >= L_sim["first_time"]]
        L_t1_post_high = float(L_post_t1["high"].max())
        L_worst = max(L_t1_post_high, t_pm_high, tt["high"])
        L_held_mae = (L_worst - L_avg) / L_avg * 100.0
        L_held_pnl = (L_avg - tt["close"]) / L_avg * 100.0
    else:
        L_held_mae = None
        L_held_pnl = None

    t1_pm_vwap = t1.get("pm_vwap")
    t1_opens_below_vwap = (t1_pm_vwap is not None) and (t1["open"] < t1_pm_vwap)
    t1_green = t1["close"] > t1["open"]
    t1_above_t2_high = (t2_high is not None) and (t1["close"] > t2_high)

    return {
        "ticker": ticker,
        "trade_date": t,
        "t_minus_1": t_minus_1,
        "atr_pct": atr_pct,
        "trade_grade": trade_grade,
        "pct_from_9ema": pct_from_9ema,
        "t1_open": t1["open"],
        "t1_close": t1["close"],
        "t1_high": t1["high"],
        "t1_pm_vwap": t1_pm_vwap,
        "t1_opens_below_vwap": t1_opens_below_vwap,
        "t1_green": t1_green,
        "t2_high": t2_high,
        "t1_above_t2_high": t1_above_t2_high,
        "t_open": tt["open"],
        "t_close": tt["close"],
        "t_high": tt["high"],
        "avg_entry": avg_entry,
        "n_legs": ladder["n_legs"],
        "leg_50_hit": ladder["leg_50_hit"],
        "leg_100_hit": ladder["leg_100_hit"],
        "t1_close_pnl_pct": ladder["t1_close_pnl_pct"],
        "t1_intraday_mae_pct": ladder["t1_intraday_mae_pct"],
        "overnight_gap_pct": overnight_gap_pct,
        "held_pnl_pct": held_pnl_pct,
        "held_mae_pct": held_mae_pct,
        # Strategy L (confirmation ladder) on T-1
        "L_fired": L_sim["fired"],
        "L_n_legs": L_sim["n_legs"],
        "L_t1_close_pnl_pct": L_sim["t1_close_pnl_pct"],
        "L_t1_intraday_mae_pct": L_sim["t1_intraday_mae_pct"],
        "L_held_pnl_pct": L_held_pnl,
        "L_held_mae_pct": L_held_mae,
    }, None


def fmt(v, suf="%", pl=2):
    return f"{v:+.{pl}f}{suf}" if v is not None and not pd.isna(v) else "n/a"


def fmt_abs(v, pl=2):
    return f"{v:.{pl}f}%" if v is not None and not pd.isna(v) else "n/a"


def build_html(df: pd.DataFrame) -> str:
    n = len(df)

    leg_50_pct = df["leg_50_hit"].mean() * 100
    leg_100_pct = df["leg_100_hit"].mean() * 100
    avg_legs = df["n_legs"].mean()

    def stats(col):
        s = df[col]
        return {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "p10": float(s.quantile(0.10)),
            "p25": float(s.quantile(0.25)),
            "p75": float(s.quantile(0.75)),
            "p90": float(s.quantile(0.90)),
            "p95": float(s.quantile(0.95)),
            "max": float(s.max()),
            "min": float(s.min()),
        }

    s_t1pnl = stats("t1_close_pnl_pct")
    s_t1mae = stats("t1_intraday_mae_pct")
    s_gap = stats("overnight_gap_pct")
    s_held = stats("held_pnl_pct")
    s_heldmae = stats("held_mae_pct")

    pct_t1_red = (df["t1_close_pnl_pct"] > 0).mean() * 100
    pct_held_win = (df["held_pnl_pct"] > 0).mean() * 100
    pct_gap_against = (df["overnight_gap_pct"] > 0).mean() * 100

    # Worst trades
    worst_overnight = df.nlargest(5, "overnight_gap_pct")
    worst_t1_mae = df.nlargest(5, "t1_intraday_mae_pct")
    worst_held = df.nsmallest(5, "held_pnl_pct")

    # === T-1 VWAP-cohort split ===
    df_below = df[df["t1_opens_below_vwap"] == True]  # noqa: E712
    df_above = df[df["t1_opens_below_vwap"] == False]  # noqa: E712

    def cohort_metrics(d: pd.DataFrame) -> Dict:
        if d.empty:
            return {}
        return {
            "n": len(d),
            "t1_pnl_mean": float(d["t1_close_pnl_pct"].mean()),
            "t1_pnl_med": float(d["t1_close_pnl_pct"].median()),
            "t1_mae_med": float(d["t1_intraday_mae_pct"].median()),
            "t1_mae_p90": float(d["t1_intraday_mae_pct"].quantile(0.90)),
            "gap_mean": float(d["overnight_gap_pct"].mean()),
            "gap_med": float(d["overnight_gap_pct"].median()),
            "gap_p90": float(d["overnight_gap_pct"].quantile(0.90)),
            "gap_max": float(d["overnight_gap_pct"].max()),
            "held_pnl_mean": float(d["held_pnl_pct"].mean()),
            "held_pnl_med": float(d["held_pnl_pct"].median()),
            "held_win_rate": float((d["held_pnl_pct"] > 0).mean() * 100),
            "held_mae_med": float(d["held_mae_pct"].median()),
            "held_mae_p90": float(d["held_mae_pct"].quantile(0.90)),
            "held_mae_max": float(d["held_mae_pct"].max()),
        }

    m_below = cohort_metrics(df_below)
    m_above = cohort_metrics(df_above)

    def cohort_row(label: str, m: Dict) -> str:
        if not m:
            return f'<tr><td style="padding:6px 10px;border:1px solid #30363d;">{label}</td><td colspan="11" style="padding:6px 10px;border:1px solid #30363d;color:#8b949e;">no trades</td></tr>'
        pnl_color = "#3fb950" if m["t1_pnl_mean"] > 0 else "#f85149"
        held_color = "#3fb950" if m["held_pnl_mean"] > 0 else "#f85149"
        return (
            f'<tr>'
            f'<td style="padding:6px 10px;border:1px solid #30363d;font-weight:bold;">{label}</td>'
            f'<td style="padding:6px 10px;border:1px solid #30363d;">{m["n"]}</td>'
            f'<td style="padding:6px 10px;border:1px solid #30363d;color:{pnl_color};font-weight:bold;">{m["t1_pnl_mean"]:+.2f}%</td>'
            f'<td style="padding:6px 10px;border:1px solid #30363d;">{m["t1_pnl_med"]:+.2f}%</td>'
            f'<td style="padding:6px 10px;border:1px solid #30363d;color:#f85149;">{m["t1_mae_med"]:.1f}%</td>'
            f'<td style="padding:6px 10px;border:1px solid #30363d;color:#f85149;">{m["t1_mae_p90"]:.1f}%</td>'
            f'<td style="padding:6px 10px;border:1px solid #30363d;color:#f85149;font-weight:bold;">{m["gap_mean"]:+.2f}%</td>'
            f'<td style="padding:6px 10px;border:1px solid #30363d;">{m["gap_p90"]:+.2f}%</td>'
            f'<td style="padding:6px 10px;border:1px solid #30363d;color:{held_color};font-weight:bold;">{m["held_pnl_mean"]:+.2f}%</td>'
            f'<td style="padding:6px 10px;border:1px solid #30363d;">{m["held_win_rate"]:.0f}%</td>'
            f'<td style="padding:6px 10px;border:1px solid #30363d;color:#f85149;">{m["held_mae_p90"]:.1f}%</td>'
            f'</tr>'
        )

    vwap_cohort_html = (
        '<table style="border-collapse:collapse;font-size:0.9em;width:100%;">'
        '<tr style="background-color:#21262d;">'
        '<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">T-1 cohort</th>'
        '<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">N</th>'
        '<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">T-1 P&L mean</th>'
        '<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">T-1 P&L med</th>'
        '<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">T-1 MAE med</th>'
        '<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">T-1 MAE P90</th>'
        '<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Overnight gap mean</th>'
        '<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Gap P90</th>'
        '<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Held P&L mean</th>'
        '<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Held Win%</th>'
        '<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Held MAE P90</th>'
        '</tr>'
        + cohort_row("T-1 opens BELOW PM VWAP (weak open)", m_below)
        + cohort_row("T-1 opens ABOVE PM VWAP (strong open)", m_above)
        + '</table>'
    )

    # Cohort verdict
    if m_below and m_above:
        cohort_verdict = (
            f"<strong>Filter test:</strong> Above-VWAP cohort overnight gap "
            f"<strong style='color:#f85149;'>{m_above['gap_mean']:+.2f}%</strong> vs "
            f"Below-VWAP <strong style='color:#3fb950;'>{m_below['gap_mean']:+.2f}%</strong>. "
            f"Held P&L: above-VWAP <strong>{m_above['held_pnl_mean']:+.2f}%</strong>, "
            f"below-VWAP <strong>{m_below['held_pnl_mean']:+.2f}%</strong>. "
        )
        if m_above["held_pnl_mean"] < m_below["held_pnl_mean"] - 3:
            cohort_verdict += "<span style='color:#3fb950;'>VWAP-at-open IS a meaningful day-early filter.</span>"
        elif m_above["held_pnl_mean"] > m_below["held_pnl_mean"] + 3:
            cohort_verdict += "<span style='color:#f85149;'>VWAP-at-open inverts — strong opens did BETTER on T-1.</span>"
        else:
            cohort_verdict += "<span style='color:#e3b341;'>VWAP regime does not meaningfully separate the cohorts (~tie).</span>"
        vwap_cohort_html = f'<div style="margin-bottom:8px;">{cohort_verdict}</div>' + vwap_cohort_html
    else:
        vwap_cohort_html = '<p style="color:#8b949e;">Only one cohort had data — VWAP split not meaningful.</p>' + vwap_cohort_html

    # === T-1 daily-close cohort: green vs red ===
    df_t1_red = df[df["t1_green"] == False]  # noqa: E712
    df_t1_green = df[df["t1_green"] == True]  # noqa: E712
    m_t1_red = cohort_metrics(df_t1_red)
    m_t1_green = cohort_metrics(df_t1_green)

    # === T-1 close vs T-2 high cohort: above (still extending) vs below (already weakened) ===
    df_above_t2 = df[df["t1_above_t2_high"] == True]  # noqa: E712
    df_below_t2 = df[df["t1_above_t2_high"] == False]  # noqa: E712
    m_above_t2 = cohort_metrics(df_above_t2)
    m_below_t2 = cohort_metrics(df_below_t2)

    def make_cohort_table(rows_data: List, header_label: str = "T-1 cohort") -> str:
        body = "".join(cohort_row(label, m) for label, m in rows_data)
        return (
            '<table style="border-collapse:collapse;font-size:0.9em;width:100%;">'
            '<tr style="background-color:#21262d;">'
            f'<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">{header_label}</th>'
            '<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">N</th>'
            '<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">T-1 P&L mean</th>'
            '<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">T-1 P&L med</th>'
            '<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">T-1 MAE med</th>'
            '<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">T-1 MAE P90</th>'
            '<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Overnight gap mean</th>'
            '<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Gap P90</th>'
            '<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Held P&L mean</th>'
            '<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Held Win%</th>'
            '<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Held MAE P90</th>'
            '</tr>'
            + body
            + '</table>'
        )

    green_red_table = make_cohort_table([
        ("T-1 closed RED (close < open)", m_t1_red),
        ("T-1 closed GREEN (close > open)", m_t1_green),
    ])
    above_t2_table = make_cohort_table([
        ("T-1 close ≤ T-2 high (held below)", m_below_t2),
        ("T-1 close > T-2 high (still extending)", m_above_t2),
    ])

    def filter_verdict(weak_cohort: Dict, strong_cohort: Dict, label_strong: str, label_weak: str) -> str:
        if not weak_cohort or not strong_cohort:
            return "<span style='color:#8b949e;'>Insufficient data in one cohort.</span>"
        delta = (weak_cohort.get("held_pnl_mean", 0) or 0) - (strong_cohort.get("held_pnl_mean", 0) or 0)
        if delta > 3:
            return (f"<span style='color:#3fb950;'><strong>Filter works.</strong> {label_weak} held P&L "
                    f"<strong>{weak_cohort['held_pnl_mean']:+.2f}%</strong> vs {label_strong} "
                    f"<strong>{strong_cohort['held_pnl_mean']:+.2f}%</strong> — delta {delta:+.2f}%.</span>")
        elif delta < -3:
            return (f"<span style='color:#f85149;'><strong>Filter inverts.</strong> {label_weak} held P&L "
                    f"<strong>{weak_cohort['held_pnl_mean']:+.2f}%</strong> vs {label_strong} "
                    f"<strong>{strong_cohort['held_pnl_mean']:+.2f}%</strong> — strong cohort did better.</span>")
        else:
            return (f"<span style='color:#e3b341;'><strong>Filter ~tie.</strong> {label_weak} held P&L "
                    f"<strong>{weak_cohort['held_pnl_mean']:+.2f}%</strong> vs {label_strong} "
                    f"<strong>{strong_cohort['held_pnl_mean']:+.2f}%</strong>.</span>")

    green_red_verdict = filter_verdict(m_t1_red, m_t1_green, "GREEN (still strong)", "RED (already weak)")
    above_t2_verdict = filter_verdict(m_below_t2, m_above_t2, "above-T-2-high (extending)", "below-T-2-high (held)")

    # === Strategy L (confirmation ladder) summary on T-1 ===
    n_total = len(df)
    L_fired_df = df[df["L_fired"] == True]  # noqa: E712
    L_fire_rate = len(L_fired_df) / n_total * 100 if n_total else 0
    L_legs_avg = float(df["L_n_legs"].mean()) if not df.empty else 0
    if not L_fired_df.empty:
        L_t1_pnl_mean = float(L_fired_df["L_t1_close_pnl_pct"].mean())
        L_t1_pnl_med = float(L_fired_df["L_t1_close_pnl_pct"].median())
        L_held_pnl_mean = float(L_fired_df["L_held_pnl_pct"].mean())
        L_held_pnl_med = float(L_fired_df["L_held_pnl_pct"].median())
        L_held_pnl_p10 = float(L_fired_df["L_held_pnl_pct"].quantile(0.10))
        L_held_win = float((L_fired_df["L_held_pnl_pct"] > 0).mean() * 100)
        L_held_mae_med = float(L_fired_df["L_held_mae_pct"].median())
        L_held_mae_p90 = float(L_fired_df["L_held_mae_pct"].quantile(0.90))
        L_held_mae_max = float(L_fired_df["L_held_mae_pct"].max())
        # Portfolio P&L: count non-fires as 0 (you didn't trade those days = saved capital)
        L_portfolio_held_pnl = float(L_fired_df["L_held_pnl_pct"].sum() / n_total)
    else:
        L_t1_pnl_mean = L_t1_pnl_med = L_held_pnl_mean = L_held_pnl_med = 0
        L_held_pnl_p10 = L_held_win = L_held_mae_med = L_held_mae_p90 = L_held_mae_max = 0
        L_portfolio_held_pnl = 0

    # Compare L vs upward ladder (J equivalent — already-existing held P&L on full dataset)
    upward_held_pnl_mean = float(df["held_pnl_pct"].mean())

    if L_fire_rate < 50:
        l_protection_msg = (f"<span style='color:#3fb950;'><strong>Massive protection:</strong> "
                            f"L fires on only <strong>{L_fire_rate:.0f}%</strong> of T-1 days — "
                            f"on the other {100-L_fire_rate:.0f}% you just didn't enter. "
                            f"Portfolio held P&L per T-1 day: <strong>{L_portfolio_held_pnl:+.2f}%</strong> "
                            f"vs upward ladder's <strong>{upward_held_pnl_mean:+.2f}%</strong>.</span>")
    else:
        l_protection_msg = (f"<span style='color:#e3b341;'><strong>Mild protection:</strong> "
                            f"L still fires on {L_fire_rate:.0f}% of T-1 days. "
                            f"Portfolio held P&L: <strong>{L_portfolio_held_pnl:+.2f}%</strong> "
                            f"vs upward ladder's <strong>{upward_held_pnl_mean:+.2f}%</strong>.</span>")

    l_summary_html = (
        f'<div style="margin-bottom:8px;">{l_protection_msg}</div>'
        '<table style="border-collapse:collapse;font-size:0.9em;width:100%;">'
        '<tr style="background-color:#21262d;">'
        '<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Metric</th>'
        '<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Value</th>'
        '<th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Notes</th>'
        '</tr>'
        f'<tr><td style="padding:6px 10px;border:1px solid #30363d;">Fire rate (any leg fires)</td>'
        f'<td style="padding:6px 10px;border:1px solid #30363d;font-weight:bold;">{len(L_fired_df)}/{n_total} ({L_fire_rate:.0f}%)</td>'
        f'<td style="padding:6px 10px;border:1px solid #30363d;color:#8b949e;">Of {n_total} day-early simulations, this many had at least one confirmation leg fire on T-1</td></tr>'
        f'<tr><td style="padding:6px 10px;border:1px solid #30363d;">Avg legs filled (when fired)</td>'
        f'<td style="padding:6px 10px;border:1px solid #30363d;">{(L_fired_df["L_n_legs"].mean() if not L_fired_df.empty else 0):.2f}/3</td>'
        f'<td style="padding:6px 10px;border:1px solid #30363d;color:#8b949e;">Higher = more confirmations triggered = stronger evidence T-1 was actually fading</td></tr>'
        f'<tr><td style="padding:6px 10px;border:1px solid #30363d;">T-1 close P&L (when fired)</td>'
        f'<td style="padding:6px 10px;border:1px solid #30363d;color:{"#3fb950" if L_t1_pnl_mean>0 else "#f85149"};font-weight:bold;">{L_t1_pnl_mean:+.2f}%</td>'
        f'<td style="padding:6px 10px;border:1px solid #30363d;color:#8b949e;">Median {L_t1_pnl_med:+.2f}%</td></tr>'
        f'<tr><td style="padding:6px 10px;border:1px solid #30363d;">Held P&L through T close (when fired)</td>'
        f'<td style="padding:6px 10px;border:1px solid #30363d;color:{"#3fb950" if L_held_pnl_mean>0 else "#f85149"};font-weight:bold;">{L_held_pnl_mean:+.2f}%</td>'
        f'<td style="padding:6px 10px;border:1px solid #30363d;color:#8b949e;">Win rate {L_held_win:.0f}%, P10 {L_held_pnl_p10:+.2f}%</td></tr>'
        f'<tr><td style="padding:6px 10px;border:1px solid #30363d;">Held MAE P90 (when fired)</td>'
        f'<td style="padding:6px 10px;border:1px solid #30363d;color:#f85149;">{L_held_mae_p90:.2f}%</td>'
        f'<td style="padding:6px 10px;border:1px solid #30363d;color:#8b949e;">Median {L_held_mae_med:.2f}%, max {L_held_mae_max:.2f}%</td></tr>'
        f'<tr style="background-color:#1c3426;"><td style="padding:6px 10px;border:1px solid #30363d;font-weight:bold;">Portfolio held P&L per T-1 day</td>'
        f'<td style="padding:6px 10px;border:1px solid #30363d;color:{"#3fb950" if L_portfolio_held_pnl>=0 else "#f85149"};font-weight:bold;">{L_portfolio_held_pnl:+.2f}%</td>'
        f'<td style="padding:6px 10px;border:1px solid #30363d;color:#8b949e;">Counts no-fire days as 0. Fair vs upward ladder ({upward_held_pnl_mean:+.2f}%) which always enters.</td></tr>'
        '</table>'
    )

    # === Trade-grade cohorts ===
    df_grade_a = df[df["trade_grade"] == "A"]
    df_grade_b = df[df["trade_grade"] == "B"]
    df_grade_cd = df[df["trade_grade"].isin(["C", "D"])]
    grade_table = make_cohort_table([
        (f"Grade A (high conviction)", cohort_metrics(df_grade_a)),
        (f"Grade B (medium)", cohort_metrics(df_grade_b)),
        (f"Grade C/D (low)", cohort_metrics(df_grade_cd)),
    ])

    # Grade verdict — does Grade A protect more than Grade C/D?
    m_a = cohort_metrics(df_grade_a)
    m_cd = cohort_metrics(df_grade_cd)
    if m_a and m_cd:
        delta_grade = (m_a.get("held_pnl_mean", 0) or 0) - (m_cd.get("held_pnl_mean", 0) or 0)
        if delta_grade > 5:
            grade_verdict = (f"<span style='color:#3fb950;'><strong>Grade A protects.</strong> "
                             f"Grade A held P&L {m_a['held_pnl_mean']:+.2f}% vs Grade C/D "
                             f"{m_cd['held_pnl_mean']:+.2f}% — delta {delta_grade:+.2f}%. "
                             f"Higher-conviction setups have less day-early risk.</span>")
        elif delta_grade < -5:
            grade_verdict = (f"<span style='color:#f85149;'><strong>Grade A is WORSE on day-early.</strong> "
                             f"A: {m_a['held_pnl_mean']:+.2f}% vs C/D: {m_cd['held_pnl_mean']:+.2f}%. "
                             f"Counterintuitive — high-grade trades have bigger day-early traps.</span>")
        else:
            grade_verdict = (f"<span style='color:#e3b341;'><strong>Grade does not separate cohorts (~tie).</strong> "
                             f"A: {m_a['held_pnl_mean']:+.2f}%, C/D: {m_cd['held_pnl_mean']:+.2f}%.</span>")
    else:
        grade_verdict = "<span style='color:#8b949e;'>Insufficient grade data.</span>"

    # === pct_from_9ema cohorts (continuous extension) ===
    df_with_9ema = df.dropna(subset=["pct_from_9ema"])
    if len(df_with_9ema) >= 6:
        median_9ema = float(df_with_9ema["pct_from_9ema"].median())
        df_low_ext = df_with_9ema[df_with_9ema["pct_from_9ema"] <= median_9ema]
        df_high_ext = df_with_9ema[df_with_9ema["pct_from_9ema"] > median_9ema]
        ema_table = make_cohort_table([
            (f"Low extension (≤ {median_9ema*100:.1f}% from 9EMA)", cohort_metrics(df_low_ext)),
            (f"High extension (> {median_9ema*100:.1f}% from 9EMA)", cohort_metrics(df_high_ext)),
        ])
        m_lo = cohort_metrics(df_low_ext)
        m_hi = cohort_metrics(df_high_ext)
        delta_ema = (m_hi.get("held_pnl_mean", 0) or 0) - (m_lo.get("held_pnl_mean", 0) or 0)
        if delta_ema > 5:
            ema_verdict = (f"<span style='color:#3fb950;'><strong>High-extension setups protect.</strong> "
                           f"High-ext held P&L {m_hi['held_pnl_mean']:+.2f}% vs Low-ext "
                           f"{m_lo['held_pnl_mean']:+.2f}% — delta {delta_ema:+.2f}%. "
                           f"Stocks more extended = T more likely the climax = day-early risk lower.</span>")
        elif delta_ema < -5:
            ema_verdict = (f"<span style='color:#f85149;'><strong>Low-extension setups are SAFER on day-early.</strong> "
                           f"High-ext: {m_hi['held_pnl_mean']:+.2f}%, Low-ext: {m_lo['held_pnl_mean']:+.2f}%. "
                           f"Counterintuitive — implies the most extended setups have the worst day-early traps.</span>")
        else:
            ema_verdict = (f"<span style='color:#e3b341;'><strong>9EMA extension does not separate cohorts (~tie).</strong> "
                           f"High: {m_hi['held_pnl_mean']:+.2f}%, Low: {m_lo['held_pnl_mean']:+.2f}%.</span>")
    else:
        ema_table = '<p style="color:#8b949e;">Insufficient 9EMA data.</p>'
        ema_verdict = "<span style='color:#8b949e;'>Insufficient data for 9EMA split.</span>"

    def trade_rows(sub_df: pd.DataFrame, sort_col: str) -> str:
        rows = []
        for _, r in sub_df.iterrows():
            rows.append(
                f'<tr>'
                f'<td style="padding:4px 8px; border:1px solid #30363d;">{r["ticker"]}</td>'
                f'<td style="padding:4px 8px; border:1px solid #30363d;">{r["trade_date"]}</td>'
                f'<td style="padding:4px 8px; border:1px solid #30363d;">{fmt(r["t1_close_pnl_pct"])}</td>'
                f'<td style="padding:4px 8px; border:1px solid #30363d; color:#f85149;">{fmt(r["t1_intraday_mae_pct"])}</td>'
                f'<td style="padding:4px 8px; border:1px solid #30363d; color:#f85149;">{fmt(r["overnight_gap_pct"])}</td>'
                f'<td style="padding:4px 8px; border:1px solid #30363d;">{fmt(r["held_pnl_pct"])}</td>'
                f'<td style="padding:4px 8px; border:1px solid #30363d; color:#f85149;">{fmt(r["held_mae_pct"])}</td>'
                f'</tr>'
            )
        return "".join(rows)

    # Verdict
    if s_t1pnl["mean"] > 0:
        verdict_color = "#3fb950"
        verdict_word = "Day-early entries were still mostly profitable on the day."
    else:
        verdict_color = "#f85149"
        verdict_word = "Day-early entries lost money on average."
    if s_gap["mean"] > 0:
        gap_color = "#f85149"
        gap_word = f"Overnight gap averaged <strong>{s_gap['mean']:+.2f}%</strong> AGAINST the short."
    else:
        gap_color = "#3fb950"
        gap_word = f"Overnight gap averaged <strong>{s_gap['mean']:+.2f}%</strong> in your favor."

    html = f"""
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;
            max-width:920px;margin:0 auto;color:#c9d1d9;background-color:#161b22;
            font-size:14px;line-height:1.5;padding:16px;">

  <div style="background-color:#21262d;padding:14px 18px;border-radius:6px;border-bottom:2px solid #30363d;">
    <h1 style="margin:0;font-size:1.5em;">Day-Early Trap Audit — 3DGapFade Ladder</h1>
    <div style="font-size:0.9em;color:#8b949e;margin-top:4px;">
      n={n} 3DGapFade reversals | strategy: 33% at 9:30 / 33% at +0.5 ATR / 33% at +1.0 ATR on day T-1
    </div>
  </div>

  <div style="margin-top:20px;padding:12px 16px;border-left:4px solid {verdict_color};background-color:#0d1117;">
    <h2 style="margin:0 0 6px 0;color:#f0f6fc;">Verdict</h2>
    <div style="font-size:1.05em;color:{verdict_color};font-weight:bold;">{verdict_word}</div>
    <div style="font-size:1em;margin-top:6px;color:{gap_color};">{gap_word}</div>
    <div style="font-size:0.95em;margin-top:6px;color:#c9d1d9;">
      <strong>Held T-1 entry → T close: avg {s_held['mean']:+.2f}%, win rate {pct_held_win:.0f}%</strong>
      — but you would have endured median MAE <strong style="color:#f85149;">{s_heldmae['median']:.2f}%</strong>
      and P90 MAE <strong style="color:#f85149;">{s_heldmae['p90']:.2f}%</strong> along the way.
    </div>
  </div>

  <h2 style="color:#f0f6fc;margin-top:28px;">T-1 Outcomes (covering EOD on day-early)</h2>
  <table style="border-collapse:collapse;font-size:0.9em;width:100%;">
    <tr style="background-color:#21262d;">
      <th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Metric</th>
      <th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Mean</th>
      <th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Median</th>
      <th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">P10</th>
      <th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">P90</th>
      <th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">P95</th>
      <th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Worst</th>
    </tr>
    <tr>
      <td style="padding:6px 10px;border:1px solid #30363d;">T-1 close P&L (short)</td>
      <td style="padding:6px 10px;border:1px solid #30363d;color:{'#3fb950' if s_t1pnl['mean']>0 else '#f85149'};font-weight:bold;">{fmt(s_t1pnl['mean'])}</td>
      <td style="padding:6px 10px;border:1px solid #30363d;">{fmt(s_t1pnl['median'])}</td>
      <td style="padding:6px 10px;border:1px solid #30363d;">{fmt(s_t1pnl['p10'])}</td>
      <td style="padding:6px 10px;border:1px solid #30363d;">{fmt(s_t1pnl['p90'])}</td>
      <td style="padding:6px 10px;border:1px solid #30363d;">{fmt(s_t1pnl['p95'])}</td>
      <td style="padding:6px 10px;border:1px solid #30363d;">{fmt(s_t1pnl['min'])}</td>
    </tr>
    <tr>
      <td style="padding:6px 10px;border:1px solid #30363d;">T-1 intraday MAE (worst point)</td>
      <td style="padding:6px 10px;border:1px solid #30363d;color:#f85149;">{fmt_abs(s_t1mae['mean'])}</td>
      <td style="padding:6px 10px;border:1px solid #30363d;color:#f85149;">{fmt_abs(s_t1mae['median'])}</td>
      <td style="padding:6px 10px;border:1px solid #30363d;color:#f85149;">{fmt_abs(s_t1mae['p10'])}</td>
      <td style="padding:6px 10px;border:1px solid #30363d;color:#f85149;font-weight:bold;">{fmt_abs(s_t1mae['p90'])}</td>
      <td style="padding:6px 10px;border:1px solid #30363d;color:#f85149;">{fmt_abs(s_t1mae['p95'])}</td>
      <td style="padding:6px 10px;border:1px solid #30363d;color:#f85149;">{fmt_abs(s_t1mae['max'])}</td>
    </tr>
  </table>
  <div style="font-size:0.9em;color:#8b949e;margin-top:6px;">
    {pct_t1_red:.0f}% of T-1 days closed below the avg ladder entry (i.e., would have been profitable to cover at 4 PM).
  </div>

  <h2 style="color:#f0f6fc;margin-top:28px;">Overnight Gap (T-1 close → T open)</h2>
  <table style="border-collapse:collapse;font-size:0.9em;width:100%;">
    <tr style="background-color:#21262d;">
      <th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Metric</th>
      <th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Mean</th>
      <th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Median</th>
      <th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">P75</th>
      <th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">P90</th>
      <th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">P95</th>
      <th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Worst</th>
    </tr>
    <tr>
      <td style="padding:6px 10px;border:1px solid #30363d;">Gap % (positive = against short)</td>
      <td style="padding:6px 10px;border:1px solid #30363d;color:{'#f85149' if s_gap['mean']>0 else '#3fb950'};font-weight:bold;">{fmt(s_gap['mean'])}</td>
      <td style="padding:6px 10px;border:1px solid #30363d;">{fmt(s_gap['median'])}</td>
      <td style="padding:6px 10px;border:1px solid #30363d;">{fmt(s_gap['p75'])}</td>
      <td style="padding:6px 10px;border:1px solid #30363d;color:#f85149;font-weight:bold;">{fmt(s_gap['p90'])}</td>
      <td style="padding:6px 10px;border:1px solid #30363d;color:#f85149;">{fmt(s_gap['p95'])}</td>
      <td style="padding:6px 10px;border:1px solid #30363d;color:#f85149;">{fmt(s_gap['max'])}</td>
    </tr>
  </table>
  <div style="font-size:0.9em;color:#8b949e;margin-top:6px;">
    {pct_gap_against:.0f}% of overnights gapped UP against the short. The 9:45 rule's whole purpose is to avoid this.
  </div>

  <h2 style="color:#f0f6fc;margin-top:28px;">Held-through outcome (T-1 entry → T 4PM close)</h2>
  <p style="color:#8b949e;font-size:0.9em;">
    If you entered T-1 with the ladder and held all the way through to T 4PM close (the actual reversal day):
  </p>
  <table style="border-collapse:collapse;font-size:0.9em;width:100%;">
    <tr style="background-color:#21262d;">
      <th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Metric</th>
      <th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Mean</th>
      <th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Median</th>
      <th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">P10</th>
      <th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">P90</th>
      <th style="padding:6px 10px;border:1px solid #30363d;text-align:left;">Worst</th>
    </tr>
    <tr>
      <td style="padding:6px 10px;border:1px solid #30363d;">P&L through T close</td>
      <td style="padding:6px 10px;border:1px solid #30363d;color:{'#3fb950' if s_held['mean']>0 else '#f85149'};font-weight:bold;">{fmt(s_held['mean'])}</td>
      <td style="padding:6px 10px;border:1px solid #30363d;">{fmt(s_held['median'])}</td>
      <td style="padding:6px 10px;border:1px solid #30363d;">{fmt(s_held['p10'])}</td>
      <td style="padding:6px 10px;border:1px solid #30363d;">{fmt(s_held['p90'])}</td>
      <td style="padding:6px 10px;border:1px solid #30363d;">{fmt(s_held['min'])}</td>
    </tr>
    <tr>
      <td style="padding:6px 10px;border:1px solid #30363d;">Held MAE (worst point ever)</td>
      <td style="padding:6px 10px;border:1px solid #30363d;color:#f85149;">{fmt_abs(s_heldmae['mean'])}</td>
      <td style="padding:6px 10px;border:1px solid #30363d;color:#f85149;">{fmt_abs(s_heldmae['median'])}</td>
      <td style="padding:6px 10px;border:1px solid #30363d;color:#f85149;">{fmt_abs(s_heldmae['p10'])}</td>
      <td style="padding:6px 10px;border:1px solid #30363d;color:#f85149;font-weight:bold;">{fmt_abs(s_heldmae['p90'])}</td>
      <td style="padding:6px 10px;border:1px solid #30363d;color:#f85149;">{fmt_abs(s_heldmae['max'])}</td>
    </tr>
  </table>
  <div style="font-size:0.9em;color:#8b949e;margin-top:6px;">
    Held win rate (P&L > 0 by T close): <strong>{pct_held_win:.0f}%</strong>.
  </div>

  <h2 style="color:#f0f6fc;margin-top:28px;">By T-1 VWAP Regime — does VWAP filter the trap?</h2>
  <p style="color:#8b949e;font-size:0.9em;">
    Splits the day-early simulations by whether the stock opened below or above PM VWAP
    on day T-1.
  </p>
  {vwap_cohort_html}

  <h2 style="color:#f0f6fc;margin-top:28px;">By T-1 Daily Close Direction — does close-color filter the trap?</h2>
  <p style="color:#8b949e;font-size:0.9em;">
    The hypothesis: a T-1 that closed RED (close &lt; open) is showing intraday weakness
    even mid-uptrend, so day-early shorts there are safer. A GREEN T-1 means the stock is
    still extending — day-early should be most punishing.
  </p>
  <div style="margin-bottom:8px;">{green_red_verdict}</div>
  {green_red_table}

  <h2 style="color:#f0f6fc;margin-top:28px;">By T-1 Close vs T-2 High — sustained breakout filter</h2>
  <p style="color:#8b949e;font-size:0.9em;">
    The hypothesis: if T-1 closed <em>above</em> T-2's high, the stock is in sustained
    breakout/momentum mode — most likely to keep extending = day-early trap. If T-1 closed
    <em>below</em> T-2's high (failed-test pattern), the rally is already weakening = safer
    to short on T-1.
  </p>
  <div style="margin-bottom:8px;">{above_t2_verdict}</div>
  {above_t2_table}

  <h2 style="color:#f0f6fc;margin-top:28px;">By Trade Grade — does setup quality reduce day-early risk?</h2>
  <p style="color:#8b949e;font-size:0.9em;">
    The hypothesis: high-grade trades (Grade A, all setup criteria firing) represent climactic
    extension — T is more likely THE day. Lower-grade trades (B/C/D) might be earlier in the
    move, where T-1 was already a setup but the climax hadn't fully built. If grade reflects
    selection conviction, day-early risk should be LOWER on Grade A trades.
  </p>
  <div style="margin-bottom:8px;">{grade_verdict}</div>
  {grade_table}

  <h2 style="color:#f0f6fc;margin-top:28px;">By 9EMA Extension — does extension level distinguish climax day?</h2>
  <p style="color:#8b949e;font-size:0.9em;">
    Splits the 35 trades by their <code>pct_from_9ema</code> on the trade day, at the median.
    The hypothesis: more extended (further from 9EMA) = T more likely the climax = day-early
    risk lower. Continuous metric — less subjective than trade_grade.
  </p>
  <div style="margin-bottom:8px;">{ema_verdict}</div>
  {ema_table}

  <h2 style="color:#f0f6fc;margin-top:28px;">Strategy L — Confirmation Ladder on T-1 (the protection test)</h2>
  <p style="color:#8b949e;font-size:0.9em;">
    Strategy L (33% on VWAP break + 33% on PM-low break + 33% on open-fail break) only fires
    legs on confirmed downside. On a typical day-early trap (stock keeps grinding up), few or no
    legs should fire — that's the protection mechanism. Compares to the upward ladder above.
  </p>
  {l_summary_html}

  <h2 style="color:#f0f6fc;margin-top:28px;">Upward-Ladder Leg Fire Rates on T-1</h2>
  <ul style="color:#c9d1d9;">
    <li>Avg legs filled: <strong>{avg_legs:.2f}/3</strong></li>
    <li>+0.5 ATR leg hit T-1: <strong>{leg_50_pct:.0f}%</strong> of trades</li>
    <li>+1.0 ATR leg hit T-1: <strong>{leg_100_pct:.0f}%</strong> of trades</li>
  </ul>

  <h2 style="color:#f0f6fc;margin-top:28px;">Worst Overnight Gaps</h2>
  <table style="border-collapse:collapse;font-size:0.85em;width:100%;">
    <tr style="background-color:#21262d;">
      <th style="padding:4px 8px;border:1px solid #30363d;text-align:left;">Ticker</th>
      <th style="padding:4px 8px;border:1px solid #30363d;text-align:left;">T date</th>
      <th style="padding:4px 8px;border:1px solid #30363d;text-align:left;">T-1 close P&L</th>
      <th style="padding:4px 8px;border:1px solid #30363d;text-align:left;">T-1 MAE</th>
      <th style="padding:4px 8px;border:1px solid #30363d;text-align:left;">Overnight gap</th>
      <th style="padding:4px 8px;border:1px solid #30363d;text-align:left;">Held P&L</th>
      <th style="padding:4px 8px;border:1px solid #30363d;text-align:left;">Held MAE</th>
    </tr>
    {trade_rows(worst_overnight, "overnight_gap_pct")}
  </table>

  <h2 style="color:#f0f6fc;margin-top:28px;">Worst T-1 Intraday MAE</h2>
  <table style="border-collapse:collapse;font-size:0.85em;width:100%;">
    <tr style="background-color:#21262d;">
      <th style="padding:4px 8px;border:1px solid #30363d;text-align:left;">Ticker</th>
      <th style="padding:4px 8px;border:1px solid #30363d;text-align:left;">T date</th>
      <th style="padding:4px 8px;border:1px solid #30363d;text-align:left;">T-1 close P&L</th>
      <th style="padding:4px 8px;border:1px solid #30363d;text-align:left;">T-1 MAE</th>
      <th style="padding:4px 8px;border:1px solid #30363d;text-align:left;">Overnight gap</th>
      <th style="padding:4px 8px;border:1px solid #30363d;text-align:left;">Held P&L</th>
      <th style="padding:4px 8px;border:1px solid #30363d;text-align:left;">Held MAE</th>
    </tr>
    {trade_rows(worst_t1_mae, "t1_intraday_mae_pct")}
  </table>

  <h2 style="color:#f0f6fc;margin-top:28px;">Worst Held Outcomes (T-1 entry → T close)</h2>
  <table style="border-collapse:collapse;font-size:0.85em;width:100%;">
    <tr style="background-color:#21262d;">
      <th style="padding:4px 8px;border:1px solid #30363d;text-align:left;">Ticker</th>
      <th style="padding:4px 8px;border:1px solid #30363d;text-align:left;">T date</th>
      <th style="padding:4px 8px;border:1px solid #30363d;text-align:left;">T-1 close P&L</th>
      <th style="padding:4px 8px;border:1px solid #30363d;text-align:left;">T-1 MAE</th>
      <th style="padding:4px 8px;border:1px solid #30363d;text-align:left;">Overnight gap</th>
      <th style="padding:4px 8px;border:1px solid #30363d;text-align:left;">Held P&L</th>
      <th style="padding:4px 8px;border:1px solid #30363d;text-align:left;">Held MAE</th>
    </tr>
    {trade_rows(worst_held, "held_pnl_pct")}
  </table>

  <h2 style="color:#f0f6fc;margin-top:32px;">Caveats</h2>
  <ul style="color:#8b949e;font-size:0.9em;">
    <li>Assumes you would have entered T-1 ladder unconditionally. In practice not every T-1 would have triggered the same setup criteria — trades with <code>consecutive_up_days=1</code> on T (n varies) wouldn't have qualified on T-1 yet. This is therefore a worst-case "always-on" estimate; the real day-early hit rate is likely lower.</li>
    <li>P&L measures hold-to-close. Real-world stops would cap the worst outcomes earlier.</li>
    <li>No transaction cost / borrow / overnight financing.</li>
  </ul>

</div>
"""
    return html


def main():
    df = pd.read_csv(PROJECT_ROOT / "data" / "reversal_data.csv")
    df = df.dropna(subset=["ticker", "date"]).copy()
    df["parsed_date"] = df["date"].apply(parse_csv_date)
    df_3dgf = df[df["setup"] == "3DGapFade"].copy()
    log.info(f"Loaded {len(df)} reversal trades; {len(df_3dgf)} are 3DGapFade")

    rows = list(df_3dgf[["ticker", "parsed_date", "atr_pct", "trade_grade", "pct_from_9ema"]].itertuples(index=False, name="R"))

    results: List[Dict] = []
    failures: List[str] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(work, r): r for r in rows}
        for i, fut in enumerate(as_completed(futs), 1):
            out, err = fut.result()
            if out is not None:
                results.append(out)
            else:
                failures.append(err or "unknown")
            if i % 10 == 0:
                log.info(f"  {i}/{len(rows)} processed (ok={len(results)}, fail={len(failures)})")

    log.info(f"Done. ok={len(results)}, fail={len(failures)}")
    if failures:
        for f in failures[:5]:
            log.warning(f"  example failure: {f}")
    if not results:
        log.error("No results — aborting.")
        return

    df_r = pd.DataFrame(results)

    html = build_html(df_r)

    out_path = PROJECT_ROOT / "scripts" / "reports" / "analyze_day_early_3dgapfade.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    log.info(f"Saved report: {out_path}")

    today_str = dt.date.today().strftime("%m/%d/%Y")
    subject = f"Day-Early Trap Audit — 3DGapFade Ladder | n={len(df_r)} | {today_str}"
    try:
        send_email(to_email="zmburr@gmail.com", subject=subject, body=html, is_html=True)
        log.info(f"Email sent: {subject}")
    except Exception as e:
        log.error(f"Failed to send email: {e}")


if __name__ == "__main__":
    main()
