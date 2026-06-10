"""Signal Scorecard v2 — validates priority-report GO/CAUTION signals against actual price action.

Replaces dispatcher.signal_scorecard. Key differences from v1:
- Evening signals are validated against the NEXT trading day (v1 scored them
  against the same day's bar, which the evening report was generated FROM).
- Multi-day outcome window (D0..D+3) instead of signal day only, so a reversal
  that cracks on day 2 is no longer scored as a miss.
- "Tradeable" = MFE >= 1.0x ATR at any point in the window, independent of
  where the day-0 close lands. Direction-of-close is reported separately.
- Tracks earliness: max adverse run (in ATRs) before the favorable move, and
  days until the 1-ATR favorable target hits.
- Holiday-aware trading calendar; backfills any signal dates it missed.
- Idempotent: re-runs update incomplete rows (signals whose 4-day window is
  still open) and skip everything already final.

Usage:
    python scripts/signal_scorecard.py            # validate/refresh + email
    python scripts/signal_scorecard.py --dry      # no CSV writes, no email
    python scripts/signal_scorecard.py --no-email # update CSV only
"""

import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas_market_calendars as mcal
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from support.config import send_email  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("signal_scorecard")

SIGNAL_DIR = PROJECT_ROOT / "data" / "priority_signals"
OUTCOMES_FILE = PROJECT_ROOT / "data" / "signal_outcomes.csv"
EMAIL_TO = "zmburr@gmail.com"

HORIZON_DAYS = 4          # D0..D+3
MFE_ATR_THRESHOLD = 1.0   # favorable excursion >= 1 ATR = tradeable
ROLLING_WINDOW = 30       # calendar days for rolling summary
ALERT_WARN = 70.0         # % — warn banner
ALERT_BAD = 50.0          # % — red banner

COLUMNS = [
    "signal_date", "session", "target_date", "ticker", "bucket", "cap",
    "recommendation", "score", "atr_pct",
    "entry_open", "d0_pct", "d1_pct", "d2_pct", "d3_pct",
    "mfe_atr_d0", "mfe_atr_3d", "mae_atr_3d",
    "adverse_before_fav_atr", "days_to_1atr",
    "dir_correct_d0", "dir_correct_final",
    "tradeable_d0", "tradeable_3d",
    "days_available", "complete",
]


# ── Trading calendar ─────────────────────────────────────────────────────────

_NYSE = mcal.get_calendar("NYSE")
_SCHEDULE_CACHE: list[str] | None = None


def trading_days() -> list[str]:
    """Sorted list of NYSE trading days (YYYY-MM-DD) covering all signal files +10d ahead."""
    global _SCHEDULE_CACHE
    if _SCHEDULE_CACHE is None:
        sched = _NYSE.schedule(start_date="2026-03-01",
                               end_date=(datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d"))
        _SCHEDULE_CACHE = [d.strftime("%Y-%m-%d") for d in sched.index]
    return _SCHEDULE_CACHE


def next_trading_day(date: str) -> str | None:
    for d in trading_days():
        if d > date:
            return d
    return None


def last_completed_trading_day() -> str:
    """Most recent trading day whose session has closed (4pm ET ~= local for this box)."""
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    days = [d for d in trading_days() if d <= today]
    if days and days[-1] == today and now.hour < 16:
        days = days[:-1]
    return days[-1]


def trading_days_from(start: str, n: int) -> list[str]:
    """Up to n trading days starting at `start` (inclusive), capped at last completed day."""
    last = last_completed_trading_day()
    return [d for d in trading_days() if start <= d <= last][:n]


# ── Polygon ──────────────────────────────────────────────────────────────────


def _polygon_client():
    from polygon import RESTClient
    return RESTClient(api_key=os.getenv("POLYGON_API_KEY"))


def fetch_daily_bars(client, ticker: str, start: str, end: str) -> dict[str, dict]:
    """Daily OHLC bars keyed by YYYY-MM-DD date."""
    try:
        aggs = client.get_aggs(ticker, 1, "day", from_=start, to=end, limit=50)
        out = {}
        for bar in aggs:
            d = datetime.fromtimestamp(bar.timestamp / 1000).strftime("%Y-%m-%d")
            out[d] = {"open": bar.open, "high": bar.high, "low": bar.low, "close": bar.close}
        return out
    except Exception as e:
        logger.warning(f"Polygon fetch failed for {ticker} {start}..{end}: {e}")
        return {}


# ── Signal collection ────────────────────────────────────────────────────────


def collect_signals() -> dict[tuple, dict]:
    """All signals from priority_signals/, keyed by (target_date, ticker, bucket).

    Morning file for date D targets D; evening file targets the next trading day.
    When both sessions flag the same ticker for the same target day, the morning
    signal wins (fresher premarket data).
    """
    signals: dict[tuple, dict] = {}
    for path in sorted(SIGNAL_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            logger.warning(f"Failed to parse {path.name}: {e}")
            continue
        date, session = path.stem.rsplit("_", 1)
        target = date if session == "morning" else next_trading_day(date)
        if target is None:
            continue
        for s in data.get("signals", []):
            if s.get("bucket") not in ("bounce", "reversal"):
                continue
            key = (target, s["ticker"], s["bucket"])
            if key in signals and signals[key]["session"] == "morning":
                continue  # morning beats evening for the same target
            signals[key] = {
                "signal_date": date,
                "session": session,
                "target_date": target,
                "ticker": s["ticker"],
                "bucket": s["bucket"],
                "cap": s.get("cap", ""),
                "recommendation": s.get("recommendation", ""),
                "score": s.get("score", ""),
                "atr_pct": s.get("metrics", {}).get("atr_pct"),
            }
    return signals


# ── Outcome computation ──────────────────────────────────────────────────────


def compute_outcome(sig: dict, bars_by_day: dict[str, dict]) -> dict:
    """Score one signal over its multi-day window. Favorable = down for reversal, up for bounce."""
    window = trading_days_from(sig["target_date"], HORIZON_DAYS)
    bars = [bars_by_day[d] for d in window if d in bars_by_day]

    row = {**sig}
    row["atr_pct"] = round(sig["atr_pct"], 6) if isinstance(sig["atr_pct"], (int, float)) else ""
    row["days_available"] = len(bars)
    # complete = full horizon scored, or window can never grow (delisted/halted with partial data)
    row["complete"] = len(bars) >= HORIZON_DAYS or (len(window) >= HORIZON_DAYS and len(bars) < len(window))
    if not bars:
        for c in COLUMNS:
            row.setdefault(c, "")
        row["complete"] = False
        return row

    o = bars[0]["open"]
    is_short = sig["bucket"] == "reversal"
    atr = sig["atr_pct"] if isinstance(sig["atr_pct"], (int, float)) and sig["atr_pct"] > 0 else None

    def fav(px):  # favorable move from entry open, as +pct
        return (o - px) / o if is_short else (px - o) / o

    row["entry_open"] = o
    closes = [fav(b["close"]) for b in bars]
    for i in range(HORIZON_DAYS):
        signed = (-closes[i] if is_short else closes[i]) if i < len(closes) else None
        row[f"d{i}_pct"] = round(signed, 6) if signed is not None else ""

    # excursions per day: best/worst price within each bar
    day_mfe = [fav(b["low"] if is_short else b["high"]) for b in bars]
    day_mae = [-fav(b["high"] if is_short else b["low"]) for b in bars]  # adverse as +pct

    mfe = max(day_mfe)
    mae = max(max(day_mae), 0.0)
    row["mfe_atr_d0"] = round(day_mfe[0] / atr, 2) if atr else ""
    row["mfe_atr_3d"] = round(mfe / atr, 2) if atr else ""
    row["mae_atr_3d"] = round(mae / atr, 2) if atr else ""

    # earliness: days until 1-ATR favorable hit, and worst adverse run before that day
    days_to, adverse_before = "", ""
    if atr:
        for i, m in enumerate(day_mfe):
            if m / atr >= MFE_ATR_THRESHOLD:
                days_to = i
                adverse_before = round(max(day_mae[: i + 1]) / atr, 2)
                break
    row["days_to_1atr"] = days_to
    row["adverse_before_fav_atr"] = adverse_before

    row["dir_correct_d0"] = closes[0] > 0
    row["dir_correct_final"] = closes[-1] > 0
    row["tradeable_d0"] = bool(closes[0] > 0 and atr and day_mfe[0] / atr >= MFE_ATR_THRESHOLD)
    row["tradeable_3d"] = bool(atr and mfe / atr >= MFE_ATR_THRESHOLD)
    return row


# ── CSV store ────────────────────────────────────────────────────────────────


def load_outcomes() -> dict[tuple, dict]:
    if not OUTCOMES_FILE.exists():
        return {}
    with open(OUTCOMES_FILE, newline="") as f:
        return {(r["target_date"], r["ticker"], r["bucket"]): r for r in csv.DictReader(f)}


def save_outcomes(rows: dict[tuple, dict]):
    with open(OUTCOMES_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow({c: rows[key].get(c, "") for c in COLUMNS})


# ── Rolling summary ──────────────────────────────────────────────────────────


def _is_true(v) -> bool:
    return v is True or str(v).lower() == "true"


def _pct(rows, field):
    if not rows:
        return None
    n = sum(1 for r in rows if _is_true(r.get(field)))
    return {"correct": n, "total": len(rows), "pct": round(n / len(rows) * 100, 1)}


def timing_curve(group: list[dict]) -> dict | None:
    """Cumulative 1-ATR hit curve: % paying on day 0, by day 1/2/3, never.

    Denominator is ALL complete signals in the group, so 'never' is honest.
    """
    n = len(group)
    if n == 0:
        return None
    days = [int(float(r["days_to_1atr"])) for r in group if r.get("days_to_1atr") != ""]
    cum = {}
    for k in range(HORIZON_DAYS):
        cum[k] = sum(1 for d in days if d <= k)
    hits = len(days)
    curve = {
        "n": n,
        "d0": {"count": cum[0], "pct": round(cum[0] / n * 100, 1)},
        "le1": {"count": cum[1], "pct": round(cum[1] / n * 100, 1)},
        "le2": {"count": cum[2], "pct": round(cum[2] / n * 100, 1)},
        "le3": {"count": cum[3], "pct": round(cum[3] / n * 100, 1)},
        "never": {"count": n - hits, "pct": round((n - hits) / n * 100, 1)},
        "median_days": sorted(days)[len(days) // 2] if days else None,
    }
    # direction-correct close per day (favorable close if held to that day's close)
    for k in range(HORIZON_DAYS):
        scored = [r for r in group if r.get(f"d{k}_pct") not in ("", None)]
        if scored:
            fav = sum(1 for r in scored
                      if (float(r[f"d{k}_pct"]) < 0) == (r["bucket"] == "reversal")
                      and float(r[f"d{k}_pct"]) != 0)
            curve[f"dir_d{k}"] = round(fav / len(scored) * 100, 1)
    return curve


def rolling_summary(rows: list[dict]) -> dict:
    cutoff = (datetime.now() - timedelta(days=ROLLING_WINDOW)).strftime("%Y-%m-%d")
    recent = [r for r in rows if r.get("target_date", "") >= cutoff and _is_true(r.get("complete"))
              and str(r.get("days_available") or "0") != "0"]  # exclude no-data (delisted) rows
    rev = [r for r in recent if r.get("bucket") == "reversal"]
    bnc = [r for r in recent if r.get("bucket") == "bounce"]
    groups = {
        "Overall": recent,
        "GO": [r for r in recent if r.get("recommendation") == "GO"],
        "CAUTION": [r for r in recent if r.get("recommendation") == "CAUTION"],
        "Reversals": rev,
        "Bounces": bnc,
    }
    timing_groups = {
        "Reversal GO": [r for r in rev if r.get("recommendation") == "GO"],
        "Reversal CAUTION": [r for r in rev if r.get("recommendation") == "CAUTION"],
        "Reversal · morning signals": [r for r in rev if r.get("session") == "morning"],
        "Reversal · evening signals": [r for r in rev if r.get("session") == "evening"],
        "Bounce GO": [r for r in bnc if r.get("recommendation") == "GO"],
        "Bounce CAUTION": [r for r in bnc if r.get("recommendation") == "CAUTION"],
    }
    out = {"total": len(recent), "groups": {}, "timing": {}}
    for label, g in groups.items():
        out["groups"][label] = {
            "tradeable_3d": _pct(g, "tradeable_3d"),
            "tradeable_d0": _pct(g, "tradeable_d0"),
            "dir_final": _pct(g, "dir_correct_final"),
        }
    for label, g in timing_groups.items():
        curve = timing_curve(g)
        if curve and curve["n"] >= 5:
            out["timing"][label] = curve
    # earliness stats for reversals that eventually worked
    rev_hits = [r for r in recent if r.get("bucket") == "reversal" and r.get("days_to_1atr") != ""]
    if rev_hits:
        days = sorted(int(float(r["days_to_1atr"])) for r in rev_hits)
        adv = sorted(float(r["adverse_before_fav_atr"]) for r in rev_hits if r.get("adverse_before_fav_atr") != "")
        out["reversal_earliness"] = {
            "n": len(rev_hits),
            "median_days_to_1atr": days[len(days) // 2],
            "pct_hit_day0": round(sum(1 for d in days if d == 0) / len(days) * 100, 0),
            "median_adverse_before_atr": adv[len(adv) // 2] if adv else None,
            "max_adverse_before_atr": max(adv) if adv else None,
        }
    return out


# ── Email ────────────────────────────────────────────────────────────────────


def format_email(new_rows: list[dict], updated: int, summary: dict, target_date: str) -> str:
    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    td = "padding:6px 10px;border-bottom:1px solid #1e2330;"

    def fmt_pct(v):
        return f"{float(v) * 100:+.2f}%" if v not in ("", None) else "—"

    if new_rows:
        body_rows = ""
        for r in new_rows:
            t3 = _is_true(r.get("tradeable_3d"))
            color = "#22c55e" if t3 else "#f59e0b" if _is_true(r.get("dir_correct_final")) else "#ef4444"
            mfe = r.get("mfe_atr_3d", "")
            mae = r.get("mae_atr_3d", "")
            dt = r.get("days_to_1atr", "")
            body_rows += f"""<tr>
              <td style="{td}color:#e8ecf4;font-weight:500;">{r['ticker']}</td>
              <td style="{td}color:#9ca3af;">{r['bucket']}</td>
              <td style="{td}color:#9ca3af;">{r['recommendation']} {r.get('score', '')}</td>
              <td style="{td}color:{color};font-weight:600;">{fmt_pct(r.get('d0_pct'))}</td>
              <td style="{td}color:#9ca3af;">{mfe}x</td>
              <td style="{td}color:#9ca3af;">{mae}x</td>
              <td style="{td}color:#9ca3af;text-align:center;">{dt if dt != '' else '—'}</td>
              <td style="{td}color:{'#22c55e' if t3 else '#ef4444'};font-weight:600;text-align:center;">{'&#10003;' if t3 else '&#10007;'}</td>
            </tr>"""
        table = f"""<table style="width:100%;border-collapse:collapse;font-size:13px;font-family:'JetBrains Mono',monospace;">
          <thead><tr style="border-bottom:2px solid #2a3040;">
            <th style="padding:8px 10px;text-align:left;color:#6b7280;">Ticker</th>
            <th style="padding:8px 10px;text-align:left;color:#6b7280;">Bucket</th>
            <th style="padding:8px 10px;text-align:left;color:#6b7280;">Rec</th>
            <th style="padding:8px 10px;text-align:left;color:#6b7280;">D0</th>
            <th style="padding:8px 10px;text-align:left;color:#6b7280;">MFE 3d</th>
            <th style="padding:8px 10px;text-align:left;color:#6b7280;">MAE 3d</th>
            <th style="padding:8px 10px;text-align:center;color:#6b7280;">Days&rarr;1ATR</th>
            <th style="padding:8px 10px;text-align:center;color:#6b7280;">Tradeable</th>
          </tr></thead><tbody>{body_rows}</tbody></table>
          <p style="color:#4b5563;font-size:11px;">MFE/MAE in ATRs over the D0&ndash;D+3 window (window may still be open for recent signals). Days&rarr;1ATR = trading days until the 1-ATR favorable target hit.</p>"""
    else:
        table = "<p style='color:#6b7280;'>No new signals validated today.</p>"

    # rolling summary
    sum_html = ""
    if summary.get("total"):
        rows_html = ""
        for label, g in summary["groups"].items():
            t3, t0, df = g["tradeable_3d"], g["tradeable_d0"], g["dir_final"]
            if not t3:
                continue
            c = "#22c55e" if t3["pct"] >= ALERT_WARN else "#f59e0b" if t3["pct"] >= ALERT_BAD else "#ef4444"
            rows_html += f"""<tr>
              <td style="{td}color:#e8ecf4;">{label}</td>
              <td style="{td}color:{c};font-weight:600;">{t3['pct']}% ({t3['correct']}/{t3['total']})</td>
              <td style="{td}color:#9ca3af;">{t0['pct']}%</td>
              <td style="{td}color:#9ca3af;">{df['pct']}%</td>
            </tr>"""
        # timing breakdown — when do signals pay?
        timing_html = ""
        if summary.get("timing"):
            t_rows = ""
            for label, t in summary["timing"].items():
                cells = ""
                for key, hdr in (("d0", None), ("le1", None), ("le2", None), ("le3", None), ("never", None)):
                    s = t[key]
                    c = "#ef4444" if key == "never" and s["pct"] > 50 else "#e8ecf4" if key == "d0" else "#9ca3af"
                    cells += f'<td style="{td}color:{c};">{s["pct"]}% <span style="color:#4b5563;">({s["count"]}/{t["n"]})</span></td>'
                t_rows += f'<tr><td style="{td}color:#e8ecf4;">{label}</td>{cells}</tr>'

            takeaways = []
            for label in ("Reversal GO", "Bounce GO"):
                t = summary["timing"].get(label)
                if t:
                    takeaways.append(
                        f"<strong>{label}:</strong> {t['d0']['pct']}% pay (&ge;1 ATR) on the signal day, "
                        f"{t['le1']['pct']}% within a day after, {t['le3']['pct']}% within 3 days; "
                        f"{t['never']['pct']}% never do."
                        + (f" Median wait when it works: {t['median_days']} day(s)." if t["median_days"] is not None else "")
                    )
            takeaway_html = "".join(f'<p style="color:#9ca3af;font-size:12px;margin:8px 0 0 0;">{x}</p>' for x in takeaways)

            dir_rows = ""
            for label, t in summary["timing"].items():
                cells = "".join(
                    f'<td style="{td}color:#9ca3af;">{t.get(f"dir_d{k}", "—")}%</td>' for k in range(HORIZON_DAYS)
                )
                dir_rows += f'<tr><td style="{td}color:#e8ecf4;">{label}</td>{cells}</tr>'

            timing_html = f"""<div style="background:#161a24;padding:12px 16px;border-radius:6px;margin:16px 0;border-left:3px solid #f59e0b;">
              <strong style="color:#f59e0b;">Timing &mdash; when do signals pay? (cumulative 1-ATR hit rate)</strong>
              <table style="width:100%;border-collapse:collapse;font-size:12px;margin-top:8px;">
                <thead><tr style="border-bottom:1px solid #2a3040;">
                  <th style="padding:6px 10px;text-align:left;color:#6b7280;"></th>
                  <th style="padding:6px 10px;text-align:left;color:#6b7280;">Day 0 (signal day)</th>
                  <th style="padding:6px 10px;text-align:left;color:#6b7280;">&le; Day 1</th>
                  <th style="padding:6px 10px;text-align:left;color:#6b7280;">&le; Day 2</th>
                  <th style="padding:6px 10px;text-align:left;color:#6b7280;">&le; Day 3</th>
                  <th style="padding:6px 10px;text-align:left;color:#6b7280;">Never</th>
                </tr></thead><tbody>{t_rows}</tbody></table>
              {takeaway_html}
              <details style="margin-top:10px;"><summary style="color:#6b7280;font-size:11px;cursor:pointer;">Direction-correct close by day (if held to that day's close)</summary>
              <table style="width:100%;border-collapse:collapse;font-size:12px;margin-top:6px;">
                <thead><tr style="border-bottom:1px solid #2a3040;">
                  <th style="padding:6px 10px;text-align:left;color:#6b7280;"></th>
                  <th style="padding:6px 10px;text-align:left;color:#6b7280;">D0 close</th>
                  <th style="padding:6px 10px;text-align:left;color:#6b7280;">D1 close</th>
                  <th style="padding:6px 10px;text-align:left;color:#6b7280;">D2 close</th>
                  <th style="padding:6px 10px;text-align:left;color:#6b7280;">D3 close</th>
                </tr></thead><tbody>{dir_rows}</tbody></table></details>
            </div>"""

        early_html = ""
        e = summary.get("reversal_earliness")
        if e:
            early_html = f"""<p style="color:#9ca3af;font-size:12px;margin:10px 0 0 0;">
              <strong style="color:#3b82f6;">Reversal earliness</strong> (n={e['n']} that hit 1 ATR):
              median {e['median_days_to_1atr']} day(s) to target, {e['pct_hit_day0']:.0f}% hit on day 0,
              median adverse run before the move {e['median_adverse_before_atr']}x ATR
              (max {e['max_adverse_before_atr']}x).</p>"""
        sum_html = f"""<div style="background:#161a24;padding:12px 16px;border-radius:6px;margin:16px 0;border-left:3px solid #3b82f6;">
          <strong style="color:#3b82f6;">Rolling {ROLLING_WINDOW}-Day (complete windows only, n={summary['total']})</strong>
          <table style="width:100%;border-collapse:collapse;font-size:12px;margin-top:8px;">
            <thead><tr style="border-bottom:1px solid #2a3040;">
              <th style="padding:6px 10px;text-align:left;color:#6b7280;"></th>
              <th style="padding:6px 10px;text-align:left;color:#6b7280;">Tradeable (3d MFE&ge;1ATR)</th>
              <th style="padding:6px 10px;text-align:left;color:#6b7280;">Tradeable D0 (legacy)</th>
              <th style="padding:6px 10px;text-align:left;color:#6b7280;">Direction (final close)</th>
            </tr></thead><tbody>{rows_html}</tbody></table>
          {early_html}
        </div>
        {timing_html}"""

    # alert banner
    alert_html = ""
    go = summary.get("groups", {}).get("GO", {}).get("tradeable_3d")
    if go and go["total"] >= 10 and go["pct"] < ALERT_WARN:
        sev = "#ef4444" if go["pct"] < ALERT_BAD else "#f59e0b"
        alert_html = f"""<div style="background:rgba(239,68,68,0.08);padding:12px 16px;border-radius:6px;margin:16px 0;border-left:3px solid {sev};">
          <strong style="color:{sev};">GO tradeable rate {go['pct']}% ({go['correct']}/{go['total']}) over rolling {ROLLING_WINDOW}d</strong>
        </div>"""

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background-color:#0a0c10;color:#c8cdd8;padding:20px;max-width:860px;margin:0 auto;">
  <div style="border-bottom:2px solid #f59e0b;padding-bottom:12px;margin-bottom:20px;">
    <h1 style="color:#e8ecf4;font-size:20px;margin:0 0 6px 0;">Signal Scorecard v2</h1>
    <span style="color:#6b7280;font-size:13px;">{now} &mdash; latest target day {target_date} &mdash; {updated} window(s) refreshed</span>
  </div>
  {alert_html}
  {table}
  {sum_html}
  <div style="border-top:1px solid #1e2330;margin-top:30px;padding-top:12px;color:#4b5563;font-size:11px;">
    Signal Scorecard v2 &mdash; Backtester &mdash; data/signal_outcomes.csv
  </div>
</body></html>"""


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    dry = "--dry" in sys.argv
    no_email = "--no-email" in sys.argv or dry
    logger.info(f"Signal Scorecard v2 starting{' (dry run)' if dry else ''}...")

    signals = collect_signals()
    outcomes = load_outcomes()
    last_day = last_completed_trading_day()
    logger.info(f"{len(signals)} signals on file, {len(outcomes)} already scored, last completed day {last_day}")

    client = _polygon_client()
    updated, new_today = 0, []
    for key, sig in sorted(signals.items()):
        target = sig["target_date"]
        if target > last_day:
            continue  # window hasn't opened yet
        existing = outcomes.get(key)
        if existing and _is_true(existing.get("complete")):
            continue
        # how many window days could be scored by now?
        possible = len(trading_days_from(target, HORIZON_DAYS))
        if existing and int(existing.get("days_available") or 0) >= possible:
            continue  # nothing new to add yet

        window = trading_days_from(target, HORIZON_DAYS)
        bars = fetch_daily_bars(client, sig["ticker"], window[0], window[-1])
        row = compute_outcome(sig, bars)
        if row["days_available"] == 0:
            if existing is not None and len(window) >= HORIZON_DAYS:
                # second straight miss after the window fully elapsed — likely
                # delisted/halted; finalize the empty row so we stop refetching
                row["complete"] = True
                logger.warning(f"  {sig['ticker']} {target}: still no price data, finalizing empty row")
            else:
                logger.warning(f"  {sig['ticker']} {target}: no price data, recording empty row")
        outcomes[key] = row
        updated += 1
        if target == last_day:
            new_today.append(row)
        if row["days_available"] > 0:
            logger.info(
                f"  {sig['ticker']:6s} {sig['bucket']:8s} {sig['recommendation']:7s} target={target} "
                f"d0={row.get('d0_pct', '')} mfe3d={row.get('mfe_atr_3d', '')}ATR "
                f"tradeable_3d={row.get('tradeable_3d')} ({row['days_available']}/{HORIZON_DAYS}d)"
            )
        time.sleep(0.12)

    if not dry and updated:
        save_outcomes(outcomes)
        logger.info(f"Saved {len(outcomes)} rows to {OUTCOMES_FILE.name} ({updated} updated)")

    summary = rolling_summary(list(outcomes.values()))
    go = summary.get("groups", {}).get("GO", {}).get("tradeable_3d")
    if go:
        logger.info(f"Rolling {ROLLING_WINDOW}d GO tradeable_3d: {go['pct']}% ({go['correct']}/{go['total']})")
    for label, t in summary.get("timing", {}).items():
        logger.info(
            f"  timing {label}: D0 {t['d0']['pct']}% | <=D1 {t['le1']['pct']}% | "
            f"<=D3 {t['le3']['pct']}% | never {t['never']['pct']}% (n={t['n']})"
        )

    if not no_email:
        html = format_email(new_today, updated, summary, last_day)
        n_trade = sum(1 for r in new_today if _is_true(r.get("tradeable_3d")))
        subject = f"Signal Scorecard — {last_day} — {n_trade}/{len(new_today)} tradeable (3d)"
        send_email(EMAIL_TO, subject, html, is_html=True)
        logger.info("Scorecard email sent.")

    logger.info("Signal Scorecard v2 complete.")


if __name__ == "__main__":
    main()
