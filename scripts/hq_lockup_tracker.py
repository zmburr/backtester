"""HQ warrant-redemption / lockup-trigger tracker.

Once-a-day (after the 4pm ET close) progress check on HQ's path to its warrant
redemption / lockup-expiry condition:

    Class A closing price >= $18.00 for 20 of any 30 consecutive TRADING days.

Hitting it lets the company force the ~3.16M public warrants ($11.50 strike) to
exercise on 30-day notice -> ~3M new Class A shares + ~$36M cash to the company,
with the sponsor warrants registered for resale on top. It is the most concrete
near-term supply event and the company controls the trigger. HQ only crossed $18
around June 15, 2026, so the 20/30 count is still accumulating.

Data: Polygon daily aggregates. Each daily bar IS one trading day, so the last
30 bars == the last 30 trading days -- no market-calendar dependency needed.

Delivery: HTML email via support.config.send_email (Gmail, from zmburr@gmail.com).

Usage:
    python scripts/hq_lockup_tracker.py            # compute + email (default)
    python scripts/hq_lockup_tracker.py --dry-run  # compute + print, no email
    python scripts/hq_lockup_tracker.py --email     # force send even on dry data
"""
from __future__ import annotations

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta, date
from pathlib import Path

import requests

# --- make `support` importable when run as a bare script (not -m) ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Importing support.config also triggers its load_dotenv(), populating
# POLYGON_API_KEY and GMAIL_PASSWORD from backtester/.env.
from support.config import send_email  # noqa: E402

log = logging.getLogger("hq_lockup_tracker")

# ----------------------------- Config -----------------------------
TICKER = "HQ"
TRIGGER_PRICE = 18.00      # $18+ close required
DAYS_REQUIRED = 20         # ... for 20 of ...
WINDOW_DAYS = 30           # ... any 30 consecutive trading days
RECIPIENT = "zmburr@gmail.com"
LOOKBACK_CALENDAR_DAYS = 120   # enough to comfortably cover 30+ trading days
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")


# ----------------------------- Data -------------------------------
def fetch_daily_closes(ticker: str):
    """Return list of (date:datetime.date, close:float) ascending, trading days only."""
    if not POLYGON_API_KEY:
        raise RuntimeError("POLYGON_API_KEY not set (expected in backtester/.env)")

    end = date.today()
    start = end - timedelta(days=LOOKBACK_CALENDAR_DAYS)
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start:%Y-%m-%d}/{end:%Y-%m-%d}"
        f"?adjusted=true&sort=asc&limit=5000&apiKey={POLYGON_API_KEY}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    results = data.get("results") or []
    bars = []
    for bar in results:
        # Polygon 't' is epoch ms (UTC) for the start of the trading day.
        d = datetime.utcfromtimestamp(bar["t"] / 1000).date()
        bars.append((d, float(bar["c"])))
    bars.sort(key=lambda x: x[0])
    return bars


# ----------------------------- Math -------------------------------
def add_trading_days(start: date, n: int) -> date:
    """Approximate a future trading date by skipping weekends (holidays ignored)."""
    d = start
    added = 0
    while added < n:
        d += timedelta(days=1)
        if d.weekday() < 5:   # Mon-Fri
            added += 1
    return d


def evaluate(bars):
    """Compute progress toward 20-of-30 trading days >= TRIGGER_PRICE.

    Returns a dict of metrics. The official condition is satisfied if ANY
    30-consecutive-trading-day window contains >= 20 qualifying closes; we also
    report the trailing-30 window since that is the live, forward-moving one.
    """
    closes = [c for _, c in bars]
    dates = [d for d, _ in bars]
    qualifies = [c >= TRIGGER_PRICE for c in closes]

    # Max qualifying count across every 30-trading-day window (official test).
    best_count = 0
    best_window_end = None
    for end_i in range(len(qualifies)):
        start_i = max(0, end_i - WINDOW_DAYS + 1)
        cnt = sum(qualifies[start_i:end_i + 1])
        if cnt >= best_count:
            best_count = cnt
            best_window_end = dates[end_i]

    # Trailing 30-trading-day window (the live one).
    trailing = qualifies[-WINDOW_DAYS:]
    trailing_count = sum(trailing)
    trailing_span = bars[-len(trailing):]

    # Current consecutive qualifying streak (most recent run of >= $18 closes).
    streak = 0
    for q in reversed(qualifies):
        if q:
            streak += 1
        else:
            break

    met = best_count >= DAYS_REQUIRED
    needed = max(0, DAYS_REQUIRED - trailing_count)

    # Projection: if every remaining day closes >= $18, each adds one qualifying
    # day to the trailing window (the dropping-off days are pre-June-15 sub-$18
    # days, so the count only rises). So `needed` more qualifying closes hits 20.
    last_date = dates[-1]
    projected_date = None if met else add_trading_days(last_date, needed)

    return {
        "last_date": last_date,
        "last_close": closes[-1],
        "last_above": closes[-1] >= TRIGGER_PRICE,
        "trailing_count": trailing_count,
        "trailing_window_start": trailing_span[0][0],
        "best_count": best_count,
        "best_window_end": best_window_end,
        "streak": streak,
        "met": met,
        "needed": needed,
        "projected_date": projected_date,
        "n_bars": len(bars),
    }


# ----------------------------- Render -----------------------------
def render_text(m) -> str:
    pct = 100.0 * m["trailing_count"] / DAYS_REQUIRED
    lines = [
        f"HQ LOCKUP / WARRANT REDEMPTION TRACKER  ({m['last_date']:%a %b %d, %Y})",
        "=" * 60,
        f"Last close:        ${m['last_close']:.2f}  "
        f"({'ABOVE' if m['last_above'] else 'below'} ${TRIGGER_PRICE:.2f})",
        "",
        f"Progress:          {m['trailing_count']} / {DAYS_REQUIRED} qualifying "
        f"closes ({pct:.0f}%)",
        f"  (trailing {WINDOW_DAYS} trading days, since {m['trailing_window_start']:%b %d})",
        f"Best 30-day window: {m['best_count']} / {DAYS_REQUIRED} "
        f"(ending {m['best_window_end']:%b %d})",
        f"Current streak:     {m['streak']} consecutive closes >= ${TRIGGER_PRICE:.2f}",
        "",
    ]
    if m["met"]:
        lines.append(">>> CONDITION MET — 20-of-30 satisfied. Company can issue the "
                     "30-day redemption notice to force public warrant exercise.")
    else:
        lines.append(f"Status:            NOT YET — need {m['needed']} more qualifying close(s).")
        lines.append(f"If price holds >= ${TRIGGER_PRICE:.2f}, eligible on ~"
                     f"{m['projected_date']:%a %b %d, %Y} "
                     f"(approx; excludes market holidays).")
    lines += [
        "",
        "-" * 60,
        "On trigger: ~3.16M public warrants ($11.50 strike) forced to exercise on",
        "30-day notice -> ~3M new Class A shares + ~$36M cash to the company;",
        "sponsor warrants (2.88M) registered for resale on top. The most concrete",
        "near-term supply event, and the company controls the trigger.",
    ]
    return "\n".join(lines)


def render_html(m) -> str:
    pct = 100.0 * m["trailing_count"] / DAYS_REQUIRED
    bar_w = min(100, pct)
    accent = "#2e7d32" if m["met"] else ("#1565c0" if m["last_above"] else "#c62828")
    if m["met"]:
        status = ('<strong style="color:#2e7d32;">CONDITION MET</strong> — 20-of-30 '
                  'satisfied. Company can issue the 30-day redemption notice to force '
                  'public warrant exercise.')
    else:
        status = (f'<strong>Not yet</strong> — need <strong>{m["needed"]}</strong> more '
                  f'qualifying close(s). If price holds &ge; ${TRIGGER_PRICE:.2f}, eligible '
                  f'on ~<strong>{m["projected_date"]:%a %b %d, %Y}</strong> '
                  f'(approx; excludes market holidays).')
    above_txt = ("ABOVE" if m["last_above"] else "below")
    above_color = "#2e7d32" if m["last_above"] else "#c62828"
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="font-family:-apple-system,'Segoe UI',Roboto,sans-serif;background:#fff;
color:#1a1a1a;padding:20px;max-width:640px;margin:0 auto;">
  <div style="border-bottom:2px solid #e0e0e0;padding-bottom:12px;margin-bottom:20px;">
    <h1 style="font-size:20px;margin:0 0 6px 0;">HQ Lockup / Warrant Redemption Tracker</h1>
    <span style="color:#666;font-size:13px;">{m['last_date']:%A, %B %d, %Y}</span>
  </div>

  <div style="background:#f5f5f5;padding:16px 18px;border-radius:8px;margin-bottom:20px;
              border-left:4px solid {accent};">
    <div style="font-size:15px;margin-bottom:10px;">
      Last close: <strong style="font-size:18px;">${m['last_close']:.2f}</strong>
      <span style="color:{above_color};font-weight:600;">({above_txt} ${TRIGGER_PRICE:.2f})</span>
    </div>
    <div style="font-size:14px;margin-bottom:6px;">
      Progress: <strong>{m['trailing_count']} / {DAYS_REQUIRED}</strong> qualifying closes
      <span style="color:#666;">(trailing {WINDOW_DAYS} trading days, since {m['trailing_window_start']:%b %d})</span>
    </div>
    <div style="background:#e0e0e0;border-radius:6px;height:14px;overflow:hidden;margin:8px 0 12px 0;">
      <div style="background:{accent};height:14px;width:{bar_w:.0f}%;"></div>
    </div>
    <div style="font-size:13px;color:#444;">
      Best 30-day window: <strong>{m['best_count']} / {DAYS_REQUIRED}</strong>
      (ending {m['best_window_end']:%b %d}) &nbsp;&middot;&nbsp;
      Current streak: <strong>{m['streak']}</strong> closes &ge; ${TRIGGER_PRICE:.2f}
    </div>
  </div>

  <p style="font-size:14px;line-height:1.6;">{status}</p>

  <div style="border-top:1px solid #e0e0e0;margin-top:24px;padding-top:14px;
              color:#555;font-size:12px;line-height:1.6;">
    <strong>On trigger:</strong> ~3.16M public warrants ($11.50 strike) forced to exercise
    on 30-day notice &rarr; ~3M new Class A shares + ~$36M cash to the company; sponsor
    warrants (2.88M) registered for resale on top. Most concrete near-term supply event;
    the company controls the trigger.
  </div>
  <div style="color:#999;font-size:11px;margin-top:14px;">
    Polygon daily aggs &middot; {m['n_bars']} trading days analyzed &middot; auto-generated after close
  </div>
</body></html>"""


# ----------------------------- Main -------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="HQ lockup/warrant redemption tracker")
    ap.add_argument("--dry-run", action="store_true", help="print report, do not send email")
    ap.add_argument("--email", action="store_true", help="force send email")
    args = ap.parse_args()

    bars = fetch_daily_closes(TICKER)
    if not bars:
        log.error("No bars returned for %s — aborting.", TICKER)
        return 1
    m = evaluate(bars)
    text = render_text(m)
    print(text)

    if args.dry_run and not args.email:
        log.info("Dry run — email not sent.")
        return 0

    standing = "MET" if m["met"] else "need {}".format(m["needed"])
    subject = (f"HQ lockup: {m['trailing_count']}/{DAYS_REQUIRED} ({standing}) "
               f"— close ${m['last_close']:.2f}")
    try:
        send_email(to_email=RECIPIENT, subject=subject, body=render_html(m), is_html=True)
        log.info("Email sent to %s", RECIPIENT)
    except Exception as e:
        log.error("Failed to send email: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
