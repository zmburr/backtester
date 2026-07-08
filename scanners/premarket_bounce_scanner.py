"""
Premarket Bounce Scanner — continuous premarket scoring with email alerts.

Why this exists (premarket-bottom study, July 2026, n=93 bounce_data.csv trades):
  - 39% of historical bounces printed their TRUE low in the premarket —
    and the rate is regime-shifted upward: 62% in 2024, 55% in 2025.
  - Premarket lows cluster 8:00-9:30 AM ET (2/3 of them), but tails reach
    4-5 AM (NBIS 2026-07-08 bottomed 5:14 AM as a 5/6 GO; the 8:43 AM
    priority report scored it 0 signals because the bounce had already
    deflated the metrics).
  - The bounce score is a decaying spike that peaks at the low. A single
    morning sample structurally misses it; this scanner re-scores the
    watchlist every few minutes through the premarket and emails on score
    CROSSINGS (a name reaching GO fires immediately, once).

Alerts are EMAIL ONLY by design — this is meant to run unattended on a
remote machine where popups/TTS are useless.

Scoring is identical to the priority report path: BouncePretrade's six V3
criteria with cap-specific thresholds — but "current price" is the latest
premarket price instead of a stale close, so gap/selloff/discount metrics
reflect the actual premarket capitulation.

NOTE on data latency: the current Polygon plan serves 15-MINUTE DELAYED
data (verified 2026-07-08: minute aggs and last-trade both lag exactly
15:00). Alerts therefore fire ~15 min after the actual score crossing.
Upgrading the Polygon plan to a real-time entitlement removes the lag with
no code change here.

Watchlist is read from scanners/stock_screener.py (single source of truth).

Usage:
    python -m scanners.premarket_bounce_scanner                       # loop 4:15->9:30 ET, email alerts
    python -m scanners.premarket_bounce_scanner --dry                 # loop, print instead of email
    python -m scanners.premarket_bounce_scanner --once                # single scan now, print table
    python -m scanners.premarket_bounce_scanner --once --email        # single scan, send alerts/brief
    python -m scanners.premarket_bounce_scanner --replay "2026-07-08 05:00"   # score as-of a past time
"""

import argparse
import ast
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from analyzers.bounce_scorer import BouncePretrade, ChecklistResult
from data_queries import polygon_queries as pq
from support.config import send_email

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('premarket_bounce_scanner')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMAIL_TO = 'zmburr@gmail.com'
ALERT_SCORE = 5              # >= this (GO) fires an immediate alert email
BRIEF_TIME = '08:00'         # daily leaderboard email (desk arrival)
BRIEF_MIN_SCORE = 3          # names at/above this appear in the brief
SCAN_START = '04:15'         # ET
SCAN_END = '09:30'           # ET
EARLY_INTERVAL_S = 600       # poll cadence before 07:00 (lows are sparse here)
LATE_INTERVAL_S = 300        # poll cadence 07:00-09:30 (2/3 of PM lows print here)
TZ = 'US/Eastern'

_DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
STATE_FILE = _DATA_DIR / 'premarket_scanner_state.json'
CAP_CACHE_FILE = _DATA_DIR / 'ticker_cap_cache.json'

KNOWN_ETFS = {'SOXL', 'SOXS', 'IBIT', 'ETHE', 'KWEB', 'EWY', 'BOT', 'DXYZ', 'QQQ',
              'SPY', 'IWM', 'TQQQ', 'SQQQ', 'TNA', 'GLD', 'SLV', 'USO', 'RVI'}


# ---------------------------------------------------------------------------
# Watchlist (parsed from stock_screener.py so there is one source of truth)
# ---------------------------------------------------------------------------

def load_watchlist() -> list:
    """Extract the `watchlist` list literal from scanners/stock_screener.py.

    AST parse instead of import — stock_screener loads CSVs at module level.
    """
    src_path = Path(__file__).resolve().parent / 'stock_screener.py'
    tree = ast.parse(src_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'watchlist':
                    return [str(x) for x in ast.literal_eval(node.value)]
    raise RuntimeError('watchlist not found in stock_screener.py')


# ---------------------------------------------------------------------------
# Market cap classification (Polygon details, cached to JSON)
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def get_cap(ticker: str, cache: dict) -> str:
    if ticker in cache:
        return cache[ticker]
    if ticker in KNOWN_ETFS:
        cache[ticker] = 'ETF'
        return 'ETF'
    cap = 'Medium'
    try:
        details = pq.poly_client.get_ticker_details(ticker)
        if getattr(details, 'type', '') in ('ETF', 'ETP', 'ETN', 'FUND'):
            cap = 'ETF'
        else:
            mc = getattr(details, 'market_cap', None)
            if mc:
                if mc >= 100e9:
                    cap = 'Large'
                elif mc >= 2e9:
                    cap = 'Medium'
                elif mc >= 300e6:
                    cap = 'Small'
                else:
                    cap = 'Micro'
    except Exception as e:
        logger.warning(f'cap lookup failed for {ticker}: {e} — defaulting Medium')
    cache[ticker] = cap
    CAP_CACHE_FILE.write_text(json.dumps(cache, indent=1))
    return cap


# ---------------------------------------------------------------------------
# Static daily context (fetched once per session — completed bars only)
# ---------------------------------------------------------------------------

def build_static(ticker: str, date: str) -> dict | None:
    """Everything score-relevant that does NOT move during the premarket.

    Mirrors setup_screener.compute_metrics using completed daily bars
    (today's partial bar, if Polygon returns one, is excluded).
    """
    levels = pq.get_levels_data(ticker, date, 310, 1, 'day')
    if levels is None or levels.empty:
        return None

    try:
        has_today = levels.index[-1].date() == pd.to_datetime(date).date()
    except Exception:
        has_today = False
    hist = levels.iloc[:-1] if has_today and len(levels) > 1 else levels
    if len(hist) < 5:
        return None

    closes = hist['close']
    s = {'prior_close': hist.iloc[-1]['close']}

    s['sma200'] = closes.rolling(200).mean().iloc[-1] if len(closes) >= 200 else None
    s['close_30_ago'] = hist.iloc[-30]['close'] if len(hist) >= 30 else None
    s['close_3_ago'] = hist.iloc[-3]['close'] if len(hist) >= 3 else None

    hl = hist['high'] - hist['low']
    hpc = (hist['high'] - hist['close'].shift(1)).abs()
    lpc = (hist['low'] - hist['close'].shift(1)).abs()
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    s['atr'] = tr.rolling(window=min(14, len(tr)), min_periods=1).mean().iloc[-1]
    s['prior_day_range'] = hist.iloc[-1]['high'] - hist.iloc[-1]['low']

    window_30 = hist.iloc[-30:] if len(hist) >= 30 else hist
    s['high_30d'] = window_30['high'].max()
    s['high_52wk'] = hist['high'].max()

    # Consecutive down closes with 1 up-day tolerance — must match
    # fetch_bounce_metrics() exactly (a single marginal green day inside the
    # selloff does not reset the streak; see analyzers/bounce_scorer.py)
    consec = 0
    up_days_used = 0
    start_idx = len(hist) - 1
    for i in range(start_idx, 0, -1):
        cur_close = hist.iloc[i]['close']
        prev_close = hist.iloc[i - 1]['close']
        if pd.isna(cur_close) or pd.isna(prev_close):
            break
        if cur_close < prev_close:
            consec += 1
        elif up_days_used < 1:
            up_days_used += 1
            consec += 1
        else:
            break
    s['consecutive_down_days'] = consec
    # Selloff leg start: first open of the down streak; if no streak yet,
    # the prior day's open (a premarket price below prior close extends/starts the leg)
    leg_start = start_idx - consec + 1 if consec > 0 else len(hist) - 1
    s['selloff_first_open'] = hist.iloc[leg_start]['open']
    return s


def metrics_from_price(static: dict, price: float) -> dict:
    """The six V3 criteria (+ classification inputs) at a live premarket price."""
    s = static
    m = {
        'current_price': price,
        'prior_close': s['prior_close'],
        'consecutive_down_days': s['consecutive_down_days'],
        'gap_pct': (price - s['prior_close']) / s['prior_close'] if s['prior_close'] > 0 else None,
        'prior_day_range_atr': s['prior_day_range'] / s['atr'] if s['atr'] > 0 else 0,
        'atr_pct': s['atr'] / price if price > 0 else 0,
        'pct_off_30d_high': (price - s['high_30d']) / s['high_30d'] if s['high_30d'] > 0 else 0,
        'pct_off_52wk_high': (price - s['high_52wk']) / s['high_52wk'] if s['high_52wk'] > 0 else 0,
    }
    if s['sma200']:
        m['pct_from_200mav'] = (price - s['sma200']) / s['sma200']
    if s['close_30_ago']:
        m['pct_change_30'] = (price - s['close_30_ago']) / s['close_30_ago']
    if s['close_3_ago']:
        m['pct_change_3'] = (price - s['close_3_ago']) / s['close_3_ago']

    if s['consecutive_down_days'] > 0 or price < s['prior_close']:
        fo = s['selloff_first_open']
        m['selloff_total_pct'] = (price - fo) / fo if fo > 0 else 0
    else:
        m['selloff_total_pct'] = 0.0
    return m


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def scan(tickers, statics, caps, checker: BouncePretrade, date: str,
         asof: pd.Timestamp) -> list[dict]:
    """One pass: live premarket price per ticker -> score. Returns row dicts."""
    open_ts = pd.Timestamp(f'{date} 09:30', tz=TZ)
    rows = []
    for ticker in tickers:
        static = statics.get(ticker)
        if static is None:
            continue
        try:
            intra = pq.get_intraday(ticker, date, 1, 'minute')
            if intra is None or intra.empty:
                continue
            upto = intra[intra.index <= asof]
            if upto.empty:
                continue
            price = upto.iloc[-1]['close']
            pm = upto[upto.index < open_ts]
            pm_low = pm['low'].min() if not pm.empty else None
            pm_low_time = str(pm['low'].idxmin().time())[:5] if not pm.empty else ''

            m = metrics_from_price(static, price)
            res: ChecklistResult = checker.validate(ticker, m, cap=caps[ticker])
            rows.append({
                'ticker': ticker, 'cap': caps[ticker], 'setup': res.setup_type,
                'score': res.score, 'rec': res.recommendation, 'price': price,
                'gap_pct': m['gap_pct'], 'selloff': m['selloff_total_pct'],
                'off_30d': m['pct_off_30d_high'], 'pct3': m.get('pct_change_3'),
                'off_52wk': m['pct_off_52wk_high'], 'range_atr': m['prior_day_range_atr'],
                'pm_low': pm_low, 'pm_low_time': pm_low_time,
                'off_pm_low': (price - pm_low) / pm_low if pm_low else None,
                'result': res,
            })
        except Exception as e:
            logger.warning(f'scan failed for {ticker}: {e}')
        time.sleep(0.05)
    rows.sort(key=lambda r: (-r['score'], r['gap_pct'] if r['gap_pct'] is not None else 0))
    return rows


# ---------------------------------------------------------------------------
# Email formatting
# ---------------------------------------------------------------------------

def _pct(x, signed=True):
    if x is None:
        return 'n/a'
    return f'{x*100:+.1f}%' if signed else f'{x*100:.1f}%'


_TD = 'padding:5px 10px;border-bottom:1px solid #1e2330;'


def _row_table(rows) -> str:
    body = ''
    for r in rows:
        color = '#22c55e' if r['rec'] == 'GO' else '#f59e0b' if r['rec'] == 'CAUTION' else '#9ca3af'
        body += (f"<tr><td style='{_TD}color:#e8ecf4;font-weight:600;'>{r['ticker']}</td>"
                 f"<td style='{_TD}color:{color};font-weight:600;'>{r['score']}/6 {r['rec']}</td>"
                 f"<td style='{_TD}'>{r['cap']}</td>"
                 f"<td style='{_TD}'>{r['price']:.2f}</td>"
                 f"<td style='{_TD}'>{_pct(r['gap_pct'])}</td>"
                 f"<td style='{_TD}'>{_pct(r['selloff'])}</td>"
                 f"<td style='{_TD}'>{_pct(r['off_30d'])}</td>"
                 f"<td style='{_TD}'>{r['pm_low_time']}</td>"
                 f"<td style='{_TD}'>{_pct(r['off_pm_low'])}</td></tr>")
    head_td = 'padding:6px 10px;text-align:left;color:#6b7280;'
    return (f"<table style='border-collapse:collapse;font-size:13px;font-family:monospace;color:#c8cdd8;'>"
            f"<thead><tr>"
            f"<th style='{head_td}'>Ticker</th><th style='{head_td}'>Score</th>"
            f"<th style='{head_td}'>Cap</th><th style='{head_td}'>Price</th>"
            f"<th style='{head_td}'>Gap</th><th style='{head_td}'>Selloff</th>"
            f"<th style='{head_td}'>Off 30d</th><th style='{head_td}'>PM Low@</th>"
            f"<th style='{head_td}'>Off PM Low</th></tr></thead><tbody>{body}</tbody></table>")


def format_alert_email(row: dict, asof: pd.Timestamp) -> tuple[str, str]:
    res: ChecklistResult = row['result']
    subject = (f"PM BOUNCE {row['rec']}: {row['ticker']} {row['score']}/6 "
               f"@ {row['price']:.2f} (gap {_pct(row['gap_pct'])})")
    crit = ''
    for item in res.items:
        icon = '&#10003;' if item.passed else '&#10007;'
        c = '#22c55e' if item.passed else '#ef4444'
        crit += (f"<tr><td style='{_TD}color:{c};'>{icon}</td>"
                 f"<td style='{_TD}'>{item.description}</td>"
                 f"<td style='{_TD}'>{item.actual_display}</td>"
                 f"<td style='{_TD}color:#6b7280;'>{item.reference}</td></tr>")
    body = f"""<html><body style="font-family:-apple-system,Segoe UI,sans-serif;background:#0a0c10;color:#c8cdd8;padding:16px;">
<h2 style="color:#e8ecf4;margin:0 0 4px 0;">{row['ticker']} — {row['score']}/6 {row['rec']} ({row['setup']}, {row['cap']} cap)</h2>
<div style="color:#6b7280;font-size:12px;margin-bottom:12px;">as of {asof.strftime('%H:%M ET %Y-%m-%d')} — premarket scanner</div>
<div style="font-size:14px;margin-bottom:12px;">
Price <strong>{row['price']:.2f}</strong> &nbsp;|&nbsp; Gap {_pct(row['gap_pct'])} vs prior close
&nbsp;|&nbsp; PM low <strong>{row['pm_low']:.2f}</strong> @ {row['pm_low_time']}
&nbsp;|&nbsp; now {_pct(row['off_pm_low'])} off the PM low
</div>
<table style="border-collapse:collapse;font-size:13px;font-family:monospace;">{crit}</table>
<div style="color:#4b5563;font-size:11px;margin-top:16px;">Premarket Bounce Scanner — score crossings alert once per level per day</div>
</body></html>"""
    return subject, body


def format_brief_email(rows: list[dict], asof: pd.Timestamp) -> tuple[str, str]:
    top = [r for r in rows if r['score'] >= BRIEF_MIN_SCORE]
    n_go = sum(1 for r in rows if r['rec'] == 'GO')
    n_caution = sum(1 for r in rows if r['rec'] == 'CAUTION')
    subject = f"Premarket Bounce Brief — {n_go} GO / {n_caution} CAUTION ({asof.strftime('%H:%M ET')})"
    body = f"""<html><body style="font-family:-apple-system,Segoe UI,sans-serif;background:#0a0c10;color:#c8cdd8;padding:16px;">
<h2 style="color:#e8ecf4;margin:0 0 4px 0;">Premarket Bounce Brief</h2>
<div style="color:#6b7280;font-size:12px;margin-bottom:12px;">{asof.strftime('%Y-%m-%d %H:%M ET')} —
watchlist scored with live premarket prices (score >= {BRIEF_MIN_SCORE} shown)</div>
{_row_table(top) if top else "<p style='color:#6b7280;'>Nothing scoring above threshold.</p>"}
<div style="color:#4b5563;font-size:11px;margin-top:16px;">Premarket Bounce Scanner — GO crossings alert in real time; this is the 8 AM landscape.</div>
</body></html>"""
    return subject, body


# ---------------------------------------------------------------------------
# Alert state (persisted so restarts don't duplicate emails)
# ---------------------------------------------------------------------------

def load_state(date: str) -> dict:
    state = _load_json(STATE_FILE)
    if state.get('date') != date:
        state = {'date': date, 'alerted': {}, 'brief_sent': False}
    return state


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=1))


def process_alerts(rows, state, asof, do_email: bool):
    """Email on score crossings: fires when a ticker sets a new high score >= ALERT_SCORE."""
    for r in rows:
        if r['score'] < ALERT_SCORE:
            continue
        prev = state['alerted'].get(r['ticker'], 0)
        if r['score'] <= prev:
            continue
        subject, body = format_alert_email(r, asof)
        logger.info(f'ALERT: {subject}')
        if do_email:
            send_email(EMAIL_TO, subject, body, is_html=True)
        state['alerted'][r['ticker']] = r['score']
    save_state(state)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(rows, asof):
    print(f"\nPremarket bounce scores as of {asof.strftime('%Y-%m-%d %H:%M ET')}")
    print(f"{'TICKER':7s}{'CAP':7s}{'SCORE':6s}{'REC':9s}{'PRICE':>9s}{'GAP':>8s}"
          f"{'SELLOFF':>9s}{'OFF30D':>8s}{'3D':>8s}{'OFF52W':>8s}{'PMLOW@':>8s}{'OFF LOW':>9s}")
    for r in rows:
        if r['score'] < 1:
            continue
        print(f"{r['ticker']:7s}{r['cap']:7s}{str(r['score'])+'/6':6s}{r['rec']:9s}"
              f"{r['price']:>9.2f}{_pct(r['gap_pct']):>8s}{_pct(r['selloff']):>9s}"
              f"{_pct(r['off_30d']):>8s}{_pct(r['pct3']):>8s}{_pct(r['off_52wk']):>8s}"
              f"{r['pm_low_time']:>8s}{_pct(r['off_pm_low']):>9s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TZ)


def main():
    ap = argparse.ArgumentParser(description='Premarket bounce scanner (email alerts)')
    ap.add_argument('--once', action='store_true', help='single scan now, print table')
    ap.add_argument('--email', action='store_true', help='with --once: send alerts/brief emails')
    ap.add_argument('--dry', action='store_true', help='loop without sending email')
    ap.add_argument('--replay', metavar='"YYYY-MM-DD HH:MM"',
                    help='score as of a past ET timestamp (no email, no state)')
    args = ap.parse_args()

    tickers = load_watchlist()
    checker = BouncePretrade()
    cap_cache = _load_json(CAP_CACHE_FILE)

    if args.replay:
        asof = pd.Timestamp(args.replay, tz=TZ)
        date = asof.strftime('%Y-%m-%d')
        logger.info(f'REPLAY {date} as of {asof.strftime("%H:%M ET")} — {len(tickers)} tickers')
        statics = {t: build_static(t, date) for t in tickers}
        caps = {t: get_cap(t, cap_cache) for t in tickers}
        rows = scan(tickers, statics, caps, checker, date, asof)
        print_table(rows, asof)
        return

    date = _now_et().strftime('%Y-%m-%d')
    logger.info(f'Building daily context for {len(tickers)} tickers...')
    statics = {t: build_static(t, date) for t in tickers}
    caps = {t: get_cap(t, cap_cache) for t in tickers}
    ok = sum(1 for v in statics.values() if v)
    logger.info(f'{ok}/{len(tickers)} tickers ready')

    state = load_state(date)

    if args.once:
        asof = _now_et()
        rows = scan(tickers, statics, caps, checker, date, asof)
        print_table(rows, asof)
        if args.email:
            process_alerts(rows, state, asof, do_email=True)
        return

    do_email = not args.dry
    start = pd.Timestamp(f'{date} {SCAN_START}', tz=TZ)
    end = pd.Timestamp(f'{date} {SCAN_END}', tz=TZ)
    brief_at = pd.Timestamp(f'{date} {BRIEF_TIME}', tz=TZ)

    now = _now_et()
    if now < start:
        wait = (start - now).total_seconds()
        logger.info(f'Sleeping {wait/60:.0f} min until {SCAN_START} ET')
        time.sleep(wait)

    while _now_et() < end:
        asof = _now_et()
        rows = scan(tickers, statics, caps, checker, date, asof)
        best = rows[0] if rows else None
        logger.info(f"scan @ {asof.strftime('%H:%M')}: {len(rows)} scored"
                    + (f" | top: {best['ticker']} {best['score']}/6 {best['rec']}" if best else ''))

        process_alerts(rows, state, asof, do_email=do_email)

        if not state['brief_sent'] and asof >= brief_at:
            subject, body = format_brief_email(rows, asof)
            logger.info(f'BRIEF: {subject}')
            if do_email:
                send_email(EMAIL_TO, subject, body, is_html=True)
            state['brief_sent'] = True
            save_state(state)

        interval = EARLY_INTERVAL_S if asof.hour < 7 else LATE_INTERVAL_S
        time.sleep(interval)

    logger.info('9:30 ET — premarket over, scanner exiting.')


if __name__ == '__main__':
    main()
