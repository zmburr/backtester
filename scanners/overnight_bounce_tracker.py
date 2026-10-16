"""
Overnight Bounce Tracker — score the bounce pre-trade checklist on live
overnight / premarket prices for a handful of names.

Why this exists: Polygon (the repo's data source) prints its last trade at
20:00 ET and resumes at 04:00 ET, so the 8 PM -> 4 AM overnight session
(Blue Ocean ATS, "BOATS", via broker 24-hour trading) is invisible to it
(verified 2026-10-14: minute bars end 19:59, zero trades after 20:00).
Trillium/SHEL is only reachable from the desk machine. Yahoo Finance's quote
API, however, returns the live Blue Ocean print when asked with
`overnightPrice=true` (fields overnightMarketPrice / overnightMarketTime,
source "BOATS Real Time Price"), so this tracker scores the SAME six V3
criteria as scanners/premarket_bounce_scanner.py on that overnight price,
logs every sample so you can see the score move through the night, and
switches to Polygon once Polygon has a fresher trade (i.e. from 04:00 ET).

Price precedence per ticker (newest timestamp wins):
  1. --price TICKER=PRICE on the command line (as-of now)
  2. data/overnight_prices.json, edited while --watch runs
       {"KORU": 10.65, "SOXL": {"price": 105.5, "asof": "2026-10-14 23:05"}}
     (an entry without "asof" is stamped with the file's modification time)
  3. Yahoo Finance overnight (Blue Ocean ATS) print, 8 PM-4 AM ET  [src Y]
  4. Polygon snapshot last trade (15-min delayed on the current plan), with
     the 1-min bar close as a fallback                              [src P]

Scoring context (prior close, ATR, 30d/52wk highs, down-day streak, SMA200,
mid Bollinger band) is built once from COMPLETED daily bars for the next
session, exactly like the premarket scanner, so the gap/selloff/discount
metrics reflect the price you supply against tonight's close.

Usage:
    python -m scanners.overnight_bounce_tracker                            # KORU SOXL SKHY on live overnight prices
    python -m scanners.overnight_bounce_tracker NVDA SMH --watch 300       # re-score every 5 min until 09:30 ET
    python -m scanners.overnight_bounce_tracker --price KORU=10.65         # override a quote by hand
    python -m scanners.overnight_bounce_tracker --watch 300 --say          # macOS voice alert on a new GO / better grade
    python -m scanners.overnight_bounce_tracker --watch 300 --chart        # + game plan chart, reloads each pass
    python -m scanners.overnight_bounce_tracker --date 2026-10-15 --price KORU=10.65   # force the bounce day

Each row carries the bounce grade (analyzers/bounce_ladder.py: GO 5+/6 with
intensity 75+ = A+, 60-75 = A, else B; 4/6 = B; 3/6 = C) and NEXT, the price of
the next better grade below the current quote. The setup type is fixed at the
prior close, so a deeper print can only raise the score.

Every scan appends one row per ticker to data/overnight_tracker_log.csv
(git-ignored); open it in a spreadsheet or `tail -f` it to watch the trajectory.
"""

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import pandas_market_calendars as mcal

from analyzers.bounce_scorer import ChecklistResult
from analyzers.bounce_exit_targets import calculate_bounce_exit_targets
from analyzers.bounce_ladder import GRADE_RANK, Ladder, build_ladder, score_at
from data_queries import polygon_queries as pq
from scanners.premarket_bounce_scanner import build_static, get_cap, _load_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('overnight_bounce_tracker')

TZ = 'US/Eastern'
DEFAULT_TICKERS = ['KORU', 'SOXL', 'SKHY']
_DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
PRICES_FILE = _DATA_DIR / 'overnight_prices.json'
LOG_FILE = _DATA_DIR / 'overnight_tracker_log.csv'
CAP_CACHE_FILE = _DATA_DIR / 'ticker_cap_cache.json'
WATCH_END = '09:30'   # premarket is over; hand off to bounce_trader / the live scanners

LOG_FIELDS = ['scan_ts_et', 'bounce_date', 'ticker', 'cap', 'setup', 'score', 'rec',
              'price', 'src', 'price_asof_et', 'gap_pct', 'selloff_pct', 'off_30d_pct',
              'pct_change_3', 'off_52wk_pct', 'range_atr', 'intensity', 'prior_close',
              'mid_bb', 'to_mid_bb_pct', 'sma200', 'grade', 'next_grade', 'next_price']


# ---------------------------------------------------------------------------
# Session / date helpers
# ---------------------------------------------------------------------------

def now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TZ)


def bounce_day(now: pd.Timestamp) -> str:
    """The session whose open the supplied price is a gap INTO.

    After the 16:00 close (and all weekend) that is the next trading day;
    overnight, premarket and regular hours it is today when today trades.
    """
    nyse = mcal.get_calendar('NYSE')
    today = now.strftime('%Y-%m-%d')
    is_trading_today = not nyse.valid_days(start_date=today, end_date=today).empty
    if is_trading_today and now.strftime('%H:%M') < '16:00':
        return today
    return pq.adjust_date_forward(today, 1)


# ---------------------------------------------------------------------------
# Prices — manual (CLI / JSON file) vs Polygon, newest wins
# ---------------------------------------------------------------------------

def parse_price_flags(flags: Optional[list], asof: pd.Timestamp) -> Dict[str, Tuple[float, pd.Timestamp]]:
    out = {}
    for item in flags or []:
        if '=' not in item:
            raise SystemExit(f'--price expects TICKER=PRICE, got {item!r}')
        t, p = item.split('=', 1)
        out[t.strip().upper()] = (float(p), asof)
    return out


def load_prices_file(path: Path = PRICES_FILE) -> Dict[str, Tuple[float, pd.Timestamp]]:
    """Manual overnight quotes. Missing 'asof' -> file mtime (ET)."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except Exception as e:
        logger.warning(f'could not parse {path.name}: {e}')
        return {}
    mtime = pd.Timestamp(datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)).tz_convert(TZ)
    out = {}
    for t, v in raw.items():
        try:
            if isinstance(v, dict):
                price = float(v['price'])
                asof = pd.Timestamp(v['asof'], tz=TZ) if v.get('asof') else mtime
            else:
                price, asof = float(v), mtime
        except Exception as e:
            logger.warning(f'{path.name}: bad entry for {t}: {e}')
            continue
        if price > 0:
            out[t.upper()] = (price, asof)
    return out


def _ns_or_ms_to_et(ts) -> Optional[pd.Timestamp]:
    if not ts:
        return None
    unit = 'ns' if ts > 1e15 else 'ms'
    return pd.Timestamp(ts, unit=unit, tz='UTC').tz_convert(TZ)


def polygon_price(ticker: str) -> Optional[Tuple[float, pd.Timestamp]]:
    """Latest Polygon print: last trade, else last 1-min bar close."""
    snap = pq.get_ticker_snapshot(ticker)
    if snap is None:
        return None
    lt = getattr(snap, 'last_trade', None)
    if lt and getattr(lt, 'price', None):
        ts = _ns_or_ms_to_et(getattr(lt, 'sip_timestamp', None) or getattr(lt, 'participant_timestamp', None))
        if ts is not None:
            return float(lt.price), ts
    mn = getattr(snap, 'min', None)
    if mn and getattr(mn, 'close', None):
        ts = _ns_or_ms_to_et(getattr(mn, 'timestamp', None))
        if ts is not None:
            return float(mn.close), ts
    return None


YAHOO_QUOTE_URL = 'https://query1.finance.yahoo.com/v7/finance/quote'
YAHOO_FIELDS = ('symbol,marketState,regularMarketPrice,regularMarketTime,postMarketPrice,postMarketTime,'
                'overnightMarketPrice,overnightMarketTime,overnightMarketChangePercent,overnightMarketSource')


def parse_yahoo_overnight(payload: dict) -> Dict[str, Tuple[float, pd.Timestamp]]:
    """Pull (price, asof ET) per symbol from a Yahoo v7 quote response's overnight fields."""
    out = {}
    for q in (payload.get('quoteResponse') or {}).get('result') or []:
        price = q.get('overnightMarketPrice')
        ts = q.get('overnightMarketTime')
        sym = q.get('symbol')
        if not (sym and price and ts):
            continue
        try:
            out[sym.upper()] = (float(price), pd.Timestamp(int(ts), unit='s', tz='UTC').tz_convert(TZ))
        except Exception:
            continue
    return out


def yahoo_overnight_prices(tickers: list) -> Dict[str, Tuple[float, pd.Timestamp]]:
    """Live Blue Ocean ATS overnight prints via Yahoo Finance (needs yfinance for the crumb/cookie).

    Outside 8 PM-4 AM ET the fields still hold the LAST overnight print, which
    is older than Polygon's — choose_price() handles that by timestamp.
    """
    try:
        from yfinance.data import YfData
    except ImportError:
        logger.warning('yfinance not installed — no Yahoo overnight prices (pip install yfinance)')
        return {}
    try:
        r = YfData().get(YAHOO_QUOTE_URL, params={'symbols': ','.join(tickers), 'overnightPrice': 'true',
                                                  'fields': YAHOO_FIELDS})
        return parse_yahoo_overnight(r.json())
    except Exception as e:
        logger.warning(f'Yahoo overnight quote failed: {e}')
        return {}


def choose_price(manual: Optional[Tuple[float, pd.Timestamp]],
                 poly: Optional[Tuple[float, pd.Timestamp]],
                 yahoo: Optional[Tuple[float, pd.Timestamp]] = None) -> Optional[Tuple[float, pd.Timestamp, str]]:
    """Newest sample wins; ties go M > Y > P.

    src: 'M' manual (broker quote), 'Y' Yahoo/Blue Ocean overnight, 'P' Polygon.
    """
    candidates = [(s, p) for s, p in (('M', manual), ('Y', yahoo), ('P', poly)) if p]
    if not candidates:
        return None
    src, (price, asof) = max(candidates, key=lambda c: c[1][1])  # stable: first max wins ties
    # max() returns the first maximal element, so the M > Y > P ordering holds on ties
    return price, asof, src


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_ticker(ticker: str, static: dict, cap: str, price: float, ladder: Optional[Ladder] = None) -> dict:
    """Checklist (setup type fixed at the prior close), intensity vs the setup's own
    reference trades, grade, and the next better grade below `price` on the ladder."""
    sc = score_at(static, cap, price, ticker)
    res: ChecklistResult = sc['result']
    targets = calculate_bounce_exit_targets(cap, price, static['atr'],
                                            prior_close=static['prior_close'],
                                            mid_bb=static.get('mid_bb'))
    mid_bb = static.get('mid_bb')
    return {
        'ticker': ticker, 'cap': cap, 'setup': res.setup_type, 'score': res.score,
        'rec': res.recommendation, 'price': price, 'metrics': sc['metrics'], 'result': res,
        'intensity': sc['intensity'], 'grade': sc['grade'],
        'next': ladder.next_level(price) if ladder is not None else None,
        'targets': targets,
        'mid_bb': mid_bb, 'to_mid_bb': ((mid_bb - price) / price) if mid_bb else None,
        'sma200': static.get('sma200'),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _pct(x, signed=True):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return 'n/a'
    return f'{x*100:+.1f}%' if signed else f'{x*100:.1f}%'


def _age(asof: pd.Timestamp, now: pd.Timestamp) -> str:
    mins = (now - asof).total_seconds() / 60
    if mins < 90:
        return f'{mins:.0f}m ago'
    return f'{mins/60:.1f}h ago'


def format_next(nxt: Optional[dict]) -> str:
    """'A+ @ 9.98 (-6.3%)' for the next better grade below the price; 'best' when there is none."""
    if not nxt:
        return 'best'
    return f"{nxt['grade']} @ {nxt['price']:.2f} ({_pct(nxt['pct_away'])})"


def print_table(rows, now):
    print(f"\nOvernight bounce scores as of {now.strftime('%Y-%m-%d %H:%M ET')}"
          f"   (src M=manual quote, Y=Yahoo/Blue Ocean overnight real-time, P=Polygon 15-min delayed)")
    print(f"{'TICKER':7s}{'CAP':7s}{'SETUP':13s}{'SCORE':6s}{'REC':9s}{'GRADE':6s}{'PRICE':>9s}{'SRC':>4s}{'AS OF':>12s}"
          f"{'GAP':>8s}{'SELLOFF':>9s}{'OFF30D':>8s}{'3D':>8s}{'OFF52W':>8s}{'RNG/ATR':>8s}"
          f"{'INTENS':>7s}{'MID-BB':>9s}{'TO MID':>8s}  NEXT")
    for r in rows:
        m = r['metrics']
        setup = 'weak' if 'weak' in r['setup'] else 'strong'
        intens = f"{r['intensity']:.0f}" if r['intensity'] is not None else 'n/a'
        mid = f"{r['mid_bb']:.2f}" if r['mid_bb'] else 'n/a'
        print(f"{r['ticker']:7s}{r['cap']:7s}{setup:13s}{str(r['score'])+'/6':6s}{r['rec']:9s}{r['grade']:6s}"
              f"{r['price']:>9.2f}{r['src']:>4s}{_age(r['asof'], now):>12s}"
              f"{_pct(m['gap_pct']):>8s}{_pct(m['selloff_total_pct']):>9s}{_pct(m['pct_off_30d_high']):>8s}"
              f"{_pct(m.get('pct_change_3')):>8s}{_pct(m['pct_off_52wk_high']):>8s}"
              f"{m['prior_day_range_atr']:>8.2f}{intens:>7s}{mid:>9s}{_pct(r['to_mid_bb']):>8s}"
              f"  {format_next(r.get('next'))}")


def print_details(r: dict):
    res: ChecklistResult = r['result']
    m = r['metrics']
    print(f"\n--- {r['ticker']}  {r['score']}/6 {r['rec']}  grade {r['grade']}  ({res.setup_type}, {r['cap']} cap)  "
          f"price {r['price']:.2f} [{r['src']}]  prior close {m['prior_close']:.2f}  gap {_pct(m['gap_pct'])}")
    if r.get('next'):
        print(f"    next grade: {format_next(r['next'])}  [{'; '.join(r['next']['events'])}]")
    for sig in res.classification_details.get('signals', []):
        print(f"    classification: {sig}")
    for item in res.items:
        flag = 'PASS' if item.passed else 'FAIL'
        print(f"    [{flag}] {item.description:38s} actual {item.actual_display:>8s}"
              f"   ({item.reference})")
    for b in res.bonuses:
        print(f"    [+] {b}")
    for w in res.warnings:
        print(f"    [!] {w}")
    if r['intensity'] is not None:
        print(f"    intensity (0-100 vs historical bounces): {r['intensity']:.0f}")
    t = r['targets']
    if t.get('tiers'):
        tiers = '  |  '.join(
            f"T{x['tier']} {x['name']} {x['target_price']:.2f} ({_pct(x['target_pct'])}, {x['hit_rate']*100:.0f}% hit)"
            for x in t['tiers'] if x.get('target_price'))
        print(f"    targets from {r['price']:.2f} (ATR {t['atr']:.2f}): {tiers}")
    if t.get('mid_bb'):
        print(f"    mean-reversion SELL (mid Bollinger / 20d SMA): {t['mid_bb']['price']:.2f} "
              f"({_pct(t['mid_bb']['pct_from_entry'])}, {t['mid_bb']['atrs_from_entry']:.1f} ATRs)")


def rotate_stale_log(path: Path) -> Optional[Path]:
    """A log written with an older LOG_FIELDS header is renamed aside (rows appended under the
    wrong header would be misaligned). Returns the new name, or None when nothing moved."""
    if not path.exists():
        return None
    with path.open(newline='') as f:
        header = next(csv.reader(f), None)
    if header == LOG_FIELDS:
        return None
    stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y%m%d-%H%M%S')
    moved = path.with_name(f'{path.stem}.{stamp}{path.suffix}')
    path.rename(moved)
    logger.info(f'{path.name} had an older header — moved to {moved.name}')
    return moved


def append_log(rows, now, date, path: Path = LOG_FILE):
    rotate_stale_log(path)
    new = not path.exists()
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            m = r['metrics']
            w.writerow({
                'scan_ts_et': now.strftime('%Y-%m-%d %H:%M:%S'), 'bounce_date': date,
                'ticker': r['ticker'], 'cap': r['cap'], 'setup': r['setup'],
                'score': r['score'], 'rec': r['rec'], 'price': r['price'], 'src': r['src'],
                'price_asof_et': r['asof'].strftime('%Y-%m-%d %H:%M:%S'),
                'gap_pct': m['gap_pct'], 'selloff_pct': m['selloff_total_pct'],
                'off_30d_pct': m['pct_off_30d_high'], 'pct_change_3': m.get('pct_change_3'),
                'off_52wk_pct': m['pct_off_52wk_high'], 'range_atr': m['prior_day_range_atr'],
                'intensity': r['intensity'], 'prior_close': m['prior_close'],
                'mid_bb': r['mid_bb'], 'to_mid_bb_pct': r['to_mid_bb'], 'sma200': r['sma200'],
                'grade': r['grade'],
                'next_grade': r['next']['grade'] if r.get('next') else '',
                'next_price': r['next']['price'] if r.get('next') else '',
            })


def say(text: str):
    """macOS voice alert (no-op elsewhere)."""
    try:
        subprocess.Popen(['say', text])
    except Exception:
        pass


# ---------------------------------------------------------------------------
# One pass
# ---------------------------------------------------------------------------

def scan(tickers, statics, caps, cli_prices, now, use_prices_file=True, use_yahoo=True,
         ladders: Optional[Dict[str, Ladder]] = None) -> list:
    manual = dict(load_prices_file() if use_prices_file else {})
    manual.update(cli_prices)   # CLI flags are the freshest statement of intent
    yahoo = yahoo_overnight_prices(tickers) if use_yahoo else {}
    rows = []
    for t in tickers:
        static = statics.get(t)
        if static is None:
            logger.warning(f'{t}: no daily context — skipped')
            continue
        try:
            poly = polygon_price(t)
        except Exception as e:
            logger.warning(f'{t}: Polygon snapshot failed: {e}')
            poly = None
        chosen = choose_price(manual.get(t), poly, yahoo.get(t))
        if chosen is None:
            logger.warning(f'{t}: no price from any source — skipped')
            continue
        price, asof, src = chosen
        r = score_ticker(t, static, caps[t], price, (ladders or {}).get(t))
        r.update({'asof': asof, 'src': src, 'poly': poly, 'manual': manual.get(t), 'yahoo': yahoo.get(t)})
        rows.append(r)
    rows.sort(key=lambda r: (-GRADE_RANK[r['grade']], -r['score'], r['metrics']['gap_pct'] or 0))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description='Score the bounce checklist on overnight / premarket prices')
    ap.add_argument('tickers', nargs='*', help=f'tickers (default {" ".join(DEFAULT_TICKERS)})')
    ap.add_argument('--price', action='append', metavar='TICKER=PRICE',
                    help='manual overnight quote (repeatable); newest sample beats Polygon')
    ap.add_argument('--date', help='bounce day YYYY-MM-DD (default: next session)')
    ap.add_argument('--watch', type=int, metavar='SECONDS', help='loop every N seconds until --until')
    ap.add_argument('--until', default=WATCH_END, help=f'ET HH:MM to stop --watch (default {WATCH_END})')
    ap.add_argument('--say', action='store_true',
                    help='macOS voice alert when a name newly reaches GO or a better grade (B or above)')
    ap.add_argument('--chart', action='store_true',
                    help='write the bounce game plan chart (charts/gameplan/) each pass; with --watch it reloads itself')
    ap.add_argument('--no-details', action='store_true', help='table only')
    ap.add_argument('--no-log', action='store_true', help='do not append to data/overnight_tracker_log.csv')
    ap.add_argument('--no-prices-file', action='store_true', help='ignore data/overnight_prices.json')
    ap.add_argument('--no-yahoo', action='store_true', help='skip the Yahoo/Blue Ocean overnight feed')
    args = ap.parse_args(argv)

    tickers = [t.upper() for t in args.tickers] or list(DEFAULT_TICKERS)
    for t in parse_price_flags(args.price, now_et()):
        if t not in tickers:
            tickers.append(t)

    now = now_et()
    date = args.date or bounce_day(now)
    logger.info(f'Bounce day {date} — building daily context for {", ".join(tickers)}')
    statics = {t: build_static(t, date) for t in tickers}
    cap_cache = _load_json(CAP_CACHE_FILE)
    caps = {t: get_cap(t, cap_cache) for t in tickers}
    ladders = {t: build_ladder(statics[t], caps[t], ticker=t) for t in tickers if statics.get(t)}

    best_seen: Dict[str, int] = {}
    best_grade: Dict[str, int] = {}
    chart_opened = False

    def one_pass():
        nonlocal chart_opened
        now_ = now_et()
        cli_prices = parse_price_flags(args.price, now_)
        rows = scan(tickers, statics, caps, cli_prices, now_,
                    use_prices_file=not args.no_prices_file, use_yahoo=not args.no_yahoo, ladders=ladders)
        print_table(rows, now_)
        if not args.no_details:
            for r in rows:
                print_details(r)
        if not args.no_log and rows:
            append_log(rows, now_, date)
        for r in rows:
            t, g = r['ticker'], GRADE_RANK[r['grade']]
            new_go = r['rec'] == 'GO' and r['score'] > best_seen.get(t, 0)
            new_grade = g > best_grade.get(t, GRADE_RANK['C'])    # first B-or-better, then each step up
            if new_go or new_grade:
                spoken = r['grade'].replace('+', ' plus')
                msg = f"{t} bounce grade {spoken}, {r['score']} of 6 at {r['price']:.2f}"
                print(f"\n  >>> NEW {'GO' if new_go else 'GRADE'}: {t} {r['grade']} {r['score']}/6 at {r['price']:.2f} <<<")
                if args.say:
                    say(msg)
            best_seen[t] = max(best_seen.get(t, 0), r['score'])
            best_grade[t] = max(best_grade.get(t, GRADE_RANK['C']), g)
        if args.chart and rows:
            from scanners import bounce_gameplan   # lazy: bounce_gameplan imports this module
            out, _ = bounce_gameplan.generate(tickers, date, {r['ticker']: (r['price'], r['asof'], r['src']) for r in rows},
                                              refresh=args.watch, now=now_)
            if not chart_opened:
                print(f'\n  chart: {out}')
                webbrowser.open(out.resolve().as_uri())
                chart_opened = True
        return rows

    one_pass()
    if not args.watch:
        return

    end = pd.Timestamp(f'{date} {args.until}', tz=TZ)
    if args.date is None and now_et() > end:
        # started after today's cutoff — the next session's cutoff is the real end
        end = pd.Timestamp(f'{pq.adjust_date_forward(now_et().strftime("%Y-%m-%d"), 1)} {args.until}', tz=TZ)
    logger.info(f'Watching every {args.watch}s until {end.strftime("%Y-%m-%d %H:%M ET")} — '
                f'Yahoo/Blue Ocean overnight prints until 04:00, then Polygon; '
                f'{PRICES_FILE.name} or --price override by hand')
    while now_et() < end:
        time.sleep(args.watch)
        try:
            one_pass()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.warning(f'pass failed: {e}')
    logger.info('Watch window over.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
