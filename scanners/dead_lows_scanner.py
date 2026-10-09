"""
Dead-lows close scanner: the overnight starter alert.

Why this exists (SNDK 2026-07-29, overnight study 2026-10-09):
  On day 4 of the semis flush SNDK closed at 0.14 of its range, 57% off its
  high. The trader waited for a gap-down GO open that never came: +11.7%
  gap, +26% day, +80% in 13 sessions. Across every liquid US name 2021-2026
  (dedup, 272 events / 160 dates in the >=$1B cell):
    - close location does NOT predict gap direction; a >=2% gap-down came
      only 17.6% of the time, a >=3% gap-up 23.2%.
    - a starter at the close with a 1-ATR stop, plus an add if it does gap
      down, was positive in every period (+.11 to +.15R/event pre-2026,
      +.24 to +.43R in 2026). A stop tight under the day's low got run.
    - holding past day 1 only paid on broad flushes (>=51 names; 7/29 had
      225) or 5+ red days.

Gate (user, 2026-10-09: "only if the priority report recognizes a bounce that
is high confidence"): a name is eligible only if the priority report rated it
a GO bounce today (morning) or in either report of the prior 2 sessions. The
look-back is what catches the motivating case: SNDK was GO 5/6 in both 7/28
reports, then CAUTION 4/6 on the 7/29 morning it closed at dead lows. No GO
bounce in the window -> the scanner exits at launch without touching the market.
In the 2026 archive (3/31-8/07) the gate fired on 4 days (6/09, 6/10, 7/16, 7/29).

What it does: at launch (Task Scheduler, 12:30 ET) it reads the gate, builds
daily context (rolling grouped-daily cache) and re-checks the gated names per
ticker; at close-13 and close-8 it counts breadth and checks them with
real-time Trillium bars, alerting names that qualify:
    ADV20 >= $250M, red, 4+ straight lower closes (or 4 of the last 5 down),
    >= 25% below the 30-session high close, last in the bottom 15% of range.
Every gated name that qualifies is spoken (max 3 lines) and emailed.
Silent when nothing new qualifies. Each name alerts once per day.
`--asof D --no-gate` replays the whole market (research only).

Breadth = names with ADV20 >= $9M that are red, 3+ straight lower closes and
>= 25% off the 30-session high close (the study's cluster definition), from
one Polygon all-tickers snapshot (15-min delayed; fine for a count).

Every final candidate is logged to data/signal_ledger.csv (source=dead_lows,
date = NEXT trading day, last price in the label) so the morning
fill_signal_outcomes pass records what happened next.

Usage:
    python -m scanners.dead_lows_scanner                  # wait for the close, alert, log (Task Scheduler)
    python -m scanners.dead_lows_scanner --dry            # same timing, print only
    python -m scanners.dead_lows_scanner --once [--dry]   # one pass now (no ledger)
    python -m scanners.dead_lows_scanner --asof 2026-07-29   # replay a past close (gated)
    python -m scanners.dead_lows_scanner --asof 2026-07-29 --no-gate   # whole market
"""

import argparse
import datetime
import json
import logging
import math
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_queries import polygon_queries as pq
from scanners.premarket_bounce_scanner import (
    CAP_CACHE_FILE, KNOWN_ETFS, _load_json, get_cap, load_watchlist,
)
from support import risk_source
from support.config import send_email
from support.signal_ledger import log_signals

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('dead_lows_scanner')

# ---------------------------------------------------------------------------
# Config (thresholds are the 10/09 study's; change them there first)
# ---------------------------------------------------------------------------

EMAIL_TO = 'zmburr@gmail.com'
TZ = 'US/Eastern'

ALERT_ADV = 250e6            # 20-day avg dollar volume floor for an alert
SPEAK_ADV = 1e9              # SNDK-like tier: spoken
BREADTH_ADV = 9e6            # breadth universe floor (no price floor, per user rule)
DD_MAX = -0.25               # close vs the prior 30-session high close
CLV_MAX = 0.15               # "dead lows": bottom 15% of today's range
STREAK_MIN = 4               # straight lower closes, OR
DOWN5_MIN = 4                # down closes among the last 5
BREADTH_STREAK = 3
BREADTH_HOLD = 51            # broad flush -> hold to D5
HOLD_STREAK = 5              # long flush -> hold to D5
GAP_ADD = -0.02              # tomorrow's open at/below this -> add on the 2-min signal
MAX_SPOKEN = 3               # spoken names per pass; the rest go in the email only

ATR_N, ADV_N, HIGH_N, SMA_N = 14, 20, 30, 20
HIST_SESSIONS = 45           # completed sessions of context needed before D0
ROLLING_KEEP = 60            # sessions kept in the rolling cache by live runs
FALLBACK_ONE_R = 3000.0

PASS_BEFORE_CLOSE_MIN = (13, 8)   # live checks; context is built at launch (12:30)

GATE_LOOKBACK = 2            # prior sessions whose GO bounce still counts

_DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
PRIORITY_DIR = _DATA_DIR / 'priority_signals'
CACHE_DIR = _DATA_DIR / 'dead_lows'
CACHE_FILE = CACHE_DIR / 'grouped_daily_rolling.pkl'
STATE_FILE = _DATA_DIR / 'dead_lows_state.json'

_ROUND_RATIOS = (2, 3, 4, 5, 10)


# ---------------------------------------------------------------------------
# Pure features (mirror scratchpad/overnight/build_events.py exactly)
# ---------------------------------------------------------------------------

def prior_features(bars: pd.DataFrame) -> dict | None:
    """Context as of the PRIOR close from completed daily bars (oldest first,
    every row strictly before D0). Columns: open, high, low, close, volume.

    Matches the study: streak = consecutive lower closes; ATR = mean true
    range of the last 14 sessions; ADV = mean close*volume of the last 20;
    max30 = highest close of the last 30. None when history is too short.
    """
    if bars is None or len(bars) < 20:
        return None
    c = bars['close'].to_numpy(dtype=float)
    h = bars['high'].to_numpy(dtype=float)
    l = bars['low'].to_numpy(dtype=float)
    v = bars['volume'].to_numpy(dtype=float)

    down = c[1:] < c[:-1]
    streak = 0
    for d in down[::-1]:
        if not d:
            break
        streak += 1
    down4 = int(down[-4:].sum()) if len(down) >= 4 else None

    prev = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - prev), np.abs(l - prev)]), axis=0)
    tr_win = tr[-ATR_N:]
    adv_win = (c * v)[-ADV_N:]
    if len(tr_win) < 10 or len(adv_win) < 15:
        return None
    return {
        'prior_close': float(c[-1]),
        'streak_prior': streak,
        'down4_prior': down4,
        'max30': float(c[-HIGH_N:].max()),
        'atr': float(np.mean(tr_win)),
        'adv20': float(np.mean(adv_win)),
        'mid_bb': float(c[-SMA_N:].mean()) if len(c) >= SMA_N else None,
    }


def day_features(prior: dict, last: float, high: float, low: float) -> dict:
    """Today's state at `last` given prior context. High/low widen to include last."""
    high = max(high, last)
    low = min(low, last)
    rng = high - low
    red = last < prior['prior_close']
    down4 = prior.get('down4_prior')
    return {
        **prior,
        'last': last, 'high': high, 'low': low,
        'ret0': last / prior['prior_close'] - 1,
        'red': red,
        'streak': prior['streak_prior'] + 1 if red else 0,
        'down5': (down4 + int(red)) if down4 is not None else None,
        'dd30': last / prior['max30'] - 1,
        'clv': (last - low) / rng if rng > 0 else 0.5,
    }


def qualify(f: dict) -> bool:
    """The alert rule (study's >=$250M dead-lows cell, either streak form)."""
    streak_ok = f['streak'] >= STREAK_MIN or (f.get('down5') or 0) >= DOWN5_MIN
    return (f['adv20'] >= ALERT_ADV and f['red'] and streak_ok
            and f['dd30'] <= DD_MAX and f['clv'] < CLV_MAX)


def breadth_hit(f: dict) -> bool:
    """Study cluster definition: any CLV."""
    return (f['adv20'] >= BREADTH_ADV and f['red']
            and f['streak'] >= BREADTH_STREAK and f['dd30'] <= DD_MAX)


def tier(f: dict) -> str:
    return 'SPEAK' if f['adv20'] >= SPEAK_ADV and f['streak'] >= STREAK_MIN else 'EMAIL'


def starter(last: float, atr: float, one_r: float) -> tuple[int, float]:
    """Shares risking 1R at a 1-ATR stop, and the stop price."""
    if not atr or atr <= 0:
        return 0, last
    return int(math.floor(one_r / atr)), last - atr


def gap_add_level(last: float) -> float:
    """Tomorrow's open at/below this (-2% vs today's close) -> add on the 2-min signal."""
    return last * (1 + GAP_ADD)


def hold_plan(breadth: int, streak: int, cap: str) -> str:
    if breadth >= BREADTH_HOLD or streak >= HOLD_STREAK:
        why = f'breadth {breadth}' if breadth >= BREADTH_HOLD else f'{streak} red days'
        if cap in ('ETF', 'Large'):
            return f'hold to D5 ({why}); exit at mid-BB'
        return f'hold to D5 ({why})'
    return 'exit next day\'s close'


def split_suspect(closes: np.ndarray) -> bool:
    """A close-to-close jump at a round split ratio means inconsistent adjustment."""
    c = np.asarray(closes, dtype=float)
    if len(c) < 2:
        return False
    with np.errstate(divide='ignore', invalid='ignore'):
        lr = np.log(c[1:] / c[:-1])
    lr = np.nan_to_num(lr)
    return any(np.any(np.abs(np.abs(lr) - math.log(k)) < 0.04) for k in _ROUND_RATIOS)


def rank_rows(rows: list[dict], watchlist: set) -> list[dict]:
    """SPEAK first, then watchlist names, then most liquid."""
    return sorted(rows, key=lambda r: (r['tier'] != 'SPEAK',
                                       r['ticker'] not in watchlist,
                                       -r['adv20']))


def build_row(ticker: str, f: dict, breadth: int, cap: str, one_r: float, src: str,
              gate: str | None = None) -> dict:
    """`gate` = why the priority report put this name in scope. Gated names are
    all spoken; ungated rows (--no-gate research replays) keep the ADV tiers."""
    shares, stop = starter(f['last'], f['atr'], one_r)
    return {
        'ticker': ticker, 'tier': 'SPEAK' if gate else tier(f), 'gate': gate or '',
        'cap': cap, 'src': src,
        'adv20': f['adv20'], 'streak': f['streak'], 'down5': f.get('down5'),
        'dd30': f['dd30'], 'clv': f['clv'], 'ret0': f['ret0'],
        'last': f['last'], 'atr': f['atr'], 'shares': shares, 'stop': stop,
        'add_below': gap_add_level(f['last']),
        'mid_bb': f.get('mid_bb'),
        'hold': hold_plan(breadth, f['streak'], cap),
        'breadth': breadth,
    }


# ---------------------------------------------------------------------------
# Daily history (rolling grouped-daily cache, whole market)
# ---------------------------------------------------------------------------

def _nyse():
    import pandas_market_calendars as mcal
    return mcal.get_calendar('NYSE')


def sessions_before(date: datetime.date, n: int) -> list[str]:
    days = _nyse().valid_days(start_date=date - datetime.timedelta(days=int(n * 1.6) + 15),
                              end_date=date - datetime.timedelta(days=1))
    return [d.strftime('%Y-%m-%d') for d in days[-n:]]


def next_trading_day(after: datetime.date) -> datetime.date:
    sched = _nyse().schedule(start_date=after + datetime.timedelta(days=1),
                             end_date=after + datetime.timedelta(days=10))
    return sched.index[0].date()


# ---------------------------------------------------------------------------
# Gate: only names the priority report rated a GO bounce
# ---------------------------------------------------------------------------

def _gate_files(date: str, lookback: int) -> list[tuple[str, str]]:
    """(date, session) report files that count for `date`, newest first.
    Today's evening report is excluded: it's written after the close (and in a
    replay it would be lookahead)."""
    prior = sessions_before(datetime.date.fromisoformat(date), lookback)
    files: list[tuple[str, str]] = [(date, 'morning')]
    for d in reversed(prior):
        files += [(d, 'evening'), (d, 'morning')]
    return files


def high_conf_bounces(date: str, lookback: int = GATE_LOOKBACK) -> dict[str, str]:
    """{ticker: why} for bounce signals the priority report rated GO in today's
    morning report or either report of the prior `lookback` sessions. The most
    recent GO wins the label, e.g. 'GO 5/6 · 2026-07-28 evening'."""
    gate: dict[str, str] = {}
    for d, sess in _gate_files(date, lookback):
        path = PRIORITY_DIR / f'{d}_{sess}.json'
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding='utf-8-sig'))
        except Exception as e:
            logger.warning(f'gate: unreadable {path.name}: {e}')
            continue
        for s in payload.get('signals', []):
            if s.get('bucket') == 'bounce' and s.get('recommendation') == 'GO':
                gate.setdefault(str(s['ticker']).upper(),
                                f"GO {s.get('score', '')} · {d} {sess}".replace('  ', ' '))
    return gate


def load_cache() -> dict:
    try:
        with open(CACHE_FILE, 'rb') as fh:
            return pickle.load(fh)
    except Exception:
        return {}


def save_cache(cache: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, 'wb') as fh:
        pickle.dump(cache, fh)


def ensure_history(cache: dict, dates: list[str]) -> int:
    """Fetch any missing sessions (one grouped-daily call each). Returns # fetched."""
    fetched = 0
    for d in dates:
        if d in cache and not cache[d].empty:
            continue
        try:
            aggs = pq.poly_client.get_grouped_daily_aggs(d, adjusted=True)
            cache[d] = pd.DataFrame([{
                'ticker': a.ticker, 'open': a.open, 'high': a.high, 'low': a.low,
                'close': a.close, 'volume': a.volume,
            } for a in (aggs or [])])
            fetched += 1
        except Exception as e:
            logger.warning(f'grouped daily fetch failed for {d}: {e}')
        time.sleep(0.15)
    return fetched


def prune_cache(cache: dict, keep_from: str) -> None:
    for d in [d for d in cache if d < keep_from]:
        del cache[d]


def market_priors(cache: dict, dates: list[str], adv_floor: float) -> dict:
    """Prior features for every ticker over `dates` with ADV20 >= adv_floor.
    Tickers with a split-like jump in the window are dropped."""
    frames = []
    for d in dates:
        df = cache.get(d)
        if df is not None and not df.empty:
            frames.append(df.assign(date=d))
    if not frames:
        return {}
    panel = pd.concat(frames, ignore_index=True)
    panel = panel[~panel['ticker'].str.contains(r'[.\-/]', regex=True, na=True)]
    panel = panel.sort_values(['ticker', 'date'])
    panel['dv'] = panel['close'] * panel['volume']
    # Cheap liquidity cut before the per-ticker work.
    adv = panel.groupby('ticker')['dv'].apply(lambda s: s.tail(ADV_N).mean())
    keep = set(adv[adv >= adv_floor].index)
    out = {}
    for t, g in panel[panel['ticker'].isin(keep)].groupby('ticker', sort=False):
        if split_suspect(g['close'].to_numpy()):
            continue
        p = prior_features(g)
        if p is not None and p['adv20'] >= adv_floor:
            out[t] = p
    return out


def recheck_prior(ticker: str, date: str) -> tuple[dict | None, dict | None]:
    """Per-ticker daily bars (consistently adjusted): prior features from rows
    before `date`, plus `date`'s own bar if Polygon has it (replays)."""
    df = pq.get_levels_data(ticker, date, 75, 1, 'day')
    if df is None or df.empty:
        return None, None
    df = df.copy()
    df['d'] = [ts.strftime('%Y-%m-%d') for ts in df.index]
    before = df[df['d'] < date]
    today = df[df['d'] == date]
    d0 = None
    if not today.empty:
        r = today.iloc[-1]
        d0 = {'last': float(r['close']), 'high': float(r['high']), 'low': float(r['low'])}
    return prior_features(before.reset_index(drop=True)), d0


# ---------------------------------------------------------------------------
# Live values
# ---------------------------------------------------------------------------

def snapshot_values() -> dict:
    """{ticker: {'last','high','low','prev_close'}} from Polygon's all-tickers
    snapshot (15-min delayed). Empty dict on failure."""
    try:
        snap = pq.poly_client.get_snapshot_all('stocks')
    except Exception as e:
        logger.warning(f'Polygon snapshot failed: {e}')
        return {}
    out = {}
    for s in snap:
        day = getattr(s, 'day', None)
        prev = getattr(s, 'prev_day', None)
        lt = getattr(s, 'last_trade', None)
        last = getattr(lt, 'price', None) or getattr(day, 'close', None)
        hi, lo = getattr(day, 'high', None), getattr(day, 'low', None)
        pc = getattr(prev, 'close', None)
        if last and hi and lo and pc:
            out[s.ticker] = {'last': float(last), 'high': float(hi), 'low': float(lo),
                             'prev_close': float(pc)}
    return out


def trillium_day_values(tickers: list, date: str) -> dict:
    """Real-time RTH high/low/last per ticker from today's 1-min bars, ONE
    shared SHEL session (pattern of premarket_bounce_scanner.trillium_prices).
    Returns {} when SHEL is unavailable so callers fall back to Polygon."""
    try:
        from data_queries.trillium_queries import HAS_SHEL, USER, PWD
        import sheldatagateway
        from sheldatagateway import environments
    except Exception:
        return {}
    if not HAS_SHEL or not PWD:
        return {}

    d = pd.to_datetime(date).date()
    rth_open = pd.Timestamp(f'{date} 09:30', tz=TZ).value
    out = {}
    try:
        with sheldatagateway.Session(environments.env_defs.Prod, USER, PWD) as session:
            for t in tickers:
                aggs = []
                try:
                    handle = session.request_data(
                        callback=aggs.append, symbol=t,
                        start_date=d, end_date=d, subscriptions=['bar-1min'])
                    handle.wait()
                    handle.raise_on_error()
                except Exception:
                    continue
                rth = [b for b in aggs if isinstance(b, dict)
                       and (b.get('close-time') or 0) > rth_open and b.get('close')]
                if rth:
                    out[t] = {'last': float(rth[-1]['close']),
                              'high': max(float(b['high']) for b in rth),
                              'low': min(float(b['low']) for b in rth)}
    except Exception as e:
        logger.warning(f'Trillium session failed ({e}) — Polygon fallback (15-min delayed)')
    return out


# ---------------------------------------------------------------------------
# Cap lookup (hold plan needs ETF/Large)
# ---------------------------------------------------------------------------

def lookup_cap(ticker: str, cache: dict, persist: bool) -> str:
    """get_cap() writes the shared cap cache; replays and dry runs must not."""
    if ticker in cache:
        return cache[ticker]
    if persist:
        return get_cap(ticker, cache)
    if ticker in KNOWN_ETFS:
        return 'ETF'
    try:
        det = pq.poly_client.get_ticker_details(ticker)
        if getattr(det, 'type', '') in ('ETF', 'ETP', 'ETN', 'FUND'):
            return 'ETF'
        mc = getattr(det, 'market_cap', None) or 0
        return 'Large' if mc >= 100e9 else 'Medium' if mc >= 2e9 else 'Small' if mc >= 300e6 else 'Micro'
    except Exception:
        return 'Medium'


def _one_r() -> float:
    r = risk_source.one_r_dollars()
    if r is None:
        logger.warning(f'ExitMonitor ONE_R unavailable — sizing with ${FALLBACK_ONE_R:,.0f}')
        return FALLBACK_ONE_R
    return r


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

class Session:
    """Daily context built once: priors for the breadth universe, the alert
    shortlist re-checked per ticker, watchlist, caps, 1R."""

    def __init__(self, date: str, persist: bool, gate: dict | None = None):
        self.date = date
        self.persist = persist
        self.gate = gate              # {ticker: why}; None = whole market (research)
        d = datetime.date.fromisoformat(date)
        self.dates = sessions_before(d, HIST_SESSIONS)
        cache = load_cache()
        n = ensure_history(cache, self.dates)
        if n:
            save_cache(cache)
        logger.info(f'history: {len(self.dates)} sessions ({n} fetched)')
        self.cache = cache
        self.priors = market_priors(cache, self.dates, BREADTH_ADV)
        logger.info(f'breadth universe: {len(self.priors)} names with ADV20 >= ${BREADTH_ADV/1e6:.0f}M')
        self.watchlist = set(load_watchlist())
        self.cap_cache = _load_json(CAP_CACHE_FILE)
        self.one_r = _one_r()
        self.rechecked: dict[str, tuple] = {}

    def shortlist(self, loose: dict | None = None) -> list[str]:
        """Names that could qualify today: liquid and already falling.
        `loose` (replay) = {ticker: day_features} to pre-screen by today's bar."""
        out = []
        for t, p in self.priors.items():
            if self.gate is not None and t not in self.gate:
                continue
            if p['adv20'] < ALERT_ADV * 0.8:
                continue
            if loose is not None:
                f = loose.get(t)
                if not f or not f['red'] or f['dd30'] > -0.20 or f['clv'] >= 0.25:
                    continue
                if f['streak'] < STREAK_MIN and (f.get('down5') or 0) < DOWN5_MIN:
                    continue
            elif p['streak_prior'] < STREAK_MIN - 1 and (p['down4_prior'] or 0) < DOWN5_MIN - 1:
                continue
            out.append(t)
        return out

    def recheck(self, tickers: list[str]) -> None:
        for t in tickers:
            if t in self.rechecked:
                continue
            try:
                self.rechecked[t] = recheck_prior(t, self.date)
            except Exception as e:
                logger.warning(f'recheck failed for {t}: {e}')
                self.rechecked[t] = (None, None)

    def breadth(self, values: dict) -> int:
        n = 0
        for t, p in self.priors.items():
            v = values.get(t)
            if not v:
                continue
            pc = v.get('prev_close')
            if pc and abs(pc / p['prior_close'] - 1) > 0.02:
                continue          # adjustment mismatch vs history -> skip
            if breadth_hit(day_features(p, v['last'], v['high'], v['low'])):
                n += 1
        return n

    def candidates(self, values: dict, breadth: int, src_of: dict) -> list[dict]:
        rows = []
        for t, (prior, _) in self.rechecked.items():
            v = values.get(t)
            if prior is None or not v:
                continue
            f = day_features(prior, v['last'], v['high'], v['low'])
            if not qualify(f):
                continue
            if self.gate is not None and t not in self.gate:
                continue
            cap = lookup_cap(t, self.cap_cache, self.persist)
            rows.append(build_row(t, f, breadth, cap, self.one_r, src_of.get(t, 'P'),
                                  gate=(self.gate or {}).get(t)))
        return rank_rows(rows, self.watchlist)


def live_pass(sess: Session) -> tuple[int, list[dict]]:
    snap = snapshot_values()
    breadth = sess.breadth(snap)
    tickers = list(sess.rechecked)
    trill = trillium_day_values(tickers, sess.date)
    values, src = {}, {}
    for t in tickers:
        if t in trill:
            values[t], src[t] = trill[t], 'T'
        elif t in snap:
            values[t], src[t] = snap[t], 'P'
    logger.info(f'pass: breadth {breadth}; live values {len(trill)} Trillium / '
                f'{len(values) - len(trill)} Polygon of {len(tickers)} shortlisted')
    return breadth, sess.candidates(values, breadth, src)


def replay(date: str, use_gate: bool = True) -> tuple[int, list[dict]]:
    """Score a past close from daily bars: the full-day bar stands in for 15:50.
    Gated by the archived priority reports unless use_gate=False.
    No alerts, no ledger, no state."""
    gate = high_conf_bounces(date) if use_gate else None
    if gate is not None:
        logger.info(f'gate: {len(gate)} GO bounce(s) in scope: '
                    + (', '.join(f'{t} ({w})' for t, w in gate.items()) or 'none'))
    sess = Session(date, persist=False, gate=gate)
    ensure_history(sess.cache, [date])
    save_cache(sess.cache)
    d0 = sess.cache.get(date)
    if d0 is None or d0.empty:
        raise SystemExit(f'no grouped daily bars for {date}')
    d0 = d0.set_index('ticker')
    values = {t: {'last': float(r['close']), 'high': float(r['high']), 'low': float(r['low'])}
              for t, r in d0.iterrows() if t in sess.priors}
    breadth = sess.breadth(values)
    loose = {t: day_features(sess.priors[t], v['last'], v['high'], v['low'])
             for t, v in values.items()}
    short = sess.shortlist(loose)
    sess.recheck(short)
    per_ticker = {t: d for t, (_, d) in sess.rechecked.items() if d is not None}
    rows = sess.candidates(per_ticker, breadth, {t: 'D' for t in per_ticker})
    return breadth, rows


# ---------------------------------------------------------------------------
# Output + alerts
# ---------------------------------------------------------------------------

def _pct(x) -> str:
    return f'{x * 100:+.1f}%' if x is not None else 'n/a'


def _money(x: float) -> str:
    return f'${x / 1e9:.1f}B' if x >= 1e9 else f'${x / 1e6:.0f}M'


def print_table(rows: list[dict], breadth: int, label: str) -> None:
    print(f'\nDead-lows close candidates — {label}   breadth {breadth}'
          f'  (src T=Trillium real-time, P=Polygon 15-min delayed, D=daily bar)')
    if not rows:
        print('  none')
        return
    print(f"{'TICK':6} {'TIER':6} {'ADV':>7} {'STK':>3} {'OFF HI':>7} {'CLV':>5} {'LAST':>9} "
          f"{'ATR':>8} {'SHRS':>6} {'STOP':>9} {'ADD<=':>9} {'MID-BB':>9} {'SRC':>3}  HOLD")
    for r in rows:
        mid = f"{r['mid_bb']:.2f}" if r['mid_bb'] else 'n/a'
        print(f"{r['ticker']:6} {r['tier']:6} {_money(r['adv20']):>7} {r['streak']:>3} "
              f"{_pct(r['dd30']):>7} {r['clv']:>5.2f} {r['last']:>9.2f} {r['atr']:>8.2f} "
              f"{r['shares']:>6} {r['stop']:>9.2f} {r['add_below']:>9.2f} {mid:>9} {r['src']:>3}  {r['hold']}")


def spoken_line(r: dict) -> str:
    return f"Dead lows close. {r['ticker']}. Starter {r['shares']} shares. Breadth {r['breadth']}."


def spoken_lines(rows: list[dict]) -> list[str]:
    """One line per SPEAK-tier name, capped: a broad flush (7/29 had 25) must
    not become a minute of speech. The rest are named in the email."""
    speak_rows = [r for r in rows if r['tier'] == 'SPEAK']
    lines = [spoken_line(r) for r in speak_rows[:MAX_SPOKEN]]
    extra = len(speak_rows) - MAX_SPOKEN
    if extra > 0:
        lines.append(f'Plus {extra} more dead lows names in the email.')
    return lines


def speak(text: str) -> None:
    safe = text.replace("'", '')
    try:
        subprocess.run(['powershell', '-Command',
                        "Add-Type -AssemblyName System.Speech; "
                        f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe}')"],
                       timeout=60, check=False)
    except Exception as e:
        logger.warning(f'speech failed: {e}')


def format_email(rows: list[dict], breadth: int, asof: pd.Timestamp) -> tuple[str, str]:
    names = ', '.join(r['ticker'] for r in rows[:3])
    more = f' (+{len(rows) - 3} more)' if len(rows) > 3 else ''
    subject = f'Dead-lows close: {names}{more} — breadth {breadth}'
    cell = 'padding:4px 8px;border-bottom:1px solid #1f2937;'
    trs = []
    for r in rows:
        color = '#f59e0b' if r['tier'] == 'SPEAK' else '#e8ecf4'
        mid = f"{r['mid_bb']:.2f}" if r['mid_bb'] else 'n/a'
        trs.append(
            f"<tr style='color:{color};'>"
            f"<td style='{cell}'><b>{r['ticker']}</b></td><td style='{cell}'>{r['tier']}</td>"
            f"<td style='{cell}'>{_money(r['adv20'])}</td><td style='{cell}'>{r['streak']}</td>"
            f"<td style='{cell}'>{_pct(r['dd30'])}</td><td style='{cell}'>{r['clv']:.2f}</td>"
            f"<td style='{cell}'>{r['last']:.2f}</td><td style='{cell}'>{r['atr']:.2f}</td>"
            f"<td style='{cell}'><b>{r['shares']}</b></td><td style='{cell}'>{r['stop']:.2f}</td>"
            f"<td style='{cell}'>{r['add_below']:.2f}</td><td style='{cell}'>{mid}</td>"
            f"<td style='{cell}'>{r['hold']}</td><td style='{cell}'>{r['breadth']}</td>"
            f"<td style='{cell}'>{r.get('gate') or '—'}</td></tr>")
    head = ''.join(f"<th style='{cell}text-align:left;color:#9ca3af;'>{h}</th>" for h in (
        'Ticker', 'Tier', 'ADV', 'Streak', 'Off high', 'CLV', 'Last', 'ATR', 'Starter',
        'Stop', 'Add if open ≤', 'Mid-BB', 'Hold plan', 'Breadth', 'Priority report'))
    body = f"""<html><body style="background:#0b0f17;color:#e8ecf4;font-family:Segoe UI,Arial;font-size:13px;">
<h2 style="color:#e8ecf4;margin:0 0 4px 0;">Dead-lows close</h2>
<div style="color:#6b7280;font-size:12px;margin-bottom:12px;">{asof.strftime('%Y-%m-%d %H:%M ET')} —
breadth {breadth} names flushing together (≥51 = broad)</div>
<table style="border-collapse:collapse;">{head}{''.join(trs)}</table>
<div style="color:#9ca3af;font-size:12px;margin-top:14px;line-height:1.5;">
Plan: buy the starter at the close (1R at a 1-ATR stop). If it opens at or below the add level
tomorrow, add on the normal 2-min signal; if not, you're already in.<br>
Why: after a dead-lows close a ≥2% gap-down came only 18% of the time in ≥$1B names
(2021-2026); starter + gap-down add was positive in every period.</div>
<div style="color:#4b5563;font-size:11px;margin-top:16px;">Dead-lows Close Scanner — only names the priority report rated a GO bounce (today or the prior 2 sessions); once per name per day; silent otherwise.</div>
</body></html>"""
    return subject, body


def load_state(date: str) -> dict:
    state = _load_json(STATE_FILE)
    if state.get('date') != date:
        state = {'date': date, 'alerted': {}}
    return state


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=1))


def new_alerts(rows: list[dict], state: dict) -> list[dict]:
    """Rows not yet alerted today, or upgraded EMAIL -> SPEAK."""
    out = []
    for r in rows:
        prev = state['alerted'].get(r['ticker'])
        if prev is None or (prev == 'EMAIL' and r['tier'] == 'SPEAK'):
            out.append(r)
    return out


def send_alerts(rows: list[dict], breadth: int, asof: pd.Timestamp, state: dict, dry: bool) -> None:
    """Speak SPEAK-tier names, one email for all new rows, record state."""
    if not rows:
        return
    for line in spoken_lines(rows):
        logger.info(f'SPEAK: {line}')
        if not dry:
            speak(line)
    subject, body = format_email(rows, breadth, asof)
    logger.info(f'ALERT: {subject}')
    if dry:
        return
    send_email(EMAIL_TO, subject, body, is_html=True)
    for r in rows:
        state['alerted'][r['ticker']] = r['tier']
    save_state(state)


def ledger_entries(rows: list[dict]) -> list[dict]:
    return [{
        'ticker': r['ticker'], 'bucket': 'bounce', 'cap': r['cap'], 'rec': r['tier'],
        'score_str': '',
        'label': (f"dead_lows {r['tier']} brd{r['breadth']} stk{r['streak']} "
                  f"clv{r['clv']:.2f} dd{r['dd30'] * 100:.0f}% px{r['last']:.2f}"),
        'metrics': {'atr_pct': r['atr'] / r['last'] if r['last'] else None},
    } for r in rows]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TZ)


def _market_close(date: str) -> pd.Timestamp | None:
    """Today's NYSE close in ET (13:00 on half-days), None on holidays/weekends."""
    sched = _nyse().schedule(start_date=date, end_date=date)
    if sched.empty:
        return None
    return pd.Timestamp(sched.iloc[0]['market_close']).tz_convert(TZ)


def _sleep_until(t: pd.Timestamp) -> None:
    wait = (t - _now_et()).total_seconds()
    if wait > 0:
        logger.info(f'sleeping {wait / 60:.1f} min until {t.strftime("%H:%M")} ET')
        time.sleep(wait)


def main():
    ap = argparse.ArgumentParser(description='Dead-lows close scanner (overnight starter alert)')
    ap.add_argument('--once', action='store_true', help='one pass now (alerts unless --dry; no ledger)')
    ap.add_argument('--dry', action='store_true', help='print only: no speech, email, ledger or state')
    ap.add_argument('--asof', metavar='YYYY-MM-DD', help='replay a past close from daily bars')
    ap.add_argument('--no-gate', action='store_true',
                    help='with --asof: whole market, ignore the priority-report gate (research)')
    args = ap.parse_args()

    if args.asof:
        breadth, rows = replay(args.asof, use_gate=not args.no_gate)
        print_table(rows, breadth, f'replay {args.asof} (full-day bar)')
        return

    now = _now_et()
    date = now.strftime('%Y-%m-%d')
    close = _market_close(date)
    if close is None:
        logger.info(f'{date}: market closed — nothing to do')
        return
    gate = high_conf_bounces(date)
    if not gate:
        logger.info(f'{date}: no GO bounce in the priority report (today or the prior '
                    f'{GATE_LOOKBACK} sessions) — nothing to watch')
        return
    logger.info(f'gate: {", ".join(f"{t} ({w})" for t, w in gate.items())}')
    if args.once:
        passes = [now]
    else:
        passes = [close - pd.Timedelta(minutes=m) for m in PASS_BEFORE_CLOSE_MIN]
        if now > passes[-1]:
            logger.info(f'started {now.strftime("%H:%M")} — past the last pass '
                        f'({passes[-1].strftime("%H:%M")}); exiting')
            return

    # Daily context needs no live data: build it at launch, well before the close.
    sess = Session(date, persist=not args.dry, gate=gate)
    keep = sessions_before(datetime.date.fromisoformat(date), ROLLING_KEEP)
    if keep:
        prune_cache(sess.cache, keep[0])
        save_cache(sess.cache)
    short = sess.shortlist()
    sess.recheck(short)
    logger.info(f'shortlist: {len(short)} names ({", ".join(short[:12])}{"..." if len(short) > 12 else ""})')

    state = load_state(date) if not args.dry else {'date': date, 'alerted': {}}
    latest: dict[str, dict] = {}
    for p in passes:
        _sleep_until(p)
        breadth, rows = live_pass(sess)
        print_table(rows, breadth, _now_et().strftime('%Y-%m-%d %H:%M ET'))
        for r in rows:
            latest[r['ticker']] = r
        send_alerts(new_alerts(rows, state), breadth, _now_et(), state, args.dry)
        if args.dry:
            for r in rows:
                state['alerted'].setdefault(r['ticker'], r['tier'])

    if args.once or args.dry or not latest:
        return
    final = rank_rows(list(latest.values()), sess.watchlist)
    target = next_trading_day(datetime.date.fromisoformat(date)).strftime('%Y-%m-%d')
    n = log_signals('dead_lows', 'close', ledger_entries(final), date_str=target)
    logger.info(f'ledger: {n} rows appended (date={target})')


if __name__ == '__main__':
    main()
