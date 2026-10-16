"""
Bounce Game Plan — what grade each name is at every possible open price, as an
interactive chart.

Why this exists (2026-10-15): the night before / morning of a gap-down bounce
the useful question is "it's here now; at what open is this an A+?". This lays
out analyzers/bounce_ladder.py for a few names on one HTML page:

  * daily candles with the grade zones shaded (A+ / A / B / C / no trade),
    prior close, mid Bollinger band, 200-day, premarket low and a marker for
    the current price (or the actual open, when reviewing a past day)
  * score (0-6) and intensity (0-100) profiles on the same price axis — hover
    any price for the criteria passing/failing, exit targets from that price,
    what that grade has meant on your past trades, and the population odds
  * a key-levels table: every price where a criterion or the grade flips

Grades: score gates, intensity grades — GO (5+/6) with intensity 75+ = A+,
60-75 = A, under 60 = B; 4/6 = B; 3/6 = C; 2/6 or less = no trade.

Prices (live mode, newest wins — same sources as the overnight tracker):
  --price TICKER=PRICE / data/overnight_prices.json  >  Yahoo Blue Ocean
  overnight print (8 PM-4 AM ET)  >  Polygon (15-min delayed).
Once the bounce day's regular session has opened (or for a past --date) the
marker is that day's ACTUAL open and the day's candle is drawn.

Usage:
    python -m scanners.bounce_gameplan KORU SOXL SKHY DRAM            # tonight / this morning
    python -m scanners.bounce_gameplan SOXL KORU --price KORU=10.65   # manual quote
    python -m scanners.bounce_gameplan SNDK SOXL DRAM --date 2026-07-28   # review a past day
    python -m scanners.bounce_gameplan KORU --no-open                 # write the file only

Output: charts/gameplan/<date>_bounce_gameplan.html (git-ignored), opened in
the browser. `overnight_bounce_tracker --watch --chart` rewrites the same page
each pass and the page reloads itself.
"""

import argparse
import html
import logging
import sys
import webbrowser
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from analyzers.bounce_exit_targets import calculate_bounce_exit_targets
from analyzers.bounce_ladder import (GRADE_RANK, Ladder, build_ladder, grade_zones, historical_grade_stats,
                                     lookup_bounce_odds)
from data_queries import polygon_queries as pq
from scanners import overnight_bounce_tracker as ot
from scanners.premarket_bounce_scanner import (CAP_CACHE_FILE, _load_json, completed_daily, get_cap,
                                               static_from_hist)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('bounce_gameplan')

OUT_DIR = Path(__file__).resolve().parent.parent / 'charts' / 'gameplan'
CANDLE_SESSIONS = 20

GRADE_COLORS = {'A+': '#0a7d32', 'A': '#3fb950', 'B': '#d29922', 'C': '#e8742c', '-': '#9aa4ae'}
GRADE_LABELS = {'A+': 'A+', 'A': 'A', 'B': 'B', 'C': 'C', '-': 'no trade'}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_prepared: Dict[Tuple[str, str], dict] = {}   # (ticker, date) -> daily context + ladder (static all night)


def prepare(ticker: str, date: str, cap_cache: dict, floor: float = -0.40) -> Optional[dict]:
    key = (ticker, date)
    if key not in _prepared:
        hist = completed_daily(ticker, date)
        if hist is None:
            logger.warning(f'{ticker}: no daily history — skipped')
            return None
        static = static_from_hist(hist)
        cap = get_cap(ticker, cap_cache)
        _prepared[key] = {'ticker': ticker, 'hist': hist, 'static': static, 'cap': cap,
                          'ladder': build_ladder(static, cap, ticker=ticker, floor=floor)}
    return _prepared[key]


def day_bars(ticker: str, date: str) -> dict:
    """Premarket low and the regular session so far for the bounce day (Polygon, 15-min delayed)."""
    out = {'pm_low': None, 'pm_low_time': None, 'rth': None}
    try:
        intra = pq.get_intraday(ticker, date, 1, 'minute')
    except Exception:
        intra = None
    if intra is None or intra.empty:
        return out
    open_ts = pd.Timestamp(f'{date} 09:30', tz=intra.index.tz)
    close_ts = pd.Timestamp(f'{date} 16:00', tz=intra.index.tz)
    pm = intra[intra.index < open_ts]
    if not pm.empty:
        out['pm_low'] = float(pm['low'].min())
        out['pm_low_time'] = pm['low'].idxmin().strftime('%H:%M')
    rth = intra[(intra.index >= open_ts) & (intra.index < close_ts)]
    if not rth.empty:
        out['rth'] = {'open': float(rth['open'].iloc[0]), 'high': float(rth['high'].max()),
                      'low': float(rth['low'].min()), 'close': float(rth['close'].iloc[-1]),
                      'low_time': rth['low'].idxmin().strftime('%H:%M'),
                      'asof': rth.index[-1]}
    return out


def live_prices(tickers, cli_prices, use_prices_file=True, use_yahoo=True) -> Dict[str, Tuple[float, pd.Timestamp, str]]:
    """Newest of manual / Yahoo overnight / Polygon per ticker (the overnight tracker's precedence)."""
    manual = dict(ot.load_prices_file() if use_prices_file else {})
    manual.update(cli_prices)
    yahoo = ot.yahoo_overnight_prices(tickers) if use_yahoo else {}
    out = {}
    for t in tickers:
        try:
            poly = ot.polygon_price(t)
        except Exception as e:
            logger.warning(f'{t}: Polygon snapshot failed: {e}')
            poly = None
        chosen = ot.choose_price(manual.get(t), poly, yahoo.get(t))
        if chosen is not None:
            out[t] = chosen
    return out


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _pct(x, digits=1) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return 'n/a'
    return f'{x * 100:+.{digits}f}%'


def _px(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return 'n/a'
    return f'{x:,.2f}' if x >= 1 else f'{x:.4f}'


def _hist_line(grade: str, stats: dict) -> str:
    s = stats.get(grade)
    if not s:
        return 'none of your trades graded here'
    return (f"n={s['n']} · O→H {_pct(s['med_open_high'], 0)} · O→C {_pct(s['med_open_close'], 0)}"
            f" · O→L {_pct(s['med_open_low'], 0)}")


def _odds_line(score: int, cap: str) -> str:
    o = lookup_bounce_odds(score, cap)
    if not o or o.get('p_bounce5') is None:
        return 'n/a'
    return f"{o['p_bounce5'] * 100:.0f}% ({o['bucket']}, n={o['n']:,})"


def _targets(ctx: dict, price: float) -> dict:
    s = ctx['static']
    return calculate_bounce_exit_targets(ctx['cap'], price, s['atr'], prior_close=s['prior_close'],
                                         mid_bb=s.get('mid_bb'))


def _targets_line(t: dict) -> str:
    bits = [f"{x['name'].split(' (')[0]} {_px(x['target_price'])}" for x in t.get('tiers', []) if x.get('target_price')]
    if t.get('mid_bb'):
        bits.append(f"mid-BB {_px(t['mid_bb']['price'])}")
    return ' · '.join(bits)


def _setup_label(static: dict) -> str:
    weak = static['setup_type'] == 'GapFade_weakstock'
    if static.get('sma200'):
        where = 'below' if static['prior_close'] < static['sma200'] else 'above'
        why = f"{where} 200-day {_px(static['sma200'])}"
    else:
        why = 'no 200-day history; 30-day change at the close'
    return f"{'weak' if weak else 'strong'} stock ({why})"


def _hover(ctx: dict, r: dict, stats: dict) -> str:
    """Hover text for one ladder price."""
    desc = r['descriptions']
    failing = [desc[c] for c, ok in r['passed'].items() if not ok]
    inten = f"{r['intensity']:.0f}" if r['intensity'] is not None else 'n/a'
    return '<br>'.join([
        f"<b>{_px(r['price'])}</b>  (gap {_pct(r['gap_pct'])})",
        f"<b>{r['score']}/6 {r['rec']} · grade {GRADE_LABELS[r['grade']]}</b> · intensity {inten}",
        f"failing: {', '.join(failing) or 'none — 6/6'}",
        f"targets: {_targets_line(_targets(ctx, r['price']))}",
        f"your trades at {GRADE_LABELS[r['grade']]}: {_hist_line(r['grade'], stats)}",
        f"population P(bounce ≥5%): {_odds_line(r['score'], ctx['cap'])}",
    ])


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def price_window(ctx: dict, marker: Optional[dict], bars: dict) -> Tuple[float, float]:
    """The ladder from just above the prior close down to its A+ level (or floor), plus the marker,
    premarket low, the bounce day's range and the last few sessions' highs."""
    lad: Ladder = ctx['ladder']
    s = ctx['static']
    rth = bars.get('rth')
    best = next((lv['price'] for lv in lad.levels if lv['grade'] == 'A+'), lad.ceiling['floor_price'])
    lows = [best, s['prior_close']] + [v for v in (marker and marker['price'], bars.get('pm_low'),
                                                   rth and rth['low']) if v]
    highs = [s['prior_close'] * 1.03, float(ctx['hist']['high'].iloc[-5:].max())] + ([rth['high']] if rth else [])
    return min(lows) * 0.97, max(highs) * 1.01


def visible_sessions(hist: pd.DataFrame, y_hi: float) -> pd.DataFrame:
    """The most recent sessions (up to CANDLE_SESSIONS) back to the first one that sits entirely
    above the price window — older candles would only leave empty dates on the axis."""
    recent = hist.iloc[-CANDLE_SESSIONS:]
    keep = 0
    for low in reversed(recent['low'].tolist()):
        if low > y_hi:
            break
        keep += 1
    return recent.iloc[len(recent) - max(keep, 1):]


def ticker_figure(ctx: dict, marker: Optional[dict], bars: dict, stats: dict) -> go.Figure:
    lad: Ladder = ctx['ladder']
    s = ctx['static']
    rth = bars.get('rth')
    y_lo, y_hi = price_window(ctx, marker, bars)
    hist = visible_sessions(ctx['hist'], y_hi)
    xs = [d.strftime('%m-%d') for d in hist.index]
    day_x = pd.Timestamp(ctx['date']).strftime('%m-%d') + ' ▸'

    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, column_widths=[0.62, 0.19, 0.19],
                        horizontal_spacing=0.012,
                        subplot_titles=('Daily · grade zones by open price', 'Score', 'Intensity'))
    fig.add_trace(go.Candlestick(x=xs, open=hist['open'], high=hist['high'], low=hist['low'],
                                 close=hist['close'], name='daily', increasing_line_color='#57606a',
                                 decreasing_line_color='#cf222e', showlegend=False), row=1, col=1)
    if rth:
        fig.add_trace(go.Candlestick(x=[day_x], open=[rth['open']], high=[rth['high']], low=[rth['low']],
                                     close=[rth['close']], name='bounce day', increasing_line_color='#0969da',
                                     decreasing_line_color='#0969da', showlegend=False), row=1, col=1)


    for grade, top, bottom in grade_zones(lad):
        if top < y_lo or bottom > y_hi:
            continue
        label = GRADE_LABELS[grade] if top >= s['prior_close'] * 1.019 else f"{GRADE_LABELS[grade]} ≤ {_px(top)}"
        fig.add_hrect(y0=max(bottom, y_lo), y1=min(top, y_hi), fillcolor=GRADE_COLORS[grade], opacity=0.13,
                      line_width=0, row=1, col=1, annotation_text=f'<b>{label}</b>',
                      annotation_position='top left', annotation_font_color=GRADE_COLORS[grade],
                      annotation_font_size=12)

    lines = [(s['prior_close'], f"prior close {_px(s['prior_close'])}", '#24292f', 'dash'),
             (s.get('mid_bb'), f"mid-BB {_px(s.get('mid_bb'))} (sell)", '#0969da', 'dot'),
             (s.get('sma200'), f"200-day {_px(s.get('sma200'))}", '#8250df', 'dot')]
    if bars.get('pm_low'):
        lines.append((bars['pm_low'], f"PM low {_px(bars['pm_low'])} @{bars['pm_low_time']}", '#bf3989', 'dash'))
    for y, text, color, dash in lines:
        if y and y_lo <= y <= y_hi:
            fig.add_hline(y=y, line_color=color, line_dash=dash, line_width=1, row=1, col=1,
                          annotation_text=text, annotation_position='bottom right',
                          annotation_font_color=color, annotation_font_size=11)

    if marker:
        mr = marker['row']
        fig.add_trace(go.Scatter(x=[day_x], y=[marker['price']], mode='markers+text',
                                 marker=dict(symbol='diamond', size=15, color=GRADE_COLORS[mr['grade']],
                                             line=dict(color='#24292f', width=1)),
                                 text=[f"{marker['label']} {_px(marker['price'])} · {GRADE_LABELS[mr['grade']]}"],
                                 textposition='middle left', textfont=dict(size=12),
                                 hovertext=[_hover(ctx, mr, stats)], hoverinfo='text', showlegend=False),
                      row=1, col=1)

    pts = lad.points[(lad.points['price'] >= y_lo) & (lad.points['price'] <= y_hi)]
    rows = [lad.at(p) for p in pts['price']]
    hover = [_hover(ctx, r, stats) for r in rows]
    colors = [GRADE_COLORS[g] for g in pts['grade']]
    width = s['prior_close'] * 0.0025 * 0.92
    fig.add_trace(go.Bar(y=pts['price'], x=pts['score'], orientation='h', marker_color=colors, width=width,
                         hovertext=hover, hoverinfo='text', showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(y=pts['price'], x=pts['intensity'].fillna(0), orientation='h', marker_color=colors,
                         width=width, hovertext=hover, hoverinfo='text', showlegend=False), row=1, col=3)
    for x in (60, 75):   # A / A+ intensity cut-offs
        fig.add_vline(x=x, line_dash='dot', line_color='#57606a', line_width=1, row=1, col=3)
    if marker:
        for col in (2, 3):
            fig.add_hline(y=marker['price'], line_color='#24292f', line_width=1.5, row=1, col=col)

    fig.update_yaxes(range=[y_lo, y_hi], tickformat=',.2f' if s['prior_close'] >= 1 else '.4f')
    fig.update_xaxes(type='category', rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(range=[0, 6.3], dtick=1, row=1, col=2)
    fig.update_xaxes(range=[0, 102], dtick=25, row=1, col=3)
    fig.update_layout(template='plotly_white', height=560, margin=dict(l=55, r=15, t=40, b=30),
                      hoverlabel=dict(align='left', font_size=12), bargap=0)
    return fig


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def levels_table(ctx: dict, marker: Optional[dict], stats: dict) -> str:
    lad: Ladder = ctx['ladder']
    rows = [dict(lv, kind='level') for lv in lad.levels]
    if marker:
        mr = marker['row']
        rows.append({'price': marker['price'], 'gap_pct': mr['gap_pct'], 'score': mr['score'],
                     'grade': mr['grade'], 'intensity': mr['intensity'],
                     'events': [f"◆ {marker['label']}"], 'kind': 'now'})
    rows.sort(key=lambda r: (-r['price'], r['kind'] != 'now'))
    body = []
    for r in rows:
        inten = f"{r['intensity']:.0f}" if r['intensity'] is not None else 'n/a'
        g = r['grade']
        body.append(
            f"<tr class='{r['kind']}'><td class='num'>{_px(r['price'])}</td><td class='num'>{_pct(r['gap_pct'])}</td>"
            f"<td>{html.escape('; '.join(r['events']))}</td><td class='num'>{r['score']}/6</td>"
            f"<td><span class='g' style='background:{GRADE_COLORS[g]}'>{GRADE_LABELS[g]}</span></td>"
            f"<td class='num'>{inten}</td><td>{_hist_line(g, stats)}</td>"
            f"<td>{_odds_line(r['score'], ctx['cap'])}</td>"
            f"<td>{html.escape(_targets_line(_targets(ctx, r['price'])))}</td></tr>")
    return ("<div class='scroll'><table><thead><tr><th>Price</th><th>Gap</th><th>What changes</th>"
            "<th>Score</th><th>Grade</th><th>Intensity</th><th>Your trades at this grade (medians)</th>"
            "<th>P(bounce ≥5%)</th><th>Targets from here</th></tr></thead><tbody>"
            + ''.join(body) + "</tbody></table></div>")


def summary_card(ctx: dict, marker: Optional[dict], bars: dict, stats: dict) -> str:
    lad: Ladder = ctx['ladder']
    s = ctx['static']
    parts = [f"<span class='tk'>{ctx['ticker']}</span> <span class='muted'>{ctx['cap']} · {_setup_label(s)}"
             f" · prior close {_px(s['prior_close'])} · ATR {_px(s['atr'])}</span>"]
    if marker:
        mr = marker['row']
        inten = f"{mr['intensity']:.0f}" if mr['intensity'] is not None else 'n/a'
        parts.append(f"<div class='big'>{marker['label']} {_px(marker['price'])} <span class='muted'>{marker['note']}</span>"
                     f" · gap {_pct(mr['gap_pct'])} · {mr['score']}/6 {mr['rec']} · intensity {inten} "
                     f"<span class='g' style='background:{GRADE_COLORS[mr['grade']]}'>{GRADE_LABELS[mr['grade']]}</span></div>")
        nxt = lad.next_level(marker['price'])
        if nxt:
            parts.append(f"<div>Next: <b>{GRADE_LABELS[nxt['grade']]} at {_px(nxt['price'])}</b> "
                         f"({_pct(nxt['pct_away'])} from here, gap {_pct(nxt['gap_pct'])})</div>")
        else:
            parts.append("<div>Next: best grade on the ladder already reached</div>")
        dip = (stats.get(mr['grade']) or {}).get('med_open_low')
        if dip is not None and mr['grade'] != '-':
            parts.append(f"<div class='muted'>Your {GRADE_LABELS[mr['grade']]} trades dipped a median {_pct(dip, 0)} "
                         f"below the open first. Stopped out? The morning watcher's re-entry call is part of the plan.</div>")
    else:
        parts.append("<div class='big muted'>no current price — ladder only</div>")
    c = lad.ceiling
    blocked = f" — {', '.join(c['blocked_desc'])} fails at every price" if c['blocked'] else ''
    parts.append(f"<div class='muted'>Ceiling: {c['score']}/6{blocked}.</div>")
    rth = bars.get('rth')
    if rth:
        parts.append(f"<div class='muted'>Session: open {_px(rth['open'])} · low {_px(rth['low'])} @{rth['low_time']}"
                     f" · high {_px(rth['high'])} · last {_px(rth['close'])} ({_pct(rth['close'] / rth['open'] - 1)} from open)</div>")
    return "<div class='card'>" + ''.join(parts) + "</div>"


CSS = """
:root { --fg:#24292f; --muted:#57606a; --line:#d0d7de; --bg:#ffffff; --now:#fff8c5; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
       color: var(--fg); background: var(--bg); margin: 0 auto; max-width: 1280px; padding: 12px 16px 40px; }
h1 { font-size: 20px; margin: 8px 0 2px; } h2 { font-size: 16px; margin: 28px 0 8px; }
.muted { color: var(--muted); } .tk { font-size: 20px; font-weight: 700; }
.card { border: 1px solid var(--line); border-radius: 8px; padding: 10px 14px; margin: 18px 0 6px; line-height: 1.6; }
.big { font-size: 16px; }
.g { color: #fff; border-radius: 4px; padding: 1px 7px; font-weight: 700; font-size: 13px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-size: 13px; width: 100%; margin: 6px 0 4px; }
th, td { border-bottom: 1px solid var(--line); padding: 4px 8px; text-align: left; white-space: nowrap; }
th { color: var(--muted); font-weight: 600; } td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.now td { background: var(--now); font-weight: 600; }
"""


def grade_legend(stats: dict) -> str:
    rows = []
    rule = {'A+': 'GO (5+/6) and intensity 75+', 'A': 'GO and intensity 60-75', 'B': 'GO under 60, or 4/6',
            'C': '3/6', '-': '2/6 or less'}
    for g in ['A+', 'A', 'B', 'C', '-']:
        s = stats.get(g) or {}
        green = 'n/a' if s.get('pct_green') is None else f"{s['pct_green'] * 100:.0f}%"
        rows.append(f"<tr><td><span class='g' style='background:{GRADE_COLORS[g]}'>{GRADE_LABELS[g]}</span></td>"
                    f"<td>{rule[g]}</td><td class='num'>{s.get('n', 0)}</td>"
                    f"<td class='num'>{_pct(s.get('med_open_high'), 0)}</td><td class='num'>{_pct(s.get('med_open_close'), 0)}</td>"
                    f"<td class='num'>{_pct(s.get('med_open_low'), 0)}</td>"
                    f"<td class='num'>{green}</td></tr>")
    return ("<div class='scroll'><table><thead><tr><th>Grade</th><th>Rule</th><th>Your trades</th><th>Med open→high</th>"
            "<th>Med open→close</th><th>Med open→low</th><th>Closed green</th></tr></thead><tbody>"
            + ''.join(rows) + "</tbody></table></div>"
            "<div class='muted'>Your bounce_data.csv trades re-graded at their open (partly in-sample: the intensity "
            "weights were fit to these trades).</div>")


def render_page(sections: list, date: str, generated: pd.Timestamp, refresh: Optional[int] = None) -> str:
    meta = f"<meta http-equiv='refresh' content='{refresh}'>" if refresh else ''
    ts = pd.Timestamp(date)
    day = f"{ts:%A %b} {ts.day}, {ts.year}"   # no %-d: it is not portable to Windows
    body = ''.join(sec for sec in sections)
    stats = historical_grade_stats()
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"{meta}<title>Bounce game plan {date}</title><style>{CSS}</style>"
            f"<script>{get_plotlyjs()}</script></head><body>"
            f"<h1>Bounce game plan — {day}</h1>"
            f"<div class='muted'>Generated {generated.strftime('%Y-%m-%d %H:%M ET')} · hover the score/intensity bars for any price · "
            f"double-click a chart to autoscale</div>"
            f"<h2>What each grade has meant</h2>{grade_legend(stats)}{body}</body></html>")


def ticker_section(ctx: dict, marker: Optional[dict], bars: dict) -> str:
    stats = historical_grade_stats()
    fig = ticker_figure(ctx, marker, bars, stats)
    chart = fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True, 'displaylogo': False})
    return summary_card(ctx, marker, bars, stats) + chart + levels_table(ctx, marker, stats)


def generate(tickers, date: str, prices: Dict[str, Tuple[float, pd.Timestamp, str]], out: Optional[Path] = None,
             refresh: Optional[int] = None, floor: float = -0.40, now: Optional[pd.Timestamp] = None) -> Tuple[Path, list]:
    """Build the page. A ticker whose bounce-day session has opened is marked at its actual
    open; otherwise at prices[ticker] (price, asof, src). Returns (path, summary rows)."""
    now = now or ot.now_et()
    cap_cache = _load_json(CAP_CACHE_FILE)
    built = []
    for t in tickers:
        ctx = prepare(t, date, cap_cache, floor=floor)
        if ctx is None:
            continue
        ctx = dict(ctx, date=date)
        bars = day_bars(t, date)
        marker = None
        if bars['rth']:
            p = bars['rth']['open']
            marker = {'price': p, 'label': 'open', 'note': '(actual)'}
        elif t in prices:
            p, asof, src = prices[t]
            marker = {'price': p, 'label': 'now', 'note': f'({src}, {ot._age(asof, now)})'}
        if marker:
            marker['row'] = ctx['ladder'].at(marker['price'])
        built.append((ctx, marker, bars))

    def rank(item):
        _, marker, _ = item
        if not marker:
            return (-1, 0)
        return (GRADE_RANK[marker['row']['grade']], marker['row']['intensity'] or 0)
    built.sort(key=rank, reverse=True)

    page = render_page([ticker_section(*b) for b in built], date, now, refresh)
    out = out or OUT_DIR / f'{date}_bounce_gameplan.html'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)

    summary = []
    for ctx, marker, _ in built:
        lad = ctx['ladder']
        row = {'ticker': ctx['ticker'], 'marker': marker, 'next': lad.next_level(marker['price']) if marker else None,
               'levels': lad.levels}
        summary.append(row)
    return out, summary


def print_summary(summary: list):
    for r in summary:
        m = r['marker']
        here = (f"{m['label']} {_px(m['price'])} {m['row']['score']}/6 {GRADE_LABELS[m['row']['grade']]}"
                if m else 'no price')
        nxt = r['next']
        nxt_s = f"next {GRADE_LABELS[nxt['grade']]} at {_px(nxt['price'])} ({_pct(nxt['pct_away'])})" if nxt else 'best grade reached'
        ladder = '  '.join(f"{GRADE_LABELS[lv['grade']]}≤{_px(lv['price'])}" for lv in r['levels']
                           if any(e.startswith('grade') for e in lv['events']))
        print(f"  {r['ticker']:6s} {here:28s} {nxt_s:34s} | {ladder}")


def main(argv=None):
    ap = argparse.ArgumentParser(description='Bounce grade at every possible open price, as an interactive chart')
    ap.add_argument('tickers', nargs='*', help=f'tickers (default {" ".join(ot.DEFAULT_TICKERS)})')
    ap.add_argument('--price', action='append', metavar='TICKER=PRICE', help='manual quote (repeatable)')
    ap.add_argument('--date', help='bounce day YYYY-MM-DD (default: next session; a past date reviews that day)')
    ap.add_argument('--out', type=Path, help='output HTML path (default charts/gameplan/<date>_bounce_gameplan.html)')
    ap.add_argument('--floor', type=float, default=-0.40, help='deepest gap to ladder, as a fraction (default -0.40)')
    ap.add_argument('--no-open', action='store_true', help='write the file without opening a browser')
    ap.add_argument('--no-yahoo', action='store_true', help='skip the Yahoo/Blue Ocean overnight feed')
    ap.add_argument('--no-prices-file', action='store_true', help='ignore data/overnight_prices.json')
    args = ap.parse_args(argv)

    now = ot.now_et()
    cli_prices = ot.parse_price_flags(args.price, now)
    tickers = [t.upper() for t in args.tickers] or list(ot.DEFAULT_TICKERS)
    for t in cli_prices:
        if t not in tickers:
            tickers.append(t)
    date = args.date or ot.bounce_day(now)
    live = pd.Timestamp(date).date() >= now.date()
    prices = live_prices(tickers, cli_prices, not args.no_prices_file, not args.no_yahoo) if live else {}

    logger.info(f'Bounce day {date} — laddering {", ".join(tickers)}')
    out, summary = generate(tickers, date, prices, out=args.out, floor=args.floor, now=now)
    print(f'\nBounce game plan {date}')
    print_summary(summary)
    print(f'\n  wrote {out}')
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
