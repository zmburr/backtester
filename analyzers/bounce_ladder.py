"""
Bounce gap ladder — the bounce grade at every hypothetical open price.

Why this exists (2026-10-15 KORU/SOXL/SKHY): the night before and the morning
of a gap-down bounce the question is "it's HERE now; if it opens at $X, is it
an A+, an A, a B?". The checklist only answered that for the current price, so
game-planning meant re-asking in chat. Everything in the pre-trade checklist
except prior-day range is a function of price, so the whole score can be laid
out as a ladder of price levels from the prior close down.

Scoring is identical to the live scanners: the premarket scanner's
build_static()/metrics_from_price(), BouncePretrade.validate() with the setup
type fixed at the PRIOR CLOSE (see setup_type_at_close), and the bounce
intensity composite against the matching setup's reference trades. With the
setup type fixed, every price-driven criterion and the intensity can only
improve as price falls, so the ladder is monotonic.

Grades — score gates, intensity grades (chosen 2026-10-15):
    score >= 5 (GO):  intensity >= 75 -> A+,  60-75 -> A,  < 60 -> B
    score == 4 (CAUTION) -> B,  score == 3 -> C,  <= 2 -> '-' (no trade)
Evidence: re-grading the GapFade rows of bounce_data.csv at the open, GO with
intensity 75+ had a median open->close of +18% vs +9% for GO under 60 (n=94).
Partly in-sample — the intensity weights were fit to these same trades.

Usage:
    from analyzers.bounce_ladder import build_ladder
    lad = build_ladder(static, cap, ticker='KORU')
    lad.levels          # one row per price where a criterion or the grade flips
    lad.next_level(10.65)
"""

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from analyzers.bounce_scorer import BouncePretrade, classify_from_setup_column
from scanners.premarket_bounce_scanner import metrics_from_price, setup_type_at_close

_DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
BOUNCE_CSV = _DATA_DIR / 'bounce_data.csv'
ODDS_FILE = _DATA_DIR / 'bounce_odds.json'

GRADES = ['-', 'C', 'B', 'A', 'A+']          # worst -> best
GRADE_RANK = {g: i for i, g in enumerate(GRADES)}
A_PLUS_INTENSITY = 75.0
A_INTENSITY = 60.0

CRITERIA = ['selloff_total_pct', 'pct_off_30d_high', 'gap_pct',
            'prior_day_range_atr', 'pct_change_3', 'pct_off_52wk_high']


def grade_for(score: int, intensity: Optional[float]) -> str:
    if score >= 5:
        if intensity is not None and intensity >= A_PLUS_INTENSITY:
            return 'A+'
        if intensity is not None and intensity >= A_INTENSITY:
            return 'A'
        return 'B'
    if score == 4:
        return 'B'
    if score == 3:
        return 'C'
    return '-'


def intensity_for(metrics: Dict, setup_type: str) -> Optional[float]:
    """0-100 composite vs the setup's own reference trades (as priority_report does)."""
    try:
        from scanners.bounce_trader import compute_bounce_intensity, _bounce_df_weak, _bounce_df_strong
    except Exception:  # pragma: no cover
        return None
    ref = _bounce_df_weak if setup_type == 'GapFade_weakstock' else _bounce_df_strong
    try:
        return compute_bounce_intensity(metrics, ref_df=ref)['composite']
    except Exception:
        return None


_CHECKER = BouncePretrade()


def score_at(static: dict, cap: str, price: float, ticker: str = '') -> dict:
    """Checklist score, intensity and grade at one price, setup type fixed at the prior close."""
    m = metrics_from_price(static, price)
    res = _CHECKER.validate(ticker, m, cap=cap, force_setup=static.get('setup_type'))
    inten = intensity_for(m, res.setup_type)
    return {
        'price': price,
        'gap_pct': m['gap_pct'],
        'score': res.score,
        'rec': res.recommendation,
        'intensity': inten,
        'grade': grade_for(res.score, inten),
        'passed': {i.name: i.passed for i in res.items},
        'descriptions': {i.name: i.description for i in res.items},
        'failing': [i.name for i in res.items if not i.passed],
        'setup_type': res.setup_type,
        'metrics': m,
        'result': res,
    }


def _tick(price: float) -> float:
    return 0.01 if price >= 1 else 0.0001


def _bisect(pred, lo: float, hi: float) -> float:
    """Highest price in [lo, hi] where pred holds; pred(lo) True, pred(hi) False, monotone."""
    tick = _tick(lo)
    while hi - lo > tick / 2:
        mid = (lo + hi) / 2
        if pred(mid):
            lo = mid
        else:
            hi = mid
    snapped = math.floor(lo / tick) * tick
    return round(snapped if pred(snapped) else lo, 4)


@dataclass
class Ladder:
    ticker: str
    cap: str
    setup_type: str
    static: dict
    points: pd.DataFrame                       # one row per grid price, high -> low
    levels: List[dict] = field(default_factory=list)   # flips, high -> low
    ceiling: dict = field(default_factory=dict)

    def at(self, price: float) -> dict:
        return score_at(self.static, self.cap, price, self.ticker)

    def next_level(self, price: float) -> Optional[dict]:
        """The first level BELOW price whose grade beats the grade at price."""
        here = GRADE_RANK[self.at(price)['grade']]
        for lv in self.levels:
            if lv['price'] < price and GRADE_RANK[lv['grade']] > here:
                return {**lv, 'pct_away': lv['price'] / price - 1}
        return None


def build_ladder(static: dict, cap: str, ticker: str = '', floor: float = -0.40,
                 step: float = 0.0025, top: float = 0.02, deep_floor: float = -0.60) -> Ladder:
    """Score every price from `top` above the prior close down to `floor` (as gap
    fractions) and find each price where a criterion or the grade flips.

    The grid extends to `deep_floor` when the best grade is not reached by `floor`.
    """
    static = dict(static)
    static.setdefault('setup_type', setup_type_at_close(static))
    pc = static['prior_close']

    def run(g_hi, g_lo):
        n = int(round((g_hi - g_lo) / step))
        return [score_at(static, cap, pc * (1 + g_hi - k * step), ticker) for k in range(n + 1)]

    rows = run(top, floor)
    if rows[-1]['grade'] != 'A+' and deep_floor < floor:
        rows += run(floor - step, deep_floor)

    points = pd.DataFrame([{
        'price': r['price'], 'gap_pct': r['gap_pct'], 'score': r['score'], 'rec': r['rec'],
        'intensity': r['intensity'], 'grade': r['grade'], 'failing': ', '.join(r['failing']),
        **{f'pass_{c}': r['passed'].get(c) for c in CRITERIA},
    } for r in rows])

    descriptions = rows[0]['descriptions']
    levels: Dict[float, dict] = {}
    for hi_r, lo_r in zip(rows, rows[1:]):
        events = []
        for c in CRITERIA:
            if not hi_r['passed'][c] and lo_r['passed'][c]:
                p = _bisect(lambda x, c=c: score_at(static, cap, x, ticker)['passed'][c],
                            lo_r['price'], hi_r['price'])
                events.append((p, f"{descriptions[c]} passes"))
        if GRADE_RANK[lo_r['grade']] > GRADE_RANK[hi_r['grade']]:
            target = GRADE_RANK[lo_r['grade']]
            p = _bisect(lambda x: GRADE_RANK[score_at(static, cap, x, ticker)['grade']] >= target,
                        lo_r['price'], hi_r['price'])
            events.append((p, f"grade {lo_r['grade']}"))
        for p, ev in events:
            levels.setdefault(p, []).append(ev)

    level_rows = []
    for p in sorted(levels, reverse=True):
        r = score_at(static, cap, p, ticker)
        level_rows.append({'price': p, 'gap_pct': r['gap_pct'], 'events': levels[p],
                           'score': r['score'], 'grade': r['grade'], 'intensity': r['intensity'],
                           'rec': r['rec']})

    best = rows[-1]
    ceiling = {'score': best['score'], 'grade': best['grade'], 'blocked': best['failing'],
               'blocked_desc': [descriptions[c] for c in best['failing']],
               'floor_price': best['price'], 'floor_gap': best['gap_pct']}
    return Ladder(ticker=ticker, cap=cap, setup_type=static['setup_type'], static=static,
                  points=points, levels=level_rows, ceiling=ceiling)


def grade_zones(lad: 'Ladder') -> list:
    """[(grade, top, bottom)] high -> low, split at the exact grade-flip prices."""
    pts = lad.points
    edges = [(lv['price'], lv['grade']) for lv in lad.levels if any(e.startswith('grade') for e in lv['events'])]
    zones, top, grade = [], float(pts['price'].iloc[0]), pts['grade'].iloc[0]
    for price, g in edges:
        zones.append((grade, top, price))
        top, grade = price, g
    zones.append((grade, top, float(pts['price'].iloc[-1])))
    return zones


# ---------------------------------------------------------------------------
# What each grade has meant historically
# ---------------------------------------------------------------------------

def grade_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Re-grade historical bounce rows at their open (metrics in bounce_data.csv are as-of the open)."""
    out = []
    for _, row in df.iterrows():
        setup_name = row.get('Setup')
        if isinstance(setup_name, str) and 'IntradayCapitch' in setup_name:
            continue
        m = row.to_dict()
        m['prior_day_range_atr'] = m.get('one_day_before_range_pct')
        setup = classify_from_setup_column(setup_name)
        cap = row.get('cap') if isinstance(row.get('cap'), str) else 'Medium'
        res = _CHECKER.validate(str(row.get('ticker', '')), m, force_setup=setup, cap=cap)
        inten = intensity_for(m, setup)
        out.append({'ticker': row.get('ticker'), 'date': row.get('date'), 'score': res.score,
                    'intensity': inten, 'grade': grade_for(res.score, inten),
                    'trade_grade': row.get('trade_grade'),
                    'open_high': row.get('bounce_open_high_pct'),
                    'open_close': row.get('bounce_open_close_pct'),
                    'open_low': row.get('bounce_open_low_pct')})
    return pd.DataFrame(out)


def summarize_grades(graded: pd.DataFrame) -> Dict[str, dict]:
    stats = {}
    for g in GRADES:
        sub = graded[graded['grade'] == g]
        if sub.empty:
            continue
        oc = sub['open_close'].dropna()
        stats[g] = {'n': int(len(sub)),
                    'med_open_high': float(sub['open_high'].median()),
                    'med_open_close': float(oc.median()) if len(oc) else None,
                    'med_open_low': float(sub['open_low'].median()),
                    'pct_green': float((oc > 0).mean()) if len(oc) else None}
    return stats


@lru_cache(maxsize=1)
def historical_grade_stats() -> Dict[str, dict]:
    """Per grade: n, median open->high/close/low and % green across your bounce_data.csv trades."""
    try:
        df = pd.read_csv(BOUNCE_CSV).dropna(subset=['ticker', 'date'])
    except Exception:
        return {}
    return summarize_grades(grade_trades(df))


# ---------------------------------------------------------------------------
# Population odds (scripts/build_bounce_odds.py) — mirrors
# scripts/priority_report._lookup_bounce_odds without importing that module
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_odds() -> Optional[dict]:
    try:
        return json.loads(ODDS_FILE.read_text())
    except Exception:
        return None


def lookup_bounce_odds(score: int, cap: str, odds: Optional[dict] = None) -> Optional[dict]:
    """(score band, cap) bucket first, then the score band alone, then the population base rate."""
    odds = odds if odds is not None else _load_odds()
    if not odds:
        return None
    band_only = None
    for b in odds.get('buckets', []):
        if not (b['score_min'] <= score <= b['score_max']):
            continue
        if b.get('cap') == cap:
            return {**_odds_fields(b), 'bucket': f"score {b['score_min']}-{b['score_max']} · {cap}"}
        if b.get('cap') is None:
            band_only = b
    if band_only is not None:
        return {**_odds_fields(band_only), 'bucket': f"score {band_only['score_min']}-{band_only['score_max']}"}
    base = odds.get('base')
    if base:
        return {**_odds_fields(base), 'bucket': 'all screened days'}
    return None


def _odds_fields(b: dict) -> dict:
    return {k: b.get(k) for k in ('n', 'p_green', 'p_bounce5', 'med_return')}
