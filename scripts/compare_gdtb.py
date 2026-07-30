"""Compare today's GapDownTrendBreak trades (MU/SNDK/SOXL) vs historical GDTB baseline.

Fetches fresh metrics from Polygon for today's trades since reversal_data.csv
rows have empty enrichment columns.
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np

from data_queries.polygon_queries import (
    get_atr, get_daily, get_ticker_mavs_open, get_levels_data
)

DATE = '2026-05-12'
TODAY_TICKERS = ['MU', 'SNDK', 'SOXL']

# ---------- Historical baseline ----------
df = pd.read_csv('data/reversal_data.csv')
hist = df[(df['setup'] == 'GapDownTrendBreak') & (df['date'] != '5/12/2026')].copy()

KEY_COLS = ['atr_pct', 'gap_pct', 'pct_from_9ema', 'pct_change_3',
            'pct_from_50mav', 'pct_from_200mav', 'one_day_before_range_pct']

baseline = {}
for c in KEY_COLS:
    v = pd.to_numeric(hist[c], errors='coerce').dropna()
    baseline[c] = {
        'median': float(v.median()),
        'p25': float(v.quantile(0.25)),
        'p75': float(v.quantile(0.75)),
        'mean': float(v.mean()),
        'std': float(v.std()),
    }

gradeA = hist[hist['trade_grade'] == 'A']
baseline_A = {}
for c in KEY_COLS:
    v = pd.to_numeric(gradeA[c], errors='coerce').dropna()
    if len(v) > 0:
        baseline_A[c] = {'median': float(v.median()), 'min': float(v.min()), 'max': float(v.max())}

# ---------- Fetch live metrics for today ----------
def safe(fn, label):
    try:
        return fn()
    except Exception as e:
        print(f"    [{label} failed: {type(e).__name__}: {e}]")
        return None

def fetch_metrics(ticker, date):
    out = {}
    d = safe(lambda: get_daily(ticker, date), 'get_daily')
    out['_daily_obj'] = d
    out['open'] = getattr(d, 'open', None) if d else None
    out['high'] = getattr(d, 'high', None) if d else None
    out['low'] = getattr(d, 'low', None) if d else None
    out['close'] = getattr(d, 'close', None) if d else None

    prior = safe(lambda: get_levels_data(ticker, date, 5, 1, 'day'), 'get_levels_data(5d)')
    if prior is not None and not prior.empty:
        idx = -2 if len(prior) >= 2 else -1
        out['prior_close'] = float(prior['close'].iloc[idx])
        out['prior_high'] = float(prior['high'].iloc[idx])
        out['prior_low'] = float(prior['low'].iloc[idx])

    if out.get('open') is not None and out.get('prior_close') is not None:
        out['gap_pct'] = (out['open'] - out['prior_close']) / out['prior_close']
    if out.get('prior_high') is not None and out.get('prior_low') is not None:
        out['one_day_before_range_pct'] = (out['prior_high'] - out['prior_low']) / out['prior_low']

    atr = safe(lambda: get_atr(ticker, date), 'get_atr')
    if atr is not None and out.get('open'):
        out['atr_pct'] = float(atr) / out['open']

    bars = safe(lambda: get_levels_data(ticker, date, 10, 1, 'day'), 'get_levels_data(10d)')
    if bars is not None and len(bars) >= 4:
        try:
            out['pct_change_3'] = float((bars['close'].iloc[-1] - bars['close'].iloc[-4]) / bars['close'].iloc[-4])
        except Exception:
            pass

    mavs = safe(lambda: get_ticker_mavs_open(ticker, date), 'get_ticker_mavs_open')
    if mavs:
        for k in ('pct_from_9ema', 'pct_from_50mav', 'pct_from_200mav'):
            if k in mavs:
                out[k] = mavs[k]

    if out.get('open') and out.get('close') is not None:
        out['reversal_open_close_pct'] = (out['close'] - out['open']) / out['open']
    if out.get('open') and out.get('low') is not None:
        out['reversal_open_low_pct'] = (out['low'] - out['open']) / out['open']
    return out

today_metrics = {}
for t in TODAY_TICKERS:
    print(f"Fetching {t} ...")
    today_metrics[t] = fetch_metrics(t, DATE)

# ---------- Report ----------
print("\n" + "=" * 80)
print(f"GapDownTrendBreak baseline: {len(hist)} historical trades")
print(f"  Grades: A={(hist['trade_grade']=='A').sum()}  B={(hist['trade_grade']=='B').sum()}  C={(hist['trade_grade']=='C').sum()}")
print(f"  Historical win rate (open->close < 0): {(pd.to_numeric(hist['reversal_open_close_pct'], errors='coerce') < 0).sum() / len(hist) * 100:.0f}%")
print(f"  Historical median open->low: {pd.to_numeric(hist['reversal_open_low_pct'], errors='coerce').median():+.2%}")
print("=" * 80)

def pct_rank_in_baseline(val, baseline_arr):
    arr = np.sort(np.asarray(baseline_arr))
    if len(arr) == 0 or val is None or pd.isna(val):
        return None
    return float(np.searchsorted(arr, val) / len(arr) * 100)

print(f"\n{'Metric':<30} {'GDTB med':>10} {'GDTB Q1-Q3':>17} {'Grade A med':>13} {'Grade A range':>22}")
print("-" * 100)
for c in KEY_COLS:
    b = baseline[c]
    a = baseline_A.get(c)
    a_med = f"{a['median']:+.4f}" if a else "  --"
    a_rng = f"[{a['min']:+.3f}, {a['max']:+.3f}]" if a else "  --"
    print(f"{c:<30} {b['median']:+10.4f}  [{b['p25']:+.3f}, {b['p75']:+.3f}]  {a_med:>13}  {a_rng:>22}")

print("\n" + "=" * 80)
print("TODAY (2026-05-12)")
print("=" * 80)

for t in TODAY_TICKERS:
    m = today_metrics[t]
    print(f"\n--- {t} ---")
    print(f"  OHLC:        open={m.get('open')!s:<10} high={m.get('high')!s:<10} low={m.get('low')!s:<10} close={m.get('close')!s:<10}")
    print(f"  Prior close: {m.get('prior_close')}")
    print(f"\n  {'Metric':<30} {'Value':>12}  {'vs GDTB median':>20}  {'In Grade A range?':>22}")
    print(f"  " + "-" * 88)
    for c in KEY_COLS:
        v = m.get(c)
        if v is None:
            print(f"  {c:<30} {'(missing)':>12}")
            continue
        b = baseline[c]
        delta = v - b['median']
        a = baseline_A.get(c)
        in_a = "  --"
        if a:
            in_a = "YES" if (a['min'] <= v <= a['max']) else f"NO ({a['min']:+.3f}..{a['max']:+.3f})"
        delta_pct = "(>>)" if v > b['p75'] else ("(<<)" if v < b['p25'] else "(within Q1-Q3)")
        print(f"  {c:<30} {v:>+12.4f}  {delta:+10.4f} {delta_pct:<11} {in_a:>22}")

    # Day result so far
    if 'reversal_open_close_pct' in m:
        print(f"\n  Day result so far: open->close={m['reversal_open_close_pct']:+.2%}  open->low={m['reversal_open_low_pct']:+.2%}")
        hist_med_open_low = pd.to_numeric(hist['reversal_open_low_pct'], errors='coerce').median()
        print(f"  vs GDTB historical median open->low: {hist_med_open_low:+.2%}")
