"""Pre-market analysis: combine yesterday's daily-bar context with today's PM
intraday data, project forward-looking metrics, and rank the complex.

Run:
    venv/Scripts/python.exe scripts/score_pm_complex.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

import pandas as pd
from data_queries.polygon_queries import get_intraday, get_levels_data
from analyzers.reversal_scorer import compute_reversal_intensity, ReversalScorer

scorer = ReversalScorer()

DATE = '2026-05-06'
PRIOR = '2026-05-05'


def analyze(ticker, cap):
    levels = get_levels_data(ticker, PRIOR, 310, 1, 'day')
    if levels is None or levels.empty:
        print(f'{ticker}: no daily data'); return None
    hist = levels.copy()
    closes = hist['close']

    intra = get_intraday(ticker, DATE, 1, 'minute')
    if intra is None or intra.empty:
        print(f'{ticker}: no PM data'); return None

    pm_open = float(intra.iloc[0]['open'])
    pm_last = float(intra.iloc[-1]['close'])
    pm_high = float(intra['high'].max())
    pm_low = float(intra['low'].min())
    pm_vol = int(intra['volume'].sum())
    pm_last_ts = intra.index[-1].strftime('%H:%M %Z')

    prior_close = float(hist.iloc[-1]['close'])

    cp = pm_last  # treat PM last as current price

    ema9 = closes.ewm(span=9, adjust=False).mean().iloc[-1]
    sma50 = closes.rolling(50).mean().iloc[-1]
    sma200 = closes.rolling(200).mean().iloc[-1] if len(closes) >= 200 else None

    pf9 = (cp - ema9) / ema9
    pf50 = (cp - sma50) / sma50
    pf200 = (cp - sma200) / sma200 if sma200 is not None else None

    hl = hist['high'] - hist['low']
    hpc = abs(hist['high'] - hist['close'].shift(1))
    lpc = abs(hist['low'] - hist['close'].shift(1))
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=1).mean().iloc[-1]
    atr_pct = atr / cp

    prior_range = hist.iloc[-1]['high'] - hist.iloc[-1]['low']
    prior_range_atr = prior_range / atr

    adv20 = hist['volume'].rolling(20, min_periods=1).mean().iloc[-1]
    prior_vol = hist.iloc[-1]['volume']
    rvol_prior = prior_vol / adv20
    pm_vol_pct = pm_vol / adv20 * 100

    pc = {}
    for n in [3, 15, 30, 90, 120]:
        if len(hist) >= n:
            old = hist.iloc[-n]['close']
            pc[f'pct_change_{n}'] = (cp - old) / old

    gap_pct = (pm_open - prior_close) / prior_close

    high_full = max(hist['high'].max(), pm_high)
    pct_from_52 = (cp - high_full) / high_full
    breaks_52 = pm_high >= hist['high'].max()

    atr_dist_9 = (cp - ema9) / atr
    atr_dist_50 = (cp - sma50) / atr
    atr_dist_200 = (cp - sma200) / atr if sma200 is not None else None

    metrics = {
        'pct_from_9ema': pf9,
        'pct_from_50mav': pf50,
        'pct_from_200mav': pf200,
        'atr_pct': atr_pct,
        'prior_day_range_atr': prior_range_atr,
        'rvol_score': rvol_prior,
        'gap_pct': gap_pct,
        'pct_change_3': pc.get('pct_change_3'),
        'pct_change_15': pc.get('pct_change_15'),
        'pct_change_30': pc.get('pct_change_30'),
        'pct_change_90': pc.get('pct_change_90'),
        'pct_change_120': pc.get('pct_change_120'),
        'breaks_fifty_two_wk': breaks_52,
        'reversal_open_close_pct': 0,
    }

    result = scorer.score_setup(ticker, DATE, cap, metrics)
    intensity = compute_reversal_intensity(metrics, cap=cap)

    print('=' * 78)
    print(f'{ticker} ({cap}) -- {DATE} {pm_last_ts} (PM data)')
    print('=' * 78)
    print(f'  Prior close (5/5):          ${prior_close:,.2f}')
    print(f'  PM open:                    ${pm_open:,.2f}   gap = {gap_pct*100:+.2f}%')
    print(f'  PM high / low / last:       ${pm_high:,.2f} / ${pm_low:,.2f} / ${pm_last:,.2f}')
    print(f'  PM move from prior close:   {(pm_last/prior_close-1)*100:+.2f}%')
    print(f'  PM fade from PM high:       {(pm_last/pm_high-1)*100:+.2f}%')
    print(f'  PM vol:                     {pm_vol:>14,}  ({pm_vol_pct:.1f}% of 20d ADV)')
    print()
    print('  Percentile ladder (PM last as current):')
    for n in [3, 15, 30, 90, 120]:
        v = pc.get(f'pct_change_{n}')
        if v is not None:
            print(f'    pct_change_{n:<3}:    {v*100:+.2f}%')
    print()
    print(f'  ATR (14d): ${atr:.2f} ({atr_pct*100:.2f}%)')
    print(f'  prior_day_range_atr: {prior_range_atr:.2f}x   prior_day_RVOL: {rvol_prior:.2f}x')
    print()
    print(f'  9EMA  ~${ema9:,.2f}   ({pf9*100:+.2f}%, {atr_dist_9:+.2f} ATR above)')
    print(f'  50MA  ~${sma50:,.2f}   ({pf50*100:+.2f}%, {atr_dist_50:+.2f} ATR above)')
    if sma200 is not None:
        print(f'  200MA ~${sma200:,.2f}   ({pf200*100:+.2f}%, {atr_dist_200:+.2f} ATR above)')
    print()
    print(f'  pct_from_52wk_high (incl PM): {pct_from_52*100:+.2f}%   breaks_52wk_high: {breaks_52}')
    print()
    print(f'  PROJECTED Parabolic Score: {result["pretrade_score"]}/{result["pretrade_max"]} '
          f'{result["pretrade_grade"]} -> {result["pretrade_recommendation"]}')
    for crit in ['pct_from_9ema', 'prior_day_range_atr', 'rvol_score', 'pct_change_3', 'gap_pct']:
        d = result['criteria_details'][crit]
        passed = 'PASS' if d['passed'] else 'FAIL'
        actual = d['actual']
        actual_str = f'{actual:.4f}' if isinstance(actual, float) else str(actual)
        print(f'    [{passed}] {crit:22s}  actual={actual_str:10s}  thresh={d["threshold"]}')
    print()
    print(f'  PROJECTED Intensity: {intensity["composite"]}/100')
    for k, v in intensity['details'].items():
        print(f'    {k:22s}  pctile={v["pctile"]!s:>6s}')
    print()

    return {
        'ticker': ticker,
        'cap': cap,
        'gap_pct': gap_pct,
        'pm_open': pm_open,
        'pm_last': pm_last,
        'pm_high': pm_high,
        'pm_fade_from_high': (pm_last / pm_high - 1),
        'score': result['pretrade_score'],
        'recommendation': result['pretrade_recommendation'],
        'intensity': intensity['composite'],
        'rvol_prior': rvol_prior,
        'atr_dist_50': atr_dist_50,
        'atr_dist_200': atr_dist_200,
        'pct_change_3': pc.get('pct_change_3'),
        'pm_vol_pct_adv': pm_vol_pct,
        'breaks_52wk': breaks_52,
    }


results = []
for tk, cap in [('SOXL', 'ETF'), ('MU', 'Large'), ('INTC', 'Large'), ('SNDK', 'Large')]:
    r = analyze(tk, cap)
    if r:
        results.append(r)

print('=' * 78)
print('SUMMARY')
print('=' * 78)
header = f'{"Ticker":<6} {"Cap":<5} {"Gap%":>7} {"PM_fade%":>10} {"PM_vol_%ADV":>12} {"Score":>5} {"Rec":>10} {"Intensity":>10} {"ATR>50":>7}'
print(header)
print('-' * len(header))
for r in sorted(results, key=lambda x: -(x['intensity'] or 0)):
    print(f'{r["ticker"]:<6} {r["cap"]:<5} {r["gap_pct"]*100:>+6.2f}% '
          f'{r["pm_fade_from_high"]*100:>+9.2f}% {r["pm_vol_pct_adv"]:>10.1f}% '
          f'{r["score"]}/5 {r["recommendation"]:>10} {r["intensity"]!s:>10} '
          f'{r["atr_dist_50"]:>+6.2f}')
