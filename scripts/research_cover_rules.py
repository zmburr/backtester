"""
Cover-rule research for gap fades (3DGapFade + 2DGapFade, 55 trades).

Question: what cover signal maximizes capture of the open->low move?
Score every rule identically: capture = (open - exit_px) / (open - day_low).
1.0 = covered the exact low; 0 = scratched at open price.

Rules tested (all enter short at 09:30 open, exit at rule trigger else 15:55):
  eod              hold to 15:55 (patience baseline)
  time_1200/1300   fixed-clock covers
  vwap_reclaim     cover when price reclaims session VWAP (the "too late?" rule)
  mom30_flip       cover when 30-min return turns positive after >=1 ATR of drop
  bounce_05atr/10atr   cover on bounce off session low >= 0.5 / 1.0 ATR
  retrace_15/25/35     cover when bounce retraces >=15/25/35% of drop-so-far
  stale_45/75      cover when no new session low for 45 / 75 min (after 10:30)
  flush_4atr       cover INTO weakness when drop-from-open >= 4 ATRs
  flush_abs35      cover INTO weakness when drop-from-open >= 35%
  volclimax        cover on 5x volume spike printing a new session low (>=2 ATR down)
  hybrid           crash-vs-grind: flush_4atr if it fires; else retrace_25 after 12:30; else EOD
  half_flush       SCALE-OUT: half into flush_4atr, half EOD (else all EOD)

Regime cuts: depth terciles (total drop in ATRs) and speed (frac of move done by 11:00).
Early-warning: does drop-by-10:30 predict final depth (can you classify crash vs grind live)?
"""
import os, sys
from datetime import time as _t
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_queries.polygon_queries import get_intraday

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETUPS = ['3DGapFade', '2DGapFade']


def load_trades():
    df = pd.read_csv(os.path.join(ROOT, 'data', 'reversal_data.csv'), encoding='utf-8-sig')
    df = df[df['setup'].isin(SETUPS)].copy()
    df['date_iso'] = pd.to_datetime(df['date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')
    df['atr_pct'] = pd.to_numeric(df['atr_pct'], errors='coerce').fillna(0.05).clip(lower=0.01)
    return df[['date_iso','ticker','cap','setup','trade_grade','atr_pct']]


def prep_session(tk, d):
    bars = get_intraday(tk, d, 1, 'minute')
    if bars is None or bars.empty:
        return None
    sess = bars[(bars.index.time >= _t(9,30)) & (bars.index.time < _t(16,0))].copy()
    if len(sess) < 60:
        return None
    tp = (sess['high'] + sess['low'] + sess['close']) / 3
    sess['vwap'] = (tp * sess['volume']).cumsum() / sess['volume'].cumsum()
    sess['run_low'] = sess['low'].cummin()
    sess['min_of_day'] = sess.index.hour * 60 + sess.index.minute
    return sess


def simulate(sess, atr):
    """Return dict rule -> (exit_minute_index, exit_px). Vector-ish scan, one pass."""
    o = sess['open'].iloc[0]
    px = sess['close'].values
    run_low = sess['run_low'].values
    vwap = sess['vwap'].values
    vol = sess['volume'].values
    mins = sess['min_of_day'].values
    n = len(sess)
    day_low = sess['low'].min()

    drop = 1 - px / o                              # + when short is winning
    bounce = px / run_low - 1                      # off session low
    denom = np.maximum(o - run_low, 1e-9)
    retrace = (px - run_low) / denom               # fraction of drop given back
    # minutes since last new session low
    new_low = sess['low'].values <= run_low + 1e-12
    last_nl = np.maximum.accumulate(np.where(new_low, np.arange(n), 0))
    since_low = np.arange(n) - last_nl
    # 30-min momentum
    ret30 = np.full(n, np.nan)
    ret30[30:] = px[30:] / px[:-30] - 1
    # volume spike vs rolling 30-min median
    vmed = pd.Series(vol).rolling(30, min_periods=10).median().values
    vspike = vol / np.maximum(vmed, 1)

    min_drop = max(atr, 0.02)                      # noise floor for bounce/retrace rules
    after = lambda hhmm: mins >= hhmm

    sigs = {
        'eod':          np.zeros(n, dtype=bool),
        'time_1200':    mins >= 720,
        'time_1300':    mins >= 780,
        'vwap_reclaim': (px > vwap) & after(600),
        'mom30_flip':   (ret30 > 0) & (drop >= atr) & after(600),
        'bounce_05atr': (bounce >= 0.5*atr) & (drop >= min_drop) & after(595),
        'bounce_10atr': (bounce >= 1.0*atr) & (drop >= min_drop) & after(595),
        'retrace_15':   (retrace >= 0.15) & (drop >= min_drop) & after(595),
        'retrace_25':   (retrace >= 0.25) & (drop >= min_drop) & after(595),
        'retrace_35':   (retrace >= 0.35) & (drop >= min_drop) & after(595),
        'stale_45':     (since_low >= 45) & (drop > 0) & after(630),
        'stale_75':     (since_low >= 75) & (drop > 0) & after(630),
        'flush_4atr':   drop >= 4*atr,
        'flush_abs35':  drop >= 0.35,
        'volclimax':    (vspike >= 5) & new_low & (drop >= 2*atr) & after(600),
        'hybrid':       np.zeros(n, dtype=bool),   # filled below
    }
    # hybrid: flush_4atr any time; else retrace_25 after 12:30
    sigs['hybrid'] = sigs['flush_4atr'] | ((retrace >= 0.25) & (drop >= min_drop) & after(750))

    out = {}
    cap = lambda p: (o - p) / max(o - day_low, 1e-9)
    for name, sig in sigs.items():
        idx = np.argmax(sig) if sig.any() else n - 1
        if not sig.any():
            idx = n - 1
        out[name] = dict(exit_min=int(mins[idx]), capture=cap(px[idx]),
                         bounce_at_exit=float(bounce[idx]))
    # scale-out: half into flush, half EOD
    f = out['flush_4atr']; e = out['eod']
    fired = sigs['flush_4atr'].any()
    out['half_flush'] = dict(exit_min=e['exit_min'],
                             capture=0.5*f['capture'] + 0.5*e['capture'] if fired else e['capture'],
                             bounce_at_exit=np.nan)
    # context for regime cuts
    by = lambda hhmm: (1 - sess.loc[mins <= hhmm, 'low'].min()/o) if (mins <= hhmm).any() else 0.0
    ctx = dict(total_drop=1 - day_low/o,
               drop_1030=by(630), drop_1100=by(660),
               frac_done_1100=by(660) / max(1 - day_low/o, 1e-9))
    return out, ctx


def main():
    trades = load_trades()
    print(f"gap fades in scope: {len(trades)} ({trades['setup'].value_counts().to_dict()})\n")
    rows = []
    for _, r in trades.iterrows():
        sess = prep_session(r['ticker'], r['date_iso'])
        if sess is None:
            print(f"  SKIP {r['ticker']} {r['date_iso']}")
            continue
        res, ctx = simulate(sess, r['atr_pct'])
        base = dict(ticker=r['ticker'], date=r['date_iso'], setup=r['setup'],
                    grade=r['trade_grade'], cap=r['cap'], atr_pct=r['atr_pct'], **ctx)
        for rule, v in res.items():
            rows.append(dict(**base, rule=rule, **v))
        print(f"  OK {r['ticker']:>6} {r['date_iso']} drop={ctx['total_drop']*100:5.1f}%  "
              f"done_by_11={ctx['frac_done_1100']*100:4.0f}%")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, 'data', 'cover_rule_results.csv'), index=False)
    print(f"\nwrote data/cover_rule_results.csv ({len(df)} rows)")

    pd.set_option('display.width', 200)
    t = df.pivot_table(index='rule', values=['capture','exit_min','bounce_at_exit'],
                       aggfunc={'capture':['mean','median',lambda s: s.quantile(0.25)],
                                'exit_min':'mean','bounce_at_exit':'mean'})
    t.columns = ['bounce@exit','cap_p25','cap_mean','cap_median','exit_min']
    t['exit_time'] = (t['exit_min']//60).astype(int).astype(str) + ':' + (t['exit_min']%60).astype(int).astype(str).str.zfill(2)
    t = t.drop(columns='exit_min').sort_values('cap_mean', ascending=False)
    print("\n=== RULE LEADERBOARD (capture of open->low move; all 55 fades) ===")
    print((t[['cap_mean','cap_median','cap_p25','bounce@exit','exit_time']]*1).round(3).to_string())

    # by setup
    print("\n=== mean capture by setup ===")
    print(df.pivot_table(index='rule', columns='setup', values='capture', aggfunc='mean').round(3)
            .sort_values('3DGapFade', ascending=False).to_string())

    # regime cuts
    one = df[df['rule']=='eod'][['ticker','date','total_drop','frac_done_1100','atr_pct']].copy()
    one['depth_atr'] = one['total_drop'] / one['atr_pct']
    one['depth_bin'] = pd.qcut(one['depth_atr'], 3, labels=['shallow','mid','deep'])
    one['speed_bin'] = np.where(one['frac_done_1100'] >= 0.75, 'front-loaded', 'grind')
    df2 = df.merge(one[['ticker','date','depth_bin','speed_bin','depth_atr']], on=['ticker','date'])
    print("\n=== mean capture: rule x depth tercile (drop in ATRs) ===")
    print(df2.pivot_table(index='rule', columns='depth_bin', values='capture', aggfunc='mean', observed=True).round(3)
             .sort_values('deep', ascending=False).to_string())
    print("\n=== mean capture: rule x speed ===")
    print(df2.pivot_table(index='rule', columns='speed_bin', values='capture', aggfunc='mean').round(3)
             .sort_values('front-loaded', ascending=False).to_string())
    print("\nregime sizes:", one.groupby(['depth_bin','speed_bin'], observed=True).size().to_dict())

    # early warning: does 10:30 drop predict final depth?
    one['drop_1030_atr'] = df[df['rule']=='eod'].set_index(['ticker','date']).loc[
        list(zip(one['ticker'], one['date'])), 'drop_1030'].values / one['atr_pct']
    c = one[['drop_1030_atr','depth_atr']].corr().iloc[0,1]
    print(f"\n=== EARLY WARNING: corr(drop-by-10:30 in ATRs, final depth in ATRs) = {c:.2f} ===")
    one['early_bin'] = pd.qcut(one['drop_1030_atr'], 3, labels=['slow start','medium','crashing by 10:30'])
    print(one.groupby('early_bin', observed=True)[['depth_atr','total_drop','frac_done_1100']].mean().round(2).to_string())


if __name__ == '__main__':
    main()
