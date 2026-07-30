"""
PROBE: Decision-node reconstruction for A-grade 3DGapFade trades.

Purpose (read-only feasibility test, NOT the matrix itself):
  - Replay each A-grade 3DGapFade trade minute-by-minute from Polygon.
  - At fixed decision nodes (every 15 min of RTH), reconstruct the *state*
    variables a live watcher would see, plus the *forward* outcome from that node.
  - Answer one question: is the intraday path rich enough to support a
    state -> action-EV matrix? Show the shape of a single decision-node record.

Anchor: a 3DGapFade is shorted into the open. We treat the 09:30 RTH open as
the entry reference. For a short, profit = -(ret_from_open).
"""
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_queries.polygon_queries import get_intraday

CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'reversal_data.csv')

# Decision nodes: every 15 min from open through the close
NODE_TIMES = [(9,30),(9,45),(10,0),(10,30),(11,0),(11,30),(12,0),(13,0),(14,0),(15,0),(15,45)]


def load_a_grade():
    df = pd.read_csv(CSV, encoding='utf-8-sig')
    df = df[(df['setup'] == '3DGapFade') & (df['trade_grade'] == 'A')].copy()
    df['date_iso'] = pd.to_datetime(df['date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')
    return df[['date_iso','ticker','cap','gap_pct','pct_from_9ema','reversal_open_low_pct']]


def rth(df):
    """Filter a minute-bar frame to regular trading hours 09:30-16:00 ET."""
    t = df.index.time
    from datetime import time as _t
    return df[(t >= _t(9,30)) & (t < _t(16,0))]


def build_nodes(ticker, date_iso):
    bars = get_intraday(ticker, date_iso, 1, 'minute')
    if bars is None or bars.empty:
        return None, "no bars"
    full = bars.copy()
    sess = rth(full)
    if sess.empty:
        return None, "no RTH bars"
    # session-cumulative VWAP from RTH bars
    tp = (sess['high'] + sess['low'] + sess['close']) / 3
    sess = sess.assign(cumvol=sess['volume'].cumsum(),
                       cumtpv=(tp * sess['volume']).cumsum())
    sess['vwap_sess'] = sess['cumtpv'] / sess['cumvol']

    open_px = sess['open'].iloc[0]
    pm = full[full.index.time < pd.Timestamp('09:30').time()]
    pm_high = pm['high'].max() if not pm.empty else np.nan
    day_low = sess['low'].min()

    rows = []
    for (hh, mm) in NODE_TIMES:
        upto = sess[sess.index.time <= pd.Timestamp(f'{hh:02d}:{mm:02d}').time()]
        if upto.empty:
            continue
        bar = upto.iloc[-1]
        px = bar['close']
        run_high = upto['high'].max()
        run_low = upto['low'].min()
        # forward window: everything strictly after this node
        fwd = sess[sess.index.time > pd.Timestamp(f'{hh:02d}:{mm:02d}').time()]
        fwd_low = fwd['low'].min() if not fwd.empty else px
        fwd_30 = sess[sess.index.time <= pd.Timestamp(f'{(hh + (mm+30)//60):02d}:{(mm+30)%60:02d}').time()]
        fwd_30_px = fwd_30['close'].iloc[-1] if not fwd_30.empty else px

        rows.append(dict(
            node=f'{hh:02d}:{mm:02d}',
            px=round(px, 2),
            # ---- STATE a live watcher sees at this node ----
            short_pnl_pct=round(-(px/open_px - 1)*100, 2),       # running short P&L from open
            below_hod_pct=round((px/run_high - 1)*100, 2),       # how far below session high (reversal progress)
            vs_vwap_pct=round((px/bar['vwap_sess'] - 1)*100, 2),  # below VWAP = short in control
            made_new_hod=bool(run_high > sess['open'].iloc[0] and upto['high'].idxmax() != upto.index[0]),
            mae_pct=round((run_high/open_px - 1)*100, 2),        # worst squeeze vs open so far
            # ---- FORWARD outcome from this node (the label the matrix needs) ----
            fwd_to_low_pct=round(-(fwd_low/px - 1)*100, 2),      # add'l short profit still available
            fwd_30min_pct=round(-(fwd_30_px/px - 1)*100, 2),     # short P&L change over next 30m
        ))
    meta = dict(open_px=round(open_px,2), pm_high=round(pm_high,2) if pd.notna(pm_high) else None,
                day_low=round(day_low,2), full_to_low_pct=round(-(day_low/open_px-1)*100,2),
                n_rth_bars=len(sess))
    return (pd.DataFrame(rows), meta), None


def main():
    a = load_a_grade()
    print(f"A-grade 3DGapFade trades: {len(a)}\n")
    detail_for = {('2020-02-27','MRNA'), ('2024-02-16','SMCI'), ('2024-03-08','NVDA')}
    summary = []
    for _, r in a.iterrows():
        tk, d = r['ticker'], r['date_iso']
        res, err = build_nodes(tk, d)
        if err:
            summary.append(dict(date=d, ticker=tk, cap=r['cap'], status=err, n_bars=0))
            continue
        nodes, meta = res
        summary.append(dict(date=d, ticker=tk, cap=r['cap'], status='OK',
                            n_bars=meta['n_rth_bars'], day_to_low=meta['full_to_low_pct'],
                            csv_open_low=round(r['reversal_open_low_pct']*100,1)))
        if (d, tk) in detail_for:
            print("="*100)
            print(f"DECISION-NODE TABLE  {tk}  {d}  cap={r['cap']}  "
                  f"open={meta['open_px']}  pm_high={meta['pm_high']}  day_low={meta['day_low']}  "
                  f"open->low={meta['full_to_low_pct']}%")
            print("-"*100)
            print(nodes.to_string(index=False))
            print()

    print("="*100)
    print("COVERAGE SUMMARY (all A-grade trades)")
    print("-"*100)
    s = pd.DataFrame(summary)
    print(s.to_string(index=False))
    ok = s[s['status']=='OK']
    print(f"\nBars reconstructed: {len(ok)}/{len(s)}   "
          f"median RTH bars: {ok['n_bars'].median() if len(ok) else 0:.0f}")


if __name__ == '__main__':
    main()
