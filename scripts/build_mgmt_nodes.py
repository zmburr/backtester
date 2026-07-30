"""
Management-node dataset builder for parabolic-short reversals.

Scope: 3DGapFade + 2DGapFade + GapDownTrendBreak (~75 curated trades, all grades).
Anchor: 09:30 RTH open; short P&L = -(price/open - 1). Matches the open->low
convention of the CSV's reversal_open_*_pct columns.

Answers (conditional on a REAL, mostly-winning fade — survivorship by design):
  - when to ADD / HOLD : nodes where forward juice (fwd_to_low) is still large
                         and the short is in control (below VWAP, deepening below HOD)
  - when to TAKE OFF   : nodes where fwd_to_low has collapsed / VWAP is being reclaimed
  - when to HOLD OVERNIGHT : per-trade overnight block (gap + next-day continuation)

Outputs:
  data/mgmt_nodes.csv          long: one row per (trade, node) — state + intraday forward labels
  data/mgmt_trade_summary.csv  per-trade outcome + overnight block
  reports/mgmt_matrix.html     aggregation tables (the first-cut matrix) + legend

NOT a risk model: this set has ~no losers, so mae/stop/blowup stats here are
winner-only and optimistic. Stops & blowup rate require a set that includes failures.
"""
import os, sys
from datetime import time as _t
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_queries.polygon_queries import get_intraday, get_daily, adjust_date_forward

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, 'data', 'reversal_data.csv')
SETUPS = ['3DGapFade', '2DGapFade', 'GapDownTrendBreak']
NODE_TIMES = [(9,30),(9,45),(10,0),(10,30),(11,0),(11,30),(12,0),(13,0),(14,0),(15,0),(15,45)]


def load_trades():
    df = pd.read_csv(CSV, encoding='utf-8-sig')
    df = df[df['setup'].isin(SETUPS)].copy()
    df['date_iso'] = pd.to_datetime(df['date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')
    return df[['date_iso','ticker','cap','setup','trade_grade','reversal_open_low_pct',
               'reversal_open_to_day_after_open_pct']]


def _at_or_before(series, hh, mm):
    sub = series[series.index.time <= _t(hh, mm)]
    return sub.iloc[-1] if len(sub) else np.nan


def build_trade(tk, date_iso):
    bars = get_intraday(tk, date_iso, 1, 'minute')
    if bars is None or bars.empty:
        return None, None, "no bars"
    full = bars.copy()
    sess = full[(full.index.time >= _t(9,30)) & (full.index.time < _t(16,0))].copy()
    if sess.empty:
        return None, None, "no RTH bars"

    tp = (sess['high'] + sess['low'] + sess['close']) / 3
    sess['vwap'] = (tp * sess['volume']).cumsum() / sess['volume'].cumsum()
    vwap_s = sess['vwap']

    open_px = sess['open'].iloc[0]
    close_px = sess['close'].iloc[-1]
    pm = full[full.index.time < _t(9,30)]
    pm_high = pm['high'].max() if not pm.empty else np.nan
    day_low = sess['low'].min()

    rows = []
    for (hh, mm) in NODE_TIMES:
        upto = sess[sess.index.time <= _t(hh, mm)]
        if upto.empty:
            continue
        px = upto['close'].iloc[-1]
        run_high = upto['high'].max()
        vwap_now = _at_or_before(vwap_s, hh, mm)
        # vwap slope over prior 15 min
        prev_min = (hh*60 + mm) - 15
        vwap_prev = _at_or_before(vwap_s, prev_min // 60, prev_min % 60)
        fwd = sess[sess.index.time > _t(hh, mm)]
        fwd_low = fwd['low'].min() if not fwd.empty else px
        f30 = (hh*60 + mm) + 30
        fwd_30 = sess[sess.index.time <= _t(f30 // 60, f30 % 60)]
        fwd_30_px = fwd_30['close'].iloc[-1] if not fwd_30.empty else px

        rows.append(dict(
            ticker=tk, date=date_iso, node=f'{hh:02d}:{mm:02d}', px=round(px, 2),
            # ---- live-visible STATE ----
            short_pnl_pct=round(-(px/open_px - 1)*100, 2),
            below_hod_pct=round((px/run_high - 1)*100, 2),
            vs_vwap_pct=round((px/vwap_now - 1)*100, 2) if pd.notna(vwap_now) else np.nan,
            vwap_slope15=round((vwap_now/vwap_prev - 1)*100, 2) if pd.notna(vwap_prev) else np.nan,
            below_open=bool(px < open_px),
            made_new_hod=bool(run_high > open_px and upto['high'].idxmax() != upto.index[0]),
            mae_pct=round((run_high/open_px - 1)*100, 2),
            dist_to_pm_high_pct=round((px/pm_high - 1)*100, 2) if pd.notna(pm_high) else np.nan,
            # ---- forward LABELS ----
            fwd_to_low_pct=round(-(fwd_low/px - 1)*100, 2),
            fwd_30min_pct=round(-(fwd_30_px/px - 1)*100, 2),
            eod_close_pct=round(-(close_px/px - 1)*100, 2),
        ))

    # ---- overnight block (per trade) ----
    overnight = dict(overnight_gap_pct=np.nan, next_day_open_to_low_pct=np.nan, hold_cc_pct=np.nan)
    try:
        nxt = adjust_date_forward(date_iso, 1)
        d2 = get_daily(tk, nxt)
        if d2 is not None and getattr(d2, 'open', None):
            n_open, n_low, n_close = d2.open, d2.low, d2.close
            overnight = dict(
                overnight_gap_pct=round(-(n_open/close_px - 1)*100, 2),       # short P&L through the gap
                next_day_open_to_low_pct=round(-(n_low/n_open - 1)*100, 2),    # continuation available next day
                hold_cc_pct=round(-(n_close/close_px - 1)*100, 2),            # close-to-close overnight short P&L
            )
    except Exception as e:
        print(f"  overnight fetch failed {tk} {date_iso}: {e}")

    meta = dict(open_px=round(open_px,2), close_px=round(close_px,2), day_low=round(day_low,2),
                pm_high=round(pm_high,2) if pd.notna(pm_high) else None,
                open_to_low_pct=round(-(day_low/open_px-1)*100,2), n_rth_bars=len(sess), **overnight)
    return pd.DataFrame(rows), meta, None


def main():
    trades = load_trades()
    print(f"Trades in scope ({'+'.join(SETUPS)}): {len(trades)}")
    print(trades['setup'].value_counts().to_string(), "\n")

    node_frames, summary = [], []
    for _, r in trades.iterrows():
        tk, d = r['ticker'], r['date_iso']
        nodes, meta, err = build_trade(tk, d)
        if err:
            print(f"  SKIP {tk} {d}: {err}")
            summary.append(dict(ticker=tk, date=d, setup=r['setup'], grade=r['trade_grade'],
                                cap=r['cap'], status=err))
            continue
        nodes['setup'] = r['setup']; nodes['grade'] = r['trade_grade']; nodes['cap'] = r['cap']
        node_frames.append(nodes)
        summary.append(dict(ticker=tk, date=d, setup=r['setup'], grade=r['trade_grade'], cap=r['cap'],
                            status='OK', **meta,
                            csv_open_low=round(r['reversal_open_low_pct']*100,1) if pd.notna(r['reversal_open_low_pct']) else None))
        print(f"  OK  {tk:>6} {d} {r['setup']:<16} g={r['trade_grade']} "
              f"o->low={meta['open_to_low_pct']:>6}%  ON_gap={meta['overnight_gap_pct']}")

    nodes_all = pd.concat(node_frames, ignore_index=True)
    summ = pd.DataFrame(summary)
    nodes_all.to_csv(os.path.join(ROOT,'data','mgmt_nodes.csv'), index=False)
    summ.to_csv(os.path.join(ROOT,'data','mgmt_trade_summary.csv'), index=False)
    print(f"\nwrote data/mgmt_nodes.csv ({len(nodes_all)} node rows)")
    print(f"wrote data/mgmt_trade_summary.csv ({len(summ)} trades)")

    write_html(nodes_all, summ.query("status=='OK'"))


def write_html(nodes, summ):
    css = """<style>body{font-family:Segoe UI,Arial,sans-serif;background:#0e1116;color:#e6edf3;margin:24px}
    h1{font-size:20px}h2{color:#58a6ff;font-size:15px;margin-top:26px}
    table{border-collapse:collapse;font-size:12px;margin:6px 0 14px}th,td{border:1px solid #30363d;padding:3px 8px;text-align:right}
    th{background:#161b22;color:#8b949e}.l{color:#8b949e;font-size:12px;line-height:1.6}</style>"""
    H = [css, "<h1>Management Matrix v1 — parabolic-short reversals</h1>",
         "<div class='l'><b>Survivorship by design:</b> ~all winners. Valid for ADD / HOLD / TAKE-OFF / OVERNIGHT "
         "management of a working fade. NOT a stop/blowup model.<br>"
         "<b>fwd_to_low_pct</b> = short profit still available from that node. When it nears 0, the move is done (TAKE OFF). "
         "<b>vs_vwap_pct</b> &lt; 0 = short in control (HOLD/ADD); reclaim toward 0 = cover signal.</div>"]

    # 1) juice-decay curve: mean forward profit remaining by setup x node
    H.append("<h2>1. When is the juice gone? (mean fwd_to_low_pct by node — TAKE OFF as it approaches 0)</h2>")
    piv = nodes.pivot_table(index='setup', columns='node', values='fwd_to_low_pct', aggfunc='mean').round(1)
    H.append(piv.to_html())
    H.append("<div class='l'>% of nodes already 'done' (fwd_to_low &lt; 1):</div>")
    done = nodes.assign(done=(nodes['fwd_to_low_pct']<1)).pivot_table(index='setup', columns='node', values='done', aggfunc='mean').round(2)*100
    H.append(done.round(0).astype('Int64').to_html())

    # 2) add/hold signal: does being below VWAP predict more remaining juice?
    H.append("<h2>2. ADD/HOLD signal — fwd_to_low conditioned on VWAP control (in-trade nodes, short_pnl &gt; 0)</h2>")
    intrade = nodes[nodes['short_pnl_pct']>0].copy()
    intrade['vwap_state'] = np.where(intrade['vs_vwap_pct']<0, 'below_vwap (short in control)', 'above_vwap')
    g = intrade.groupby(['setup','vwap_state'])['fwd_to_low_pct'].agg(['count','mean','median']).round(2)
    H.append(g.to_html())

    # 3) overnight block per setup
    H.append("<h2>3. HOLD OVERNIGHT — per-setup overnight outcomes (short P&amp;L, + = good for the short)</h2>")
    ov = summ.groupby('setup')[['overnight_gap_pct','next_day_open_to_low_pct','hold_cc_pct']].agg(['mean','median','min']).round(2)
    H.append(ov.to_html())
    H.append("<div class='l'>overnight_gap = short P&amp;L through the open gap (negative = gapped against you). "
             "hold_cc = close-to-close overnight short P&amp;L. next_day_open_to_low = continuation still available next day.</div>")

    # 4) overnight split by cap x setup — variance/tail view
    H.append("<h2>4. OVERNIGHT by cap &times; setup — tails live here, not in the means</h2>")
    ov2 = summ.groupby(['cap','setup'])[['overnight_gap_pct','hold_cc_pct']].agg(['count','mean','median','min']).round(1)
    H.append(ov2.to_html())
    H.append("<div class='l'>Read the <b>min</b> column, not the mean: ETF/Large worst overnight gap is &minus;6.5%; "
             "Med/Small/Micro worst is &minus;65%. Means are similar — the cap effect is tail width (variance), not EV. "
             "GapDownTrendBreak is overnight-negative at <i>every</i> cap. Cells with count &le; 3 are anecdotes, not stats.</div>")
    H.append("<div class='l'><b>All overnight gaps worse than &minus;5%:</b></div>")
    tail = summ[summ['overnight_gap_pct'] < -5][['ticker','date','setup','grade','cap','overnight_gap_pct','hold_cc_pct']] \
        .sort_values('overnight_gap_pct')
    H.append(tail.to_html(index=False))

    out = os.path.join(ROOT,'reports','mgmt_matrix.html')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out,'w',encoding='utf-8') as f:
        f.write('\n'.join(H))
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
