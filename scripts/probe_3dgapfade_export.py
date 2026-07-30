"""
Export the A-grade 3DGapFade decision-node tables to viewable files:
  - data/probe_3dgapfade_nodes.csv   (long format: one row per trade x node)
  - reports/probe_3dgapfade_nodes.html  (per-trade styled tables)
Reuses build_nodes() from the probe so the numbers are identical.
"""
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_3dgapfade_nodes import load_a_grade, build_nodes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_OUT = os.path.join(ROOT, 'data', 'probe_3dgapfade_nodes.csv')
HTML_OUT = os.path.join(ROOT, 'reports', 'probe_3dgapfade_nodes.html')

CSS = """
<style>
 body{font-family:-apple-system,Segoe UI,Arial,sans-serif;background:#0e1116;color:#e6edf3;margin:24px}
 h2{margin:28px 0 4px;color:#58a6ff;font-size:16px}
 .meta{color:#8b949e;font-size:12px;margin-bottom:6px}
 table{border-collapse:collapse;margin-bottom:10px;font-size:12px}
 th,td{border:1px solid #30363d;padding:3px 8px;text-align:right}
 th{background:#161b22;color:#8b949e;position:sticky;top:0}
 td.node{font-weight:600;color:#e6edf3;text-align:center}
 .legend{color:#8b949e;font-size:12px;margin:8px 0 20px;line-height:1.6}
</style>
"""

LEGEND = """
<div class="legend">
<b>How to read a node row (state a live watcher would see at that minute):</b><br>
&nbsp;&nbsp;<b>short_pnl_pct</b> — running P&amp;L if shorted at the 09:30 open (green = working).<br>
&nbsp;&nbsp;<b>below_hod_pct</b> — distance below the session high so far (how far the reversal has progressed).<br>
&nbsp;&nbsp;<b>vs_vwap_pct</b> — price vs session VWAP (negative = short in control).<br>
&nbsp;&nbsp;<b>made_new_hod</b> — has it printed a new high AFTER the open (squeeze/stop-hunt before the fade)?<br>
&nbsp;&nbsp;<b>mae_pct</b> — worst adverse move vs open so far (max squeeze against the short).<br>
<b>Forward labels (the outcome the matrix would learn — NOT visible live):</b><br>
&nbsp;&nbsp;<b>fwd_to_low_pct</b> — additional short profit still available from this node to the day's low.<br>
&nbsp;&nbsp;<b>fwd_30min_pct</b> — short P&amp;L change over the next 30 minutes.<br>
</div>
"""


def color_pnl(v):
    try: v = float(v)
    except: return ''
    if v > 0: return 'color:#3fb950'
    if v < 0: return 'color:#f85149'
    return ''

def color_fwd(v):
    try: v = float(v)
    except: return ''
    if v >= 5: return 'background:#0f5132;color:#d4ffe0'
    if v >= 1: return 'background:#163d2b'
    return 'color:#8b949e'


def main():
    a = load_a_grade()
    html = [CSS, f"<h1 style='color:#e6edf3;font-size:20px'>A-grade 3DGapFade — decision-node tables ({len(a)} trades)</h1>", LEGEND]
    long_rows = []
    for _, r in a.iterrows():
        tk, d = r['ticker'], r['date_iso']
        res, err = build_nodes(tk, d)
        if err:
            html.append(f"<h2>{tk} — {d} ({r['cap']})</h2><div class='meta'>SKIPPED: {err}</div>")
            continue
        nodes, meta = res
        nodes_out = nodes.copy()
        nodes_out.insert(0, 'ticker', tk); nodes_out.insert(1, 'date', d); nodes_out.insert(2, 'cap', r['cap'])
        long_rows.append(nodes_out)

        sty = (nodes.style
               .applymap(color_pnl, subset=['short_pnl_pct','vs_vwap_pct','fwd_30min_pct'])
               .applymap(color_fwd, subset=['fwd_to_low_pct'])
               .set_table_attributes('cellspacing="0"')
               .hide(axis='index'))
        html.append(f"<h2>{tk} — {d} ({r['cap']})</h2>")
        html.append(f"<div class='meta'>open={meta['open_px']} &nbsp; pm_high={meta['pm_high']} &nbsp; "
                    f"day_low={meta['day_low']} &nbsp; <b>open&rarr;low={meta['full_to_low_pct']}%</b> &nbsp; "
                    f"({meta['n_rth_bars']} RTH bars)</div>")
        html.append(sty.to_html())

    pd.concat(long_rows, ignore_index=True).to_csv(CSV_OUT, index=False)
    os.makedirs(os.path.dirname(HTML_OUT), exist_ok=True)
    with open(HTML_OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    print(f"wrote {CSV_OUT}")
    print(f"wrote {HTML_OUT}")


if __name__ == '__main__':
    main()
