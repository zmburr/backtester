"""
Classifier audit: what would reversal_trader have said about every A-grade
3DGapFade in the database?

For each trade, instantiate ReversalTradeManager(ticker, cap, date) — the
exact init path the live tool runs (metrics from daily history ->
classify_reversal_setup -> pretrade/scorer). Record the verdict and, on a
classification miss, which gate failed, comparing live-computed metrics
against the hand-labeled CSV values.

Gates (classify_reversal_setup):
  core:  consecutive_up_days >= 2, gap_pct >= 0.04, pct_from_9ema >= 0.30
  supp:  pct_from_50mav >= 0.4, atr_pct in [0.04, 0.20], not closed-green-at-highs
"""
import os, sys, io, logging
from contextlib import redirect_stdout

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)  # quiet the manager's init chatter

from scanners.reversal_trader import ReversalTradeManager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gate_diagnosis(m: dict) -> str:
    """Which classification gate(s) fail on these metrics?"""
    fails = []
    cu, gap, ema = m.get('consecutive_up_days'), m.get('gap_pct'), m.get('pct_from_9ema')
    p50, atr = m.get('pct_from_50mav'), m.get('atr_pct')
    fr, cp = m.get('fade_day_return'), m.get('fade_day_close_position')

    def bad(v):
        return v is None or (isinstance(v, float) and pd.isna(v))

    if bad(cu) or cu < 2:           fails.append(f'up_days={cu}<2')
    if bad(gap) or gap < 0.04:      fails.append(f'gap={gap if bad(gap) else round(gap,3)}<0.04')
    if bad(ema) or ema < 0.30:      fails.append(f'9ema={ema if bad(ema) else round(ema,3)}<0.30')
    if not bad(p50) and p50 < 0.4:  fails.append(f'50mav={round(p50,3)}<0.4')
    if not bad(atr) and (atr < 0.04 or atr > 0.20): fails.append(f'atr={round(atr,3)} outside [.04,.20]')
    if not bad(fr) and not bad(cp) and fr > 0.02 and cp > 0.75:
        fails.append('continuation-reject (closed green at highs)')
    return '; '.join(fails) if fails else 'all gates pass?!'


def main():
    df = pd.read_csv(os.path.join(ROOT, 'data', 'reversal_data.csv'), encoding='utf-8-sig')
    a = df[(df['setup'] == '3DGapFade') & (df['trade_grade'] == 'A')].copy()
    a['date_iso'] = pd.to_datetime(a['date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')
    print(f'Auditing {len(a)} A-grade 3DGapFades through the live init path...\n')

    rows = []
    for _, r in a.iterrows():
        tk, d = r['ticker'], r['date_iso']
        try:
            with redirect_stdout(io.StringIO()):
                mgr = ReversalTradeManager(tk, cap=r['cap'], date=d)
            m = mgr.metrics
            rows.append(dict(
                ticker=tk, date=d, cap=r['cap'],
                detected=mgr.setup_type or 'Generic',
                score=mgr.reversal_score, rec=mgr.recommendation,
                live_up_days=m.get('consecutive_up_days'),
                live_gap=round(m.get('gap_pct', float('nan')), 3),
                live_9ema=round(m.get('pct_from_9ema', float('nan')), 3),
                csv_gap=round(float(r['gap_pct']), 3),
                csv_9ema=round(float(r['pct_from_9ema']), 3),
                csv_up_days=r.get('consecutive_up_days'),
                miss_reason='' if mgr.setup_type == '3DGapFade' else gate_diagnosis(m),
            ))
            status = 'OK  ' if mgr.setup_type == '3DGapFade' else 'MISS'
            print(f'{status} {tk:>6} {d}  detected={mgr.setup_type or "Generic":<10} '
                  f'{mgr.reversal_score}/5 {mgr.recommendation:<7} '
                  f'{rows[-1]["miss_reason"]}')
        except Exception as e:
            rows.append(dict(ticker=tk, date=d, cap=r['cap'], detected='ERROR',
                             score=None, rec=str(e)[:60], miss_reason='init failed'))
            print(f'ERR  {tk:>6} {d}  {str(e)[:70]}')

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROOT, 'data', 'classifier_audit_a_grades.csv'), index=False)

    ok = out[out['detected'] == '3DGapFade']
    miss = out[(out['detected'] != '3DGapFade') & (out['detected'] != 'ERROR')]
    print('\n' + '=' * 90)
    print(f'CLASSIFIED CORRECTLY: {len(ok)}/{len(out)}   '
          f'MISSED: {len(miss)}   ERRORS: {len(out) - len(ok) - len(miss)}')
    print(f'GO when classified:   {(ok["rec"] == "GO").sum()}/{len(ok)}')
    print(f'NO-GO on misses:      {(miss["rec"] == "NO-GO").sum()}/{len(miss)}')
    if len(miss):
        print('\nMISS REASONS:')
        for _, r in miss.iterrows():
            print(f'  {r["ticker"]:>6} {r["date"]}: {r["miss_reason"]}'
                  f'   [live gap={r["live_gap"]} 9ema={r["live_9ema"]} up={r["live_up_days"]}'
                  f' | csv gap={r["csv_gap"]} 9ema={r["csv_9ema"]}]')
    print('\nwrote data/classifier_audit_a_grades.csv')


if __name__ == '__main__':
    main()
