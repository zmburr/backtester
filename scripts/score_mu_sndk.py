"""Pull capitulation case metrics for MU and SNDK using setup_screener."""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timedelta
from scanners.setup_screener import SetupScreener


def fmt_pct(v):
    return f"{v*100:.2f}%" if v is not None else "N/A"


def fmt_x(v):
    return f"{v:.2f}x" if v is not None else "N/A"


def fmt_money(v):
    if v is None:
        return "N/A"
    return f"${v:,.2f}"


def report(ticker, date, cap):
    s = SetupScreener()
    r = s.screen_ticker(ticker, date, cap)
    if r.error:
        print(f"\n{'='*70}\n{ticker} ERROR: {r.error}\n{'='*70}")
        return None
    m = r.metrics
    print(f"\n{'='*70}")
    print(f"{ticker} ({cap}) — {date}")
    print(f"{'='*70}")
    print(f"Current price: {fmt_money(m.get('current_price'))}")
    print(f"Prior close:   {fmt_money(m.get('prior_close'))}")
    print()
    print("--- Percentile ladder ---")
    for k in ['pct_change_3', 'pct_change_15', 'pct_change_30',
              'pct_change_90', 'pct_change_120']:
        print(f"  {k}: {fmt_pct(m.get(k))}")
    print()
    print("--- Range / Volume ---")
    print(f"  ATR: {fmt_money(m.get('atr'))} ({fmt_pct(m.get('atr_pct'))})")
    print(f"  Prior day range / ATR: {fmt_x(m.get('prior_day_range_atr'))}")
    print(f"  RVOL (prior day): {fmt_x(m.get('rvol_score'))}")
    print(f"  Avg daily vol (20d): {m.get('avg_daily_vol'):,.0f}" if m.get('avg_daily_vol') else "  Avg daily vol: N/A")
    print(f"  Gap %: {fmt_pct(m.get('gap_pct'))}")
    print(f"  Consecutive up days: {m.get('consecutive_up_days')}")
    print()
    print("--- Distance from MAVs ---")
    print(f"  9EMA:  {fmt_pct(m.get('pct_from_9ema'))}")
    print(f"  50MA:  {fmt_pct(m.get('pct_from_50mav'))}")
    print(f"  200MA: {fmt_pct(m.get('pct_from_200mav'))}")
    print()
    print("--- Bollinger ---")
    print(f"  Closed outside upper band: {m.get('closed_outside_upper_band')}")
    print(f"  Bollinger width: {fmt_pct(m.get('bollinger_width'))}")
    print()
    print("--- Prior day positioning ---")
    print(f"  Close vs low (% of range): {fmt_pct(m.get('prior_day_close_vs_low_pct'))}")
    print(f"  pct_from_52wk_high: {fmt_pct(m.get('pct_from_52wk_high'))}")
    print(f"  breaks_52wk_high (today's intraday high >= prior period max): {m.get('breaks_52wk_high')}")
    print()
    print("--- Setup classification & score ---")
    print(f"  Parabolic short: {r.parabolic_score}/{r.parabolic_max_score}  grade={r.parabolic_grade}  -> {r.parabolic_recommendation}")
    print(f"  Setup type detected: {r.parabolic_setup_type}")
    print(f"  Is parabolic candidate: {r.is_parabolic_candidate}")

    # Compute intensity (cap-stratified ATR-adjusted percentile vs Grade-A reference)
    from analyzers.reversal_scorer import compute_reversal_intensity, check_archetype
    intensity_result = compute_reversal_intensity(m, cap=cap)
    print(f"  Intensity (0-100): {intensity_result.get('composite')}")
    if intensity_result.get('details'):
        print("  Intensity components:")
        for k, v in intensity_result['details'].items():
            print(f"    {k}: pctile={v.get('pctile')}  actual={v.get('actual')}  weight={v.get('weight')}")
    archetype_passed, archetype_detail = check_archetype(m)
    print(f"  Archetype gate (parabolic-at-highs): {'PASS' if archetype_passed else 'FAIL'}")
    print(f"    by_52wk={archetype_detail['passed_by_52wk']}  by_200mav={archetype_detail['passed_by_200mav']}  by_30d={archetype_detail['passed_by_30d']}")
    print()
    print("--- Parabolic criteria ---")
    for k, v in r.parabolic_criteria.items():
        passed = "PASS" if v['passed'] else "FAIL"
        actual = v['actual']
        threshold = v['threshold']
        if isinstance(actual, float):
            actual_str = f"{actual:.4f}"
        else:
            actual_str = str(actual)
        print(f"  [{passed}] {v['name']}: actual={actual_str}, threshold={threshold}")
    return r


if __name__ == '__main__':
    today = datetime.now().strftime('%Y-%m-%d')
    if len(sys.argv) > 1:
        today = sys.argv[1]
    print(f"Run date: {today}")
    # MU and SNDK are large-cap memory chip names
    report('MU', today, 'Large')
    report('SNDK', today, 'Large')
