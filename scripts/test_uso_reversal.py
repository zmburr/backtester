"""
USO Overnight Gap Reversal Analysis
====================================
Simulates USO opening +18% on Monday 2026-03-09 (oil futures up 18% overnight)
and runs it through the full reversal scoring pipeline to evaluate GO/NO-GO.

USO is classified as ETF cap.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime

from data_queries.polygon_queries import get_levels_data, adjust_date_to_market
from analyzers.reversal_scorer import (
    ReversalScorer, compute_reversal_intensity, print_score_report,
    CAP_THRESHOLDS,
)
from analyzers.reversal_pretrade import ReversalPretrade, classify_reversal_setup
from support.config import send_email

# ── Config ──────────────────────────────────────────────────────────────────
TICKER = 'USO'
CAP = 'ETF'
GAP_PCT = 0.18  # +18% overnight gap
LAST_TRADING_DAY = '2026-03-06'  # Friday
SIMULATED_DATE = '2026-03-09'    # Monday open


def fetch_and_compute(ticker, date, gap_pct):
    """Fetch historical data and compute metrics with a simulated gap-up open."""

    print(f"Fetching {ticker} historical data through {date}...")
    levels = get_levels_data(ticker, date, 310, 1, 'day')
    if levels is None or levels.empty:
        print("ERROR: No data returned from Polygon.")
        return None

    hist = levels.copy()
    print(f"  Got {len(hist)} daily bars, last bar: {hist.index[-1].date()}")

    # Reference prices
    last_close = hist.iloc[-1]['close']
    simulated_open = round(last_close * (1 + gap_pct), 2)
    print(f"\n  Last close (Fri 3/6): ${last_close:.2f}")
    print(f"  Simulated Mon open:   ${simulated_open:.2f} (+{gap_pct*100:.0f}%)")

    closes = hist['close']
    metrics = {}
    metrics['current_price'] = simulated_open
    metrics['prior_close'] = last_close
    metrics['simulated_open'] = simulated_open

    # ── Moving averages (computed from hist, but % distance uses simulated open) ──
    if len(closes) >= 9:
        ema_9 = closes.ewm(span=9, adjust=False).mean().iloc[-1]
        metrics['ema_9'] = ema_9
        metrics['pct_from_9ema'] = (simulated_open - ema_9) / ema_9

    if len(closes) >= 50:
        sma_50 = closes.rolling(50).mean().iloc[-1]
        metrics['sma_50'] = sma_50
        metrics['pct_from_50mav'] = (simulated_open - sma_50) / sma_50

    if len(closes) >= 200:
        sma_200 = closes.rolling(200).mean().iloc[-1]
        metrics['sma_200'] = sma_200
        metrics['pct_from_200mav'] = (simulated_open - sma_200) / sma_200

    # ── ATR (14-day) ──
    if len(hist) >= 2:
        hl = hist['high'] - hist['low']
        hpc = abs(hist['high'] - hist['close'].shift(1))
        lpc = abs(hist['low'] - hist['close'].shift(1))
        tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        atr_window = min(14, len(tr))
        atr = tr.rolling(window=atr_window, min_periods=1).mean().iloc[-1]
        metrics['atr'] = atr
        metrics['atr_pct'] = atr / simulated_open if simulated_open > 0 else 0

        # Prior day range as multiple of ATR
        prior_range = hist.iloc[-1]['high'] - hist.iloc[-1]['low']
        metrics['prior_day_range_atr'] = prior_range / atr if atr > 0 else 0

    # ── Consecutive up days ──
    consecutive_up = 0
    for i in range(len(hist) - 1, 0, -1):
        if hist.iloc[i]['close'] > hist.iloc[i - 1]['close']:
            consecutive_up += 1
        else:
            break
    metrics['consecutive_up_days'] = consecutive_up

    # ── RVOL ──
    adv_window = min(20, len(hist))
    adv = hist['volume'].rolling(window=adv_window, min_periods=1).mean().iloc[-1]
    prior_day_vol = hist.iloc[-1]['volume']
    metrics['avg_daily_vol'] = adv
    metrics['rvol_score'] = prior_day_vol / adv if adv > 0 else 0
    # Note: Monday's RVOL unknown pre-market; using Friday's as baseline

    # ── Gap % (simulated 18% gap) ──
    metrics['gap_pct'] = gap_pct

    # ── Percent changes over lookback windows (relative to simulated open) ──
    for days, key in [(3, 'pct_change_3'), (15, 'pct_change_15'),
                      (30, 'pct_change_30'), (90, 'pct_change_90'),
                      (120, 'pct_change_120')]:
        if len(hist) >= days:
            old_close = hist.iloc[-days]['close']
            if old_close > 0:
                metrics[key] = (simulated_open - old_close) / old_close

    # ── Percent off highs ──
    window_30 = hist.iloc[-30:] if len(hist) >= 30 else hist
    high_30d = window_30['high'].max()
    metrics['pct_off_30d_high'] = (simulated_open - high_30d) / high_30d if high_30d > 0 else 0

    high_52 = hist['high'].max()
    metrics['pct_off_52wk_high'] = (simulated_open - high_52) / high_52 if high_52 > 0 else 0

    return metrics


def format_pct(val, decimals=1):
    if val is None:
        return 'N/A'
    return f"{val * 100:+.{decimals}f}%"


def format_mult(val, decimals=1):
    if val is None:
        return 'N/A'
    return f"{val:.{decimals}f}x"


def build_console_report(metrics, scorer_result, pretrade_result, intensity_result):
    """Print a detailed console report."""
    print("\n" + "=" * 72)
    print(f"  USO OVERNIGHT GAP REVERSAL ANALYSIS — {SIMULATED_DATE}")
    print(f"  Oil Futures +18% -> Simulated Open ${metrics['simulated_open']:.2f}")
    print("=" * 72)

    print(f"\n  Last Close:   ${metrics['prior_close']:.2f}")
    print(f"  Sim Open:     ${metrics['simulated_open']:.2f} ({format_pct(GAP_PCT)})")
    print(f"  9 EMA:        ${metrics.get('ema_9', 0):.2f}  ({format_pct(metrics.get('pct_from_9ema'))} above)")
    print(f"  50 SMA:       ${metrics.get('sma_50', 0):.2f}  ({format_pct(metrics.get('pct_from_50mav'))} above)")
    print(f"  ATR:          ${metrics.get('atr', 0):.2f}  ({format_pct(metrics.get('atr_pct'))} of price)")

    print(f"\n  CONTEXTUAL METRICS:")
    print(f"  {'─' * 50}")
    print(f"  3-day momentum:     {format_pct(metrics.get('pct_change_3'))}")
    print(f"  15-day change:      {format_pct(metrics.get('pct_change_15'))}")
    print(f"  30-day change:      {format_pct(metrics.get('pct_change_30'))}")
    print(f"  Prior range/ATR:    {format_mult(metrics.get('prior_day_range_atr'))}")
    print(f"  RVOL (Fri):         {format_mult(metrics.get('rvol_score'))}")
    print(f"  Consec up days:     {metrics.get('consecutive_up_days', 0)}")
    print(f"  Gap %:              {format_pct(metrics.get('gap_pct'))}")
    print(f"  % off 30d high:     {format_pct(metrics.get('pct_off_30d_high'))}")
    print(f"  % off 52wk high:    {format_pct(metrics.get('pct_off_52wk_high'))}")

    # Scorer result
    print(f"\n  {'=' * 50}")
    print(f"  REVERSAL SCORER (ETF thresholds)")
    print(f"  {'=' * 50}")
    print(f"  Pre-Trade:  {scorer_result['pretrade_score']}/{scorer_result['pretrade_max']} "
          f"({scorer_result['pretrade_grade']}) → {scorer_result['pretrade_recommendation']}")
    print(f"  Readiness:  {'PASS' if scorer_result['readiness_passed'] else 'FAIL'}")
    if scorer_result.get('intensity') is not None:
        print(f"  Intensity:  {scorer_result['intensity']:.0f}/100")
    else:
        print(f"  Intensity:  N/A (not GO or insufficient data)")

    print(f"\n  CRITERIA BREAKDOWN:")
    print(f"  {'─' * 60}")
    for criterion, details in scorer_result['criteria_details'].items():
        is_pretrade = criterion != 'reversal_pct'
        tag = "PRE" if is_pretrade else "OUT"
        status = "PASS" if details['passed'] else "FAIL"
        actual = details['actual']
        threshold = details['threshold']

        if criterion in ['pct_from_9ema', 'gap_pct', 'reversal_pct', 'pct_change_3']:
            actual_str = f"{actual*100:.1f}%" if actual is not None else "N/A"
            threshold_str = f"{threshold*100:.1f}%"
        else:
            actual_str = f"{actual:.2f}x" if actual is not None else "N/A"
            threshold_str = f"{threshold:.1f}x"

        mark = "PASS" if details['passed'] else "FAIL"
        print(f"  [{mark}] [{tag}] {details['name']}")
        print(f"           Required: >= {threshold_str} | Actual: {actual_str}")

    # Intensity breakdown
    if intensity_result and intensity_result.get('composite') is not None:
        print(f"\n  INTENSITY BREAKDOWN (cap-stratified percentile vs Grade A trades):")
        print(f"  {'─' * 60}")
        for col, info in intensity_result['details'].items():
            pctile = info.get('pctile')
            weight = info.get('weight', 0)
            actual = info.get('actual')
            pctile_str = f"{pctile:.0f}th" if pctile is not None else "N/A"
            actual_str = f"{actual:.2f}" if actual is not None else "N/A"
            print(f"  {col:25s} actual={actual_str:>8s}  pctile={pctile_str:>5s}  weight={weight:.0%}")
        print(f"\n  COMPOSITE INTENSITY: {intensity_result['composite']:.0f}/100")

    # Pretrade typed classification
    if pretrade_result:
        print(f"\n  {'=' * 50}")
        print(f"  3DGapFade TYPED CLASSIFICATION")
        print(f"  {'=' * 50}")
        setup_type = classify_reversal_setup(metrics)
        if setup_type:
            print(f"  Classified as: {setup_type}")
            print(f"  Score: {pretrade_result.score}/{pretrade_result.max_score} → {pretrade_result.recommendation}")
            for item in pretrade_result.items:
                mark = "PASS" if item.passed else "FAIL"
                print(f"  [{mark}] {item.description}")
                print(f"           Required: >= {item.threshold_display} | Actual: {item.actual_display}")
        else:
            print(f"  Does not classify as 3DGapFade (needs 2+ consec up days + gap >= 4% + 30%+ above 9EMA)")


def build_email_html(metrics, scorer_result, intensity_result, pretrade_result):
    """Build an HTML email matching the priority report style."""
    now = datetime.now()
    rec = scorer_result['pretrade_recommendation']
    rec_colors = {"GO": "#3fb950", "CAUTION": "#e3b341", "NO-GO": "#f85149"}
    rec_color = rec_colors.get(rec, "#8b949e")

    score_str = f"{scorer_result['pretrade_score']}/{scorer_result['pretrade_max']}"
    grade = scorer_result['pretrade_grade']
    intensity = scorer_result.get('intensity')
    intensity_str = f"{intensity:.0f}/100" if intensity is not None else "N/A"

    # Build criteria rows
    criteria_rows = ""
    for criterion, details in scorer_result['criteria_details'].items():
        if criterion == 'reversal_pct':
            continue  # skip outcome criterion
        status_color = "#3fb950" if details['passed'] else "#f85149"
        status_icon = "✓" if details['passed'] else "✗"
        actual = details['actual']
        threshold = details['threshold']
        if criterion in ['pct_from_9ema', 'gap_pct', 'pct_change_3']:
            actual_str = f"{actual*100:.1f}%" if actual is not None else "N/A"
            threshold_str = f"{threshold*100:.1f}%"
        else:
            actual_str = f"{actual:.2f}x" if actual is not None else "N/A"
            threshold_str = f"{threshold:.1f}x"

        criteria_rows += f"""
        <tr>
            <td style="padding: 6px 10px; border: 1px solid #30363d; color: {status_color}; font-weight: bold; text-align: center;">{status_icon}</td>
            <td style="padding: 6px 10px; border: 1px solid #30363d;">{details['name']}</td>
            <td style="padding: 6px 10px; border: 1px solid #30363d; text-align: center;">&ge; {threshold_str}</td>
            <td style="padding: 6px 10px; border: 1px solid #30363d; text-align: center; font-weight: bold; color: {status_color};">{actual_str}</td>
        </tr>"""

    # Intensity breakdown rows
    intensity_rows = ""
    if intensity_result and intensity_result.get('composite') is not None:
        for col, info in intensity_result['details'].items():
            pctile = info.get('pctile')
            weight = info.get('weight', 0)
            actual = info.get('actual')
            pctile_str = f"{pctile:.0f}th" if pctile is not None else "N/A"
            actual_str = f"{actual:.2f}" if actual is not None else "N/A"
            # Color code percentile
            if pctile is not None:
                if pctile >= 80:
                    p_color = "#3fb950"
                elif pctile >= 50:
                    p_color = "#e3b341"
                else:
                    p_color = "#8b949e"
            else:
                p_color = "#8b949e"
            intensity_rows += f"""
            <tr>
                <td style="padding: 4px 10px; border: 1px solid #30363d;">{col}</td>
                <td style="padding: 4px 10px; border: 1px solid #30363d; text-align: center;">{actual_str}</td>
                <td style="padding: 4px 10px; border: 1px solid #30363d; text-align: center; color: {p_color}; font-weight: bold;">{pctile_str}</td>
                <td style="padding: 4px 10px; border: 1px solid #30363d; text-align: center;">{weight:.0%}</td>
            </tr>"""

    # Context metrics
    context_html = f"""
    <table style="border-collapse: collapse; font-size: 0.9em; color: #c9d1d9; margin: 10px 0; width: 100%;">
        <tr style="background-color: #21262d;">
            <th style="padding: 6px 10px; border: 1px solid #30363d; text-align: left;" colspan="2">Market Context</th>
        </tr>
        <tr><td style="padding: 4px 10px; border: 1px solid #30363d;">Last Close (Fri 3/6)</td>
            <td style="padding: 4px 10px; border: 1px solid #30363d; font-weight: bold;">${metrics['prior_close']:.2f}</td></tr>
        <tr><td style="padding: 4px 10px; border: 1px solid #30363d;">Simulated Open (Mon 3/9)</td>
            <td style="padding: 4px 10px; border: 1px solid #30363d; font-weight: bold; color: #f85149;">${metrics['simulated_open']:.2f} (+{GAP_PCT*100:.0f}%)</td></tr>
        <tr><td style="padding: 4px 10px; border: 1px solid #30363d;">9 EMA</td>
            <td style="padding: 4px 10px; border: 1px solid #30363d;">${metrics.get('ema_9', 0):.2f} ({format_pct(metrics.get('pct_from_9ema'))} above)</td></tr>
        <tr><td style="padding: 4px 10px; border: 1px solid #30363d;">50 SMA</td>
            <td style="padding: 4px 10px; border: 1px solid #30363d;">${metrics.get('sma_50', 0):.2f} ({format_pct(metrics.get('pct_from_50mav'))} above)</td></tr>
        <tr><td style="padding: 4px 10px; border: 1px solid #30363d;">ATR</td>
            <td style="padding: 4px 10px; border: 1px solid #30363d;">${metrics.get('atr', 0):.2f} ({format_pct(metrics.get('atr_pct'))} of price)</td></tr>
        <tr><td style="padding: 4px 10px; border: 1px solid #30363d;">3-Day Momentum</td>
            <td style="padding: 4px 10px; border: 1px solid #30363d;">{format_pct(metrics.get('pct_change_3'))}</td></tr>
        <tr><td style="padding: 4px 10px; border: 1px solid #30363d;">30-Day Change</td>
            <td style="padding: 4px 10px; border: 1px solid #30363d;">{format_pct(metrics.get('pct_change_30'))}</td></tr>
        <tr><td style="padding: 4px 10px; border: 1px solid #30363d;">Prior Day Range/ATR</td>
            <td style="padding: 4px 10px; border: 1px solid #30363d;">{format_mult(metrics.get('prior_day_range_atr'))}</td></tr>
        <tr><td style="padding: 4px 10px; border: 1px solid #30363d;">RVOL (Friday)</td>
            <td style="padding: 4px 10px; border: 1px solid #30363d;">{format_mult(metrics.get('rvol_score'))}</td></tr>
        <tr><td style="padding: 4px 10px; border: 1px solid #30363d;">Consecutive Up Days</td>
            <td style="padding: 4px 10px; border: 1px solid #30363d;">{metrics.get('consecutive_up_days', 0)}</td></tr>
        <tr><td style="padding: 4px 10px; border: 1px solid #30363d;">% Off 30d High</td>
            <td style="padding: 4px 10px; border: 1px solid #30363d;">{format_pct(metrics.get('pct_off_30d_high'))}</td></tr>
        <tr><td style="padding: 4px 10px; border: 1px solid #30363d;">% Off 52wk High</td>
            <td style="padding: 4px 10px; border: 1px solid #30363d;">{format_pct(metrics.get('pct_off_52wk_high'))}</td></tr>
    </table>
    """

    # ETF thresholds context
    etf_thresh = CAP_THRESHOLDS['ETF']
    thresholds_note = (
        f"ETF thresholds: 9EMA &ge; {etf_thresh.pct_from_9ema*100:.0f}%, "
        f"Range &ge; {etf_thresh.prior_day_range_atr:.1f}x ATR, "
        f"RVOL &ge; {etf_thresh.rvol_score:.1f}x, "
        f"3D mom &ge; {etf_thresh.pct_change_3*100:.0f}%, "
        f"Gap &ge; {etf_thresh.gap_pct*100:.1f}%"
    )

    # Readiness gate
    readiness = scorer_result.get('readiness_passed', False)
    readiness_color = "#3fb950" if readiness else "#f85149"
    readiness_str = "PASS" if readiness else "FAIL"
    readiness_thresh = scorer_result.get('readiness_threshold')
    readiness_note = ""
    if readiness_thresh is not None:
        readiness_note = f" (need {readiness_thresh*100:.0f}% 3D momentum for ETF)"

    html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                max-width: 860px; margin: 0 auto; color: #c9d1d9;
                background-color: #161b22; font-size: 14px; line-height: 1.5; padding: 16px;">

        <!-- Header -->
        <div style="background-color: #21262d; color: #ffffff; padding: 12px 16px;
                    border-radius: 6px 6px 0 0; margin-bottom: 16px; border-bottom: 2px solid #30363d;">
            <h1 style="margin: 0; font-size: 1.4em;">⚠ USO Overnight Gap Analysis</h1>
            <div style="font-size: 0.9em; color: #8b949e; margin-top: 4px;">
                {now.strftime("%A, %B %d %Y  %I:%M %p")} ET &nbsp;|&nbsp;
                Oil Futures +18% Overnight
            </div>
        </div>

        <!-- Ticker banner -->
        <div style="border-top: 3px solid {rec_color}; margin-top: 10px; padding-top: 10px;">
            <h2 style="margin: 0 0 4px 0; color: #f0f6fc;">
                USO
                <span style="color: {rec_color}; font-size: 0.8em;">{rec}</span>
                <span style="color: #58a6ff; font-size: 0.65em; font-weight: normal;">REVERSAL (ETF)</span>
            </h2>
        </div>

        <!-- Narrative -->
        <div style="background-color: #1c2333; border-left: 4px solid {rec_color};
                    padding: 10px 14px; margin: 8px 0; border-radius: 0 6px 6px 0;
                    font-size: 0.95em; color: #e6edf3; line-height: 1.5;">
            <strong>Oil futures surged +18% overnight.</strong> USO would open at ~${metrics['simulated_open']:.2f}
            vs Friday close of ${metrics['prior_close']:.2f}. This is an unprecedented single-session move for
            a commodity ETF. Scoring against the reversal framework with ETF-specific thresholds below.
            <br><br>
            <strong>Pre-Trade Score: {score_str} ({grade}) → <span style="color: {rec_color};">{rec}</span></strong>
            &nbsp;|&nbsp; Intensity: {intensity_str}
            &nbsp;|&nbsp; Readiness: <span style="color: {readiness_color};">{readiness_str}</span>{readiness_note}
        </div>

        <!-- Context table -->
        {context_html}

        <!-- Criteria table -->
        <h3 style="color: #f0f6fc; margin: 16px 0 8px 0;">Pre-Trade Criteria ({score_str})</h3>
        <div style="font-size: 0.8em; color: #8b949e; margin-bottom: 8px;">{thresholds_note}</div>
        <table style="border-collapse: collapse; font-size: 0.9em; color: #c9d1d9; margin: 10px 0; width: 100%;">
            <tr style="background-color: #21262d;">
                <th style="padding: 6px 10px; border: 1px solid #30363d; width: 40px;"></th>
                <th style="padding: 6px 10px; border: 1px solid #30363d; text-align: left;">Criterion</th>
                <th style="padding: 6px 10px; border: 1px solid #30363d;">Required</th>
                <th style="padding: 6px 10px; border: 1px solid #30363d;">Actual</th>
            </tr>
            {criteria_rows}
        </table>

        <!-- Intensity breakdown -->
        {"" if not intensity_rows else f'''
        <h3 style="color: #f0f6fc; margin: 16px 0 8px 0;">Intensity Score: {intensity_str}</h3>
        <div style="font-size: 0.8em; color: #8b949e; margin-bottom: 8px;">
            Cap-stratified percentile ranking vs historical Grade A reversal trades (Large/ETF group)
        </div>
        <table style="border-collapse: collapse; font-size: 0.9em; color: #c9d1d9; margin: 10px 0; width: 100%;">
            <tr style="background-color: #21262d;">
                <th style="padding: 4px 10px; border: 1px solid #30363d; text-align: left;">Metric</th>
                <th style="padding: 4px 10px; border: 1px solid #30363d;">ATR-Adj Value</th>
                <th style="padding: 4px 10px; border: 1px solid #30363d;">Percentile</th>
                <th style="padding: 4px 10px; border: 1px solid #30363d;">Weight</th>
            </tr>
            {intensity_rows}
        </table>
        '''}

        <!-- Footer -->
        <div style="margin-top: 24px; padding-top: 12px; border-top: 1px solid #30363d;
                    font-size: 0.8em; color: #8b949e;">
            Generated by test_uso_reversal.py &nbsp;|&nbsp; Simulated scenario, not live data.
            <br>Metrics assume USO tracks oil futures 1:1 on the gap (actual tracking may differ).
            <br>RVOL uses Friday's value — Monday's actual RVOL will likely be much higher.
        </div>
    </div>
    """
    return html


def main():
    # 1. Fetch data and compute metrics
    metrics = fetch_and_compute(TICKER, LAST_TRADING_DAY, GAP_PCT)
    if not metrics:
        print("Failed to fetch data. Exiting.")
        return

    # 2. Run through ReversalScorer (generic ETF thresholds)
    scorer = ReversalScorer()
    scorer_result = scorer.score_setup(
        ticker=TICKER,
        date=SIMULATED_DATE,
        cap=CAP,
        metrics=metrics,
        setup='3DGapFade',  # treat as euphoric setup for readiness gate
    )

    # 3. Compute intensity
    intensity_result = compute_reversal_intensity(metrics, cap=CAP)

    # 4. Run through ReversalPretrade (typed classification)
    pretrade = ReversalPretrade()
    setup_type = classify_reversal_setup(metrics)
    pretrade_result = None
    if setup_type:
        pretrade_result = pretrade.validate(
            ticker=TICKER, metrics=metrics,
            setup_type=setup_type, cap=CAP,
        )

    # 5. Console report
    build_console_report(metrics, scorer_result, pretrade_result, intensity_result)

    # 6. Build and send email
    html = build_email_html(metrics, scorer_result, intensity_result, pretrade_result)

    subject = f"⚠ USO {scorer_result['pretrade_recommendation']} — Oil +18% Overnight Gap Analysis"
    print(f"\nSending email to zmburr@gmail.com...")
    send_email(
        to_email='zmburr@gmail.com',
        subject=subject,
        body=html,
        is_html=True,
    )
    print("Done.")


if __name__ == '__main__':
    main()
