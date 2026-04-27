"""Smoke test: verify reversal intensity rendering on today's live watchlist.

Mirrors the priority_report wiring without sending email or running LLMs.
Writes one HTML file with all reversal tickers' intensity sections for visual
inspection.
"""
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import generate_report as gr
from scanners import stock_screener as ss
from analyzers.bounce_scorer import classify_stock  # noqa: F401 (matches priority_report imports)


def main():
    watchlist = ss.watchlist
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    print(f"Watchlist: {len(watchlist)} tickers")

    print("Phase 1: screener data...")
    all_data = ss.get_all_stocks_data(watchlist)

    print("Phase 2: pretrade metrics + routing + intensity...")
    sections = []
    summary_rows = []
    for ticker in watchlist:
        td = all_data.get(ticker, {}) or {}
        pct_data = td.get('pct_data', {}) or {}
        mav_data = td.get('mav_data', {}) or {}
        bucket, _reason = gr.route_playbook(pct_data, mav_data)
        if bucket != 'reversal':
            continue

        pm = gr.get_pretrade_metrics(ticker, today)
        if not pm:
            print(f"  {ticker}: no pretrade metrics, skip")
            continue

        cap = gr.get_ticker_cap(ticker)
        score_result = gr.score_pretrade_setup(ticker, pm, cap=cap)

        intensity_metrics = {
            'atr_pct':             pm.get('atr_pct'),
            'pct_from_9ema':       pm.get('pct_from_9ema'),
            'pct_change_3':        pm.get('pct_change_3'),
            'gap_pct':             pm.get('gap_pct'),
            'prior_day_range_atr': pm.get('prior_day_range_atr'),
            'rvol_score':          pm.get('prior_day_rvol'),
            'pct_from_50mav':      mav_data.get('pct_from_50mav'),
        }
        intensity = gr.compute_reversal_intensity(intensity_metrics, cap=cap)
        composite = intensity.get('composite')
        rec = score_result.get('recommendation')

        print(f"  {ticker:6s} cap={cap or 'NA':6s} rec={rec:8s} "
              f"intensity={composite if composite is not None else 'N/A'}")

        summary_rows.append((ticker, cap, rec, composite))
        sections.append(
            f"<h2>{ticker} ({cap}) - {rec}</h2>" +
            gr.format_reversal_intensity_html(intensity)
        )

    if not sections:
        print("\nNo reversal tickers in watchlist today.")
        return

    out_path = Path(__file__).resolve().parent.parent / 'reports' / 'smoke_reversal_intensity.html'
    out_path.parent.mkdir(exist_ok=True)
    html = (
        '<html><head><meta charset="utf-8"><style>'
        'body { background: #0d1117; color: #c9d1d9; font-family: ui-sans-serif, system-ui; padding: 20px; }'
        'table { border-collapse: collapse; }'
        '</style></head><body>'
        '<h1>Reversal Intensity smoke test</h1>'
        f'<p>Generated {datetime.datetime.now().isoformat(timespec="seconds")}</p>'
        + '\n'.join(sections)
        + '</body></html>'
    )
    out_path.write_text(html, encoding='utf-8')
    print(f"\nWrote {out_path} ({len(html)} chars, {len(sections)} reversal sections)")
    print("\n=== Summary ===")
    for t, c, r, i in summary_rows:
        bar = '#' * int((i or 0) / 5) if i is not None else ''
        print(f"  {t:6s} {c or 'NA':6s} {r:8s} {str(i or 'N/A'):>5s}  {bar}")


if __name__ == '__main__':
    main()
