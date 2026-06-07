"""One-shot: email all Grade-A strongstock bounce-trade charts + SOXL/MU/KORU comp charts."""
import os
import sys
import pandas as pd
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

from analyzers.charter import create_daily_chart
from support.config import send_email

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger("ss_review")

CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'bounce_data.csv')
CHART_DIR = os.path.join(PROJECT_ROOT, 'charts', 'strongstock_review')
COMP_DATE = '2026-06-05'  # latest completed bar (Fri before Mon 6/8)
COMP_TICKERS = ['SOXL', 'MU', 'KORU']
TO_EMAIL = 'zmburr@gmail.com'

os.makedirs(CHART_DIR, exist_ok=True)


def to_iso(date_str: str) -> str:
    """Convert M/D/YYYY (CSV format) to YYYY-MM-DD."""
    return pd.to_datetime(date_str).strftime('%Y-%m-%d')


def main():
    df = pd.read_csv(CSV_PATH)
    # Strongstock = any setup containing 'strongstock' but NOT IntradayCapitch (different setup, 11% WR)
    mask = (
        df['Setup'].str.contains('strongstock', case=False, na=False)
        & ~df['Setup'].str.contains('IntradayCapitch', case=False, na=False)
        & (df['trade_grade'] == 'A')
    )
    strong = df.loc[mask].copy().sort_values('bounce_open_close_pct', ascending=False)
    log.info(f"{len(strong)} Grade A strongstock trades to chart")

    # --- Generate per-trade historical charts (1y window ending on the bounce date) ---
    inline_images = {}
    rows = []  # for body table/section list
    for i, (_, r) in enumerate(strong.iterrows(), start=1):
        ticker = str(r['ticker']).strip()
        trade_date_iso = to_iso(str(r['date']))
        setup = str(r['Setup'])
        bounce_pct = float(r.get('bounce_open_close_pct') or 0) * 100
        gap_pct = float(r.get('gap_pct') or 0) * 100
        cap = str(r.get('cap', '') or '')
        label = f"{trade_date_iso.replace('-', '')}_setup"
        cid = f"trade_{i}"
        try:
            png = create_daily_chart(
                ticker, output_dir=CHART_DIR, end_date=trade_date_iso, label=label,
            )
            inline_images[cid] = png
            rows.append({
                'cid': cid, 'i': i, 'ticker': ticker, 'date': trade_date_iso,
                'setup': setup, 'cap': cap, 'gap_pct': gap_pct, 'bounce_pct': bounce_pct,
            })
            log.info(f"[{i}/{len(strong)}] {ticker} {trade_date_iso} -> {png}")
        except Exception as e:
            log.error(f"[{i}/{len(strong)}] {ticker} {trade_date_iso} FAILED: {e}")

    # --- Generate comp charts (SOXL/MU/KORU as of 6/5) ---
    comp_rows = []
    for j, t in enumerate(COMP_TICKERS, start=1):
        cid = f"comp_{j}"
        try:
            png = create_daily_chart(
                t, output_dir=CHART_DIR, end_date=COMP_DATE, label='compare_2026-06-05',
            )
            inline_images[cid] = png
            comp_rows.append({'cid': cid, 'ticker': t})
            log.info(f"COMP {t} -> {png}")
        except Exception as e:
            log.error(f"COMP {t} FAILED: {e}")

    # --- Build HTML body ---
    style = (
        "body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#222;}"
        "h1{font-size:20px;}h2{font-size:16px;margin-top:32px;border-bottom:1px solid #ddd;padding-bottom:4px;}"
        "h3{font-size:14px;margin-bottom:4px;}"
        ".meta{color:#555;font-size:12px;margin-bottom:6px;}"
        ".chart{max-width:900px;margin-bottom:24px;}"
        "table{border-collapse:collapse;font-size:12px;}td,th{padding:4px 10px;border-bottom:1px solid #eee;text-align:left;}"
    )
    html = [
        f"<html><head><style>{style}</style></head><body>",
        f"<h1>Strongstock bounce review — {len(rows)} Grade-A trades + SOXL / MU / KORU comps</h1>",
        f"<p class='meta'>Sorted best-to-worst by open-to-close P&amp;L. Each chart is the 1-year daily candle window ending on the bounce day "
        f"(MAs: SMA200/100/50/10 + EMA9, grey dashed line = 1Y high).</p>",
        "<h2>Comparison candidates — current setup as of 2026-06-05</h2>",
    ]
    for cr in comp_rows:
        html.append(f"<h3>{cr['ticker']}</h3>")
        html.append(f"<img class='chart' src='cid:{cr['cid']}' /><br/>")

    html.append(f"<h2>Best strongstock bounces (Grade A, {len(rows)} trades)</h2>")
    # Quick summary table up top
    html.append("<table><tr><th>#</th><th>Ticker</th><th>Date</th><th>Setup</th><th>Cap</th>"
                "<th>Gap %</th><th>Open→Close %</th></tr>")
    for r in rows:
        html.append(
            f"<tr><td>{r['i']}</td><td>{r['ticker']}</td><td>{r['date']}</td>"
            f"<td>{r['setup']}</td><td>{r['cap']}</td>"
            f"<td>{r['gap_pct']:+.1f}%</td><td>{r['bounce_pct']:+.1f}%</td></tr>"
        )
    html.append("</table>")
    # Then full chart list
    for r in rows:
        html.append(
            f"<h3>{r['i']}. {r['ticker']} — {r['date']} ({r['setup']}, {r['cap']} cap)</h3>"
            f"<div class='meta'>Gap: {r['gap_pct']:+.2f}% &nbsp; Open→Close: {r['bounce_pct']:+.2f}%</div>"
            f"<img class='chart' src='cid:{r['cid']}' /><br/>"
        )
    html.append("</body></html>")
    body = "\n".join(html)

    subject = f"[Strongstock Review] {len(rows)} Grade-A bounces + SOXL/MU/KORU comps"
    send_email(
        to_email=TO_EMAIL,
        subject=subject,
        body=body,
        is_html=True,
        inline_images=inline_images,
    )
    log.info(f"Sent {len(rows)} trade charts + {len(comp_rows)} comp charts to {TO_EMAIL}")


if __name__ == '__main__':
    main()
