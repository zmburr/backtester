"""EQS Screener Report — pulls Bloomberg EQS screens, scores, and reports GO/CAUTION setups.

Pulls tickers from 3 Bloomberg EQS equity screens (parabolic short, bounce strong,
bounce weak), scores them through the existing pre-trade validators, filters to
GO + CAUTION only, and generates a focused HTML report with scoring, exit targets,
percentile rankings, and charts.

Sent via email to zmburr@gmail.com.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import datetime
import os
import shutil
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import scripts.generate_report as gr
from scanners import stock_screener as ss
from analyzers.bounce_scorer import BouncePretrade, fetch_bounce_metrics
from analyzers.charter import create_daily_chart, cleanup_charts
from support.config import send_email
from support.signal_ledger import log_signals, current_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s – %(message)s")
log = logging.getLogger(__name__)

_MAX_WORKERS = 8

# ---------------------------------------------------------------------------
# EQS Screen definitions — map screen name -> bucket + label
# ---------------------------------------------------------------------------
EQS_SCREENS = {
    "parabolic short screener": {
        "bucket": "reversal",
        "setup_type": None,
        "label": "Parabolic Short (EQS)",
    },
    "bounce screener -strong": {
        "bucket": "bounce",
        "setup_type": "GapFade_strongstock",
        "label": "Bounce Strong (EQS)",
    },
    "Bounce screener - weak": {
        "bucket": "bounce",
        "setup_type": "GapFade_weakstock",
        "label": "Bounce Weak (EQS)",
    },
}


# ---------------------------------------------------------------------------
# Bloomberg EQS pull
# ---------------------------------------------------------------------------

def pull_eqs_tickers() -> Dict[str, dict]:
    """Pull tickers from all EQS screens. Returns {ticker: screen_config}.

    Cleans Bloomberg tickers ("AAPL US" -> "AAPL"), deduplicates (first screen wins).
    Logs warnings for screens that fail and continues with the rest.
    """
    from xbbg import blp

    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    result: Dict[str, dict] = {}

    for screen_name, config in EQS_SCREENS.items():
        try:
            df = pd.DataFrame(blp.beqs(screen=screen_name, asof=today_str))
            if df.empty or "ticker" not in df.columns:
                print(f"  {screen_name}: 0 tickers")
                continue
            tickers = df["ticker"].str.replace(r"\s+\w+$", "", regex=True).tolist()
            added = 0
            for t in tickers:
                t = t.strip()
                if t and t not in result:
                    result[t] = config
                    added += 1
            print(f"  {screen_name}: {len(tickers)} raw -> {added} new")
        except Exception as e:
            log.warning(f"EQS screen '{screen_name}' failed: {e}")

    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_screener_report() -> str | None:
    """Pull EQS tickers, score, filter to GO/CAUTION, build and send report."""
    timings: Dict[str, float] = {}
    t_total = time.time()
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    charts_dir = "charts"
    Path(charts_dir).mkdir(exist_ok=True)

    # === Phase 0: Pull EQS tickers ===
    t0 = time.time()
    print("Phase 0: Pulling Bloomberg EQS screens...")
    ticker_map = pull_eqs_tickers()
    timings["eqs_pull"] = time.time() - t0

    if not ticker_map:
        print("No EQS hits today — exiting without report.")
        return None

    tickers = list(ticker_map.keys())
    print(f"  Total unique tickers: {len(tickers)}")

    # Install API response cache
    cache = gr.ReportCache()
    cache.install()

    try:
        # === Phase 1: Screener metrics ===
        t0 = time.time()
        print("Phase 1: Collecting screener metrics...")
        all_data = ss.get_all_stocks_data(tickers)
        timings["screener_metrics"] = time.time() - t0

        # Split tickers by bucket
        reversal_tickers = [t for t in tickers if ticker_map[t]["bucket"] == "reversal"]
        bounce_tickers = [t for t in tickers if ticker_map[t]["bucket"] == "bounce"]
        print(f"  Reversal: {len(reversal_tickers)} | Bounce: {len(bounce_tickers)}")

        # === Phase 2: Pre-trade metrics for reversal tickers ===
        t0 = time.time()
        print("Phase 2: Fetching reversal pretrade metrics (parallel)...")
        pretrade_metrics_all: Dict[str, Dict] = {}
        if reversal_tickers:
            with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
                futs = {executor.submit(gr.get_pretrade_metrics, t, today): t for t in reversal_tickers}
                for f in as_completed(futs):
                    tk = futs[f]
                    try:
                        pretrade_metrics_all[tk] = f.result()
                    except Exception as e:
                        log.warning(f"{tk}: pretrade metrics failed – {e}")
                        pretrade_metrics_all[tk] = {}
        timings["pretrade_metrics"] = time.time() - t0

        # === Phase 3: Bounce metrics for bounce tickers ===
        t0 = time.time()
        print("Phase 3: Fetching bounce metrics (parallel)...")
        bounce_metrics_all: Dict[str, Dict] = {}
        if bounce_tickers:
            with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
                futs = {executor.submit(fetch_bounce_metrics, t, today): t for t in bounce_tickers}
                for f in as_completed(futs):
                    tk = futs[f]
                    try:
                        bounce_metrics_all[tk] = f.result()
                    except Exception as e:
                        log.warning(f"{tk}: bounce metrics failed – {e}")
        timings["bounce_metrics"] = time.time() - t0

        # === Phase 4: Score + Filter ===
        t0 = time.time()
        print("Phase 4: Scoring all tickers...")
        scored: List[Dict] = []

        for ticker in tickers:
            config = ticker_map[ticker]
            bucket = config["bucket"]
            cap = gr.get_ticker_cap(ticker)

            if bucket == "reversal":
                pm = pretrade_metrics_all.get(ticker, {})
                score_result = gr.score_pretrade_setup(ticker, pm, cap=cap)
                rec = score_result["recommendation"]
                scored.append({
                    "ticker": ticker,
                    "bucket": bucket,
                    "cap": cap,
                    "rec": rec,
                    "score_result": score_result,
                    "metrics": pm,
                    "score_str": f"{score_result['score']}/{score_result['max_score']}",
                    "label": config["label"],
                })
            elif bucket == "bounce":
                bm = bounce_metrics_all.get(ticker, {})
                if bm:
                    # Supplement pct_from_200mav from screener data if missing
                    td = all_data.get(ticker, {})
                    mav_data = td.get("mav_data", {})
                    if bm.get("pct_from_200mav") is None and mav_data.get("pct_from_200mav") is not None:
                        bm["pct_from_200mav"] = mav_data["pct_from_200mav"]
                    checker = BouncePretrade()
                    bounce_result = checker.validate(ticker, bm, cap=cap)
                    rec = bounce_result.recommendation
                    scored.append({
                        "ticker": ticker,
                        "bucket": bucket,
                        "cap": cap,
                        "rec": rec,
                        "score_result": bounce_result,
                        "metrics": bm,
                        "score_str": f"{bounce_result.score}/{bounce_result.max_score}",
                        "label": config["label"],
                    })
                else:
                    scored.append({
                        "ticker": ticker,
                        "bucket": bucket,
                        "cap": cap,
                        "rec": "NO-GO",
                        "score_result": None,
                        "metrics": {},
                        "score_str": "N/A",
                        "label": config["label"],
                    })

        # Unified signal ledger: log EVERY scored signal (incl. NO-GO). The EQS
        # screen name (entry["label"]) is folded into the ledger's setup_type column.
        log_signals("eqs_screener", current_session(), scored)

        # Filter: keep only GO + CAUTION
        priority = [s for s in scored if s["rec"] in ("GO", "CAUTION")]
        priority.sort(key=lambda x: (0 if x["rec"] == "GO" else 1, x["score_str"]), reverse=False)

        go_ct = sum(1 for p in priority if p["rec"] == "GO")
        cau_ct = sum(1 for p in priority if p["rec"] == "CAUTION")
        nogo_ct = len(scored) - len(priority)
        print(f"Phase 4: {len(priority)} pass ({go_ct} GO, {cau_ct} CAUTION) from {len(scored)} screened ({nogo_ct} NO-GO dropped)")
        timings["scoring"] = time.time() - t0

        if not priority:
            print("No GO/CAUTION tickers from EQS screens — no report sent.")
            return None

        # === Phase 5: Build ticker HTML (parallel) ===
        t0 = time.time()
        print("Phase 5: Building ticker HTML sections (parallel)...")
        ticker_results: Dict[str, Tuple[str, list]] = {}
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = {}
            for item in priority:
                ticker = item["ticker"]
                bucket = item["bucket"]
                label = item["label"]
                futures[executor.submit(
                    gr._build_ticker_html,
                    ticker,
                    all_data.get(ticker, {}),
                    pretrade_metrics_all.get(ticker, {}),
                    bounce_metrics_all.get(ticker),
                    bucket_override=bucket,
                    bucket_reason_override=label,
                )] = ticker
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    ticker_results[ticker] = future.result()
                except Exception as e:
                    log.warning(f"Error building section for {ticker}: {e}")
                    ticker_results[ticker] = (f"<h2>{ticker}</h2><p>Error: {e}</p>", [], None)
        timings["build_html"] = time.time() - t0

        # === Phase 6: Charts (sequential — matplotlib not thread-safe) ===
        t0 = time.time()
        print("Phase 6: Generating charts (sequential)...")
        sections: List[str] = []
        for item in priority:
            ticker = item["ticker"]
            html, chart_hlines, _signal_row = ticker_results.get(ticker, ("", [], None))
            try:
                chart_path = Path(create_daily_chart(ticker, output_dir=charts_dir, extra_hlines=chart_hlines or None))
                img_tag = (
                    f'<img src="{gr._png_to_data_uri(chart_path)}" alt="{ticker} chart" '
                    f'style="max-width:800px; display:block; margin-top:10px; border-radius:4px;">'
                )
                html += "\n" + img_tag
            except Exception as e:
                log.warning(f"Chart failed for {ticker}: {e}")
                html += f"\n<p><em>Chart unavailable for {ticker}</em></p>"
            sections.append(html)
        timings["charts"] = time.time() - t0

        # === Phase 7: Assemble HTML ===
        t0 = time.time()
        date_str_display = datetime.datetime.now().strftime("%A, %B %d %Y  %I:%M %p")
        banner = (
            f'<div style="background-color: #3d2f0a; color: #e3b341; padding: 12px 14px; '
            f'font-size: 0.95em; border-radius: 4px 4px 0 0; margin-bottom: 16px; '
            f'border-bottom: 2px solid #e3b341;">'
            f'<strong>EQS Screener Report</strong> &nbsp;|&nbsp; '
            f'{go_ct} GO, {cau_ct} CAUTION from {len(scored)} screened &nbsp;|&nbsp; '
            f'{date_str_display} ET'
            f'</div>'
        )

        body_content = banner + "<br><br>\n".join(sections)
        html_report = (
            '<div style="font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', '
            'Arial, sans-serif; max-width: 860px; margin: 0 auto; color: #c9d1d9; '
            'background-color: #161b22; font-size: 14px; line-height: 1.5; padding: 16px;">'
            + body_content
            + '</div>'
        )

        # === Phase 8: Save PDF ===
        pdf_path = _save_screener_pdf(html_report)
        if pdf_path:
            print(f"PDF saved: {pdf_path}")

        # === Phase 9: Send email ===
        date_email = datetime.datetime.now().strftime("%m/%d/%Y")
        subject = f"EQS Screener Report — {go_ct} GO, {cau_ct} CAUTION | {date_email}"
        try:
            send_email(
                to_email="zmburr@gmail.com",
                subject=subject,
                body=html_report,
                is_html=True,
            )
            print(f"Email sent: {subject}")
        except Exception as e:
            log.error(f"Failed to send email: {e}")

        timings["assemble_send"] = time.time() - t0

    finally:
        cache.uninstall()

    timings["total"] = time.time() - t_total

    print("\n" + "=" * 60)
    print("EQS SCREENER REPORT TIMING")
    print("=" * 60)
    for phase, elapsed in timings.items():
        print(f"  {phase:30s} {elapsed:6.1f}s")
    print("=" * 60)

    return html_report


# ---------------------------------------------------------------------------
# PDF export (mirrors gr._save_report_pdf with screener-specific filename)
# ---------------------------------------------------------------------------

def _save_screener_pdf(html_report: str, output_dir: str = "reports") -> str | None:
    """Save HTML report as screener_report_YYYYMMDD_HHMMSS.pdf."""
    Path(output_dir).mkdir(exist_ok=True)
    filename = f"screener_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    path = Path(output_dir) / filename

    # Reuse generate_report's PDF backends
    try:
        from weasyprint import HTML as WeasyprintHTML
        WeasyprintHTML(string=html_report).write_pdf(str(path))
        return str(path)
    except (ImportError, OSError):
        pass

    try:
        import pdfkit
        wk_path = os.environ.get("WKHTMLTOPDF_PATH")
        if not wk_path:
            default = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
            if os.path.isfile(default):
                wk_path = default
            else:
                wk_path = shutil.which("wkhtmltopdf")
        if wk_path:
            config = pdfkit.configuration(wkhtmltopdf=wk_path)
            pdfkit.from_string(html_report, str(path), configuration=config)
            return str(path)
        else:
            pdfkit.from_string(html_report, str(path))
            return str(path)
    except Exception as e:
        log.warning(f"PDF export failed: {e}")

    return None


if __name__ == "__main__":
    generate_screener_report()
    cleanup_charts()
    print("EQS screener report complete.")
