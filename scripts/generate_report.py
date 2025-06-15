"""Generate a textual watch-list report plus daily charts.

The report logic re-uses helper functions defined in *scanners.stock_screener* so we don't
repeat code.  It produces a giant string suitable for plain-text e-mail (or logging) and
saves one PNG chart per ticker in the *charts/* directory.
"""

from pathlib import Path
from textwrap import indent
from typing import List
from support.llm_client import llm
from support.config import send_email
import pandas as pd
import os

from scanners import stock_screener as ss
from data_collectors.combined_data_collection import reversal_df, momentum_df
from analyzers.charter import create_daily_chart, cleanup_charts

# Columns we want to compare (same list used inside stock_screener)
COLUMNS_TO_COMPARE = ss.columns_to_compare


def _format_percentile_dict(d: dict) -> str:
    if not d:
        return "    None"
    lines = [f"    {k}: {v:.1f}" for k, v in sorted(d.items())]
    return "\n".join(lines)


def _generate_ticker_section(ticker: str, data: dict, charts_dir: str) -> str:
    """Return formatted string section for one ticker and create its chart."""

    rev_pcts = ss.calculate_percentiles(reversal_df, data, COLUMNS_TO_COMPARE)
    mom_pcts = ss.calculate_percentiles(momentum_df, data, COLUMNS_TO_COMPARE)

    pct_data = data.get("pct_data", {})
    range_data = data.get("range_data", {})

    # Build section text
    lines: List[str] = [f"Ticker: {ticker}"]

    lines.append("Reversal Percentiles:")
    lines.append(_format_percentile_dict(rev_pcts))
    lines.append("Momentum Percentiles:")
    lines.append(_format_percentile_dict(mom_pcts))

    def _fmt(val):
        try:
            return f"{float(val):.2f}"
        except (TypeError, ValueError):
            return str(val)

    if pct_data:
        lines.append("Absolute PCT Changes:")
        lines.append(indent("\n".join([f"{k}: {_fmt(v)}" for k, v in pct_data.items()]), "    "))

    if range_data:
        lines.append("Range Data:")
        lines.append(indent("\n".join([f"{k}: {_fmt(v)}" for k, v in range_data.items()]), "    "))

    # Generate chart
    chart_path = create_daily_chart(ticker, output_dir=charts_dir)
    lines.append(f"Chart: {chart_path}")

    return "\n".join(lines)


def generate_report() -> str:
    """Return a giant string report and create all charts under *charts/*."""

    watchlist = ss.watchlist
    charts_dir = "charts"  # same default as charter
    Path(charts_dir).mkdir(exist_ok=True)

    # Collect new metrics for watchlist tickers
    all_data = ss.get_all_stocks_data(watchlist)

    sections = [
        _generate_ticker_section(ticker, all_data[ticker], charts_dir) for ticker in watchlist
    ]

    report = "\n\n".join(sections)

    # Collect chart paths for attachments
    attachments = list(Path(charts_dir).glob("*.png"))

    # Send email (configure recipient & subject as needed)
    try:
        send_email(
            to_email=os.getenv("REPORT_RECIPIENT", "zburr@trlm.com"),
            subject="Daily Watchlist Report",
            body=report,
            attachments=[str(p) for p in attachments],
        )
    except Exception as e:
        print(f"Email send failed: {e}")

    return report


if __name__ == "__main__":
    rep = generate_report()
    print(rep)
    # Uncomment this after e-mailing if you want to remove chart files automatically
    cleanup_charts()