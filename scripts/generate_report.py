"""Generate a textual watch-list report plus daily charts.

The report logic re-uses helper functions defined in *scanners.stock_screener* so we don't
repeat code.  It produces a giant string suitable for plain-text e-mail (or logging) and
saves one PNG chart per ticker in the *charts/* directory.
"""

from pathlib import Path
from textwrap import indent
from typing import List
from support.config import send_email
import pandas as pd
import base64
import datetime
import os
import shutil

try:
    from weasyprint import HTML
    _PDF_AVAILABLE = True
except (ImportError, OSError):
    # Import failed either because WeasyPrint isn't installed or its runtime
    # dependencies (e.g., GTK/Pango/Cairo) are missing on this system.
    _PDF_AVAILABLE = False
    HTML = None  # type: ignore

# Try pdfkit / wkhtmltopdf as a lighter-weight alternative (single exe on Windows)
try:
    import pdfkit  # type: ignore
    _PDFKIT_AVAILABLE = True
except ImportError:
    _PDFKIT_AVAILABLE = False

# Resolve wkhtmltopdf location once so we can reuse it
_WKHTMLTOPDF_PATH = os.getenv("WKHTMLTOPDF_PATH")  # user override

if not _WKHTMLTOPDF_PATH:
    # Try typical Windows install location
    default_win_path = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
    if os.path.isfile(default_win_path):
        _WKHTMLTOPDF_PATH = default_win_path
    else:
        # Fallback to `which` / `where` search in PATH
        _WKHTMLTOPDF_PATH = shutil.which("wkhtmltopdf")

# None -> still unresolved

from scanners import stock_screener as ss
from data_collectors.combined_data_collection import reversal_df, momentum_df
from analyzers.charter import create_daily_chart, cleanup_charts

# Columns we want to compare (same list used inside stock_screener)
COLUMNS_TO_COMPARE = ss.columns_to_compare

# ---------------------------------------------
# Report header (trading rules & daily checklist)
# ---------------------------------------------

HEADER_HTML = """<h1 style="text-align:center;">Daily Trading Rules & Checklist</h1>

<h2>Rules</h2>
<ol>
  <li>Quality in everything – end day with quality & take breaks to maintain quality</li>
  <li>Push size in liquid names</li>
  <li>Start orderpipe</li>
  <li>Start cnbc</li>
  <li>It's ok to consciously risk 30-40K on bread and butter / ETF aggression</li>
  <li>Let the upside take care of itself</li>
  <li>Selectivity - trust your instincts - reactive trades always best (don't change tiers)</li>
  <li>Use the 2-minute bar for high volume good news / 1 min for scalp - after VOLUME</li>
  <li>Liquidity focus</li>
  <li>Who gets paid? → That's my trade.</li>
  <li>Avoid Swing trading if mental perception is not performing well at work</li>
  <li>Expected Value over First Prints - Push size in your bread and butter</li>
  <li>Every single trade was not within .2% of reference after a minute unless it was a dissem issue</li>
  <li>Single stocks - 50% of them last 21.5 minutes - On my biggest trades - 50% = 35 mins</li>
  <li>If it breaks upper or lower bound trend- hold until it fails trend as it's a pos signal (good RRR to see if it goes para)</li>
</ol>

<h2>News Rules / Reminders</h2>
<ol>
  <li>CP on canada deal with US / CAR on any car tariff changes / STZ+EWW or TNA on Mexico / XLE short / MT LONG / KYIV on Russia Deal</li>
  <li>XRT SHW for Trump tariffs</li>
  <li>IBIT on Rieder / Short market on Warsh</li>

</ol>

<h2>Morning Checklist</h2>
<ul>
  <li>Read overnight news</li>
  <li>Look at all stocks gapping up or down 5 %+ (Stockfetcher, MAT, NLRTs)</li>
  <li>Go through rules and reminders</li>
  <li>Check Trump schedule</li>
  <li>Create one explicit process-oriented goal for the day</li>
  <li>Go through all events / ECO for the day – call out important ones to group</li>
  <li>Write down any tasks you want to accomplish today</li>
</ul>"""

# Custom ordering for percentile keys
PERCENTILE_ORDER = [
    "pct_change_120",
    "pct_change_90",
    "pct_change_30",
    "pct_change_15",
    "pct_change_3",
    "pct_from_10mav", "pct_from_20mav", "pct_from_50mav", "pct_from_200mav",
]

def _format_percentile_dict(d: dict) -> str:
    """Return nicely-formatted percentile dict respecting the desired order."""
    if not d:
        return "    None"
    lines: List[str] = []
    # Add keys in the specified order first
    for key in PERCENTILE_ORDER:
        if key in d:
            lines.append(f"    {key}: {d[key]:.1f}")
    # Append any remaining (non-PCT) keys alphabetically
    for key in sorted(k for k in d if k not in PERCENTILE_ORDER):
        lines.append(f"    {key}: {d[key]:.1f}")
    return "\n".join(lines)

# +++++ NEW helper ++++++++++++++++++++++++++++++++++++++++++++++++++++++
def _fmt_pct(val):
    """
    Convert a fractional distance-from-MA (e.g. 0.042) to a percentage
    string like '4.2%'.  Falls back to str(val) if conversion fails.
    """
    try:
        return f"{float(val) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(val)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def _generate_ticker_section(ticker: str, data: dict, charts_dir: str) -> str:
    """Return formatted string section for one ticker and create its chart."""

    rev_pcts = ss.calculate_percentiles(reversal_df, data, COLUMNS_TO_COMPARE)
    mom_pcts = ss.calculate_percentiles(momentum_df, data, COLUMNS_TO_COMPARE)

    pct_data = data.get("pct_data", {})
    range_data = data.get("range_data", {})
    mav_data = data.get("mav_data", {})

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

    if mav_data:
        lines.append("Distance from Moving Averages:")
        # prettier labels: “10 MA” instead of “pct_from_10mav”
        label = lambda k: k.removeprefix("pct_from_").removesuffix("mav") + " MA"
        lines.append(indent(
            "\n".join(f"{label(k)}: {_fmt_pct(v)}" for k, v in mav_data.items()),
            "    "
        ))

    # Generate chart and embed inline
    chart_path = Path(create_daily_chart(ticker, output_dir=charts_dir))
    img_tag = f'<img src="{_png_to_data_uri(chart_path)}" alt="{ticker} chart" style="max-width:800px;">'
    lines.append(img_tag)

    # Return HTML block
    return "<br>\n".join(lines)


def _png_to_data_uri(path: Path) -> str:
    """Return a data URI string for embedding PNG inline."""
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{encoded}"

# -------------------------------------------------
# PDF export helper
# -------------------------------------------------

def _save_report_pdf(html_report: str, output_dir: str = "reports") -> str | None:
    """Save the given HTML report to *output_dir* as a timestamped PDF.

    Uses WeasyPrint if available (pip install weasyprint).  If WeasyPrint is
    not installed the function prints a warning and does nothing.
    Returns the path of the written file or *None* if skipped/failed.
    """

    if not _PDF_AVAILABLE and not _PDFKIT_AVAILABLE:
        print("No PDF backend available (install `weasyprint` or `pdfkit` + wkhtmltopdf); skipping PDF export.")
        return None

    Path(output_dir).mkdir(exist_ok=True)
    filename = f"watchlist_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    path = Path(output_dir) / filename

    if _PDF_AVAILABLE:
        try:
            HTML(string=html_report).write_pdf(str(path))
            print(f"PDF report saved to {path} using WeasyPrint")
            return str(path)
        except Exception as e:
            print(f"WeasyPrint failed ({e}). Attempting pdfkit fallback...")

    # Fallback to pdfkit if available
    if _PDFKIT_AVAILABLE:
        if not _WKHTMLTOPDF_PATH or not os.path.isfile(_WKHTMLTOPDF_PATH):
            print("wkhtmltopdf executable not found. Set WKHTMLTOPDF_PATH environment variable to its location.")
        try:
            # pdfkit requires wkhtmltopdf binary; if not on PATH you can set pdfkit.configuration()
            try:
                # First try default invocation (assumes binary on PATH)
                pdfkit.from_string(html_report, str(path))
            except (OSError, IOError):
                # Fallback to explicit path if provided or default Windows install location
                if _WKHTMLTOPDF_PATH:
                    config = pdfkit.configuration(wkhtmltopdf=_WKHTMLTOPDF_PATH)
                    pdfkit.from_string(html_report, str(path), configuration=config)
                else:
                    raise
            print(f"PDF report saved to {path} using pdfkit/wkhtmltopdf")
            return str(path)
        except Exception as e:
            print(f"pdfkit failed to write PDF: {e}")

    return None

#TODO - add historical context to the report - "largest volume day since xyz" or "highest PCT change since abc"
#TODO - add news overnight from context and google into llm summary to report.
def generate_report() -> str:
    """Return a giant string report and create all charts under *charts/*."""

    watchlist = ss.watchlist
    charts_dir = "charts"  # same default as charter
    Path(charts_dir).mkdir(exist_ok=True)

    # Collect new metrics for watchlist tickers
    all_data = ss.get_all_stocks_data(watchlist)

    # ---- NEW: sort by distance from 10-day MA (descending) -----------------
    def _safe_float(x, default=float('-inf')):
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    def _mav10_key(ticker: str) -> float:
        return _safe_float((all_data.get(ticker, {}).get("mav_data") or {}).get("pct_from_10mav"))

    # If you prefer absolute distance (above or below), use key=lambda t: abs(_mav10_key(t))
    sorted_watchlist = sorted(watchlist, key=_mav10_key, reverse=True)
    # -----------------------------------------------------------------------

    sections = [
        _generate_ticker_section(ticker, all_data[ticker], charts_dir) for ticker in sorted_watchlist
    ]

    # Prepend header information
    html_report = HEADER_HTML + "<hr>" + "<br><br>\n".join(sections)

    # Persist a copy of the report locally before (or even if) e-mailing
    _save_report_pdf(html_report)

    # Send email as HTML
    send_email(
        to_email="zburr@trlm.com",
        subject="Daily Watchlist Report",
        body=html_report,
        is_html=True,
    )

    return html_report

def project_choice():
    send_email(
        to_email="zburr@trlm.com",
        subject="Your One Current Focus Project",
        body="""Jupiter improvements - focus on making the pipeline cleaner, should review the code and understand it.""",
        is_html=True,
    )
if __name__ == "__main__":
    rep = generate_report()
    # project_choice()
    # Print plain-text fallback (strip HTML tags) if desired
    print("Report generated, saved, and (attempted) e-mailed.")
    cleanup_charts() 