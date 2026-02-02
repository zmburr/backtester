"""Generate a textual watch-list report plus daily charts.

The report logic re-uses helper functions defined in *scanners.stock_screener* so we don't
repeat code.  It produces a giant string suitable for plain-text e-mail (or logging) and
saves one PNG chart per ticker in the *charts/* directory.
"""

from pathlib import Path
from textwrap import indent
from typing import List, Dict, Optional
from support.config import send_email
import pandas as pd
import base64
import datetime
import os
import shutil
import logging

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
from data_queries.polygon_queries import get_levels_data, get_daily, get_atr

# -------------------------------------------------
# Pre-Trade Reversal Scoring (5 of 6 criteria - excludes reversal_pct since pre-trade)
# -------------------------------------------------

# Cap-adjusted thresholds for pre-trade screening
# Same as reversal_scorer.py but without reversal_pct
PRETRADE_THRESHOLDS = {
    'Micro': {'pct_from_9ema': 0.80, 'prior_day_range_atr': 3.0, 'rvol_score': 2.0, 'consecutive_up_days': 3, 'gap_pct': 0.15},
    'Small': {'pct_from_9ema': 0.40, 'prior_day_range_atr': 2.0, 'rvol_score': 2.0, 'consecutive_up_days': 2, 'gap_pct': 0.10},
    'Medium': {'pct_from_9ema': 0.15, 'prior_day_range_atr': 1.0, 'rvol_score': 1.5, 'consecutive_up_days': 2, 'gap_pct': 0.05},
    'Large': {'pct_from_9ema': 0.08, 'prior_day_range_atr': 0.8, 'rvol_score': 1.0, 'consecutive_up_days': 1, 'gap_pct': 0.00},
    'ETF': {'pct_from_9ema': 0.04, 'prior_day_range_atr': 1.0, 'rvol_score': 1.5, 'consecutive_up_days': 1, 'gap_pct': 0.00},
}

# Known ETFs (type detection from Polygon can be unreliable)
KNOWN_ETFS = {'GLD', 'SLV', 'GDXJ', 'QQQ', 'SPY', 'XLE', 'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU',
              'IWM', 'DIA', 'VTI', 'VOO', 'VXX', 'UVXY', 'SQQQ', 'TQQQ', 'SPXU', 'SPXL', 'TLT', 'HYG', 'LQD',
              'EEM', 'EWZ', 'EWJ', 'FXI', 'KWEB', 'SMH', 'XBI', 'IBB', 'ARKK', 'ARKG'}

# Cache for market cap lookups (avoid repeated API calls)
_market_cap_cache: Dict[str, str] = {}


def get_ticker_cap(ticker: str) -> str:
    """
    Get market cap category for a ticker by fetching from Polygon API.

    Categories:
        - Large: >= $100B
        - Medium: $2B - $100B
        - Small: $300M - $2B
        - Micro: < $300M
        - ETF: Exchange-traded funds
    """
    # Check cache first
    if ticker in _market_cap_cache:
        return _market_cap_cache[ticker]

    # Check known ETFs
    if ticker.upper() in KNOWN_ETFS:
        _market_cap_cache[ticker] = 'ETF'
        return 'ETF'

    try:
        from polygon.rest import RESTClient
        client = RESTClient('pcwUY7TnSF66nYAPIBCApPMyVrXTckJY')

        details = client.get_ticker_details(ticker)

        # Check if ETF by type
        if hasattr(details, 'type') and details.type in ('ETF', 'ETN'):
            _market_cap_cache[ticker] = 'ETF'
            return 'ETF'

        mc = details.market_cap

        if mc is None:
            cap = 'Medium'  # Default if no market cap data
        elif mc >= 100_000_000_000:  # >= $100B
            cap = 'Large'
        elif mc >= 2_000_000_000:    # >= $2B
            cap = 'Medium'
        elif mc >= 300_000_000:      # >= $300M
            cap = 'Small'
        else:
            cap = 'Micro'

        _market_cap_cache[ticker] = cap
        logging.info(f"{ticker}: Market cap ${mc/1e9:.1f}B -> {cap}")
        return cap

    except Exception as e:
        logging.warning(f"Failed to get market cap for {ticker}: {e}, defaulting to Medium")
        _market_cap_cache[ticker] = 'Medium'
        return 'Medium'


def get_pretrade_metrics(ticker: str, date: str) -> Dict:
    """
    Fetch additional metrics needed for pre-trade scoring that aren't in stock_screener.
    Returns: dict with pct_from_9ema, consecutive_up_days, gap_pct, prior_day_range_atr, rvol_score
    """
    metrics = {}
    try:
        # Get historical data (35 days for calculations)
        df = get_levels_data(ticker, date, 35, 1, 'day')
        if df is None or len(df) < 5:
            return metrics

        # Calculate 9-day EMA
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()

        # Get prior day values (second to last row since last row is "today")
        if len(df) >= 2:
            prior_close = df['close'].iloc[-2]
            prior_high = df['high'].iloc[-2]
            prior_low = df['low'].iloc[-2]
            prior_ema9 = df['ema_9'].iloc[-2]
            today_open = df['open'].iloc[-1]

            # 1. Distance from 9EMA
            if prior_ema9 and prior_ema9 > 0:
                metrics['pct_from_9ema'] = (prior_close - prior_ema9) / prior_ema9

            # 2. Gap % (today's open vs prior close)
            if prior_close and prior_close > 0:
                metrics['gap_pct'] = (today_open - prior_close) / prior_close

            # 3. Prior day range as multiple of ATR
            prior_range = prior_high - prior_low
            atr = get_atr(ticker, date)
            if atr and atr > 0:
                metrics['prior_day_range_atr'] = prior_range / atr

            # 4. Consecutive up days
            consecutive_up = 0
            for i in range(len(df) - 2, -1, -1):  # Start from prior day going back
                if df['close'].iloc[i] > df['open'].iloc[i]:
                    consecutive_up += 1
                else:
                    break
            metrics['consecutive_up_days'] = consecutive_up

            # 5. RVOL (volume vs 20-day average)
            if len(df) >= 20:
                avg_vol = df['volume'].iloc[-21:-1].mean()  # 20 days before today
                today_vol = df['volume'].iloc[-1]
                if avg_vol and avg_vol > 0:
                    metrics['rvol_score'] = today_vol / avg_vol

    except Exception as e:
        logging.warning(f"Error getting pretrade metrics for {ticker}: {e}")

    return metrics


def score_pretrade_setup(ticker: str, metrics: Dict, cap: str = None) -> Dict:
    """
    Score a potential reversal setup using 5 pre-trade criteria.

    Returns dict with:
        - score: 0-5
        - max_score: 5
        - recommendation: GO/CAUTION/NO-GO
        - criteria: list of (name, passed, threshold, actual) tuples
    """
    if cap is None:
        cap = get_ticker_cap(ticker)

    if cap not in PRETRADE_THRESHOLDS:
        cap = 'Medium'

    thresholds = PRETRADE_THRESHOLDS[cap]

    criteria = []
    score = 0

    # Check each criterion
    criteria_checks = [
        ('9EMA Distance', 'pct_from_9ema', thresholds['pct_from_9ema'], False),
        ('Range (ATR)', 'prior_day_range_atr', thresholds['prior_day_range_atr'], False),
        ('RVOL', 'rvol_score', thresholds['rvol_score'], False),
        ('Consec Up Days', 'consecutive_up_days', thresholds['consecutive_up_days'], False),
        ('Gap Up', 'gap_pct', thresholds['gap_pct'], False),
    ]

    for name, key, threshold, _ in criteria_checks:
        actual = metrics.get(key)
        if actual is not None and not pd.isna(actual):
            passed = actual >= threshold
        else:
            passed = False
            actual = None

        if passed:
            score += 1

        criteria.append({
            'name': name,
            'key': key,
            'passed': passed,
            'threshold': threshold,
            'actual': actual,
        })

    # Determine recommendation (adjusted for 5 criteria)
    if score >= 4:
        recommendation = 'GO'
    elif score == 3:
        recommendation = 'CAUTION'
    else:
        recommendation = 'NO-GO'

    return {
        'ticker': ticker,
        'cap': cap,
        'score': score,
        'max_score': 5,
        'recommendation': recommendation,
        'criteria': criteria,
    }


# Historical performance statistics by score (from 61 Grade A trades analysis)
SCORE_STATISTICS = {
    5: {'trades': 25, 'win_rate': 100.0, 'avg_pnl': 13.3},
    4: {'trades': 20, 'win_rate': 85.0, 'avg_pnl': 15.9},
    3: {'trades': 6, 'win_rate': 83.3, 'avg_pnl': 10.5},
    2: {'trades': 1, 'win_rate': 0.0, 'avg_pnl': -4.6},
    1: {'trades': 0, 'win_rate': 0.0, 'avg_pnl': 0.0},
    0: {'trades': 0, 'win_rate': 0.0, 'avg_pnl': 0.0},
}


def format_pretrade_score_html(score_result: Dict) -> str:
    """Format pre-trade score as HTML for the report."""
    rec = score_result['recommendation']
    score = score_result['score']
    cap = score_result['cap']

    # Color coding
    if rec == 'GO':
        color = '#28a745'  # green
        rec_text = 'GO'
    elif rec == 'CAUTION':
        color = '#ffc107'  # yellow
        rec_text = 'CAUTION'
    else:
        color = '#dc3545'  # red
        rec_text = 'NO-GO'

    # Get historical stats for this score
    stats = SCORE_STATISTICS.get(score, {'trades': 0, 'win_rate': 0.0, 'avg_pnl': 0.0})
    stats_text = f"Historical: {stats['win_rate']:.0f}% win rate, {stats['avg_pnl']:+.1f}% avg P&L (n={stats['trades']})"

    lines = [
        f'<div style="border: 2px solid {color}; padding: 10px; margin: 10px 0; border-radius: 5px;">',
        f'<strong style="color: {color}; font-size: 1.2em;">{rec_text}</strong> ',
        f'<span>Score: {score}/5 ({cap} Cap)</span>',
        f'<br><span style="font-size: 0.85em; color: #666;">{stats_text}</span>',
        '<table style="margin-top: 8px; font-size: 0.9em;">',
    ]

    for c in score_result['criteria']:
        status = '✓' if c['passed'] else '✗'
        status_color = '#28a745' if c['passed'] else '#dc3545'

        # Format values
        if c['actual'] is not None:
            if c['key'] in ['pct_from_9ema', 'gap_pct']:
                actual_str = f"{c['actual']*100:.1f}%"
                thresh_str = f"{c['threshold']*100:.0f}%"
            elif c['key'] == 'consecutive_up_days':
                actual_str = f"{int(c['actual'])}"
                thresh_str = f"{int(c['threshold'])}"
            else:
                actual_str = f"{c['actual']:.1f}x"
                thresh_str = f"{c['threshold']:.1f}x"
        else:
            actual_str = 'N/A'
            thresh_str = f"{c['threshold']}"

        lines.append(
            f'<tr><td style="color: {status_color}; padding-right: 10px;">{status}</td>'
            f'<td style="padding-right: 10px;">{c["name"]}</td>'
            f'<td style="padding-right: 10px;">≥{thresh_str}</td>'
            f'<td>{actual_str}</td></tr>'
        )

    lines.append('</table></div>')
    return '\n'.join(lines)

# Columns we want to compare (same list used inside stock_screener)
COLUMNS_TO_COMPARE = ss.columns_to_compare

# ---------------------------------------------
# Report header (trading rules & daily checklist)
# ---------------------------------------------

HEADER_HTML = """<h1 style="text-align:center;">Daily Trading Rules & Checklist</h1>

<h2>Reversal Setup Scoring Guide</h2>
<p>Each stock is scored on <strong>5 pre-trade criteria</strong> (cap-adjusted thresholds):</p>
<ol>
  <li><strong>9EMA Distance</strong> - Price elevated above 9-day EMA</li>
  <li><strong>Range (ATR)</strong> - Prior day range vs ATR</li>
  <li><strong>RVOL</strong> - Volume vs 20-day average</li>
  <li><strong>Consecutive Up Days</strong> - Momentum into the top</li>
  <li><strong>Gap Up</strong> - Gap up on reversal day</li>
</ol>

<h3>Historical Performance by Score (61 Grade A Trades)</h3>
<table border="1" cellpadding="8" style="border-collapse: collapse; margin: 10px 0;">
<tr style="background-color: #f0f0f0;"><th>Score</th><th>Trades</th><th>Win Rate</th><th>Avg P&L</th><th>Recommendation</th></tr>
<tr style="background-color: #d4edda;"><td><strong>5/5</strong></td><td>25</td><td>100%</td><td>+13.3%</td><td style="color: #28a745;"><strong>GO</strong></td></tr>
<tr style="background-color: #d4edda;"><td><strong>4/5</strong></td><td>20</td><td>85%</td><td>+15.9%</td><td style="color: #28a745;"><strong>GO</strong></td></tr>
<tr style="background-color: #fff3cd;"><td><strong>3/5</strong></td><td>6</td><td>83%</td><td>+10.5%</td><td style="color: #ffc107;"><strong>CAUTION</strong></td></tr>
<tr style="background-color: #f8d7da;"><td><strong>&lt;3</strong></td><td>1</td><td>0%</td><td>-4.6%</td><td style="color: #dc3545;"><strong>NO-GO</strong></td></tr>
</table>

<p><strong style="color: #28a745;">GO (4-5/5)</strong>: 45 trades, 93% win rate, +14.5% avg |
<strong style="color: #ffc107;">CAUTION (3/5)</strong>: 6 trades, 83% win, +10.5% avg |
<strong style="color: #dc3545;">NO-GO (&lt;3)</strong>: Skip</p>
<hr>

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
  <li>Expected Value over First Prints - Push size in your bread and butter</li>
  <li>Every single trade was not within .2% of reference after a minute unless it was a dissem issue</li>
  <li>Single stocks - 50% of them last 21.5 minutes - On my biggest trades - 50% = 35 mins</li>
  <li>If it breaks upper or lower bound trend- hold until it fails trend as it's a pos signal (good RRR to see if it goes para)</li>
</ol>

<h2>News Rules / Reminders</h2>
<ol>
  <li>CP on canada deal with US / CAR on any car tariff changes / STZ+EWW or TNA on Mexico / XLE short / MT LONG / KYIV on Russia Deal</li>
  <li>XRT SWK for Trump tariffs</li>
  <li>SILJ PAAS on Silver </li>

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

def _generate_ticker_section(ticker: str, data: dict, charts_dir: str, pretrade_metrics: dict = None) -> str:
    """Return formatted string section for one ticker and create its chart."""

    rev_pcts = ss.calculate_percentiles(reversal_df, data, COLUMNS_TO_COMPARE)
    mom_pcts = ss.calculate_percentiles(momentum_df, data, COLUMNS_TO_COMPARE)

    pct_data = data.get("pct_data", {})
    range_data = data.get("range_data", {})
    mav_data = data.get("mav_data", {})

    # Build section text
    lines: List[str] = [f"<h2>Ticker: {ticker}</h2>"]

    # Add pre-trade reversal scoring if metrics available
    if pretrade_metrics:
        score_result = score_pretrade_setup(ticker, pretrade_metrics)
        lines.append("<strong>Reversal Setup Score:</strong>")
        lines.append(format_pretrade_score_html(score_result))

    lines.append("<strong>Reversal Percentiles:</strong>")
    lines.append(_format_percentile_dict(rev_pcts))
    lines.append("<strong>Momentum Percentiles:</strong>")
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
    try:
        chart_path = Path(create_daily_chart(ticker, output_dir=charts_dir))
        img_tag = f'<img src="{_png_to_data_uri(chart_path)}" alt="{ticker} chart" style="max-width:800px;">'
        lines.append(img_tag)
    except Exception as e:
        print(f"Failed to generate chart for {ticker}: {e}")
        lines.append(f"<p><em>Chart unavailable for {ticker}</em></p>")

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

    # Collect pre-trade metrics for reversal scoring
    print("Fetching pre-trade reversal metrics...")
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    pretrade_metrics_all = {}
    for ticker in watchlist:
        try:
            pretrade_metrics_all[ticker] = get_pretrade_metrics(ticker, today)
            print(f"  {ticker}: {pretrade_metrics_all[ticker]}")
        except Exception as e:
            print(f"  {ticker}: Failed to get pretrade metrics - {e}")
            pretrade_metrics_all[ticker] = {}

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
        _generate_ticker_section(ticker, all_data[ticker], charts_dir, pretrade_metrics_all.get(ticker, {}))
        for ticker in sorted_watchlist
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