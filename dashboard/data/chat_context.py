"""
System prompt builder for the Trading Chat interface.

Assembles context about strategies, data stats, trading memory,
and report data into a system prompt for Claude.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRADING_MEMORY_PATH = r"C:\Users\zmbur\PycharmProjects\orderPipe\analytics\trading_memory.md"


def _load_csv_stats(csv_path: Path, pnl_col: str, is_short: bool = False) -> dict:
    """Load a CSV and compute summary stats."""
    try:
        df = pd.read_csv(csv_path).dropna(subset=['ticker', 'date'])
        pnl = pd.to_numeric(df.get(pnl_col), errors='coerce').dropna()
        if is_short:
            pnl = -pnl
        pnl_pct = pnl * 100
        total = len(df)
        wins = int((pnl_pct > 0).sum())
        win_rate = round((pnl_pct > 0).mean() * 100, 1) if len(pnl_pct) > 0 else 0
        avg_pnl = round(pnl_pct.mean(), 1) if len(pnl_pct) > 0 else 0

        # By setup/cap breakdown
        by_cap = {}
        if 'cap' in df.columns:
            for cap in df['cap'].dropna().unique():
                sub = df[df['cap'] == cap]
                sub_pnl = pd.to_numeric(sub.get(pnl_col), errors='coerce').dropna()
                if is_short:
                    sub_pnl = -sub_pnl
                sub_pnl_pct = sub_pnl * 100
                if len(sub_pnl_pct) > 0:
                    by_cap[str(cap)] = {
                        'trades': len(sub),
                        'win_rate': round((sub_pnl_pct > 0).mean() * 100, 1),
                        'avg_pnl': round(sub_pnl_pct.mean(), 1),
                    }

        return {
            'total': total, 'wins': wins, 'win_rate': win_rate,
            'avg_pnl': avg_pnl, 'by_cap': by_cap,
        }
    except Exception as e:
        log.warning(f"Error loading {csv_path}: {e}")
        return {'total': 0, 'wins': 0, 'win_rate': 0, 'avg_pnl': 0, 'by_cap': {}}


def _load_bounce_setup_stats() -> str:
    """Load bounce_data.csv and produce setup-type level stats."""
    try:
        from analyzers.bounce_scorer import classify_from_setup_column
        df = pd.read_csv(REPO_ROOT / "data" / "bounce_data.csv").dropna(subset=['ticker', 'date'])
        df['pnl'] = pd.to_numeric(df.get('bounce_open_close_pct'), errors='coerce') * 100
        df['_profile'] = df['Setup'].apply(classify_from_setup_column)
        df = df[df['_profile'] != 'IntradayCapitch']

        lines = []
        for profile in ['GapFade_weakstock', 'GapFade_strongstock']:
            sub = df[df['_profile'] == profile]
            if len(sub) == 0:
                continue
            pnl = sub['pnl'].dropna()
            wr = round((pnl > 0).mean() * 100, 1) if len(pnl) > 0 else 0
            avg = round(pnl.mean(), 1) if len(pnl) > 0 else 0
            lines.append(f"  - {profile}: {len(sub)} trades, {wr}% WR, {avg:+.1f}% avg P&L")
        return "\n".join(lines)
    except Exception as e:
        return f"  (Could not load bounce setup stats: {e})"


def _load_trading_memory() -> str:
    """Load trading memory/rules file."""
    try:
        with open(TRADING_MEMORY_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        return content[:3000]
    except FileNotFoundError:
        return "(trading_memory.md not found)"
    except Exception as e:
        return f"(Error loading trading memory: {e})"


def _format_report_summary(report_data) -> str:
    """Format current report data for the system prompt with full detail."""
    if not report_data:
        return "No report data loaded for today."

    lines = []

    for r in report_data:
        ticker_lines = [f"\n--- {r.ticker} ({r.cap}) — {r.bucket.upper()} ---"]
        ticker_lines.append(f"  Reason: {r.bucket_reason}")

        if r.bucket == 'bounce':
            if r.bounce_setup_type:
                ticker_lines.append(f"  Setup type: {r.bounce_setup_type}")
            if r.bounce_result:
                br = r.bounce_result
                ticker_lines.append(f"  Score: {br.score}/{br.max_score} — {br.recommendation}")
                for item in getattr(br, 'items', []):
                    status = "PASS" if item.passed else "FAIL"
                    ticker_lines.append(f"    {item.name}: {status} (actual: {item.actual_display}, threshold: {item.threshold_display})")
                if getattr(br, 'bonuses', None):
                    ticker_lines.append(f"  Bonuses: {', '.join(br.bonuses)}")
                if getattr(br, 'warnings', None):
                    ticker_lines.append(f"  Warnings: {', '.join(br.warnings)}")
            if r.bounce_intensity and r.bounce_intensity.get('composite') is not None:
                ticker_lines.append(f"  Intensity: {r.bounce_intensity['composite']:.0f}/100")
        else:
            if r.score_result:
                sr = r.score_result
                ticker_lines.append(f"  Score: {sr.get('score', '?')}/{sr.get('max_score', '?')} — {sr.get('recommendation', 'N/A')}")
                for c in sr.get('criteria', []):
                    status = "PASS" if c.get('passed') else "FAIL"
                    actual = c.get('display_actual') or c.get('actual', '')
                    thresh = c.get('display_threshold') or c.get('threshold', '')
                    ticker_lines.append(f"    {c.get('name', '')}: {status} (actual: {actual}, threshold: {thresh})")

        # Exit targets
        if r.exit_targets and r.exit_targets.targets:
            et = r.exit_targets
            if et.entry_price:
                ticker_lines.append(f"  Entry: ${et.entry_price:.2f} ({et.entry_source})" + (f" | ATR: ${et.atr:.2f}" if et.atr else ""))
            for key, val in et.targets.items():
                if isinstance(val, dict) and val.get('price') is not None:
                    hr = f"{val['hit_rate']:.0f}%" if val.get('hit_rate') is not None else "N/A"
                    ticker_lines.append(f"    {val.get('label', key)}: ${val['price']:.2f} (hit rate: {hr})")

        # Key MA distances
        if r.mav_data:
            ma_parts = []
            for k, label in [('pct_from_10mav', '10MA'), ('pct_from_20mav', '20MA'), ('pct_from_50mav', '50MA')]:
                v = r.mav_data.get(k)
                if v is not None:
                    try:
                        ma_parts.append(f"{label}: {float(v):+.1%}")
                    except (ValueError, TypeError):
                        pass
            if ma_parts:
                ticker_lines.append(f"  MA distances: {', '.join(ma_parts)}")

        lines.extend(ticker_lines)

    return "\n".join(lines) if lines else "No tickers in today's report."


def build_system_prompt(report_data=None) -> str:
    """Build the system prompt for the trading chat assistant."""

    # Load data stats
    bounce_stats = _load_csv_stats(
        REPO_ROOT / "data" / "bounce_data.csv",
        'bounce_open_close_pct', is_short=False,
    )
    reversal_stats = _load_csv_stats(
        REPO_ROOT / "data" / "reversal_data.csv",
        'reversal_open_close_pct', is_short=True,
    )

    bounce_setup_stats = _load_bounce_setup_stats()
    trading_memory = _load_trading_memory()
    report_summary = _format_report_summary(report_data)

    # Build cap breakdown strings
    def _cap_lines(stats: dict) -> str:
        lines = []
        for cap, s in sorted(stats.get('by_cap', {}).items()):
            lines.append(f"    {cap}: {s['trades']} trades, {s['win_rate']}% WR, {s['avg_pnl']:+.1f}% avg")
        return "\n".join(lines) if lines else "    (no cap breakdown)"

    prompt = f"""You are a trading assistant for a day trader specializing in bounce and reversal (parabolic short) setups. You have access to tools to analyze live market data, score setups, search trading journals, and query historical trade databases.

HISTORICAL PERFORMANCE:
======================
Bounce Strategy (Long): {bounce_stats['total']} trades, {bounce_stats['win_rate']}% WR, {bounce_stats['avg_pnl']:+.1f}% avg P&L
  By setup type:
{bounce_setup_stats}
  By cap:
{_cap_lines(bounce_stats)}

Reversal Strategy (Short): {reversal_stats['total']} trades, {reversal_stats['win_rate']}% WR, {reversal_stats['avg_pnl']:+.1f}% avg P&L
  By cap:
{_cap_lines(reversal_stats)}

TODAY'S REPORT:
==============
{report_summary}

SCORING RULES:
=============
Bounce Pre-Trade (6 criteria - V3):
  1. Deep selloff (selloff_total_pct <= threshold)
  2. Discount from 30d high (pct_off_30d_high <= threshold)
  3. Gap down capitulation (gap_pct <= threshold)
  4. Prior day range expansion (range/ATR >= 1.0x)
  5. 3-day momentum crash (pct_change_3 <= threshold)
  6. Discount from 52wk high (pct_off_52wk_high <= threshold)
  GO = 5-6/6, CAUTION = 4/6, NO-GO = 0-3/6
  Two profiles: GapFade_weakstock (below 200MA) and GapFade_strongstock (above 200MA)

Reversal Pre-Trade (5 criteria):
  1. Extension above 9EMA
  2. Range expansion (prior day range vs ATR)
  3. 3-day run-up
  4. Gap up
  5. Volume signal (prior day RVOL or premarket RVOL)
  GO = 4-5/5, CAUTION = 3/5, NO-GO = 0-2/5

EXIT TARGET FRAMEWORK:
====================
Bounce (Long): ATR-based targets ABOVE open (0.5x, 1.0x ATR) + gap fill
Reversal (Short): ATR-based targets BELOW open (1.0x, 1.5x, 2.0x ATR) + gap fill

TRADING RULES & MEMORY:
======================
{trading_memory}

AVAILABLE TOOLS:
===============
- score_bounce_setup: Live bounce checklist scoring
- score_reversal_setup: Live reversal scoring
- get_live_price: Current price from Polygon
- calculate_exit_targets_tool: ATR-based exit targets
- get_percentile_rankings: Percentile vs historical
- query_bounce_data: Filter bounce_data.csv
- query_reversal_data: Filter reversal_data.csv
- find_similar_trades: Historical comps
- search_news: Perplexity web search
- get_trading_memory: Read trading rules file
- get_note_by_date: Obsidian journal entry (if available)
- search_notes: Search Obsidian notes (if available)
- get_recent_notes: Recent journal entries (if available)
- search_trades: MongoDB trade records (if available)
- get_trading_stats: Aggregate P&L stats (if available)
- semantic_search_notes: AI search across notes (if available)

GUIDELINES:
- TODAY'S REPORT DATA IS ALREADY ABOVE — use it directly when answering questions about today's watchlist tickers. Do NOT call scoring tools for tickers already in the report unless the user asks for a fresh/live re-score.
- Only use score_bounce_setup or score_reversal_setup for tickers NOT in today's report, or when explicitly asked to re-score.
- Always provide the recommendation (GO/CAUTION/NO-GO) with context
- Reference historical stats when giving recommendations
- Use find_similar_trades to provide comparable historical outcomes
- Be concise and action-oriented — this is for active trading decisions
- If a tool returns an error, explain what happened and suggest alternatives
- When discussing exit targets, always mention the historical hit rates
- Minimize tool calls — answer from report data and context first, only call tools when you need data you don't have
"""

    return prompt
