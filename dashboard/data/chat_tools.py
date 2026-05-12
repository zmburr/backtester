"""
Chat tool definitions and executors for the Trading Chat interface.

Defines Anthropic tool_use format tools and their executor functions.
Follows the pattern from orderPipe/analytics/trading_chat.py.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP imports (graceful degradation)
# ---------------------------------------------------------------------------
OBSIDIAN_AVAILABLE = False
_vault_manager = None
_search_engine = None

try:
    sys.path.insert(0, r"C:\Users\zmbur\Documents\obsidian-trading-mcp\obsidian-trading-mcp\src")
    from obsidian_trading_mcp.vault_manager import VaultManager
    from obsidian_trading_mcp.search_engine import NoteSearchEngine, MultiVaultSearchEngine
    OBSIDIAN_AVAILABLE = True
except ImportError:
    pass

MONGO_AVAILABLE = False
_trade_search_engine = None

try:
    sys.path.insert(0, r"C:\Users\zmbur\Documents\mongodb-trades-mcp\src")
    from mongodb_trades_mcp.database import get_database
    from mongodb_trades_mcp.search_engine import TradeSearchEngine
    MONGO_AVAILABLE = True
except ImportError:
    pass

SEMANTIC_AVAILABLE = False
_semantic_search = None

try:
    sys.path.insert(0, r"C:\Users\zmbur\PycharmProjects\orderPipe")
    from analytics.semantic_search import SemanticNoteSearch, get_semantic_search
    SEMANTIC_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Backtester imports
# ---------------------------------------------------------------------------
from analyzers.bounce_scorer import BouncePretrade, fetch_bounce_metrics
from analyzers.exit_targets import calculate_exit_targets, format_exit_targets_text
from analyzers.bounce_exit_targets import calculate_bounce_exit_targets
from scanners import stock_screener as ss
from data_queries.polygon_queries import get_actual_current_price
from support.llm_client import perplexity_search

from dashboard.data.report_engine import (
    compute_bounce_intensity,
    get_ticker_cap,
    get_pretrade_metrics,
    score_pretrade_setup,
    get_exit_target_data,
    BOUNCE_DF_WEAK,
    BOUNCE_DF_STRONG,
    BOUNCE_DF_ALL,
)

# ---------------------------------------------------------------------------
# Obsidian vault setup
# ---------------------------------------------------------------------------
OBSIDIAN_VAULT_PATHS = [
    r"C:\Users\zmbur\OneDrive\Documents\Obsidian Vault\Evernote",
    r"C:\Users\zmbur\OneDrive\Documents\Obsidian Vault\Top Opps",
]


def _get_vault_manager():
    global _vault_manager
    if not OBSIDIAN_AVAILABLE:
        return None
    if _vault_manager is None:
        try:
            _vault_manager = VaultManager(OBSIDIAN_VAULT_PATHS[0])
        except Exception:
            return None
    return _vault_manager


def _get_search_engine():
    global _search_engine
    if not OBSIDIAN_AVAILABLE:
        return None
    if _search_engine is None:
        try:
            engines = []
            for path in OBSIDIAN_VAULT_PATHS:
                vm = VaultManager(path)
                engines.append(NoteSearchEngine(vm))
            _search_engine = MultiVaultSearchEngine(engines) if len(engines) > 1 else engines[0]
        except Exception:
            return None
    return _search_engine


def _get_trade_search():
    global _trade_search_engine
    if not MONGO_AVAILABLE:
        return None
    if _trade_search_engine is None:
        try:
            db = get_database()
            _trade_search_engine = TradeSearchEngine(db)
        except Exception:
            return None
    return _trade_search_engine


def _get_semantic():
    global _semantic_search
    if not SEMANTIC_AVAILABLE:
        return None
    if _semantic_search is None:
        try:
            _semantic_search = get_semantic_search()
        except Exception:
            return None
    return _semantic_search


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def define_chat_tools() -> list:
    """Return list of tool definitions in Anthropic tool_use format."""
    tools = []

    # --- MCP Tools ---

    if OBSIDIAN_AVAILABLE:
        tools.extend([
            {
                "name": "get_note_by_date",
                "description": "Fetch an Obsidian trading journal entry for a specific date. Returns the full note content.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format"
                        }
                    },
                    "required": ["date"]
                }
            },
            {
                "name": "search_notes",
                "description": "Search Obsidian trading notes by keywords, date range, ticker, or P&L. Returns matching note snippets.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search keywords"},
                        "date_from": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                        "date_to": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                        "ticker": {"type": "string", "description": "Filter by ticker symbol"},
                        "min_pnl": {"type": "number", "description": "Minimum P&L filter"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_recent_notes",
                "description": "Get the last N days of trading journal entries from Obsidian.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of days to look back (default 7)"
                        }
                    },
                    "required": []
                }
            },
        ])

    if MONGO_AVAILABLE:
        tools.extend([
            {
                "name": "search_trades",
                "description": "Search MongoDB trade records. Filter by ticker, date range, or trade type.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Ticker symbol to filter"},
                        "date_from": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                        "date_to": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                        "trade_type": {"type": "string", "description": "Trade type filter (e.g., 'bounce', 'reversal')"}
                    },
                    "required": []
                }
            },
            {
                "name": "get_trading_stats",
                "description": "Get aggregate P&L statistics from MongoDB trades for a period.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "period": {
                            "type": "string",
                            "description": "Time period (e.g., 'last_week', 'last_month', 'last_quarter', '2024', '2025')"
                        }
                    },
                    "required": []
                }
            },
        ])

    if SEMANTIC_AVAILABLE:
        tools.append({
            "name": "semantic_search_notes",
            "description": "AI-powered semantic search across trading notes using embeddings. Finds conceptually related content even without exact keyword matches.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    }
                },
                "required": ["query"]
            }
        })

    # --- Trading memory ---
    tools.append({
        "name": "get_trading_memory",
        "description": "Read the trading rules and lessons file (trading_memory.md). Contains hard-won trading rules and guidelines.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    })

    # --- Backtester-specific tools ---

    tools.extend([
        {
            "name": "score_bounce_setup",
            "description": "Run the live bounce pre-trade checklist for a ticker. Fetches current data from Polygon, auto-classifies as weakstock/strongstock, and scores against 6 criteria. Use when asked about bounce setups or capitulation trades.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., 'NVDA')"},
                    "cap": {"type": "string", "description": "Market cap category (ETF/Large/Medium/Small/Micro). Auto-detected if not provided."}
                },
                "required": ["ticker"]
            }
        },
        {
            "name": "score_reversal_setup",
            "description": "Run the live reversal (parabolic short) pre-trade scoring for a ticker. Fetches current data and scores against 5 criteria. Use when asked about reversal or short setups.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "cap": {"type": "string", "description": "Market cap category. Auto-detected if not provided."}
                },
                "required": ["ticker"]
            }
        },
        {
            "name": "get_live_price",
            "description": "Get the current/most recent price for a ticker from Polygon.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["ticker"]
            }
        },
        {
            "name": "calculate_exit_targets_tool",
            "description": "Calculate ATR-based exit target price levels for a ticker. Shows tiered targets with historical hit rates. Works for both bounce (long) and reversal (short) trades.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "cap": {"type": "string", "description": "Market cap category. Auto-detected if not provided."},
                    "trade_type": {"type": "string", "description": "Trade type: 'bounce' (long) or 'reversal' (short). Default: 'bounce'"}
                },
                "required": ["ticker"]
            }
        },
        {
            "name": "get_percentile_rankings",
            "description": "Get percentile rankings for a ticker compared to historical trades. Shows how extreme current metrics are vs past data.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["ticker"]
            }
        },
        {
            "name": "query_bounce_data",
            "description": "Query and filter the historical bounce trades dataset (bounce_data.csv). Can filter by ticker, P&L range, setup type, and cap.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Filter by ticker symbol"},
                    "min_pnl": {"type": "number", "description": "Minimum P&L % filter"},
                    "max_pnl": {"type": "number", "description": "Maximum P&L % filter"},
                    "setup_type": {"type": "string", "description": "Filter by setup type (GapFade_weakstock or GapFade_strongstock)"},
                    "cap": {"type": "string", "description": "Filter by cap (ETF/Large/Medium/Small/Micro)"}
                },
                "required": []
            }
        },
        {
            "name": "query_reversal_data",
            "description": "Query and filter the historical reversal trades dataset (reversal_data.csv). Can filter by ticker, setup type, cap, and grade.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Filter by ticker symbol"},
                    "setup": {"type": "string", "description": "Filter by setup type (e.g., '3DGapFade', '2DBreakoutIB')"},
                    "cap": {"type": "string", "description": "Filter by cap"},
                    "grade": {"type": "string", "description": "Filter by trade grade (A/B/C/D/F)"}
                },
                "required": []
            }
        },
        {
            "name": "find_similar_trades",
            "description": "Find historical trades most similar to a given ticker's current metrics. Returns closest comparable past trades with their outcomes.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Ticker to find comps for (fetches live data)"},
                    "trade_type": {"type": "string", "description": "Trade type: 'bounce' or 'reversal'. Default: 'reversal'"}
                },
                "required": ["ticker"]
            }
        },
        {
            "name": "search_news",
            "description": "Search for recent news about a topic using Perplexity AI. Good for finding catalysts, earnings info, or market context.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (e.g., 'NVDA earnings results' or 'market selloff catalyst')"}
                },
                "required": ["query"]
            }
        },
    ])

    return tools


# ---------------------------------------------------------------------------
# Tool executors
# ---------------------------------------------------------------------------

def _today() -> str:
    """Get today's date as YYYY-MM-DD, adjusted for weekends."""
    now = datetime.now()
    if now.weekday() == 5:
        now -= timedelta(days=1)
    elif now.weekday() == 6:
        now -= timedelta(days=2)
    return now.strftime('%Y-%m-%d')


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy/pandas types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp,)):
            return str(obj)
        return super().default(obj)


def _sanitize(obj):
    """Convert numpy/pandas types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def _dumps(obj, **kwargs):
    """json.dumps with numpy/pandas type handling."""
    return json.dumps(_sanitize(obj), cls=_NumpyEncoder, **kwargs)


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return string result."""
    log.info(f"Executing tool: {tool_name} with input: {tool_input}")

    try:
        # --- MCP tools ---

        if tool_name == "get_note_by_date":
            if not OBSIDIAN_AVAILABLE:
                return _dumps({"error": "Obsidian integration not available"})
            date_str = tool_input.get("date", "")
            vm = _get_vault_manager()
            if vm is None:
                return _dumps({"error": "Could not initialize vault manager"})
            try:
                note = vm.get_note_by_date(date_str)
                return _dumps({"date": date_str, "content": note or "No note found for this date"})
            except Exception as e:
                return _dumps({"error": str(e)})

        elif tool_name == "search_notes":
            if not OBSIDIAN_AVAILABLE:
                return _dumps({"error": "Obsidian integration not available"})
            se = _get_search_engine()
            if se is None:
                return _dumps({"error": "Could not initialize search engine"})
            try:
                results = se.search(
                    query=tool_input.get("query", ""),
                    date_from=tool_input.get("date_from"),
                    date_to=tool_input.get("date_to"),
                    ticker=tool_input.get("ticker"),
                    min_pnl=tool_input.get("min_pnl"),
                )
                if hasattr(results, '__iter__') and not isinstance(results, str):
                    return _dumps([
                        {"date": getattr(r, 'date', ''), "snippet": getattr(r, 'snippet', str(r))[:500]}
                        for r in list(results)[:10]
                    ], indent=2)
                return _dumps({"results": str(results)[:2000]})
            except Exception as e:
                return _dumps({"error": str(e)})

        elif tool_name == "get_recent_notes":
            if not OBSIDIAN_AVAILABLE:
                return _dumps({"error": "Obsidian integration not available"})
            vm = _get_vault_manager()
            if vm is None:
                return _dumps({"error": "Could not initialize vault manager"})
            days = tool_input.get("days", 7)
            try:
                notes = vm.get_recent_notes(days=days)
                if hasattr(notes, '__iter__') and not isinstance(notes, str):
                    return _dumps([
                        {"date": getattr(n, 'date', ''), "content": (getattr(n, 'content', str(n)))[:1000]}
                        for n in list(notes)[:days]
                    ], indent=2)
                return _dumps({"notes": str(notes)[:3000]})
            except Exception as e:
                return _dumps({"error": str(e)})

        elif tool_name == "search_trades":
            if not MONGO_AVAILABLE:
                return _dumps({"error": "MongoDB integration not available"})
            ts = _get_trade_search()
            if ts is None:
                return _dumps({"error": "Could not initialize trade search"})
            try:
                results = ts.search(
                    ticker=tool_input.get("ticker"),
                    date_from=tool_input.get("date_from"),
                    date_to=tool_input.get("date_to"),
                    trade_type=tool_input.get("trade_type"),
                )
                if hasattr(results, '__iter__') and not isinstance(results, str):
                    return _dumps([
                        {k: str(v) for k, v in (r if isinstance(r, dict) else vars(r)).items()}
                        for r in list(results)[:20]
                    ], indent=2)
                return _dumps({"results": str(results)[:3000]})
            except Exception as e:
                return _dumps({"error": str(e)})

        elif tool_name == "get_trading_stats":
            if not MONGO_AVAILABLE:
                return _dumps({"error": "MongoDB integration not available"})
            ts = _get_trade_search()
            if ts is None:
                return _dumps({"error": "Could not initialize trade search"})
            try:
                period = tool_input.get("period", "last_month")
                stats = ts.get_stats(period=period)
                return _dumps(stats if isinstance(stats, dict) else {"stats": str(stats)[:2000]}, indent=2)
            except Exception as e:
                return _dumps({"error": str(e)})

        elif tool_name == "semantic_search_notes":
            if not SEMANTIC_AVAILABLE:
                return _dumps({"error": "Semantic search not available"})
            ss_engine = _get_semantic()
            if ss_engine is None:
                return _dumps({"error": "Could not initialize semantic search"})
            try:
                results = ss_engine.search(tool_input.get("query", ""))
                if hasattr(results, '__iter__') and not isinstance(results, str):
                    return _dumps([
                        {"score": getattr(r, 'score', 0), "snippet": getattr(r, 'text', str(r))[:500]}
                        for r in list(results)[:10]
                    ], indent=2)
                return _dumps({"results": str(results)[:3000]})
            except Exception as e:
                return _dumps({"error": str(e)})

        elif tool_name == "get_trading_memory":
            memory_path = r"C:\Users\zmbur\PycharmProjects\orderPipe\analytics\trading_memory.md"
            try:
                with open(memory_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return _dumps({"content": content[:5000]})
            except FileNotFoundError:
                return _dumps({"error": "trading_memory.md not found"})
            except Exception as e:
                return _dumps({"error": str(e)})

        # --- Backtester tools ---

        elif tool_name == "score_bounce_setup":
            ticker = tool_input.get("ticker", "").upper()
            if not ticker:
                return _dumps({"error": "ticker is required"})

            date = _today()
            cap = tool_input.get("cap") or get_ticker_cap(ticker)

            try:
                metrics = fetch_bounce_metrics(ticker, date)
                checker = BouncePretrade()
                result = checker.validate(ticker, metrics, cap=cap)

                # Intensity score
                ref_df = BOUNCE_DF_WEAK if result.setup_type == 'GapFade_weakstock' else BOUNCE_DF_STRONG
                intensity = compute_bounce_intensity(metrics, ref_df=ref_df)

                return _dumps({
                    "ticker": ticker,
                    "cap": cap,
                    "setup_type": result.setup_type,
                    "score": result.score,
                    "max_score": result.max_score,
                    "recommendation": result.recommendation,
                    "summary": result.summary,
                    "criteria": [
                        {
                            "name": item.name,
                            "description": item.description,
                            "passed": item.passed,
                            "threshold": item.threshold_display,
                            "actual": item.actual_display,
                            "reference": item.reference,
                        }
                        for item in result.items
                    ],
                    "bonuses": result.bonuses,
                    "warnings": result.warnings,
                    "intensity_score": intensity.get('composite', 0),
                    "classification": result.classification_details,
                }, indent=2)
            except Exception as e:
                return _dumps({"error": f"Failed to score bounce setup for {ticker}: {str(e)}"})

        elif tool_name == "score_reversal_setup":
            ticker = tool_input.get("ticker", "").upper()
            if not ticker:
                return _dumps({"error": "ticker is required"})

            date = _today()
            cap = tool_input.get("cap") or get_ticker_cap(ticker)

            try:
                metrics = get_pretrade_metrics(ticker, date)
                result = score_pretrade_setup(ticker, metrics, cap=cap)

                return _dumps({
                    "ticker": ticker,
                    "cap": result.get('cap', cap),
                    "score": result['score'],
                    "max_score": result['max_score'],
                    "recommendation": result['recommendation'],
                    "criteria": [
                        {
                            "name": c['name'],
                            "key": c['key'],
                            "passed": c['passed'],
                            "threshold": c['threshold'],
                            "actual": c['actual'],
                            "display_actual": c.get('display_actual'),
                            "display_threshold": c.get('display_threshold'),
                        }
                        for c in result['criteria']
                    ],
                }, indent=2)
            except Exception as e:
                return _dumps({"error": f"Failed to score reversal setup for {ticker}: {str(e)}"})

        elif tool_name == "get_live_price":
            ticker = tool_input.get("ticker", "").upper()
            if not ticker:
                return _dumps({"error": "ticker is required"})

            date = _today()
            try:
                price = get_actual_current_price(ticker, date)
                if price is not None:
                    return _dumps({"ticker": ticker, "price": round(float(price), 2), "date": date})
                else:
                    return _dumps({"ticker": ticker, "error": "Could not fetch price", "date": date})
            except Exception as e:
                return _dumps({"error": f"Failed to get price for {ticker}: {str(e)}"})

        elif tool_name == "calculate_exit_targets_tool":
            ticker = tool_input.get("ticker", "").upper()
            if not ticker:
                return _dumps({"error": "ticker is required"})

            trade_type = tool_input.get("trade_type", "bounce").lower()
            date = _today()
            cap = tool_input.get("cap") or get_ticker_cap(ticker)
            prefer_open = (trade_type == "bounce")

            try:
                exit_data = get_exit_target_data(ticker, date, prefer_open=prefer_open)
                if not exit_data.get('open_price') or not exit_data.get('atr'):
                    return _dumps({"error": f"Could not fetch price/ATR data for {ticker}"})

                if trade_type == "bounce":
                    targets = calculate_bounce_exit_targets(
                        cap=cap,
                        entry_price=exit_data['open_price'],
                        atr=exit_data['atr'],
                        prior_close=exit_data.get('prior_close'),
                        prior_high=exit_data.get('prior_high'),
                    )
                else:
                    targets = calculate_exit_targets(
                        cap=cap,
                        entry_price=exit_data['open_price'],
                        atr=exit_data['atr'],
                        prior_close=exit_data.get('prior_close'),
                        prior_low=exit_data.get('prior_low'),
                        ema_4=exit_data.get('ema_4'),
                    )

                # Serialize tier objects
                tiers_out = []
                for t in targets.get('tiers', []):
                    tiers_out.append({
                        "tier": t.get('tier'),
                        "name": t.get('name'),
                        "target_price": round(t['target_price'], 2) if t.get('target_price') else None,
                        "target_pct": round(t['target_pct'] * 100, 1) if t.get('target_pct') else None,
                        "hit_rate": round(t['hit_rate'] * 100) if t.get('hit_rate') else None,
                        "position_pct": round(t['position_pct'] * 100) if t.get('position_pct') else None,
                        "note": t.get('note'),
                    })

                return _dumps({
                    "ticker": ticker,
                    "cap": cap,
                    "trade_type": trade_type,
                    "entry_price": round(exit_data['open_price'], 2),
                    "atr": round(exit_data['atr'], 2),
                    "prior_close": round(exit_data['prior_close'], 2) if exit_data.get('prior_close') else None,
                    "tiers": tiers_out,
                    "time_stop": targets.get('time_stop'),
                    "notes": targets.get('notes'),
                }, indent=2)
            except Exception as e:
                return _dumps({"error": f"Failed to calculate exit targets for {ticker}: {str(e)}"})

        elif tool_name == "get_percentile_rankings":
            ticker = tool_input.get("ticker", "").upper()
            if not ticker:
                return _dumps({"error": "ticker is required"})

            try:
                stock_data = ss.get_all_stocks_data([ticker]).get(ticker, {})
                if not stock_data:
                    return _dumps({"error": f"Could not fetch data for {ticker}"})

                from data_collectors.combined_data_collection import reversal_df
                pcts = ss.calculate_percentiles(reversal_df, stock_data, ss.columns_to_compare)

                return _dumps({
                    "ticker": ticker,
                    "percentiles": {k: round(v, 1) for k, v in pcts.items()},
                    "reference": "reversal dataset",
                }, indent=2)
            except Exception as e:
                return _dumps({"error": f"Failed to get percentiles for {ticker}: {str(e)}"})

        elif tool_name == "query_bounce_data":
            try:
                df = pd.read_csv(REPO_ROOT / "data" / "bounce_data.csv")
                df = df.dropna(subset=['ticker', 'date'])

                if tool_input.get("ticker"):
                    df = df[df['ticker'].str.upper() == tool_input['ticker'].upper()]
                if tool_input.get("setup_type"):
                    from analyzers.bounce_scorer import classify_from_setup_column
                    df['_profile'] = df['Setup'].apply(classify_from_setup_column)
                    df = df[df['_profile'] == tool_input['setup_type']]
                if tool_input.get("cap"):
                    df = df[df['cap'].str.strip() == tool_input['cap']]

                if 'bounce_open_close_pct' in df.columns:
                    df['pnl'] = pd.to_numeric(df['bounce_open_close_pct'], errors='coerce') * 100
                    if tool_input.get("min_pnl") is not None:
                        df = df[df['pnl'] >= tool_input['min_pnl']]
                    if tool_input.get("max_pnl") is not None:
                        df = df[df['pnl'] <= tool_input['max_pnl']]

                # Summary
                total = len(df)
                if total == 0:
                    return _dumps({"message": "No matching trades found", "total": 0})

                pnl = pd.to_numeric(df.get('bounce_open_close_pct'), errors='coerce').dropna() * 100
                win_rate = float((pnl > 0).mean() * 100) if len(pnl) > 0 else 0
                avg_pnl = float(pnl.mean()) if len(pnl) > 0 else 0

                # Return last 20 trades
                display_cols = ['date', 'ticker', 'cap', 'Setup', 'trade_grade', 'bounce_open_close_pct']
                display_cols = [c for c in display_cols if c in df.columns]
                records = df.tail(20)[display_cols].to_dict('records')

                return _dumps({
                    "total_trades": total,
                    "win_rate": round(win_rate, 1),
                    "avg_pnl": round(avg_pnl, 1),
                    "recent_trades": records,
                }, indent=2, default=str)
            except Exception as e:
                return _dumps({"error": str(e)})

        elif tool_name == "query_reversal_data":
            try:
                df = pd.read_csv(REPO_ROOT / "data" / "reversal_data.csv")
                df = df.dropna(subset=['ticker', 'date'])

                if tool_input.get("ticker"):
                    df = df[df['ticker'].str.upper() == tool_input['ticker'].upper()]
                if tool_input.get("setup"):
                    df = df[df['setup'] == tool_input['setup']]
                if tool_input.get("cap"):
                    df = df[df['cap'].str.strip() == tool_input['cap']]
                if tool_input.get("grade"):
                    df = df[df['trade_grade'] == tool_input['grade']]

                total = len(df)
                if total == 0:
                    return _dumps({"message": "No matching trades found", "total": 0})

                pnl = -pd.to_numeric(df.get('reversal_open_close_pct'), errors='coerce').dropna() * 100
                win_rate = float((pnl > 0).mean() * 100) if len(pnl) > 0 else 0
                avg_pnl = float(pnl.mean()) if len(pnl) > 0 else 0

                display_cols = ['date', 'ticker', 'cap', 'setup', 'trade_grade', 'reversal_open_close_pct']
                display_cols = [c for c in display_cols if c in df.columns]
                records = df.tail(20)[display_cols].to_dict('records')

                return _dumps({
                    "total_trades": total,
                    "win_rate": round(win_rate, 1),
                    "avg_pnl": round(avg_pnl, 1),
                    "recent_trades": records,
                }, indent=2, default=str)
            except Exception as e:
                return _dumps({"error": str(e)})

        elif tool_name == "find_similar_trades":
            ticker = tool_input.get("ticker", "").upper()
            trade_type = tool_input.get("trade_type", "reversal").lower()
            if not ticker:
                return _dumps({"error": "ticker is required"})

            try:
                stock_data = ss.get_all_stocks_data([ticker]).get(ticker, {})
                if not stock_data:
                    return _dumps({"error": f"Could not fetch data for {ticker}"})

                if trade_type == "reversal":
                    from data_collectors.combined_data_collection import reversal_df as ref_df
                    comp_cols = ss.columns_to_compare
                else:
                    ref_df = BOUNCE_DF_ALL
                    comp_cols = [
                        'pct_change_3', 'pct_change_15', 'pct_change_30',
                        'pct_from_10mav', 'pct_from_50mav', 'pct_from_200mav',
                    ]
                    comp_cols = [c for c in comp_cols if c in ref_df.columns]

                # Get current values for comparison columns
                current_vals = {}
                for col in comp_cols:
                    for src_dict_name in ['pct_data', 'mav_data', 'range_data']:
                        src_dict = stock_data.get(src_dict_name, {})
                        if src_dict and col in src_dict:
                            val = src_dict[col]
                            try:
                                current_vals[col] = float(val)
                            except (TypeError, ValueError):
                                pass
                            break

                if not current_vals:
                    return _dumps({"error": f"No metric data available for {ticker}"})

                # Calculate distance to each historical trade
                available_cols = [c for c in comp_cols if c in current_vals and c in ref_df.columns]
                if not available_cols:
                    return _dumps({"error": "No overlapping columns for comparison"})

                import numpy as np
                distances = []
                for idx, row in ref_df.iterrows():
                    dist = 0
                    valid = 0
                    for col in available_cols:
                        ref_val = row.get(col)
                        if ref_val is not None and not (isinstance(ref_val, float) and pd.isna(ref_val)):
                            try:
                                d = (float(ref_val) - current_vals[col]) ** 2
                                dist += d
                                valid += 1
                            except (TypeError, ValueError):
                                pass
                    if valid > 0:
                        distances.append((idx, np.sqrt(dist / valid)))

                distances.sort(key=lambda x: x[1])
                top_10 = distances[:10]

                results = []
                pnl_col = 'bounce_open_close_pct' if trade_type == 'bounce' else 'reversal_open_close_pct'
                for idx, dist in top_10:
                    row = ref_df.loc[idx]
                    pnl = row.get(pnl_col)
                    if trade_type == 'reversal' and pnl is not None:
                        try:
                            pnl = -float(pnl) * 100
                        except (TypeError, ValueError):
                            pnl = None
                    elif trade_type == 'bounce' and pnl is not None:
                        try:
                            pnl = float(pnl) * 100
                        except (TypeError, ValueError):
                            pnl = None

                    results.append({
                        "ticker": str(row.get('ticker', '')),
                        "date": str(row.get('date', '')),
                        "cap": str(row.get('cap', '')),
                        "grade": str(row.get('trade_grade', '')),
                        "pnl": round(pnl, 1) if pnl is not None else None,
                        "distance": round(dist, 4),
                    })

                return _dumps({
                    "ticker": ticker,
                    "trade_type": trade_type,
                    "current_metrics": {k: round(v, 4) for k, v in current_vals.items()},
                    "similar_trades": results,
                }, indent=2)
            except Exception as e:
                return _dumps({"error": f"Failed to find similar trades for {ticker}: {str(e)}"})

        elif tool_name == "search_news":
            query = tool_input.get("query", "")
            if not query:
                return _dumps({"error": "query is required"})

            try:
                result = perplexity_search(query)
                # Extract the text response
                if isinstance(result, dict):
                    choices = result.get('choices', [])
                    if choices:
                        msg = choices[0].get('message', {})
                        content = msg.get('content', '')
                        return _dumps({"query": query, "response": content[:3000]})
                return _dumps({"query": query, "response": str(result)[:3000]})
            except Exception as e:
                return _dumps({"error": f"News search failed: {str(e)}"})

        else:
            return _dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        log.error(f"Error executing {tool_name}: {e}")
        return _dumps({"error": str(e)})
