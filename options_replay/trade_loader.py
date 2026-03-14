"""
Load and parse trade_data.csv from ExitMonitor.
Provides trade selection for the Options Replay dashboard.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SETUP_KEYWORDS = ["news", "momo", "capitulation", "3DGapFade", "ccall"]


def parse_setup_type(tags: str) -> str:
    """Extract primary setup type from tags string.

    Returns: 'news', 'momo', 'capitulation', '3DGapFade', 'ccall', or 'other'
    """
    if not tags or not isinstance(tags, str):
        return "other"
    tags_lower = tags.lower()
    for kw in SETUP_KEYWORDS:
        if kw.lower() in tags_lower:
            return kw
    return "other"

TRADE_CSV_PATHS = [
    Path(r"C:\Users\zmbur\PycharmProjects\ExitMonitor\data\trade_data.csv"),
    Path(r"C:\Users\zmbur\PycharmProjects\ExitMonitor\trade_data.csv"),
    Path.home() / "OneDrive" / "trade_data.csv",
]


def _find_csv() -> Path:
    for p in TRADE_CSV_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"trade_data.csv not found in any of: {[str(p) for p in TRADE_CSV_PATHS]}"
    )


def load_trades() -> pd.DataFrame:
    """Load and normalize the trade CSV.

    Returns DataFrame with columns:
        date, symbol, entry_time (datetime), side (1/-1),
        net_pnl, gross_pnl, avg_price, avg_exit_price, max_size,
        tags, news_summary, news_type, mkt_cap, lqa_score
    """
    path = _find_csv()
    raw = pd.read_csv(path)

    df = pd.DataFrame()
    df["symbol"] = raw["Symbol"].astype(str).str.strip().str.upper()

    # Parse date
    df["date"] = pd.to_datetime(raw["Date"], format="mixed").dt.date

    # Parse entry time from Start column
    # Start has formats like "6/5/2023 14:57" or "2023-06-05 14:57:00-04:00"
    start_parsed = pd.to_datetime(raw["Start"], format="mixed", utc=True)
    # Convert to Eastern and strip timezone for uniform handling
    start_parsed = start_parsed.dt.tz_convert("US/Eastern").dt.tz_localize(None)
    df["entry_time"] = start_parsed

    # If headline_time exists and is populated, prefer it
    if "headline_time" in raw.columns:
        ht = pd.to_datetime(raw["headline_time"], format="mixed", utc=True, errors="coerce")
        mask = ht.notna()
        if mask.any():
            ht_eastern = ht[mask].dt.tz_convert("US/Eastern").dt.tz_localize(None)
            df.loc[mask, "entry_time"] = ht_eastern

    # Extract just the time component for display
    df["entry_time_str"] = df["entry_time"].dt.strftime("%H:%M")

    # Side: 1 = long, -1 = short
    df["side"] = pd.to_numeric(raw.get("Side", 1), errors="coerce").fillna(1).astype(int)

    # Financials
    df["net_pnl"] = pd.to_numeric(raw.get("Net P&L", 0), errors="coerce").fillna(0)
    df["gross_pnl"] = pd.to_numeric(raw.get("Gross P&L", 0), errors="coerce").fillna(0)
    df["avg_price"] = pd.to_numeric(raw.get("Avg Price at Max", 0), errors="coerce").fillna(0)
    df["avg_exit_price"] = pd.to_numeric(raw.get("Avg Exit Price", 0), errors="coerce").fillna(0)
    df["max_size"] = pd.to_numeric(raw.get("Max Size", 0), errors="coerce").fillna(0).astype(int)

    # Metadata
    df["tags"] = raw.get("Tags", "").fillna("")
    df["setup_type"] = df["tags"].apply(parse_setup_type)
    df["news_summary"] = raw.get("news_summary", "").fillna("")
    df["news_type"] = raw.get("news_type", "").fillna("")
    df["mkt_cap"] = pd.to_numeric(raw.get("mkt_cap", 0), errors="coerce").fillna(0)
    df["lqa_score"] = pd.to_numeric(raw.get("LQA_Score", 0), errors="coerce").fillna(0)

    # Sort most recent first
    df = df.sort_values(["date", "entry_time"], ascending=[False, False]).reset_index(drop=True)

    return df


def get_trade_options(df: pd.DataFrame, min_pnl: float = 10_000) -> list:
    """Build dropdown options list from trades DataFrame.

    Only includes trades with net P&L >= min_pnl.
    Returns list of dicts: {label: "2023-06-05 U 14:57 LONG +$107,440", value: idx}
    """
    options = []
    for idx, row in df.iterrows():
        if row["net_pnl"] < min_pnl:
            continue
        direction = "LONG" if row["side"] == 1 else "SHORT"
        pnl = row["net_pnl"]
        pnl_str = f"+${pnl:,.0f}" if pnl >= 0 else f"-${abs(pnl):,.0f}"
        label = f"{row['date']}  {row['symbol']}  {row['entry_time_str']}  {direction}  {pnl_str}"
        if row["news_summary"]:
            # Truncate long summaries
            summary = row["news_summary"][:50]
            label += f"  ({summary})"
        options.append({"label": label, "value": int(idx)})
    return options


def parse_manual_input(ticker: str, date_str: str, time_str: str, direction: str) -> dict:
    """Validate and return a trade dict from manual input fields.

    Returns dict with same keys as a DataFrame row.
    """
    date = pd.Timestamp(date_str).date()
    entry_time = pd.Timestamp(f"{date_str} {time_str}")
    side = 1 if direction.upper() in ("LONG", "BUY", "1") else -1

    return {
        "symbol": ticker.upper().strip(),
        "date": date,
        "entry_time": entry_time,
        "entry_time_str": entry_time.strftime("%H:%M"),
        "side": side,
        "net_pnl": 0,
        "gross_pnl": 0,
        "avg_price": 0,
        "avg_exit_price": 0,
        "max_size": 0,
        "tags": "",
        "setup_type": "other",
        "news_summary": "",
        "news_type": "",
        "mkt_cap": 0,
        "lqa_score": 0,
    }
