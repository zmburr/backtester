"""Fetch + cache 1-min bars for every (ticker, date) in bounce_data.csv.

One parquet per day: data/bounce_entry_study/{TICKER}_{YYYY-MM-DD}.pkl
(full session incl. premarket, ET index). Fetch once, rerun the grid free.

    python -m bounce_entry_study.fetch_bars            # fetch all missing
    python -m bounce_entry_study.fetch_bars --refetch-today
"""
from __future__ import annotations

import argparse
import logging
from datetime import date as _date
from pathlib import Path

import pandas as pd

from data_queries.polygon_queries import get_intraday

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "bounce_data.csv"
CACHE = ROOT / "data" / "bounce_entry_study"

BAR_COLS = ["open", "high", "low", "close", "volume", "vwap"]


def load_universe() -> pd.DataFrame:
    """Rows of bounce_data.csv usable for the study (skips .T tickers)."""
    df = pd.read_csv(CSV, usecols=["date", "ticker", "trade_grade", "cap", "Setup",
                                   "atr_pct", "time_of_low_price"])
    df["ticker"] = df["ticker"].str.strip()
    df = df[~df["ticker"].str.upper().str.endswith(".T")]
    df["date_iso"] = pd.to_datetime(df["date"], format="%m/%d/%Y").dt.strftime("%Y-%m-%d")
    return df.reset_index(drop=True)


def cache_path(ticker: str, date_iso: str) -> Path:
    return CACHE / f"{ticker.upper()}_{date_iso}.pkl"


def fetch_day(ticker: str, date_iso: str, refetch: bool = False) -> pd.DataFrame | None:
    """Cached 1-min bars for one (ticker, date). None if unavailable."""
    path = cache_path(ticker, date_iso)
    if path.exists() and not refetch:
        return pd.read_pickle(path)
    df = get_intraday(ticker, date_iso, 1, "minute")
    if df is None or df.empty:
        log.warning(f"no bars: {ticker} {date_iso}")
        return None
    keep = [c for c in BAR_COLS if c in df.columns]
    out = df[keep].copy()
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_pickle(path)
    return out


def fetch_all(refetch_today: bool = False) -> None:
    uni = load_universe()
    today_iso = _date.today().isoformat()
    ok = missing = 0
    for _, row in uni.iterrows():
        refetch = refetch_today and row["date_iso"] == today_iso
        bars = fetch_day(row["ticker"], row["date_iso"], refetch=refetch)
        if bars is None:
            missing += 1
        else:
            ok += 1
    log.info(f"bars cached for {ok}/{len(uni)} days ({missing} unavailable)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--refetch-today", action="store_true",
                    help="refetch today's (still-forming) bars even if cached")
    args = ap.parse_args()
    fetch_all(refetch_today=args.refetch_today)
