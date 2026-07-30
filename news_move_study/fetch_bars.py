"""Fetch + cache 1-min bars for every event day in the news population.

One pickle per (ticker, date): data/news_move_study/bars/{TICKER}_{DATE}.pkl,
full session INCLUDING pre/post market — a news gun fires at 7am as readily
as at 2pm, and the RTH-only filter used by the bounce analog chart would clip
roughly one event in twelve.

    python -m news_move_study.fetch_bars              # fetch all missing
    python -m news_move_study.fetch_bars --limit 200  # pilot
"""
from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from pytz import timezone

from data_queries.polygon_queries import poly_client
from news_move_study.events import build_events

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "news_move_study" / "bars"
MISSING = ROOT / "data" / "news_move_study" / "missing.txt"

BAR_COLS = ["open", "high", "low", "close", "volume"]

ET = timezone("US/Eastern")


def _fetch_unadjusted(ticker: str, date_iso: str) -> Optional[pd.DataFrame]:
    """1-min bars in the prices that actually traded that day.

    NOT data_queries.get_intraday: that omits the ``adjusted`` flag, so Polygon
    defaults to split-adjusted-to-today. ref_price in the trade log is the raw
    price at the time, so adjusted bars silently corrupt every ticker that has
    split since — SOXS reverse-splits constantly and came back as a 200x move.
    A forward 2:1 split would look like a plausible 50% error, which is worse.
    """
    aggs = []
    for a in poly_client.list_aggs(ticker=ticker, multiplier=1, timespan="minute",
                                   from_=date_iso, to=date_iso, adjusted=False,
                                   limit=50_000):
        aggs.append(a)
    if not aggs:
        return None
    df = pd.DataFrame([vars(a) for a in aggs])
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(ET)
    return df.set_index("timestamp")


def cache_path(ticker: str, date_iso: str) -> Path:
    return CACHE / f"{ticker.upper()}_{date_iso}.pkl"


def _load_missing() -> set:
    """(ticker, date) pairs Polygon has already told us it has no data for —
    skipped on rerun so a full refetch doesn't re-pay for known holes."""
    if not MISSING.exists():
        return set()
    out = set()
    for line in MISSING.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) == 2:
            out.add((parts[0], parts[1]))
    return out


def _note_missing(ticker: str, date_iso: str) -> None:
    MISSING.parent.mkdir(parents=True, exist_ok=True)
    with MISSING.open("a") as fh:
        fh.write(f"{ticker},{date_iso}\n")


def fetch_day(ticker: str, date_iso: str, refetch: bool = False) -> Optional[pd.DataFrame]:
    """Cached 1-min bars for one (ticker, date). None when unavailable."""
    path = cache_path(ticker, date_iso)
    if path.exists() and not refetch:
        try:
            return pd.read_pickle(path)
        except Exception:
            pass  # corrupt cache entry — refetch below

    try:
        df = _fetch_unadjusted(ticker, date_iso)
    except Exception as e:
        log.warning(f"fetch failed {ticker} {date_iso}: {e}")
        return None
    if df is None or df.empty:
        return None

    cols = [c for c in BAR_COLS if c in df.columns]
    df = df[cols].sort_index()
    CACHE.mkdir(parents=True, exist_ok=True)
    try:
        df.to_pickle(path)
    except Exception as e:
        log.debug(f"cache write failed {ticker} {date_iso}: {e}")
    return df


def fetch_all(variant: str = "first", limit: Optional[int] = None,
              sleep: float = 0.0) -> dict:
    """Walk the population, filling the cache. Returns counts."""
    ev = build_events(variant)
    if limit:
        # Spread the pilot across years rather than taking the oldest N.
        ev = ev.sample(n=min(limit, len(ev)), random_state=7).sort_values("date_iso")

    known_missing = _load_missing()
    stats = {"total": len(ev), "cached": 0, "fetched": 0, "missing": 0}
    for i, row in enumerate(ev.itertuples(index=False), 1):
        key = (row.symbol, row.date_iso)
        if cache_path(row.symbol, row.date_iso).exists():
            stats["cached"] += 1
            continue
        if key in known_missing:
            stats["missing"] += 1
            continue
        df = fetch_day(row.symbol, row.date_iso)
        if df is None:
            stats["missing"] += 1
            _note_missing(row.symbol, row.date_iso)
        else:
            stats["fetched"] += 1
        if sleep:
            time.sleep(sleep)
        if i % 50 == 0:
            log.info(f"{i}/{len(ev)} — {stats}")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Cache 1-min bars for news events")
    ap.add_argument("--variant", default="first", choices=("first", "distinct"))
    ap.add_argument("--limit", type=int, default=None, help="pilot: only N events")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")
    stats = fetch_all(args.variant, args.limit, args.sleep)
    log.info(f"done — {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
