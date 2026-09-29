"""Trillium 5s bar cache: one pickle per ticker-day, full session.

    orderPipe\\venv\\Scripts\\python -m checkpoint_study.fetch_bars            # fetch missing
    orderPipe\\venv\\Scripts\\python -m checkpoint_study.fetch_bars --import-dir <dir>

Trillium bars carry every print (the print filter handles bad ones) and are
not split-adjusted, so they match the trade log's raw prices. ~3.5 s per call.
Don't run a large fetch while a live monitor is streaming from Trillium.
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd

from checkpoint_study import config
from checkpoint_study.population import build_episodes

log = logging.getLogger(__name__)
COLS = ["open", "high", "low", "close", "volume"]


def bar_path(symbol: str, date: str) -> Path:
    return config.BARS_DIR / f"{symbol}_{date}.pkl"


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Close-time index (tz ET), unique (last message wins), sorted, OHLCV floats."""
    if df is None or df.empty or "close" not in df:
        return pd.DataFrame(columns=COLS)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    return df[COLS].astype(float)


def load_day(symbol: str, date: str):
    p = bar_path(symbol, date)
    return normalize(pd.read_pickle(p)) if p.exists() else None


def fetch_missing(limit: int = 0) -> None:
    config.orderpipe_on_path()
    from trillium.trlm_data_queries import get_intraday   # orderPipe (sheldatagateway)
    config.BARS_DIR.mkdir(parents=True, exist_ok=True)
    ep = build_episodes()
    days = sorted({(s, d) for s, d, ok in zip(ep.symbol, ep.date, ep.in_scope) if ok})
    todo = [(s, d) for s, d in days if not bar_path(s, d).exists()]
    missing_log = config.DATA_DIR / "missing.txt"
    done_missing = set(missing_log.read_text().split()) if missing_log.exists() else set()
    todo = [(s, d) for s, d in todo if f"{s}_{d}" not in done_missing]
    if limit:
        todo = todo[:limit]
    log.info("ticker-days: %d total, %d to fetch", len(days), len(todo))
    for i, (s, d) in enumerate(todo):
        try:
            df = get_intraday(s, d, "bar-5s")
            if df is None or df.empty:
                raise ValueError("empty")
            df.to_pickle(bar_path(s, d))
        except Exception as e:                      # keep going; record and move on
            log.warning("FAIL %s %s: %r", s, d, e)
            with open(missing_log, "a") as fh:
                fh.write(f"{s}_{d}\n")
        if i % 50 == 0:
            log.info("%d / %d", i, len(todo))
    log.info("done")


def import_dir(src: Path) -> int:
    """Copy already-fetched Trillium 5s pickles ({SYM}_{YYYY-MM-DD}.pkl)."""
    config.BARS_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in Path(src).glob("*_*.pkl"):
        dst = config.BARS_DIR / p.name
        if dst.exists():
            continue
        if normalize(pd.read_pickle(p)).empty:
            continue
        shutil.copy2(p, dst)
        n += 1
    return n


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--import-dir", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.import_dir:
        print("imported", import_dir(a.import_dir))
    else:
        fetch_missing(a.limit)
