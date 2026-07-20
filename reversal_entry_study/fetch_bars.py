"""Fetch + cache 1-min bars for the reversal entry study.

Universe = every (ticker, date) in reversal_data.csv PLUS a seeded sample of
N control days (setup_type == 3DGapFade, not in the curated book) from the
reversal universe CSVs. One pickle per day:
data/reversal_entry_study/{TICKER}_{YYYY-MM-DD}.pkl (full session incl.
premarket, ET index). Fetch once, rerun the grid free.

    python -m reversal_entry_study.fetch_bars
"""
from __future__ import annotations

import argparse
import glob
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from data_queries.polygon_queries import get_intraday

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "reversal_data.csv"
UNIVERSE_GLOB = str(ROOT / "data" / "reversal_universe_*.csv")
CACHE = ROOT / "data" / "reversal_entry_study"

BAR_COLS = ["open", "high", "low", "close", "volume", "vwap"]
N_CONTROLS = 150
CONTROL_SEED = 26
CONTROL_SETUP = "3DGapFade"


def load_curated() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["ticker"] = df["ticker"].str.strip()
    df = df[~df["ticker"].str.upper().str.endswith(".T")]
    df["date_iso"] = pd.to_datetime(df["date"], format="%m/%d/%Y").dt.strftime("%Y-%m-%d")
    keep = ["ticker", "date_iso", "cap", "setup", "trade_grade", "atr_pct"]
    out = df[[c for c in keep if c in df.columns]].copy()
    out["cohort"] = "curated"
    return out.reset_index(drop=True)


def load_controls(n: int = N_CONTROLS, seed: int = CONTROL_SEED) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(glob.glob(UNIVERSE_GLOB))]
    uni = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date", "ticker"])
    typed = uni[uni["setup_type"] == CONTROL_SETUP].copy()
    flag = typed.get("in_reversal_csv")
    if flag is not None:
        typed = typed[~flag.astype(str).str.lower().isin({"true", "1"})]
    # Exclude anything that IS in the curated book by (ticker, date) too —
    # belt and braces against a stale in_reversal_csv flag.
    cur = load_curated()
    cur_keys = set(zip(cur["ticker"].str.upper(), cur["date_iso"]))
    typed["date_iso"] = pd.to_datetime(typed["date"]).dt.strftime("%Y-%m-%d")
    typed = typed[~typed.apply(
        lambda r: (str(r["ticker"]).upper(), r["date_iso"]) in cur_keys, axis=1)]
    sample = typed.sample(n=min(n, len(typed)), random_state=seed)
    out = pd.DataFrame({
        "ticker": sample["ticker"].str.strip(),
        "date_iso": sample["date_iso"],
        "cap": sample["cap"],
        "setup": sample["setup_type"],
        "trade_grade": "",
        "atr_pct": sample["atr_pct"],
    })
    out["cohort"] = "control"
    return out.reset_index(drop=True)


def load_universe() -> pd.DataFrame:
    return pd.concat([load_curated(), load_controls()], ignore_index=True)


def cache_path(ticker: str, date_iso: str) -> Path:
    return CACHE / f"{ticker.upper()}_{date_iso}.pkl"


def fetch_day(ticker: str, date_iso: str) -> pd.DataFrame | None:
    path = cache_path(ticker, date_iso)
    if path.exists():
        return pd.read_pickle(path)
    try:
        df = get_intraday(ticker, date_iso, 1, "minute")
    except Exception as e:
        log.warning(f"fetch failed: {ticker} {date_iso}: {e}")
        return None
    if df is None or df.empty:
        log.warning(f"no bars: {ticker} {date_iso}")
        return None
    keep = [c for c in BAR_COLS if c in df.columns]
    out = df[keep].copy()
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_pickle(path)
    return out


def fetch_all(workers: int = 8) -> None:
    uni = load_universe()
    ok = missing = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_day, r["ticker"], r["date_iso"]): i
                for i, r in uni.iterrows()}
        for f in as_completed(futs):
            if f.result() is None:
                missing += 1
            else:
                ok += 1
            if (ok + missing) % 50 == 0:
                log.info(f"  {ok + missing}/{len(uni)} fetched")
    log.info(f"bars cached for {ok}/{len(uni)} days ({missing} unavailable)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argparse.ArgumentParser().parse_args()
    fetch_all()
