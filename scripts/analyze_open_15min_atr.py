"""scripts/analyze_open_15min_atr.py

For each row in data/reversal_data.csv, measure the first-15-min open-to-high
excursion in ATR units on:
  - The reversal day (D)
  - D-1, D-2, D-3 (leading days)
  - 5 random non-reversal trading days from the 90 sessions prior (control)

Output: how often the move is >= 1 ATR on each bucket. Lets us see whether a
sharp 1+ATR open spike is an euphoric-top tell or just routine noise on these
names.

Run:
    python scripts/analyze_open_15min_atr.py
"""
from __future__ import annotations

import datetime as dt
import logging
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pandas_market_calendars as mcal
import pytz

from data_queries.polygon_queries import get_intraday

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")
RTH_OPEN = dt.time(9, 30)
WINDOW_END = dt.time(9, 45)  # 9:30–9:44 inclusive = first 15 min
CONTROL_LOOKBACK_DAYS = 90
CONTROL_SAMPLE_SIZE = 5
MAX_WORKERS = 10
RANDOM_SEED = 42

NYSE = mcal.get_calendar("NYSE")


def parse_csv_date(s: str) -> dt.date:
    return dt.datetime.strptime(str(s).strip(), "%m/%d/%Y").date()


def trading_days_back(target: dt.date, n: int) -> dt.date:
    """Return the trading day n sessions before target."""
    schedule = NYSE.valid_days(
        start_date=target - dt.timedelta(days=int(n * 2) + 14),
        end_date=target - dt.timedelta(days=1),
    )
    if len(schedule) < n:
        raise ValueError(f"Not enough trading days before {target} for n={n}")
    return schedule[-n].date()


def trading_days_in_range(end_date: dt.date, lookback: int) -> List[dt.date]:
    schedule = NYSE.valid_days(
        start_date=end_date - dt.timedelta(days=int(lookback * 1.6) + 14),
        end_date=end_date - dt.timedelta(days=1),
    )
    return [d.date() for d in schedule[-lookback:]]


def first_15min_excursion(ticker: str, date: dt.date, atr_pct: float) -> Optional[Dict]:
    """Fetch 1-min bars and compute (15min_high - open) / open / atr_pct.

    Returns None if data missing or first 15 min is incomplete.
    """
    if atr_pct is None or atr_pct <= 0:
        return None
    df = get_intraday(ticker, date.strftime("%Y-%m-%d"), 1, "minute")
    if df is None or df.empty:
        return None
    open_ts = ET.localize(dt.datetime.combine(date, RTH_OPEN))
    end_ts = ET.localize(dt.datetime.combine(date, WINDOW_END))
    window = df[(df.index >= open_ts) & (df.index < end_ts)]
    if window.empty:
        return None
    open_price = float(window.iloc[0]["open"])
    if open_price <= 0:
        return None
    high_price = float(window["high"].max())
    move_pct = (high_price - open_price) / open_price
    in_atr = move_pct / atr_pct
    return {
        "date": date,
        "ticker": ticker,
        "open": open_price,
        "high": high_price,
        "move_pct": move_pct,
        "in_atr": in_atr,
        "ge_1atr": in_atr >= 1.0,
    }


def process_trade(row: pd.Series, control_pool: List[dt.date]) -> Dict:
    ticker = str(row["ticker"]).strip().upper()
    trade_date = parse_csv_date(row["date"])
    atr_pct = float(row["atr_pct"]) if pd.notna(row["atr_pct"]) else None

    out: Dict = {
        "ticker": ticker,
        "date": trade_date,
        "trade_grade": row.get("trade_grade"),
        "intraday_setup": row.get("intraday_setup"),
        "atr_pct": atr_pct,
    }

    if atr_pct is None or atr_pct <= 0:
        return out

    # Reversal day + leading days
    for offset in (0, 1, 2, 3):
        try:
            d = trade_date if offset == 0 else trading_days_back(trade_date, offset)
        except ValueError:
            continue
        rec = first_15min_excursion(ticker, d, atr_pct)
        if rec is not None:
            out[f"d{-offset if offset else 0}_in_atr"] = rec["in_atr"]
            out[f"d{-offset if offset else 0}_move_pct"] = rec["move_pct"]
            out[f"d{-offset if offset else 0}_ge_1atr"] = rec["ge_1atr"]

    # Control: 5 random non-reversal-window days
    forbidden = {trade_date}
    for offset in (1, 2, 3):
        try:
            forbidden.add(trading_days_back(trade_date, offset))
        except ValueError:
            pass
    candidates = [d for d in control_pool if d not in forbidden]
    if len(candidates) >= CONTROL_SAMPLE_SIZE:
        rng = random.Random(RANDOM_SEED + hash((ticker, trade_date)) % 10_000)
        sampled = rng.sample(candidates, CONTROL_SAMPLE_SIZE)
        ctl_in_atrs: List[float] = []
        ctl_hits: List[bool] = []
        for d in sampled:
            rec = first_15min_excursion(ticker, d, atr_pct)
            if rec is not None:
                ctl_in_atrs.append(rec["in_atr"])
                ctl_hits.append(rec["ge_1atr"])
        if ctl_in_atrs:
            out["ctl_n"] = len(ctl_in_atrs)
            out["ctl_mean_in_atr"] = sum(ctl_in_atrs) / len(ctl_in_atrs)
            out["ctl_max_in_atr"] = max(ctl_in_atrs)
            out["ctl_hit_rate"] = sum(ctl_hits) / len(ctl_hits)
    return out


def main() -> None:
    csv_path = PROJECT_ROOT / "data" / "reversal_data.csv"
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["ticker", "date", "atr_pct"]).reset_index(drop=True)
    log.info("Loaded %d reversal trades", len(df))

    # Pre-compute control pool per (ticker, trade_date) on the fly via lookback days
    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {}
        for _, row in df.iterrows():
            try:
                trade_date = parse_csv_date(row["date"])
            except Exception:
                continue
            pool = trading_days_in_range(trade_date, CONTROL_LOOKBACK_DAYS)
            futures[ex.submit(process_trade, row, pool)] = (row["ticker"], row["date"])

        for i, fut in enumerate(as_completed(futures), 1):
            tk, dt_ = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                log.warning("Failed %s %s: %s", tk, dt_, e)
            if i % 20 == 0:
                log.info("Processed %d / %d", i, len(futures))

    res_df = pd.DataFrame(results)
    out_path = PROJECT_ROOT / "data" / "open_15min_atr_analysis.csv"
    res_df.to_csv(out_path, index=False)
    log.info("Saved per-trade results -> %s", out_path)

    print()
    print("=" * 80)
    print("OPEN -> 15-MIN HIGH EXCURSION (in ATR units)")
    print("=" * 80)

    def hit_rate(col: str) -> Tuple[int, int, float, float, float]:
        if col not in res_df.columns:
            return (0, 0, 0.0, 0.0, 0.0)
        s = res_df[col].dropna()
        if s.empty:
            return (0, 0, 0.0, 0.0, 0.0)
        n = len(s)
        hits = int((s >= 1.0).sum())
        return (hits, n, hits / n if n else 0.0, float(s.mean()), float(s.median()))

    print(f"\n{'Bucket':<22} {'>=1ATR':>10} {'n':>6} {'hit_rate':>10} {'mean_ATR':>10} {'median_ATR':>12}")
    print("-" * 80)
    for label, col in [
        ("Reversal day (D)", "d0_in_atr"),
        ("D-1 (1 day prior)", "d-1_in_atr"),
        ("D-2 (2 days prior)", "d-2_in_atr"),
        ("D-3 (3 days prior)", "d-3_in_atr"),
    ]:
        hits, n, rate, mean_, med = hit_rate(col)
        print(f"{label:<22} {hits:>10} {n:>6} {rate:>10.1%} {mean_:>10.2f} {med:>12.2f}")

    # Control
    if "ctl_hit_rate" in res_df.columns:
        c = res_df.dropna(subset=["ctl_hit_rate"])
        if not c.empty:
            mean_hit = float(c["ctl_hit_rate"].mean())
            mean_in_atr = float(c["ctl_mean_in_atr"].mean())
            n_trades = len(c)
            tot_samples = int(c["ctl_n"].sum())
            print(
                f"{'Control (random)':<22} {'-':>10} {tot_samples:>6} "
                f"{mean_hit:>10.1%} {mean_in_atr:>10.2f} {'-':>12}"
            )

    # Cross-tab by trade_grade and intraday_setup
    if "d0_ge_1atr" in res_df.columns:
        print()
        print("Reversal-day >=1ATR hit rate by trade_grade:")
        gb = res_df.dropna(subset=["d0_ge_1atr"]).groupby("trade_grade")["d0_ge_1atr"].agg(["sum", "count", "mean"])
        gb.columns = ["hits", "n", "rate"]
        gb["rate"] = (gb["rate"] * 100).round(1).astype(str) + "%"
        print(gb.to_string())

        print()
        print("Reversal-day >=1ATR hit rate by intraday_setup:")
        gb2 = res_df.dropna(subset=["d0_ge_1atr"]).groupby("intraday_setup")["d0_ge_1atr"].agg(["sum", "count", "mean"])
        gb2.columns = ["hits", "n", "rate"]
        gb2 = gb2[gb2["n"] >= 3]
        gb2["rate"] = (gb2["rate"] * 100).round(1).astype(str) + "%"
        print(gb2.to_string())


if __name__ == "__main__":
    main()
