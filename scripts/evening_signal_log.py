"""Evening bounce board -> signal ledger.

Why this exists (SOXL 2026-07-30): the checklist's entry mechanism assumes a
capitulation extends into the next RTH open (gap down). On 7/29 SOXL was the
top intensity reading on the post-close board (77.5/100) and traded THROUGH
its computed GO-gap price in the after-hours session ($86.48 vs $88.31) — then
an overnight catalyst gapped it +16.7% and the bounce happened with no
tradeable gap-down open. bounce_data.csv can't measure how often that happens
because it only records days that qualified as the setup. This script removes
that selection bias: every evening it scores the watchlist at the after-hours
price and logs ALL rows (GO through NO-GO) to the signal ledger, so over time
we can answer "how often does a top-decile evening reading resolve overnight
without ever giving the gap-down entry?"

Row semantics:
  * date        = NEXT trading day (the session the signal is for; the outcome
                  filler scores each row against its date's own OHLC, so the
                  morning bat's fill_signal_outcomes pass fills these with the
                  correct day automatically).
  * session     = "evening", source = "evening_board".
  * gap_pct     = after-hours price vs today's close — the provisional
                  overnight gap at snapshot time.
  * setup_type  = checklist setup + "[i<intensity> go<trigger>%]" label, e.g.
                  "GapFade_strongstock [i78 go-4.0%]": i = the 0-100 intensity
                  composite (same SPEC as the morning watcher), go = the
                  largest gap-down at which the checklist reaches 5/6 GO.
                  A row whose gap_pct is already <= its go trigger was
                  through-the-trigger in the after-hours session.

Scoring reuses the premarket scanner's own build_static/metrics_from_price
(statics are requested as-of the NEXT trading day so today's completed bar is
the "prior day", exactly as tomorrow's 4:15 AM scan will see it) and Trillium
for real-time after-hours prices (Polygon minute bars as delayed fallback).

Usage:
    python -m scripts.evening_signal_log            # score + append to ledger
    python -m scripts.evening_signal_log --dry      # print only, no write
"""

from __future__ import annotations

import argparse
import datetime
import logging

import numpy as np
import pandas as pd

from analyzers.bounce_scorer import BouncePretrade
from data_queries import polygon_queries as pq
from scanners.premarket_bounce_scanner import (
    build_static, get_cap, load_watchlist, metrics_from_price,
    trillium_prices, _load_json, CAP_CACHE_FILE,
)
from support.signal_ledger import log_signals

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("evening_signal_log")

# Intensity composite — mirrors morning_watcher/rules/bounce_score_rules.py SPEC
# (and scanners/bounce_trader.py compute_bounce_intensity).
_SPEC = [
    ("pct_change_3",      False, 0.30),
    ("pct_change_15",     False, 0.20),
    ("selloff_total_pct", False, 0.15),
    ("gap_pct",           False, 0.15),
    ("pct_off_30d_high",  False, 0.15),
    ("pct_off_52wk_high", False, 0.05),
]

def _bounce_data_path():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent / "data" / "bounce_data.csv"


def _pctrank(arr: np.ndarray, score: float) -> float:
    if arr.size == 0:
        return float("nan")
    return 100.0 * (np.sum(arr < score) + 0.5 * np.sum(arr == score)) / arr.size


def load_reference() -> dict:
    df = pd.read_csv(_bounce_data_path()).dropna(subset=["ticker", "date"])
    df = df[~df["Setup"].str.contains("IntradayCapitch", case=False, na=False)]
    return {col: df[col].dropna().to_numpy(dtype=float)
            for col, _, _ in _SPEC if col in df.columns}


def intensity(metrics: dict, ref: dict) -> float | None:
    weighted = total = 0.0
    for col, higher_better, weight in _SPEC:
        v, arr = metrics.get(col), ref.get(col)
        if v is None or arr is None or arr.size == 0:
            continue
        p = _pctrank(arr, v)
        if p != p:
            continue
        weighted += (p if higher_better else 100.0 - p) * weight
        total += weight
    return round(weighted / total, 1) if total > 0 else None


def go_gap_trigger(ticker: str, static: dict, cap: str, checker: BouncePretrade) -> float | None:
    """Largest gap-down (as a negative fraction) at which the checklist first
    reaches 5+/6 GO tomorrow, holding everything but price fixed. Pure
    computation — sweeps 0% to -20% in 0.1% steps through the scanner's own
    metrics_from_price. None if 5/6 is unreachable in that range."""
    for g10 in range(0, 201):
        g = g10 / 1000.0
        price = static["prior_close"] * (1 - g)
        m = metrics_from_price(static, price)
        r = checker.validate(ticker, m, cap=cap)
        if r.score >= 5:
            return -g
    return None


def ah_low(ticker: str, date_str: str) -> float | None:
    """Low of the after-hours session (>= 16:00 ET) from Polygon minute bars.

    The AH LOW, not the last price, decides whether a name traded through its
    GO-gap trigger overnight (SOXL 7/29: $86.48 at 5:39 PM, through the -4%
    trigger, then an overnight catalyst erased the dip — the 8 PM close showed
    +0.3%). Polygon is 15-min delayed, so the 5:15 PM run sees AH bars through
    ~5:00 PM; later dips are missed until a backfill.
    """
    try:
        intra = pq.get_intraday(ticker, date_str, 1, "minute")
        if intra is None or intra.empty:
            return None
        ah = intra[intra.index >= pd.Timestamp(f"{date_str} 16:00", tz=intra.index.tz)]
        return float(ah["low"].min()) if not ah.empty else None
    except Exception:
        return None


def next_trading_day(after: datetime.date) -> datetime.date:
    import pandas_market_calendars as mcal
    cal = mcal.get_calendar("NYSE")
    sched = cal.schedule(start_date=after + datetime.timedelta(days=1),
                         end_date=after + datetime.timedelta(days=10))
    return sched.index[0].date()


def run(dry: bool = False, asof: str | None = None) -> list[dict]:
    """asof: score a past evening (YYYY-MM-DD) — Trillium's last 1-min bar for
    that date is the after-hours close, so a backfill reproduces what the
    snapshot would have logged that night."""
    today = datetime.date.fromisoformat(asof) if asof else datetime.date.today()
    target = next_trading_day(today)
    today_str = today.strftime("%Y-%m-%d")
    target_str = target.strftime("%Y-%m-%d")

    tickers = load_watchlist()
    cap_cache = _load_json(CAP_CACHE_FILE)
    checker = BouncePretrade()
    ref = load_reference()

    live = trillium_prices(tickers, today_str)
    if live:
        logger.info(f"Trillium after-hours prices for {len(live)}/{len(tickers)} tickers")
    else:
        logger.warning("Trillium unavailable — falling back to delayed Polygon minute bars")

    rows = []
    for t in tickers:
        try:
            # Statics as-of the NEXT session: today's completed bar becomes the
            # "prior day", matching what tomorrow's 4:15 AM scan will compute.
            static = build_static(t, target_str)
            if static is None:
                continue
            price = live.get(t)
            src = "T"
            if price is None:
                intra = pq.get_intraday(t, today_str, 1, "minute")
                if intra is None or intra.empty:
                    continue
                price = float(intra.iloc[-1]["close"])
                src = "P"
            m = metrics_from_price(static, price)
            res = checker.validate(t, m, cap=get_cap(t, cap_cache))
            i = intensity(m, ref)
            go = go_gap_trigger(t, static, get_cap(t, cap_cache), checker)
            lo = ah_low(t, today_str)
            lo_gap = (lo / static["prior_close"] - 1) if lo and static["prior_close"] > 0 else None
            label_bits = [f"i{i:.0f}" if i is not None else "i?"]
            if go is not None:
                label_bits.append(f"go{go * 100:.1f}%")
            if lo_gap is not None:
                label_bits.append(f"ahlo{lo_gap * 100:.1f}%")
            rows.append({
                "ticker": t,
                "bucket": "bounce",
                "cap": get_cap(t, cap_cache),
                "rec": res.recommendation,
                "score_str": f"{res.score}/{res.max_score}",
                "metrics": m,
                "score_result": res,
                "label": " ".join(label_bits),
                "_intensity": i if i is not None else -1,
                "_go": go,
                "_lo_gap": lo_gap,
                "_price": price,
                "_src": src,
            })
        except Exception as e:
            logger.warning(f"evening scan failed for {t}: {e}")

    rows.sort(key=lambda r: -r["_intensity"])

    print(f"\nEvening bounce board {today_str} (for session {target_str})")
    print(f"{'TICK':6} {'PRICE':>9} {'SRC':>3} {'AHgap':>7} {'AHlo':>7} {'SCORE':>6} "
          f"{'REC':>8} {'i/100':>6} {'goGap':>7}  flag")
    for r in rows:
        gap = r["metrics"].get("gap_pct")
        go, lo_gap = r["_go"], r["_lo_gap"]
        through = (go is not None and (
            (lo_gap is not None and lo_gap <= go) or (gap is not None and gap <= go)))
        print(f"{r['ticker']:6} {r['_price']:>9.2f} {r['_src']:>3} "
              f"{(gap or 0) * 100:>6.1f}% "
              f"{(lo_gap * 100 if lo_gap is not None else float('nan')):>6.1f}% "
              f"{r['score_str']:>6} {r['rec']:>8} {r['_intensity']:>6.1f} "
              f"{(go * 100 if go is not None else float('nan')):>6.1f}%  "
              f"{'THROUGH-TRIGGER' if through else ''}")

    if dry:
        print(f"\n--dry: {len(rows)} rows scored, ledger NOT written")
        return rows

    n = log_signals("evening_board", "evening", rows, date_str=target_str)
    print(f"\nledger: {n} rows appended (session=evening, date={target_str})")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Score the evening bounce board and log it to the signal ledger.")
    parser.add_argument("--dry", action="store_true", help="print only, do not write the ledger")
    parser.add_argument("--asof", help="backfill a past evening (YYYY-MM-DD)")
    args = parser.parse_args()
    run(dry=args.dry, asof=args.asof)


if __name__ == "__main__":
    main()
