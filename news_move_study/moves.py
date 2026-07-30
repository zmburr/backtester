"""Per-event move geometry, measured from the starting gun.

For each event: take the 1-min bars from ``ref_ts`` through 20:00 ET the same
day and measure the move the way your stack already does it
(orderPipe/calculators/pct_captured_calculator.get_total_move):

    total_move = max(high) - ref_price          (long)
               = ref_price - min(low)           (short)

so the numbers tie out with ``pct_total_move_captured`` in the trade log.

What this module adds over that function is the *path*: how much of the total
was already in hand at minute k. ``pct_complete(k) = mfe_at(k) / total_move``
is the quantity the live feature has to estimate, and ``1 - pct_complete(k)``
is the number the display would show.

Deliberate choices worth knowing:

  * Events where ``total_move <= 0`` are KEPT. Those are guns that never once
    traded through the reference in the trade's direction — the losers that
    ExitMonitor's ``lo_to_hi`` silently drops by filtering on gpnl > 0. They
    are the reason the winners-only median runs long.
  * The extreme is located on the bar's high/low, not its close, matching
    get_total_move.
  * Halt bars (null open) are dropped, as in ``remove_halts``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from news_move_study.events import build_events
from news_move_study.fetch_bars import cache_path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "news_move_study"

# Minutes after the gun at which the path is sampled.
CHECKPOINTS = [1, 2, 5, 10, 15, 30, 60, 120]

# Below this, the news did nothing and "% of the move remaining" is a ratio of
# two noise terms. Tracked separately rather than silently averaged in.
DEGENERATE_MOVE_PCT = 0.005  # 0.5% of ref price

# ref_price came off the reference bar itself, so it should sit inside that
# bar's range. A large divergence means the log row and the bars disagree about
# what the stock was worth — a split the bars carry and ref_price doesn't, a
# reused ticker symbol, or a bad date. Bars are fetched unadjusted to prevent
# the split case; this stays as the safety net for the rest.
REF_MISMATCH_TOL = 0.25  # |ref_price / ref_bar_close - 1|


def load_bars(symbol: str, date_iso: str) -> Optional[pd.DataFrame]:
    path = cache_path(symbol, date_iso)
    if not path.exists():
        return None
    try:
        df = pd.read_pickle(path)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if "open" in df.columns:
        df = df[df["open"].notna()]
    return df if not df.empty else None


def measure_event(symbol: str, date_iso: str, ref_ts: pd.Timestamp,
                  ref_price: float, direction: int) -> Optional[Dict]:
    """Move geometry for one event. None when the day has no usable window."""
    bars = load_bars(symbol, date_iso)
    if bars is None:
        return None

    end = pd.Timestamp(f"{date_iso} 20:00:00").tz_localize(ref_ts.tz)
    win = bars[(bars.index >= ref_ts) & (bars.index <= end)]
    if len(win) < 2:
        return None

    if direction == 1:
        extreme_ts = win["high"].idxmax()
        total_move = float(win["high"].max()) - ref_price
        adverse = ref_price - float(win["low"].min())
        run = win["high"].cummax() - ref_price
    else:
        extreme_ts = win["low"].idxmin()
        total_move = ref_price - float(win["low"].min())
        adverse = float(win["high"].max()) - ref_price
        run = ref_price - win["low"].cummin()

    # Safety net: does the log's ref_price agree with the bar it came from?
    ref_bar_close = float(win["close"].iloc[0])
    ref_ratio = (ref_price / ref_bar_close) if ref_bar_close > 0 else float("nan")
    suspect = not (abs(ref_ratio - 1.0) <= REF_MISMATCH_TOL)

    minutes = (win.index - ref_ts).total_seconds() / 60.0
    rec: Dict = {
        "symbol": symbol,
        "date_iso": date_iso,
        "direction": direction,
        "ref_price": ref_price,
        "ref_bar_close": ref_bar_close,
        "ref_ratio": ref_ratio,
        "suspect": bool(suspect),
        "total_move": total_move,
        "total_move_pct": total_move / ref_price,
        "mae": adverse,
        "mae_pct": adverse / ref_price,
        "time_to_extreme_min": float((extreme_ts - ref_ts).total_seconds() / 60.0),
        "n_bars": int(len(win)),
        "window_min": float(minutes[-1]),
        "worked": bool(total_move > 0),
        "degenerate": bool(total_move / ref_price < DEGENERATE_MOVE_PCT),
    }

    # Adverse excursion path, mirrored, for the "is this move clean" feature.
    if direction == 1:
        draw = ref_price - win["low"].cummin()
    else:
        draw = win["high"].cummax() - ref_price

    # Pre-gun volume baseline: mean per-minute volume in the 30 minutes before
    # the gun. Thin/absent premarket tape leaves this None and the volume
    # feature drops out for that event rather than inventing a denominator.
    pre = bars[(bars.index < ref_ts) & (bars.index >= ref_ts - pd.Timedelta(minutes=30))]
    base_vpm = float(pre["volume"].mean()) if len(pre) >= 5 and "volume" in pre else None

    # Path: favorable excursion in hand at each checkpoint, the share of the
    # eventual total that represents, and the state a live display would see.
    for k in CHECKPOINTS:
        mask = minutes <= k
        if not mask.any():
            for f in ("mfe", "pct_complete", "mae", "stall", "volr"):
                rec[f"{f}_{k}"] = None
            continue
        mfe_k = float(run[mask].max())
        rec[f"mfe_{k}"] = mfe_k
        rec[f"pct_complete_{k}"] = (mfe_k / total_move) if total_move > 0 else None
        rec[f"mae_{k}"] = float(draw[mask].max())
        # Minutes since the running max was last set, as of k — a stalling move.
        sub = run[mask]
        rec[f"stall_{k}"] = float(minutes[mask][-1] - minutes[mask][int(sub.values.argmax())])
        if base_vpm and base_vpm > 0:
            vol_k = float(win.loc[mask, "volume"].sum())
            rec[f"volr_{k}"] = vol_k / (base_vpm * max(k, 1))
        else:
            rec[f"volr_{k}"] = None

    # Candidate scale unit: the opening thrust. Floored so slow-dissemination
    # news (a near-zero first 2 minutes) can't produce an explosive ratio.
    rec["impulse"] = max(rec.get("mfe_2") or 0.0, 0.001 * ref_price)
    return rec


def build_move_table(variant: str = "first", refresh: bool = False) -> pd.DataFrame:
    """Measure every event. Cached to data/news_move_study/moves_{variant}.pkl."""
    out_path = OUT / f"moves_{variant}.pkl"
    if out_path.exists() and not refresh:
        return pd.read_pickle(out_path)

    ev = build_events(variant)
    recs: List[Dict] = []
    no_bars = 0
    for row in ev.itertuples(index=False):
        rec = measure_event(row.symbol, row.date_iso, row.ref_ts,
                            float(row.ref_price), int(row.direction))
        if rec is None:
            no_bars += 1
            continue
        rec["year"] = row.year
        rec["gross_pnl"] = getattr(row, "gross_pnl", None)
        recs.append(rec)

    df = pd.DataFrame(recs)
    log.info(f"measured {len(df)} events, {no_bars} without usable bars")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out_path)
    return df
