"""Episodes: one per (symbol, date, side) first news entry in the TRAC trade log.

Source: ExitMonitor/data/trade_data.csv, rows tagged `news` (not `[OPT]`).
Anchor = when live monitoring would start (trac_trader's headline_time is the
position-detection time, ~seconds after the fill):
  * Start with real seconds (newer rows)  -> Start + DETECT_LAG_S
  * Start at minute precision (older)     -> Start + 30 s + DETECT_LAG_S
    (fill uniform inside the minute; ±30 s jitter is a sensitivity variant)
Ref is recomputed with orderPipe's live logic (ref_price_calculator on 30s bars
over the 40 min before the anchor, clamped at 9:30) plus the trac_trader /
replay wrong-side guard, so the ref phase and R unit are what live would have
used. The logged ref is kept as a sensitivity variant.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from checkpoint_study import config

log = logging.getLogger(__name__)
ET = "US/Eastern"

# Mongo headline_time - trade_data Start on rows with real seconds: n=23,
# median 11 s, IQR 6.5-24.5 s (calibrate_anchor, 2026-09-29). Thin n — the
# ±30 s anchor jitter variant covers it.
DETECT_LAG_S = 11
REF_LOOKBACK_MIN = 40     # ref_price_calculator.lookback_minutes


def _parse_start(s) -> tuple:
    """(tz-aware ET Timestamp, has_seconds)."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None, False
    s = str(s).strip()
    try:
        ts = pd.Timestamp(s)
    except (ValueError, TypeError):
        return None, False
    ts = ts.tz_localize(ET) if ts.tzinfo is None else ts.tz_convert(ET)
    return ts, ts.second != 0


def load_trades(path=config.TRADE_LOG) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    tags = df["Tags"].fillna("").str.lower()
    df = df[tags.str.contains(r"\bnews\b", regex=True) & ~tags.str.contains(r"\[opt\]")].copy()
    parsed = [_parse_start(s) for s in df["Start"]]
    df["start_ts"] = [p[0] for p in parsed]
    df["start_has_seconds"] = [p[1] for p in parsed]
    df["symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    df["side"] = np.sign(pd.to_numeric(df["Side"], errors="coerce"))
    df["entry"] = pd.to_numeric(df["Avg Price at Max"], errors="coerce")
    df["max_size"] = pd.to_numeric(df["Max Size"], errors="coerce").abs()
    df["ref_logged"] = pd.to_numeric(df["ref_price"], errors="coerce")
    df["gross_pnl"] = pd.to_numeric(df["Gross P&L"], errors="coerce")
    for c in ("mkt_cap", "ADTV_$", "spread_%", "is_ETF"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["start_ts", "side", "entry"])
    df = df[(df["side"] != 0) & (df["entry"] > 0)]
    df["date"] = df["start_ts"].map(lambda t: t.strftime("%Y-%m-%d"))
    return df


def build_episodes(path=config.TRADE_LOG) -> pd.DataFrame:
    """First entry per (symbol, date, side), anchored, RTH-scoped. Refs/R are
    filled later by attach_ref_and_r() because they need the bars."""
    df = load_trades(path).sort_values("start_ts")
    ep = df.drop_duplicates(subset=["symbol", "date", "side"], keep="first").copy()
    pad = np.where(ep["start_has_seconds"], 0, 30) + DETECT_LAG_S
    ep["anchor"] = [(t + pd.Timedelta(seconds=int(s))).round("5s") for t, s in zip(ep["start_ts"], pad)]
    ep["anchor_method"] = np.where(ep["start_has_seconds"], "seconds", "minute+30s")
    tod = ep["anchor"].map(lambda t: t.strftime("%H:%M"))
    ep["in_scope"] = (tod >= config.RTH_START) & (tod <= config.RTH_END)
    ep["split"] = np.where(ep["date"] <= config.TRAIN_END, "train", "test")
    ep["episode_id"] = ep["symbol"] + "_" + ep["date"] + "_" + ep["side"].map({1.0: "L", -1.0: "S"})
    cols = ["episode_id", "symbol", "date", "side", "anchor", "anchor_method", "start_ts",
            "entry", "max_size", "ref_logged", "gross_pnl", "mkt_cap", "ADTV_$", "spread_%",
            "is_ETF", "in_scope", "split"]
    return ep[[c for c in cols if c in ep.columns]].reset_index(drop=True)


def bars_30s(fives: pd.DataFrame) -> pd.DataFrame:
    return (fives.resample("30s", label="right", closed="right")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last",
                  "volume": "sum"}).dropna(subset=["close"]))


def live_ref(fives: pd.DataFrame, anchor: pd.Timestamp, side: int, entry: float,
             ref_fn) -> tuple:
    """(ref, source, guarded) the way trac_trader + replay compute it live."""
    start = max(anchor - pd.Timedelta(minutes=REF_LOOKBACK_MIN),
                anchor.normalize() + pd.Timedelta(hours=9, minutes=30))
    b = bars_30s(fives[(fives.index > start - pd.Timedelta(seconds=30)) & (fives.index <= anchor)])
    b = b[(b.index >= start) & (b.index <= anchor)]
    info = ref_fn(b, side="long" if side > 0 else "short", entry=float(entry)) if len(b) else None
    first = fives[fives.index > anchor]
    last = float(first["close"].iloc[0]) if len(first) else float(entry)
    if info is None or info.get("price") is None:
        return last, "fallback last", True
    ref, src = float(info["price"]), info.get("source", "")
    # replay/trac_trader guard: a ref on the wrong side of the last price
    if side > 0 and ref > last:
        return round(last - 0.05, 2), src + " | guarded", True
    if side < 0 and ref < last:
        return round(last + 0.05, 2), src + " | guarded", True
    return ref, src, False


def calibrate_anchor(mongo_docs: pd.DataFrame, trades: pd.DataFrame) -> pd.Series:
    """Detection lag = Mongo headline_time - trade_data Start, on rows whose Start
    has real seconds. Returns the lag distribution (seconds)."""
    t = trades[trades["start_has_seconds"]][["symbol", "date", "start_ts"]]
    m = mongo_docs.merge(t, on=["symbol", "date"])
    lag = (m["headline_ts"] - m["start_ts"]).dt.total_seconds()
    return lag[(lag > -120) & (lag < 180)]
