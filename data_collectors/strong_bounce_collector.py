"""
Scanner feature extractor for catalyst/breakout setups (SHEL-only)
------------------------------------------------------------------
- Backend: trillium_queries (SHEL) ONLY
- Intraday never hard-fails; if missing, features fall back to None.
- Pre-selloff reference: 30 trading-day high strictly BEFORE the anchor date.
- Deep logging: shapes, timestamps, sample OHLC; use --log-level DEBUG.
- Optional per-row debug JSON: --debug-json writes scanner_debug_dump.jsonl.

Outputs a CSV (scanner_training_set.csv or a timestamped fallback on Windows).
"""
from __future__ import annotations

import os
import sys
import json
import logging
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Backend import (SHEL only) + helpful sys.path hints for common layouts
# -----------------------------------------------------------------------------
HAS_SHEL = False
try:
    import data_queries.trillium_queries as shel  # type: ignore
    HAS_SHEL = True
except Exception:
    # try a couple of common alt paths
    for d in [
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd(),
        os.path.join(os.getcwd(), "backtester"),
        os.path.join(os.getcwd(), "backtester", "data_collectors"),
    ]:
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
    try:
        from backtester.data_collectors import trillium_queries as shel  # type: ignore
        HAS_SHEL = True
    except Exception as e:
        raise ImportError(
            "Could not import trillium_queries (SHEL). "
            "Ensure it's on PYTHONPATH or in the working dir."
        ) from e

# Logger
log = logging.getLogger("scanner_shel")

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _to_iso(date_str: str) -> str:
    """Accept 'YYYY-MM-DD' or 'MM/DD/YYYY' and return 'YYYY-MM-DD'."""
    try:
        if "/" in date_str:
            return datetime.strptime(date_str, "%m/%d/%Y").strftime("%Y-%m-%d")
        return date_str
    except Exception:
        return date_str

def _as_dt(x) -> pd.Timestamp:
    if isinstance(x, pd.Timestamp):
        return x.tz_localize(None) if x.tz is not None else x
    return pd.Timestamp(x)

def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly coerce a DataFrame to a tz-naive DatetimeIndex when possible.
    If we can't, we return the DataFrame as-is and the caller must avoid time filters.
    """
    if df is None or df.empty:
        return df

    # Already datetime?
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
        except Exception:
            pass
        return df

    # Promote a known datetime-like column to index if present
    for col in ["timestamp", "time", "t", "start", "end", "date", "datetime", "Date", "tradingDay", "day"]:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df = df.set_index(col, drop=True)
                if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                    df.index = df.index.tz_convert(None)
                return df
            except Exception:
                continue

    # Last resort: try to parse the index itself
    try:
        df.index = pd.to_datetime(df.index, errors="coerce")
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_convert(None)
    except Exception:
        pass
    return df

def _log_df(name: str, df: pd.DataFrame):
    if df is None:
        log.debug(f"{name}: None")
        return
    if df.empty:
        log.debug(f"{name}: empty")
        return
    first = df.index.min() if isinstance(df.index, pd.DatetimeIndex) else None
    last  = df.index.max() if isinstance(df.index, pd.DatetimeIndex) else None
    cols  = list(df.columns)
    sample = {}
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            try:
                sample[c] = {
                    "head": float(pd.Series(df[c]).head(1).iloc[0]),
                    "tail": float(pd.Series(df[c]).tail(1).iloc[0]),
                }
            except Exception:
                pass
    log.debug(json.dumps({
        "name": name, "rows": int(len(df)), "cols": cols,
        "first_ts": str(first) if first is not None else None,
        "last_ts": str(last) if last is not None else None,
        "samples": sample
    }))

def _safe_pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    try:
        if a is None or b is None or b == 0:
            return None
        return (a - b) / b
    except Exception:
        return None

def _between(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df.iloc[0:0]
    try:
        return df.between_time(start, end)
    except Exception:
        return df.iloc[0:0]

# -----------------------------------------------------------------------------
# SHEL (trillium_queries) wrappers
# -----------------------------------------------------------------------------
def _shel_get_daily(ticker: str, date: str):
    return shel.get_daily(ticker, date)

def _shel_get_levels(ticker: str, date: str, window: int) -> pd.DataFrame:
    return shel.get_levels_data(ticker, date, window, "bar-1day")

def _shel_minute(ticker: str, date: str) -> pd.DataFrame:
    return shel.get_intraday(ticker, date, "bar-1min")

def _shel_price_fb(ticker: str, base_date: str, days: int) -> Optional[float]:
    return shel.get_price_with_fallback(ticker, base_date, days)

# -----------------------------------------------------------------------------
# Data access helpers (SHEL-only)
# -----------------------------------------------------------------------------
def get_daily_ohlc(ticker: str, date: str) -> (Dict[str, Optional[float]], str):
    """
    Try SHEL daily; if that fails or is missing, synthesize from intraday.
    """
    try:
        res = _shel_get_daily(ticker, date)
        if isinstance(res, dict):
            out = {k: res.get(k) for k in ["open","close","high","low","volume"]}
        else:
            out = {
                "open": getattr(res, "open", None),
                "close": getattr(res, "close", None),
                "high": getattr(res, "high", None),
                "low": getattr(res, "low", None),
                "volume": getattr(res, "volume", None),
            }
        log.debug(f"daily_ohlc[shel] -> {out}")
        return out, "shel"
    except Exception as e:
        log.debug(f"daily_ohlc[shel] raised: {repr(e)}; attempting synth from intraday")

    # synthesize from intraday if daily not available
    intra, _ = get_intraday_minute(ticker, date)
    if intra is not None and not intra.empty:
        reg = _between(intra, "09:30", "16:00")
        first_bar = reg.iloc[0] if not reg.empty else intra.iloc[0]
        last_bar  = reg.iloc[-1] if not reg.empty else intra.iloc[-1]
        out = {
            "open": float(first_bar.get("open", np.nan)),
            "close": float(last_bar.get("close", np.nan)),
            "high": float(intra["high"].max()),
            "low": float(intra["low"].min()),
            "volume": float(intra["volume"].sum()),
        }
        log.debug(f"daily_ohlc[synth_from_intraday] -> {out}")
        return out, "synth_from_intraday"

    return {"open": None, "close": None, "high": None, "low": None, "volume": None}, "none"

def get_intraday_minute(ticker: str, date: str) -> (pd.DataFrame, str):
    try:
        df = _shel_minute(ticker, date)
        df = _ensure_dt_index(df)
        df = df.sort_index() if df is not None and not df.empty else df
        _log_df("intraday_minute[shel]", df)
        return df, "shel"
    except Exception as e:
        log.debug(f"intraday_minute[shel] raised: {repr(e)}")
        return pd.DataFrame(columns=["open","high","low","close","volume"]), "none"

def get_levels_window_prior(ticker: str, date: str, window: int) -> (pd.DataFrame, str):
    """
    Fetch up to 'window' daily bars using SHEL and then return ONLY rows
    strictly before the anchor date.
    """
    try:
        df = _shel_get_levels(ticker, date, window)
        df = _ensure_dt_index(df)
        cutoff = _as_dt(_to_iso(date))
        if isinstance(df.index, pd.DatetimeIndex):
            df_prior = df[df.index < cutoff]
        else:
            log.debug("levels_window: no datetime index; returning empty prior window")
            df_prior = df.iloc[0:0]
        _log_df(f"levels_window_prior[shel][{window}]", df_prior)
        return df_prior.sort_index(), "shel"
    except Exception as e:
        log.debug(f"levels_window_prior[shel] raised: {repr(e)}")
        return pd.DataFrame(columns=["open","high","low","close","volume"]), "none"

# -----------------------------------------------------------------------------
# Feature engineering (pre-selloff baselines)
# -----------------------------------------------------------------------------
def _atr_pct(df: pd.DataFrame, window: int = 30) -> Optional[float]:
    if df is None or df.empty:
        return None
    tail = df.tail(window)
    if tail.empty:
        return None
    tr = np.maximum(
        tail["high"] - tail["low"],
        np.maximum(abs(tail["high"] - tail["close"].shift(1)),
                   abs(tail["low"] - tail["close"].shift(1)))
    )
    tr = tr.dropna()
    if tr.empty or tail["close"].dropna().empty:
        return None
    return float(tr.mean() / tail["close"].dropna().mean())

def _log_slope_per_day(prices: pd.Series) -> Optional[float]:
    s = prices.dropna()
    if len(s) < 10:
        return None
    y = np.log(s.values)
    x = np.arange(len(s))
    return float(np.polyfit(x, y, 1)[0])

@dataclass
class Setup:
    ticker: str
    date: str  # 'YYYY-MM-DD' or 'MM/DD/YYYY'

def compute_features(setup: Setup, debug_json: bool = False) -> Dict[str, Optional[float]]:
    ticker = setup.ticker.upper()
    date   = _to_iso(setup.date)
    dbg: Dict[str, object] = {"ticker": ticker, "date": date}

    # Anchor day snapshot
    d_ohlc, daily_backend = get_daily_ohlc(ticker, date)
    dbg["daily_backend"] = daily_backend
    open_px, close_px, high_px, low_px, day_vol = (
        d_ohlc.get("open"), d_ohlc.get("close"), d_ohlc.get("high"), d_ohlc.get("low"), d_ohlc.get("volume")
    )
    log.debug(f"anchor_ohlc: {d_ohlc}")

    # Pre-selloff baselines strictly before anchor
    pre20,  _ = get_levels_window_prior(ticker, date, 40)    # enough to get 20 prior
    pre30,  _ = get_levels_window_prior(ticker, date, 50)
    pre60,  _ = get_levels_window_prior(ticker, date, 80)
    pre63,  _ = get_levels_window_prior(ticker, date, 90)
    pre126, _ = get_levels_window_prior(ticker, date, 140)
    pre252, _ = get_levels_window_prior(ticker, date, 300)
    pre1200,_ = get_levels_window_prior(ticker, date, 1250)

    for nm, df in [("pre20",pre20),("pre30",pre30),("pre60",pre60),("pre63",pre63),("pre126",pre126),("pre252",pre252)]:
        _log_df(nm, df)

    # Reference 30D high (pre-selloff only)
    ref30_high = float(pre30["high"].max()) if not pre30.empty else None
    try:
        ref30_high_dt = pre30["high"].idxmax().strftime("%Y-%m-%d") if not pre30.empty else None
    except Exception:
        ref30_high_dt = None
    pct_off_ref30H = _safe_pct(open_px, ref30_high)

    # 3m/6m % move vs earliest close in the prior window
    def _earliest_close(df: pd.DataFrame) -> Optional[float]:
        if df is None or df.empty:
            return None
        try:
            return float(df.iloc[0]["close"])  # ascending order assumed
        except Exception:
            return None

    base63  = _earliest_close(pre63)
    base126 = _earliest_close(pre126)
    pct_change_63  = _safe_pct(open_px, base63)
    pct_change_126 = _safe_pct(open_px, base126)

    # Off longer highs (context)
    def _max_high(df: pd.DataFrame) -> Optional[float]:
        try:
            return float(df["high"].max()) if not df.empty else None
        except Exception:
            return None

    h63_pre   = _max_high(pre63)
    h252_pre  = _max_high(pre252)
    hAll_pre  = _max_high(pre1200)

    pct_off_63H_pre  = _safe_pct(open_px, h63_pre)
    pct_off_252H_pre = _safe_pct(open_px, h252_pre)
    pct_off_ATH_pre  = _safe_pct(open_px, hAll_pre)

    # Price slopes/accel using pre-anchor windows only
    slope_20 = _log_slope_per_day(pre20["close"]) if not pre20.empty else None
    slope_60 = _log_slope_per_day(pre60["close"]) if not pre60.empty else None
    price_accel = (slope_20 - slope_60) if (slope_20 is not None and slope_60 is not None) else None

    # ATR% over pre-period
    atr_pct_30 = _atr_pct(pre60 if not pre60.empty else pre30, window=30)
    price_accel_norm_atr = (price_accel / atr_pct_30) if (price_accel is not None and atr_pct_30 not in (None, 0)) else None

    # Volume baselines from pre-period only
    def _adv(df: pd.DataFrame, n: int) -> Optional[float]:
        if df is None or df.empty:
            return None
        tail = df.tail(n)
        if tail.empty:
            return None
        return float(tail["volume"].mean())

    vol_adv_20 = _adv(pre252, 20)
    vol_adv_60 = _adv(pre252, 60)
    vol_accel_20_60 = (vol_adv_20 / vol_adv_60) if (vol_adv_20 and vol_adv_60 and vol_adv_60 != 0) else None

    # Intraday context (optional; never hard-fails)
    intra, intra_backend = get_intraday_minute(ticker, date)
    dbg["intraday_backend"] = intra_backend
    pre   = _between(intra, "06:00", "09:29:59")
    reg   = _between(intra, "09:30", "16:00")

    vol_pre  = float(pre["volume"].sum()) if pre is not None and not pre.empty else None
    vol_5    = float(reg.between_time("09:30", "09:35")["volume"].sum()) if reg is not None and not reg.empty else None
    vol_10   = float(reg.between_time("09:30", "09:40")["volume"].sum()) if reg is not None and not reg.empty else None
    vol_15   = float(reg.between_time("09:30", "09:45")["volume"].sum()) if reg is not None and not reg.empty else None
    vol_30   = float(reg.between_time("09:30", "10:00")["volume"].sum()) if reg is not None and not reg.empty else None

    day_vol_calc = float(intra["volume"].sum()) if intra is not None and not intra.empty else d_ohlc.get("volume")

    def _pct_of_adv(x: Optional[float], adv: Optional[float]) -> Optional[float]:
        return (x / adv) if (x is not None and adv not in (None, 0)) else None

    pre_vs_adv20  = _pct_of_adv(vol_pre, vol_adv_20)
    v5_vs_adv20   = _pct_of_adv(vol_5, vol_adv_20)
    v10_vs_adv20  = _pct_of_adv(vol_10, vol_adv_20)
    v15_vs_adv20  = _pct_of_adv(vol_15, vol_adv_20)
    v30_vs_adv20  = _pct_of_adv(vol_30, vol_adv_20)
    day_vs_adv20  = _pct_of_adv(day_vol_calc, vol_adv_20)

    # Day context (gap/O->H/O->C) using prev close strictly before anchor
    prev_close = float(pre252["close"].iloc[-1]) if not pre252.empty else None
    gap_pct = _safe_pct(open_px, prev_close)
    open_high_pct = _safe_pct(high_px, open_px)
    open_close_pct = _safe_pct(close_px, open_px)

    # Days since reference high
    try:
        days_since_ref30_high = int((pd.Timestamp(date) - pd.Timestamp(ref30_high_dt)).days) if ref30_high_dt else None
    except Exception:
        days_since_ref30_high = None

    out = {
        "ticker": ticker,
        "date": date,
        "backend_daily": daily_backend,
        "backend_intraday": intra_backend,
        # Anchor snapshot
        "open": open_px, "close": close_px, "high": high_px, "low": low_px,
        # Pre-selloff reference
        "ref30_high": ref30_high,
        "ref30_high_date": ref30_high_dt,
        "pct_off_ref30H": pct_off_ref30H,
        "days_since_ref30_high": days_since_ref30_high,
        # Trend state
        "pct_change_63": pct_change_63,
        "pct_change_126": pct_change_126,
        # Longer off-highs (context, pre-anchor)
        "pct_off_63H_pre": pct_off_63H_pre,
        "pct_off_252H_pre": pct_off_252H_pre,
        "pct_off_ATH_pre": pct_off_ATH_pre,
        # Slope/accel
        "price_slope_20": slope_20,
        "price_slope_60": slope_60,
        "price_accel": price_accel,
        "atr_pct_30": atr_pct_30,
        "price_accel_norm_atr": price_accel_norm_atr,
        # Volume baselines & accelerants
        "day_volume": day_vol_calc,
        "vol_adv_20": vol_adv_20,
        "vol_adv_60": vol_adv_60,
        "vol_accel_20_60": vol_accel_20_60,
        "premarket_vol": vol_pre,
        "vol_first_5m": vol_5,
        "vol_first_10m": vol_10,
        "vol_first_15m": vol_15,
        "vol_first_30m": vol_30,
        "premarket_vs_adv20": pre_vs_adv20,
        "v5_vs_adv20": v5_vs_adv20,
        "v10_vs_adv20": v10_vs_adv20,
        "v15_vs_adv20": v15_vs_adv20,
        "v30_vs_adv20": v30_vs_adv20,
        "day_vol_vs_adv20": day_vs_adv20,
        # Day performance
        "gap_pct": gap_pct,
        "open_high_pct": open_high_pct,
        "open_close_pct": open_close_pct,
    }

    if debug_json:
        out["_debug"] = dbg
    return out

# -----------------------------------------------------------------------------
# Alert scaffolding (bands to be learned later)
# -----------------------------------------------------------------------------
def evaluate_alerts(row: pd.Series, cfg: Dict[str, Optional[float]]) -> bool:
    def ok(key: str, cmp) -> bool:
        th = cfg.get(key)
        return True if th is None else cmp(float(row.get(key, np.nan)), float(th))

    return (
        ok("pct_change_63",  lambda x, t: x >= t)
        and ok("pct_change_126", lambda x, t: x >= t)
        and ok("pct_off_ref30H", lambda x, t: x >= -abs(t))
        and ok("price_accel",  lambda x, t: x >= t)
        and ok("price_accel_norm_atr", lambda x, t: x >= t)
        and ok("vol_accel_20_60", lambda x, t: x >= t)
        and ok("v30_vs_adv20", lambda x, t: x >= t)
        and ok("gap_pct", lambda x, t: x >= t)
        and ok("open_high_pct", lambda x, t: x >= t)
    )

DEFAULT_CONFIG = {
    "pct_change_63": None,
    "pct_change_126": None,
    "pct_off_ref30H": None,      # e.g., within X% of 30D pre-high (set later)
    "price_accel": None,
    "price_accel_norm_atr": None,
    "vol_accel_20_60": None,
    "v30_vs_adv20": None,
    "gap_pct": None,
    "open_high_pct": None,
}

# -----------------------------------------------------------------------------
# Seed and runner
# -----------------------------------------------------------------------------
@dataclass
class Setup:
    ticker: str
    date: str

SEED_SETUPS: List[Setup] = [
    Setup("NVAX", "8/12/2020"),
    Setup("PLTR", "8/20/2025"),
    Setup("PLTR", "2/20/2025"),
    Setup("LAES", "1/8/2025"),
    Setup("MRNA", "5/27/2020"),
    Setup("ETHE", "12/20/2024"),
]

def build_training_table(setups: List[Setup], debug_json: bool = False) -> pd.DataFrame:
    rows = []
    for s in setups:
        try:
            feats = compute_features(s, debug_json=debug_json)
            rows.append(feats)
        except Exception as e:
            log.exception(f"compute_features failed for {s.ticker} on {s.date}")
            rows.append({"ticker": s.ticker, "date": _to_iso(s.date), "error": str(e)})
    df = pd.DataFrame(rows)
    df["alert"] = df.apply(lambda r: evaluate_alerts(r, DEFAULT_CONFIG), axis=1)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default=os.getenv("SCANNER_LOG_LEVEL", "INFO"))
    parser.add_argument("--debug-json", action="store_true")
    args = parser.parse_args()

    # Proper logging formatter: time format via datefmt, not in format string
    lvl = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log.info(f"Backend (SHEL) available: {HAS_SHEL}")

    df = build_training_table(SEED_SETUPS, debug_json=args.debug_json)

    # Safe CSV write (handles Windows file lock)
    out_path = "scanner_training_set.csv"
    try:
        df.to_csv(out_path, index=False)
        wrote_path = out_path
    except PermissionError:
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        wrote_path = f"scanner_training_set_{ts}.csv"
        df.to_csv(wrote_path, index=False)
        log.warning(f"Permission denied on {out_path}; wrote to {wrote_path} instead")

    cols = [
        "ticker","date","backend_daily","backend_intraday",
        "pct_change_63","pct_change_126","pct_off_ref30H","ref30_high","ref30_high_date",
        "price_slope_20","price_slope_60","price_accel","price_accel_norm_atr",
        "vol_adv_20","vol_adv_60","vol_accel_20_60","v30_vs_adv20","day_vol_vs_adv20",
        "gap_pct","open_high_pct","open_close_pct","alert"
    ]
    view = [c for c in cols if c in df.columns]
    with pd.option_context('display.width', 160, 'display.max_columns', None, 'display.float_format', '{:.4f}'.format):
        try:
            print(df[view])
        except Exception:
            print(df)

    log.info(f"Wrote {wrote_path} with {len(df)} rows.")

    if args.debug_json:
        dbg_path = "scanner_debug_dump.jsonl"
        with open(dbg_path, "w") as f:
            for _, r in df.iterrows():
                j = r.to_dict()
                if "_debug" in j:
                    f.write(json.dumps(j["_debug"]) + "\n")
        log.info(f"Wrote per-row debug JSON to {dbg_path}")

if __name__ == "__main__":
    main()
