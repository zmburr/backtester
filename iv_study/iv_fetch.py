"""Fetch a constant-moneyness intraday ATM IV series for one trade.

Approach (see PLAN.md section 0): a parabolic underlying traverses 20-40%
intraday, so a single fixed contract's IV drifts across the skew surface and
fake-signals acceleration. Instead we pull a ladder of strikes bracketing the
day's range and, at each minute, interpolate IV at that bar's spot (the greeks
payload carries a synchronized underlying_price). Put and call are averaged.
A fixed-9:35-ATM-strike series is kept alongside as a drift cross-check.

Strike discovery gotcha: /list/strikes and /list/expirations are ALL-TIME, not
as-of-date, so the tradeable ladder is discovered from a chain snapshot on the
trade date. The snapshot also yields a parity-based spot estimate (strike where
|call mid - put mid| is smallest), avoiding any equity-data dependency.
"""

import logging

import numpy as np
import pandas as pd
import requests

from options_replay import theta_client
from options_replay.cache import load_cached, save_to_cache
from iv_study import config

logger = logging.getLogger(__name__)


def _pick_expirations(expirations: list, trade_date: str) -> list:
    """Candidate expirations: DTE in [MIN_DTE, MAX_DTE], nearest TARGET_DTE first."""
    base = pd.Timestamp(trade_date)
    scored = []
    for exp in expirations:
        try:
            dte = (pd.Timestamp(exp) - base).days
        except Exception:
            continue
        if config.MIN_DTE <= dte <= config.MAX_DTE:
            scored.append((abs(dte - config.TARGET_DTE), dte, exp))
    scored.sort()
    return [(exp, dte) for _, dte, exp in scored[:config.EXP_CANDIDATES]]


def _chain_at_time(symbol: str, date_iso: str, expiration: str,
                   time_of_day: str = None) -> pd.DataFrame:
    """NBBO chain for one expiration at a point in time on the trade date.

    Like theta_client.get_chain_snapshot but for an explicit expiration, so the
    expiration policy (MIN_DTE gate) stays in this module. Theta's 472 "no
    data" comes back as an empty frame instead of raising, so callers can fall
    through to the next candidate expiration.
    """
    date_fmt = date_iso.replace("-", "")
    exp_fmt = expiration.replace("-", "")
    time_fmt = time_of_day or config.SNAPSHOT_TIME

    cached = load_cached(symbol, date_fmt, "iv_study_chain",
                         exp=exp_fmt, time=time_fmt.replace(":", ""))
    if cached is not None:
        return cached

    try:
        rows = theta_client._paginated_get(f"{theta_client.V3}/option/at_time/quote", {
            "symbol": symbol.upper(),
            "expiration": exp_fmt,
            "strike": "*",
            "right": "both",
            "start_date": date_fmt,
            "end_date": date_fmt,
            "time_of_day": time_fmt,
        })
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 472:
            rows = []
        else:
            raise

    flat = []
    for row in rows:
        if isinstance(row, dict) and "contract" in row and "data" in row:
            if isinstance(row["data"], list) and row["data"]:
                flat.append({**row["contract"], **row["data"][0]})
        elif isinstance(row, dict):
            flat.append(row)

    df = pd.DataFrame(flat)
    if not df.empty:
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        for col in ("bid", "ask", "strike"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "right" in df.columns:
            df["right"] = df["right"].astype(str).str.upper().str.strip().str[0]
        df["mid"] = (df["bid"] + df["ask"]) / 2

    save_to_cache(df, symbol, date_fmt, "iv_study_chain",
                  exp=exp_fmt, time=time_fmt.replace(":", ""))
    return df


# Fallback snapshot times: at 09:35 a halted name (GME/AMC on the meme-peak
# days) shows a full chain with all-zero quotes, so strike discovery retries
# later in the session.
SNAPSHOT_TIMES = [None, "09:50:00", "10:15:00", "11:00:00", "13:00:00"]


def _discover(symbol: str, date_iso: str):
    """Yield (exp, dte, chain, spot) candidates: expirations x snapshot times."""
    try:
        expirations = theta_client.get_expirations(symbol)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 472:
            return
        raise
    for exp, dte in _pick_expirations(expirations, date_iso):
        for t in SNAPSHOT_TIMES:
            chain = _chain_at_time(symbol, date_iso, exp, t)
            if chain.empty:
                break  # nothing traded for this exp on this date -> next exp
            spot = _parity_spot(chain)
            if spot is not None:
                yield exp, dte, chain, spot
                break  # greeks failure falls through to the NEXT expiration


def _parity_spot(chain: pd.DataFrame):
    """Spot estimate: strike where |call mid - put mid| is smallest (put-call parity)."""
    if chain.empty or "right" not in chain.columns:
        return None
    live = chain[(chain["bid"] > 0) | (chain["ask"] > 0)]
    calls = live[live["right"] == "C"].set_index("strike")["mid"]
    puts = live[live["right"] == "P"].set_index("strike")["mid"]
    common = calls.index.intersection(puts.index)
    if len(common) == 0:
        return None
    diff = (calls.loc[common] - puts.loc[common]).abs()
    return float(diff.idxmin())


def _pick_ladder(chain: pd.DataFrame, spot: float) -> list:
    """Up to LADDER_N strikes evenly spanning [LADDER_LO, LADDER_HI] x spot,
    always including the nearest-ATM strike."""
    strikes = sorted(chain["strike"].dropna().unique())
    lo, hi = spot * config.LADDER_LO, spot * config.LADDER_HI
    band = [k for k in strikes if lo <= k <= hi]
    if not band:
        return []
    atm = min(band, key=lambda k: abs(k - spot))
    if len(band) <= config.LADDER_N:
        return band
    idx = np.unique(np.linspace(0, len(band) - 1, config.LADDER_N).round().astype(int))
    ladder = {band[i] for i in idx}
    ladder.add(atm)
    return sorted(ladder)


def _clean_greeks(g: pd.DataFrame) -> pd.DataFrame:
    """Drop failed-solve and dead-quote bars, plus the junk 09:30 opening bar."""
    if g.empty or "implied_vol" not in g.columns:
        return pd.DataFrame()
    g = g.copy()
    for col in ("implied_vol", "iv_error", "bid", "ask", "underlying_price"):
        if col in g.columns:
            g[col] = pd.to_numeric(g[col], errors="coerce")
    mask = g["implied_vol"] > 0
    if "iv_error" in g.columns:
        mask &= g["iv_error"].fillna(0) < config.IV_ERR_MAX
    if "bid" in g.columns and "ask" in g.columns:
        mask &= ~((g["bid"] == 0) & (g["ask"] == 0))
    g = g[mask]
    if g.empty:
        return pd.DataFrame()
    t = g.index
    keep = (t.hour * 60 + t.minute >= 9 * 60 + 31) & (t.hour < 16)
    return g[keep]


def _mask_bad_spot(spot: pd.Series, tol: float = 0.10) -> pd.Series:
    """Drop garbage underlying prints (LULD-halt artifacts: stale blocks like a
    flat $14 stretch on a $26 stock).

    A bar is dropped only if it deviates > tol from the last accepted value in
    BOTH a forward and a backward pass. Real fast moves and halt-reopen gaps
    always agree with their neighbors on at least one side, so they survive;
    garbage blocks of any length agree with neither side.
    """
    vals = spot.to_numpy(float)
    if len(vals) < 3:
        return spot

    def flags(v, seed):
        last = seed
        out = np.zeros(len(v), dtype=bool)
        for i, x in enumerate(v):
            if abs(x / last - 1) <= tol:
                last = x
            else:
                out[i] = True
        return out

    fwd = flags(vals, np.median(vals[:30]))
    bwd = flags(vals[::-1], np.median(vals[-30:]))[::-1]
    bad = fwd & bwd
    if bad.any():
        logger.info("Dropped %d garbage spot bars (halt artifacts)", int(bad.sum()))
    return spot[~bad]


def _interp_iv(strikes: np.ndarray, ivs: np.ndarray, spot: float):
    """IV at spot from valid (strike, iv) points; clamps outside the span."""
    if len(strikes) == 0:
        return np.nan, False
    if spot <= strikes[0]:
        return float(ivs[0]), spot < strikes[0]
    if spot >= strikes[-1]:
        return float(ivs[-1]), spot > strikes[-1]
    return float(np.interp(spot, strikes, ivs)), False


def _atm_series(frames: dict, spot935: float) -> pd.DataFrame:
    """Constant-moneyness ATM IV per minute from {(strike, right): greeks_df}.

    Columns: spot, atm_iv, atm_iv_c, atm_iv_p, fixed_iv, extrapolated, n_quotes.
    """
    iv_wide = {"C": {}, "P": {}}
    spot_cols = []
    for (strike, right), g in frames.items():
        iv_wide[right][strike] = g["implied_vol"]
        if "underlying_price" in g.columns:
            spot_cols.append(g["underlying_price"])

    spot = pd.concat(spot_cols, axis=1).median(axis=1).dropna().sort_index()
    spot = _mask_bad_spot(spot)
    panels = {r: pd.DataFrame(d).sort_index(axis=1).reindex(spot.index)
              for r, d in iv_wide.items() if d}
    if not panels or spot.empty:
        return pd.DataFrame()

    fixed_strike = None
    all_strikes = sorted({k for d in iv_wide.values() for k in d})
    if all_strikes:
        fixed_strike = min(all_strikes, key=lambda k: abs(k - spot935))

    rows = []
    for ts, s in spot.items():
        row = {"spot": s}
        extrap = False
        for right, panel in panels.items():
            vals = panel.loc[ts]
            valid = vals.dropna()
            iv, ex = _interp_iv(valid.index.to_numpy(float), valid.to_numpy(float), s)
            row[f"atm_iv_{right.lower()}"] = iv
            extrap |= ex
        both = [row.get("atm_iv_c"), row.get("atm_iv_p")]
        both = [v for v in both if v is not None and not np.isnan(v)]
        row["atm_iv"] = float(np.mean(both)) if both else np.nan
        row["extrapolated"] = extrap
        row["n_quotes"] = int(sum(panel.loc[ts].notna().sum() for panel in panels.values()))
        fixed = []
        for right, panel in panels.items():
            if fixed_strike in panel.columns and not np.isnan(panel.at[ts, fixed_strike]):
                fixed.append(panel.at[ts, fixed_strike])
        row["fixed_iv"] = float(np.mean(fixed)) if fixed else np.nan
        rows.append((ts, row))

    out = pd.DataFrame({ts: r for ts, r in rows}).T
    out.index.name = "timestamp"
    out.attrs["fixed_strike"] = fixed_strike
    return out


def fetch_trade_iv(symbol: str, date_iso: str) -> tuple:
    """Full pipeline for one trade day. Returns (series_df_or_None, meta_dict).

    meta["status"]: ok | thin | no_data | no_exp | no_chain
    """
    meta = {"ticker": symbol, "date": date_iso, "status": "no_data",
            "exp": "", "dte": "", "spot935": "", "fixed_strike": "",
            "n_strikes": 0, "strikes": "", "n_bars": 0, "valid_frac": 0.0,
            "extrap_frac": 0.0, "err": ""}

    meta["status"] = "no_chain"
    for exp, dte, chain, spot935 in _discover(symbol, date_iso):
        ladder = _pick_ladder(chain, spot935)
        if not ladder:
            meta["status"] = "no_chain"
            continue

        frames = {}
        for strike in ladder:
            for right in ("C", "P"):
                try:
                    g = theta_client.get_option_greeks(
                        symbol, exp, strike, right, date_iso, interval=config.INTERVAL)
                except theta_client.ThetaTerminalOfflineError:
                    raise
                except Exception as e:
                    logger.info("greeks failed %s %s %s%s: %s", symbol, exp, strike, right, e)
                    continue
                g = _clean_greeks(g)
                if not g.empty:
                    frames[(strike, right)] = g

        if not frames:
            meta["status"] = "no_data"
            continue

        series = _atm_series(frames, spot935)
        if series.empty:
            meta["status"] = "no_data"
            continue

        n_session_min = 389  # 09:31 .. 15:59
        valid = series["atm_iv"].notna()
        meta.update({
            "exp": exp, "dte": dte, "spot935": round(spot935, 2),
            "fixed_strike": series.attrs.get("fixed_strike"),
            "n_strikes": len(ladder), "strikes": "|".join(str(k) for k in ladder),
            "n_bars": int(len(series)),
            "valid_frac": round(float(valid.sum()) / n_session_min, 3),
            "extrap_frac": round(float(series["extrapolated"].mean()), 3),
        })
        meta["status"] = "ok" if meta["valid_frac"] >= config.MIN_VALID_FRAC else "thin"
        return series, meta

    return None, meta
