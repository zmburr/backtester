"""Phase 02 -- per-trade IV features from the fetched series -> iv_features.csv.

Track A (timing, RTH tops): smoothed/z-scored ATM IV aligned to the top;
  iv_lead = t_iv_peak - t_top is the pre-registered primary statistic.
Track B (day-level, all trades): open IV vs the same ticker's pseudo-control
  run-up days (open-vs-prior-close IV gap, open-IV percentile).

Usage (from project root, after fetch_iv --all and pseudo_controls):
    venv/Scripts/python.exe -m iv_study.build_features
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from iv_study import config, trade_loader

logger = logging.getLogger(__name__)

SMOOTH_BARS = 5


def load_series(ticker: str, date_iso: str):
    base = config.DATA_DIR / f"{ticker}_{date_iso.replace('-', '')}"
    for path in (base.with_suffix(".parquet"), base.with_suffix(".pkl")):
        if path.exists():
            return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_pickle(path)
    return None


def tau_frame(series: pd.DataFrame, t_top) -> pd.DataFrame:
    """Series with tau (minutes relative to the top) plus smoothed / z-scored IV."""
    df = series.copy()
    df["tau"] = ((df.index - t_top).total_seconds() / 60.0).round().astype(int)
    df["iv_s"] = df["atm_iv"].astype(float).rolling(SMOOTH_BARS, min_periods=3).median()

    base = df[(df["tau"] >= config.BASELINE[0]) & (df["tau"] <= config.BASELINE[1])]["iv_s"].dropna()
    if len(base) < 10:  # top too early for the [-60,-30] baseline -> first 15 session bars
        base = df["iv_s"].dropna().iloc[:15]
        base = base[df.loc[base.index, "tau"] <= -5]  # never let the top leak into its own baseline
    if len(base) < 5 or base.std() == 0 or np.isnan(base.std()):
        df["iv_z"] = np.nan
    else:
        df["iv_z"] = (df["iv_s"] - base.mean()) / base.std()
    df.attrs["iv_base"] = float(base.mean()) if len(base) else np.nan
    return df


def track_a_features(df: pd.DataFrame) -> dict:
    """Timing features on the [-WIN_PRE, +WIN_POST] window. NaNs if no window."""
    out = {k: np.nan for k in (
        "iv_lead", "iv_lead_pre", "t_vel_zero_cross", "accel_sign_pre",
        "iv_base", "iv_at_top", "iv_runup_slope", "iv_divergence", "win_bars")}
    win = df[(df["tau"] >= -config.WIN_PRE) & (df["tau"] <= config.WIN_POST)].dropna(subset=["iv_s"])
    out["win_bars"] = len(win)
    if len(win) < 30 or win["iv_z"].isna().all():
        return out

    out["iv_base"] = df.attrs["iv_base"]
    at_top = win[win["tau"].between(-2, 2)]["atm_iv"]
    out["iv_at_top"] = float(at_top.median()) if len(at_top) else np.nan

    out["iv_lead"] = float(win.loc[win["iv_s"].idxmax(), "tau"])
    pre = win[win["tau"] <= 0]
    if len(pre) >= 10:
        out["iv_lead_pre"] = float(pre.loc[pre["iv_s"].idxmax(), "tau"])
        # divergence: IV printed a clearly higher high >=5 min before the price top
        early_max = pre[pre["tau"] <= -5]["iv_s"].max()
        top_iv = pre["iv_s"].iloc[-1]
        if not np.isnan(early_max) and not np.isnan(top_iv):
            out["iv_divergence"] = bool(early_max > top_iv * 1.01)

    vel = win.set_index("tau")["iv_z"].diff()
    accel = vel.diff()
    pre_vel = vel[vel.index <= 0].dropna()
    crosses = pre_vel[(pre_vel < 0) & (pre_vel.shift(1) > 0)]
    if len(crosses):
        out["t_vel_zero_cross"] = float(crosses.index[-1])
    pre_acc = accel[(accel.index >= -15) & (accel.index <= 0)].dropna()
    if len(pre_acc):
        out["accel_sign_pre"] = float(pre_acc.mean())

    runup = win[(win["tau"] >= -30) & (win["tau"] <= 0)].dropna(subset=["iv_z"])
    if len(runup) >= 10:
        out["iv_runup_slope"] = float(np.polyfit(runup["tau"], runup["iv_z"], 1)[0])
    return out


def track_b_features(series: pd.DataFrame, ticker: str, date_iso: str,
                     controls: pd.DataFrame) -> dict:
    """Open-IV behavior vs the same ticker's prior run-up days."""
    out = {k: np.nan for k in (
        "open_iv", "prior_close_iv", "iv_gap_open", "n_controls",
        "ctrl_gap_mean", "ctrl_gap_std", "iv_gap_z", "open_iv_pctile",
        "iv_runup_chg", "iv_ramp_slope", "iv_ramp_final2d", "iv_ramp_final2d_pctile")}

    first_bars = series["atm_iv"].dropna().iloc[:10]
    if len(first_bars):
        out["open_iv"] = float(first_bars.median())

    ctrl = controls[(controls["ticker"] == ticker) & (controls["date"] < date_iso)]
    ctrl = ctrl.sort_values("date").tail(config.PSEUDO_CONTROL_DAYS)
    out["n_controls"] = len(ctrl)
    if ctrl.empty:
        return out

    out["prior_close_iv"] = float(ctrl["close_iv"].iloc[-1])
    if not np.isnan(out["open_iv"]) and out["prior_close_iv"] > 0:
        out["iv_gap_open"] = out["open_iv"] / out["prior_close_iv"] - 1

    # control-day gaps: each control day's open vs the previous control day's close
    gaps = (ctrl["open_iv"].to_numpy()[1:] / ctrl["close_iv"].to_numpy()[:-1]) - 1
    gaps = gaps[np.isfinite(gaps)]
    if len(gaps) >= 2:
        out["ctrl_gap_mean"] = float(gaps.mean())
        out["ctrl_gap_std"] = float(gaps.std(ddof=1))
        if out["ctrl_gap_std"] > 0 and not np.isnan(out["iv_gap_open"]):
            out["iv_gap_z"] = (out["iv_gap_open"] - out["ctrl_gap_mean"]) / out["ctrl_gap_std"]
    if not np.isnan(out["open_iv"]):
        opens = ctrl["open_iv"].dropna()
        if len(opens):
            out["open_iv_pctile"] = float(100.0 * (opens < out["open_iv"]).mean())

    # The lead-up hypothesis: IV RAMPS during the run-up days into the reversal
    # day. All of this is knowable before the reversal day's open (close marks
    # only), so it is a candidate day-selection signal.
    closes = ctrl["close_iv"].to_numpy(float)
    closes = closes[np.isfinite(closes)]
    if len(closes) >= 4:
        out["iv_runup_chg"] = float(closes[-1] / closes[0] - 1)
        x = np.arange(len(closes))
        out["iv_ramp_slope"] = float(np.polyfit(x, closes / closes.mean(), 1)[0])
        two_day = closes[2:] / closes[:-2] - 1  # 2-day IV change ending each day
        out["iv_ramp_final2d"] = float(two_day[-1])
        if len(two_day) >= 3:
            # did the ramp STEEPEN right before the top vs earlier in the run-up?
            out["iv_ramp_final2d_pctile"] = float(
                100.0 * (two_day[:-1] < two_day[-1]).mean())
    return out


BASELINE_COLS = [
    "rvol_score", "atr_pct", "atr_pct_move", "gap_pct", "consecutive_up_days",
    "uvxy_close", "pct_change_3", "pct_change_15", "pct_from_20mav",
    "day_of_range_pct", "percent_of_premarket_vol", "vol_ratio_5min_to_pm",
    "spy_5day_return", "gap_from_pm_high", "reversal_duration",
    "reversal_open_low_pct", "reversal_open_close_pct", "setup", "trade_grade", "cap",
]


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    trades = trade_loader.optionable_trades()
    controls_path = config.CONTROLS_DIR / "control_marks.csv"
    controls = (pd.read_csv(controls_path) if controls_path.exists()
                else pd.DataFrame(columns=["ticker", "date", "open_iv", "close_iv"]))
    logger.info("Loaded %d control marks", len(controls))

    rows = []
    for t in trades.itertuples():
        series = load_series(t.ticker, t.date_iso)
        row = {"ticker": t.ticker, "date": t.date_iso, "top_bucket": t.top_bucket,
               "t_top": str(t.t_top), "has_series": series is not None}
        if series is not None and not series.empty:
            row.update(track_b_features(series, t.ticker, t.date_iso, controls))
            if t.top_bucket in ("open30", "post10") and pd.notna(t.t_top):
                row.update(track_a_features(tau_frame(series, t.t_top)))
        rows.append(row)

    feats = pd.DataFrame(rows)
    csv = trades[["ticker", "date_iso"] + [c for c in BASELINE_COLS if c in trades.columns]]
    feats = feats.merge(csv, left_on=["ticker", "date"], right_on=["ticker", "date_iso"],
                        how="left").drop(columns="date_iso")

    out = config.DATA_DIR / "iv_features.csv"
    feats.to_csv(out, index=False)
    logger.info("%d rows (%d with series, %d with iv_lead) -> %s",
                len(feats), int(feats["has_series"].sum()),
                int(feats["iv_lead"].notna().sum()) if "iv_lead" in feats else 0, out)


if __name__ == "__main__":
    main()
