"""IV study on capitulation BOUNCE trades (bounce_data.csv, long side).

Mirror of the reversal-top study with t=0 at time_of_low_price: does IV blow
out into the capitulation low, and does the intraday IV peak mark the low?
For selloffs a rising IV is mechanical (spot-vol correlation), so the
interesting quantities are magnitude/extremity vs the name's own range and
the timing of the IV peak vs the price low -- not the direction of the ramp.

Reuses the reversal pipeline's building blocks; outputs are suffixed _bounce
so nothing collides with the top study. Per-trade minute series share the
same cache/naming (a series is defined by ticker+date, not by strategy).

Usage (from project root, Theta Terminal running):
    venv/Scripts/python.exe -m iv_study.bounce_study fetch     # intraday + controls
    venv/Scripts/python.exe -m iv_study.bounce_study analyze   # features + report
"""

import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from options_replay import theta_client
from iv_study import config
from iv_study.iv_fetch import fetch_trade_iv
from iv_study.fetch_iv import _save_series
from iv_study.pseudo_controls import fetch_control_day, _trading_days_before
from iv_study.build_features import load_series, tau_frame, track_a_features, track_b_features
from iv_study.trade_loader import _parse_et

logger = logging.getLogger(__name__)

BOUNCE_CSV = config.PROJECT_ROOT / "data" / "bounce_data.csv"
MANIFEST = config.DATA_DIR / "manifest_bounce.csv"
CONTROLS = config.CONTROLS_DIR / "control_marks_bounce.csv"
FEATURES = config.DATA_DIR / "iv_features_bounce.csv"
REPORT = config.REPORTS_DIR / "iv_report_bounce.html"

BASELINE_COLS = ["selloff_total_pct", "consecutive_down_days", "gap_pct", "rvol_score",
                 "atr_pct_move", "pct_off_30d_high", "closed_outside_lower_band",
                 "uvxy_close", "bounce_open_low_pct", "bounce_open_close_pct",
                 "bounce_duration", "trade_grade", "cap", "Setup"]


def _bucket(t) -> str:
    if pd.isna(t):
        return "unknown"
    m = t.hour * 60 + t.minute
    if m < 9 * 60 + 30:
        return "premarket"
    if m < 10 * 60:
        return "open30"
    return "post10"


def load_bounce_trades() -> pd.DataFrame:
    df = pd.read_csv(BOUNCE_CSV, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df["date_iso"] = df["date"].dt.strftime("%Y-%m-%d")
    df["t_low"] = _parse_et(df["time_of_low_price"])
    df["low_bucket"] = df["t_low"].apply(_bucket)
    logger.info("Loaded %d bounce trades | low buckets: %s",
                len(df), df["low_bucket"].value_counts().to_dict())
    return df


def fetch():
    if not theta_client.check_terminal_running():
        raise SystemExit("Theta Terminal is not reachable on localhost:25503.")
    trades = load_bounce_trades()

    rows = []
    for i, t in enumerate(trades.itertuples(), 1):
        logger.info("[%d/%d] %s %s (low %s, %s)",
                    i, len(trades), t.ticker, t.date_iso, t.t_low, t.low_bucket)
        try:
            series, meta = fetch_trade_iv(t.ticker, t.date_iso)
        except theta_client.ThetaTerminalOfflineError:
            raise SystemExit("Theta Terminal went offline; re-run to resume (cached).")
        except Exception as e:
            logger.exception("%s %s failed", t.ticker, t.date_iso)
            series, meta = None, {"ticker": t.ticker, "date": t.date_iso,
                                  "status": "error", "err": str(e)[:200]}
        meta["low_bucket"] = t.low_bucket
        meta["t_low"] = str(t.t_low)
        meta["path"] = _save_series(series, t.ticker, t.date_iso) if series is not None else ""
        rows.append(meta)
        logger.info("  -> %s (valid_frac=%s)", meta["status"], meta.get("valid_frac"))
    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST, index=False)
    logger.info("Status counts: %s", manifest["status"].value_counts().to_dict())

    # controls: the selloff days before each bounce
    wanted = sorted({(t.ticker, d) for t in trades.itertuples()
                     for d in _trading_days_before(t.date_iso, config.PSEUDO_CONTROL_DAYS)})
    logger.info("Fetching %d control ticker-days", len(wanted))

    def _one(key):
        try:
            return fetch_control_day(*key)
        except theta_client.ThetaTerminalOfflineError:
            raise
        except Exception:
            logger.exception("control %s %s failed", *key)
            return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        marks = [r for r in pool.map(_one, wanted) if r]
    config.CONTROLS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(marks).to_csv(CONTROLS, index=False)
    logger.info("%d/%d control marks -> %s", len(marks), len(wanted), CONTROLS)


def analyze():
    import base64
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from iv_study.event_study import (plot_trajectories, plot_iv_lead, lead_stats,
                                      _fig_to_b64, TAU_GRID)

    trades = load_bounce_trades()
    controls = pd.read_csv(CONTROLS)

    rows = []
    for t in trades.itertuples():
        series = load_series(t.ticker, t.date_iso)
        row = {"ticker": t.ticker, "date": t.date_iso, "low_bucket": t.low_bucket,
               "t_low": str(t.t_low), "has_series": series is not None}
        if series is not None and not series.empty:
            row.update(track_b_features(series, t.ticker, t.date_iso, controls))
            if t.low_bucket in ("open30", "post10") and pd.notna(t.t_low):
                row.update(track_a_features(tau_frame(series, t.t_low)))
        rows.append(row)
    feats = pd.DataFrame(rows)
    from iv_study.build_reference import prior_close_iv_pctile
    feats["prior_close_iv_pctile"] = [
        prior_close_iv_pctile(t, d, controls) for t, d in zip(feats["ticker"], feats["date"])
    ]
    csv = trades[["ticker", "date_iso"] + [c for c in BASELINE_COLS if c in trades.columns]]
    feats = feats.merge(csv, left_on=["ticker", "date"], right_on=["ticker", "date_iso"],
                        how="left").drop(columns="date_iso")
    feats.to_csv(FEATURES, index=False)
    logger.info("%d rows (%d with series, %d with iv_lead) -> %s",
                len(feats), int(feats["has_series"].sum()),
                int(feats["iv_lead"].notna().sum()), FEATURES)

    # ---- report ----
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html = ["<html><head><meta charset='utf-8'><title>IV at Capitulation Lows</title>",
            "<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;"
            "padding:0 16px;color:#222}h1,h2{color:#1a355e}table{border-collapse:collapse;margin:10px 0}"
            "td,th{border:1px solid #ccc;padding:4px 10px;font-size:13px}img{max-width:100%}"
            ".note{background:#f6f8fa;border-left:4px solid #1a355e;padding:8px 12px;font-size:14px}"
            "</style></head><body>",
            "<h1>IV at Capitulation Lows &mdash; bounce_data.csv</h1>",
            "<p class='note'>t=0 is time_of_low_price. iv_lead = t(IV peak) &minus; t(price low). "
            "Rising IV into a selloff is mechanical, so read magnitude/extremity and peak timing, "
            "not ramp direction. All trades bounced by construction (no non-bounce controls).</p>"]

    # Track A: trajectories + iv_lead per bucket
    html.append("<h2>1. ATM IV aligned to the intraday low</h2>")
    for bucket, label in (("post10", "Lows after 10:00 (real lead room)"),
                          ("open30", "Lows 9:30-10:00")):
        sub = trades[trades["low_bucket"] == bucket]
        cols = {}
        for t in sub.itertuples():
            series = load_series(t.ticker, t.date_iso)
            if series is None or series.empty or pd.isna(t.t_low):
                continue
            df = tau_frame(series, t.t_low)
            z = df.drop_duplicates(subset="tau").set_index("tau")["iv_z"].reindex(TAU_GRID)
            if z.notna().sum() >= 30:
                cols[f"{t.ticker} {t.date_iso}"] = z
        mat = pd.DataFrame(cols, index=TAU_GRID)
        if mat.shape[1]:
            html.append(f"<img src='data:image/png;base64,{_fig_to_b64(plot_trajectories(mat, label))}'>")

    html.append("<h2>2. iv_lead: does the IV peak mark the low?</h2>")
    stats_rows = []
    for bucket in ("post10", "open30"):
        sub = feats[feats["low_bucket"] == bucket]
        s = lead_stats(sub["iv_lead"])
        s["bucket"] = bucket
        stats_rows.append(s)
        if s["n"] >= 5:
            vals = sub["iv_lead"].dropna().to_numpy(float)
            title = f"iv_lead distribution — {bucket} (n={s['n']})"
            html.append(f"<img src='data:image/png;base64,{_fig_to_b64(plot_iv_lead(vals, title))}'>")
    html.append(pd.DataFrame(stats_rows).set_index("bucket").to_html())

    # Track B: selloff IV ramp + extremity
    html.append("<h2>3. The selloff lead-up: IV extremity into the bounce day</h2>")
    b = feats[feats["n_controls"].fillna(0) >= 4]
    tbl = {}
    for col, desc in (("iv_runup_chg", "close-IV change across the selloff window (%)"),
                      ("iv_ramp_final2d", "final 2-day IV change (%)"),
                      ("iv_ramp_final2d_pctile", "final 2-day change pctile within own window"),
                      ("prior_close_iv_pctile", "prior-close IV pctile vs own window"),
                      ("open_iv_pctile", "bounce-day open IV pctile vs own window"),
                      ("iv_gap_open", "IV gap at the bounce-day open (%)")):
        col_vals = b.get(col)
        if not isinstance(col_vals, pd.Series):
            continue
        v = pd.to_numeric(col_vals, errors="coerce").dropna()
        if len(v):
            tbl[desc] = {"n": len(v), "median": round(float(v.median()), 3),
                         "q25": round(float(v.quantile(.25)), 3),
                         "q75": round(float(v.quantile(.75)), 3),
                         "frac>0": round(float((v > 0).mean()), 2)}
    html.append(pd.DataFrame(tbl).T.to_html())

    html.append("</body></html>")
    REPORT.write_text("\n".join(html), encoding="utf-8")
    logger.info("Report -> %s", REPORT)

    print("\n=== iv_lead vs price low ===")
    print(pd.DataFrame(stats_rows).set_index("bucket").to_string())
    print("\n=== selloff lead-up ===")
    print(pd.DataFrame(tbl).T.to_string())


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    mode = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    if mode == "fetch":
        fetch()
    elif mode == "analyze":
        analyze()
    else:
        raise SystemExit("usage: python -m iv_study.bounce_study [fetch|analyze]")


if __name__ == "__main__":
    main()
