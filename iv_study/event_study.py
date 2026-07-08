"""Phase 03 -- event study, plots, and self-contained HTML report.

Pre-registered primary statistic: iv_lead = t_iv_peak - t_top (minutes) on the
post-10am RTH-top bucket. Negative median => IV peaks before price => the
hypothesis is supported. Everything else is exploratory. All numbers carry
their n; no significance language (n forbids it).

Usage (from project root, after build_features):
    venv/Scripts/python.exe -m iv_study.event_study
"""

import base64
import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iv_study import config, trade_loader
from iv_study.build_features import load_series, tau_frame

logger = logging.getLogger(__name__)

TAU_GRID = np.arange(-config.WIN_PRE, config.WIN_POST + 1)

IV_FEATURES = ["iv_lead", "iv_lead_pre", "t_vel_zero_cross", "accel_sign_pre",
               "iv_runup_slope", "iv_gap_open", "iv_gap_z", "open_iv_pctile",
               "iv_runup_chg", "iv_ramp_slope", "iv_ramp_final2d", "iv_ramp_final2d_pctile"]
BASELINE_NUM = ["rvol_score", "atr_pct_move", "gap_pct", "consecutive_up_days",
                "pct_change_3", "pct_change_15", "percent_of_premarket_vol",
                "vol_ratio_5min_to_pm", "gap_from_pm_high", "reversal_open_low_pct"]


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _bootstrap_median_ci(vals: np.ndarray, n_boot: int = 10000, ci: float = 0.90):
    rng = np.random.default_rng(7)
    meds = np.median(rng.choice(vals, size=(n_boot, len(vals)), replace=True), axis=1)
    lo, hi = np.percentile(meds, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def aligned_matrix(trades: pd.DataFrame) -> pd.DataFrame:
    """iv_z on the common tau grid, one column per trade."""
    cols = {}
    for t in trades.itertuples():
        series = load_series(t.ticker, t.date_iso)
        if series is None or series.empty or pd.isna(t.t_top):
            continue
        df = tau_frame(series, t.t_top)
        z = (df.drop_duplicates(subset="tau").set_index("tau")["iv_z"]
             .reindex(TAU_GRID))
        if z.notna().sum() < 30:
            continue
        cols[f"{t.ticker} {t.date_iso}"] = z
    return pd.DataFrame(cols, index=TAU_GRID)


def plot_trajectories(mat: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for c in mat.columns:
        ax.plot(mat.index, mat[c], color="steelblue", alpha=0.25, lw=0.8)
    med = mat.median(axis=1)
    q1, q3 = mat.quantile(0.25, axis=1), mat.quantile(0.75, axis=1)
    ax.fill_between(mat.index, q1, q3, color="steelblue", alpha=0.25, label="IQR")
    ax.plot(mat.index, med, color="navy", lw=2.2, label="median")
    ax.axvline(0, color="red", ls="--", lw=1.2, label="price top (t=0)")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("minutes relative to price top")
    ax.set_ylabel("ATM IV (z vs own pre-event baseline)")
    ax.set_title(f"{title} (n={mat.shape[1]})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return fig


def plot_iv_lead(vals: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(vals, bins=np.arange(-65, 36, 5), color="steelblue", edgecolor="white")
    ax.axvline(0, color="red", ls="--", lw=1.2)
    ax.axvline(np.median(vals), color="navy", lw=2, label=f"median {np.median(vals):+.0f} min")
    ax.set_xlabel("iv_lead = t(IV peak) - t(price top), minutes")
    ax.set_ylabel("trades")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    return fig


def lead_stats(vals: pd.Series) -> dict:
    v = vals.dropna().to_numpy(float)
    if len(v) == 0:
        return {"n": 0}
    lo, hi = _bootstrap_median_ci(v) if len(v) >= 5 else (np.nan, np.nan)
    return {
        "n": len(v),
        "median_min": float(np.median(v)),
        "iqr": f"[{np.percentile(v, 25):+.0f}, {np.percentile(v, 75):+.0f}]",
        "boot90_ci": f"[{lo:+.1f}, {hi:+.1f}]",
        "frac_strictly_before": float((v < 0).mean()),
        "frac_within_5min": float((np.abs(v) <= 5).mean()),
    }


def spearman_table(feats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ivf in IV_FEATURES:
        if ivf not in feats.columns:
            continue
        for base in BASELINE_NUM:
            if base not in feats.columns:
                continue
            sub = feats[[ivf, base]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 8:
                continue
            rho = sub[ivf].corr(sub[base], method="spearman")
            if abs(rho) >= 0.35:
                rows.append({"iv_feature": ivf, "baseline": base,
                             "spearman_rho": round(rho, 2), "n": len(sub)})
    out = pd.DataFrame(rows)
    return out.sort_values("spearman_rho", key=abs, ascending=False) if len(out) else out


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    feats = pd.read_csv(config.DATA_DIR / "iv_features.csv")
    trades = trade_loader.optionable_trades()
    manifest = pd.read_csv(config.MANIFEST_CSV)

    html = ["<html><head><meta charset='utf-8'><title>IV Top-Timing Study</title>",
            "<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;"
            "padding:0 16px;color:#222}h1,h2{color:#1a355e}table{border-collapse:collapse;margin:10px 0}"
            "td,th{border:1px solid #ccc;padding:4px 10px;font-size:13px}img{max-width:100%}"
            ".note{background:#f6f8fa;border-left:4px solid #1a355e;padding:8px 12px;font-size:14px}"
            "</style></head><body>",
            "<h1>IV Top-Timing Study &mdash; does IV acceleration predict when a parabolic stock tops?</h1>",
            "<p class='note'>Descriptive event study / hypothesis generator on "
            f"{len(trades)} optionable historical reversal trades. Pre-registered primary statistic: "
            "<b>iv_lead = t(IV peak) &minus; t(price top)</b> on the post-10am bucket. "
            "Sample sizes are small; nothing here is a validated edge.</p>"]

    # -- attrition ------------------------------------------------------------
    html.append("<h2>1. Data attrition</h2>")
    html.append(manifest["status"].value_counts().rename_axis("status")
                .to_frame("trades").to_html())
    html.append(manifest[manifest["status"] != "ok"][["ticker", "date", "status", "valid_frac"]]
                .to_html(index=False))

    # -- Track A trajectories -------------------------------------------------
    html.append("<h2>2. Track A &mdash; ATM IV aligned to the intraday top</h2>")
    for bucket, label in (("post10", "Tops after 10:00 (primary — real lead room)"),
                          ("open30", "Tops 9:30–10:00 (secondary — little lead room)")):
        sub = trades[trades["top_bucket"] == bucket]
        mat = aligned_matrix(sub)
        if mat.shape[1] == 0:
            html.append(f"<p>{label}: no usable series.</p>")
            continue
        html.append(f"<img src='data:image/png;base64,{_fig_to_b64(plot_trajectories(mat, label))}'>")

    # -- iv_lead --------------------------------------------------------------
    html.append("<h2>3. The key statistic: iv_lead</h2>")
    stats_rows = []
    for bucket in ("post10", "open30"):
        sub = feats[feats["top_bucket"] == bucket]
        s = lead_stats(sub["iv_lead"]) if "iv_lead" in sub else {"n": 0}
        s["bucket"] = bucket
        stats_rows.append(s)
        if s["n"] >= 5:
            vals = sub["iv_lead"].dropna().to_numpy(float)
            fig = plot_iv_lead(vals, f"iv_lead distribution — {bucket} (n={s['n']})")
            html.append(f"<img src='data:image/png;base64,{_fig_to_b64(fig)}'>")
    html.append(pd.DataFrame(stats_rows).set_index("bucket").to_html())
    html.append("<p class='note'>Reading: negative median &rArr; IV peaks before price. "
                "frac_strictly_before is the sign test. iv_lead is measured over "
                f"[-{config.WIN_PRE}, +{config.WIN_POST}] min, so +30 clustering means IV peaked "
                "during the crash, not before the top.</p>")

    # -- exploratory timing features -------------------------------------------
    html.append("<h2>4. Exploratory timing features (post10 bucket)</h2>")
    p10 = feats[feats["top_bucket"] == "post10"]
    expl = {}
    for col, desc in (("iv_lead_pre", "pre-top IV peak time (max over tau<=0)"),
                      ("t_vel_zero_cross", "last pre-top IV-velocity +to- cross"),
                      ("accel_sign_pre", "mean IV accel, tau in [-15,0] (sigma/min^2)"),
                      ("iv_runup_slope", "IV slope, tau in [-30,0] (sigma/min)")):
        v = pd.to_numeric(p10.get(col), errors="coerce").dropna() if col in p10 else pd.Series(dtype=float)
        if len(v):
            expl[desc] = {"n": len(v), "median": round(float(v.median()), 3),
                          "q25": round(float(v.quantile(.25)), 3),
                          "q75": round(float(v.quantile(.75)), 3)}
    html.append(pd.DataFrame(expl).T.to_html())
    if "iv_divergence" in p10.columns:
        div = p10["iv_divergence"].dropna()
        if len(div):
            html.append(f"<p><b>IV divergence</b> (IV printed a &ge;1% higher high &ge;5 min before "
                        f"the price top): {int(div.sum())}/{len(div)} trades.</p>")

    # -- incremental value ------------------------------------------------------
    html.append("<h2>5. Do IV features just proxy price/volume? (|rho| &ge; 0.35 shown)</h2>")
    corr = spearman_table(feats)
    html.append(corr.to_html(index=False) if len(corr)
                else "<p>No |rho| &ge; 0.35 overlaps with baseline features — "
                     "IV features look non-redundant at this n.</p>")

    # -- Track B -----------------------------------------------------------------
    html.append("<h2>6. Track B &mdash; the lead-up: IV behavior across the run-up days</h2>")
    html.append("<p class='note'>The core day-level hypothesis: IV ramps during the run-up "
                "days into the reversal day. iv_runup_chg / iv_ramp_slope use daily close-IV "
                "marks only, so they are knowable before the reversal day opens. "
                "iv_ramp_final2d_pctile asks whether the ramp <i>steepened</i> right before "
                "the top relative to earlier in the same run-up. Caveat: every trade here IS "
                "a reversal, so 'IV rose into the top' cannot be separated from 'IV rises "
                "during any parabolic move' without non-reversal parabolic days — the "
                "within-run-up steepening percentile is the closest proxy this dataset allows.</p>")
    b = feats[feats["n_controls"].fillna(0) >= 2].copy()
    if len(b):
        rows = {}
        for col, desc in (("iv_runup_chg", "total close-IV change across run-up window (%)"),
                          ("iv_ramp_slope", "IV ramp slope (frac of mean IV per day)"),
                          ("iv_ramp_final2d", "final 2-day IV change into the reversal day (%)"),
                          ("iv_ramp_final2d_pctile", "final 2-day change pctile vs earlier run-up"),
                          ("iv_gap_open", "trade-day open IV vs prior close (gap %)"),
                          ("ctrl_gap_mean", "control-day mean gap %"),
                          ("iv_gap_z", "trade gap z vs control gaps"),
                          ("open_iv_pctile", "trade open IV percentile vs control opens")):
            v = pd.to_numeric(b[col], errors="coerce").dropna()
            if len(v):
                rows[desc] = {"n": len(v), "median": round(float(v.median()), 3),
                              "q25": round(float(v.quantile(.25)), 3),
                              "q75": round(float(v.quantile(.75)), 3)}
        html.append(pd.DataFrame(rows).T.to_html())
        pct = pd.to_numeric(b["open_iv_pctile"], errors="coerce").dropna()
        if len(pct):
            html.append(f"<p>Trades opening at the <b>top</b> of their own 5-day open-IV range "
                        f"(pctile=100): {int((pct == 100).sum())}/{len(pct)} "
                        f"({(pct == 100).mean():.0%}).</p>")
        ramp = pd.to_numeric(b["iv_runup_chg"], errors="coerce").dropna()
        if len(ramp):
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.hist(100 * ramp, bins=20, color="steelblue", edgecolor="white")
            ax.axvline(0, color="red", ls="--", lw=1.2)
            ax.axvline(100 * ramp.median(), color="navy", lw=2,
                       label=f"median {100 * ramp.median():+.0f}%")
            ax.set_xlabel(f"close-IV change across the {config.PSEUDO_CONTROL_DAYS}-day run-up window (%)")
            ax.set_ylabel("trades")
            ax.set_title(f"Did IV ramp into the reversal day? (n={len(ramp)})")
            ax.legend()
            ax.grid(alpha=0.25)
            html.append(f"<img src='data:image/png;base64,{_fig_to_b64(fig)}'>")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(pct, bins=np.arange(0, 101, 10), color="steelblue", edgecolor="white")
        ax.set_xlabel("trade-day open IV percentile vs own run-up days")
        ax.set_ylabel("trades")
        ax.set_title(f"Reversal-day open IV vs run-up days (n={len(pct)})")
        ax.grid(alpha=0.25)
        html.append(f"<img src='data:image/png;base64,{_fig_to_b64(fig)}'>")
    else:
        html.append("<p>No control marks available — run pseudo_controls first.</p>")

    html.append("</body></html>")
    out = config.REPORTS_DIR / "iv_report.html"
    out.write_text("\n".join(html), encoding="utf-8")
    logger.info("Report -> %s", out)

    print("\n=== iv_lead (primary) ===")
    print(pd.DataFrame(stats_rows).set_index("bucket").to_string())


if __name__ == "__main__":
    main()
