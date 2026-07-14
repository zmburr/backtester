"""Stage 06: aggregate the three studies + HTML report.

  1. Announcement pop: day-0 / 2-day-high reaction, by year & sector
  2. Redemptions: redemption-% buckets vs flip-day and post-flip returns
  3. Ticker-flip trade: buy the last old-ticker close, exits at flip open /
     close / +1d / +5d; conditioned on price-vs-trust and float

Run:  python -m despac_study.analysis
"""

import base64
import io
import logging

import numpy as np
import pandas as pd

from despac_study.config import MASTER_CSV, REPORT_HTML

logging.getLogger("matplotlib").setLevel(logging.WARNING)

RED_BUCKETS = [(-0.1, 25), (25, 50), (50, 75), (75, 90), (90, 100.1)]
RED_LABELS = ["0-25%", "25-50%", "50-75%", "75-90%", "90-100%"]


def load() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV)
    df = df[df["is_spac"] == True].copy()
    # a co-filer (e.g. sponsor holdco) can resolve to the same new listing via
    # the name-search fallback; keep the row that owns the SPAC-side history
    df = df.sort_values(["old_ticker", "has_425"], na_position="last")
    dup = df["flip_ticker"].notna() & df["flip_date"].notna()
    df = pd.concat([
        df[dup].drop_duplicates(subset=["flip_ticker", "flip_date"], keep="first"),
        df[~dup],
    ])
    df["year"] = df["close_date"].astype(str).str[:4]
    df["red_bucket"] = pd.cut(df["redemption_pct_best"],
                              [b[0] for b in RED_BUCKETS] + [RED_BUCKETS[-1][1]],
                              labels=RED_LABELS)
    df["near_trust"] = df["last_old_close"] <= 10.5
    return df


def _agg(g: pd.DataFrame, cols) -> pd.Series:
    out = {"n": len(g)}
    for c in cols:
        if c in g:
            s = g[c].dropna()
            out[f"{c}_med"] = round(s.median(), 2) if len(s) else None
            out[f"{c}_win"] = round(100 * (s > 0).mean(), 0) if len(s) else None
    return pd.Series(out)


def announcement_study(df):
    cols = ["ann_gap_pct", "ann_day0_ret_pct", "ann_2d_high_ret_pct", "ann_to_vote_ret_pct"]
    t1 = df.groupby("year").apply(_agg, cols, include_groups=False)
    t2 = df.groupby("is_hightech").apply(_agg, cols, include_groups=False)
    t3 = df.groupby("sector").apply(_agg, cols, include_groups=False).sort_values("n", ascending=False)
    return {"by year": t1, "hightech vs not": t2, "by sector": t3}


def redemption_study(df):
    cols = ["flip_gap_pct", "flip_day_ret_pct", "flip_high_ret_pct",
            "post_flip_ret_5d_pct", "max_runup_10d_pct"]
    t1 = df.groupby("red_bucket", observed=True).apply(_agg, cols, include_groups=False)
    d = df.dropna(subset=["redemption_pct_best", "max_runup_10d_pct"])
    corr = d["redemption_pct_best"].corr(d["max_runup_10d_pct"], method="spearman") if len(d) > 5 else np.nan
    sq = df.assign(big_run=df["max_runup_10d_pct"] > 25).groupby("red_bucket", observed=True)["big_run"].mean().mul(100).round(0)
    t1["pct_runup_gt25"] = sq
    return {"by redemption bucket": t1, "spearman_red_vs_runup": corr}


def flip_study(df):
    cols = ["flip_gap_pct", "flip_day_ret_pct", "flip_high_ret_pct",
            "post_flip_ret_1d_pct", "post_flip_ret_3d_pct", "post_flip_ret_5d_pct",
            "post_flip_ret_10d_pct", "max_runup_10d_pct", "max_drawdown_10d_pct"]
    overall = _agg(df, cols).to_frame("all deals").T
    t_year = df.groupby("year").apply(_agg, cols[:3] + ["post_flip_ret_5d_pct"], include_groups=False)
    t_trust = df.groupby("near_trust").apply(_agg, cols, include_groups=False)
    t_tech = df.groupby("is_hightech").apply(_agg, cols[:3] + ["max_runup_10d_pct"], include_groups=False)
    lag = df["flip_lag_cal_days"].dropna()
    lag_dist = lag.value_counts().sort_index().head(15)
    return {"overall": overall, "by year": t_year, "near trust (<=10.50) vs above": t_trust,
            "hightech vs not": t_tech, "close->flip lag days (calendar)": lag_dist.to_frame("deals")}


def _fig64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def charts(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    imgs = []
    d = df["flip_high_ret_pct"].dropna().clip(-50, 150)
    if len(d) > 10:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.hist(d, bins=40, color="#3b7dd8", alpha=0.85)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title("Flip-day high vs last old-ticker close (%)")
        imgs.append(("Flip-day pop distribution", _fig64(fig)))
    d = df.dropna(subset=["redemption_pct_best", "max_runup_10d_pct"])
    if len(d) > 10:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.scatter(d["redemption_pct_best"], d["max_runup_10d_pct"].clip(-60, 300),
                   s=14, alpha=0.5, color="#d84b3b")
        ax.set_xlabel("redemption % of public shares")
        ax.set_ylabel("max 10d runup %")
        ax.set_title("Redemptions vs post-flip runup")
        imgs.append(("Redemptions vs runup", _fig64(fig)))
    d = df.dropna(subset=["ann_day0_ret_pct"])
    if len(d) > 10:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        med = d.groupby("year")["ann_day0_ret_pct"].median()
        ax.bar(med.index, med.values, color="#38a169")
        ax.set_title("Median announcement day-0 return by year (%)")
        imgs.append(("Announcement pop by year", _fig64(fig)))
    return imgs


def build_report(df, sections, imgs):
    css = """<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;max-width:1100px}
    table{border-collapse:collapse;font-size:13px;margin:8px 0 20px}
    th,td{border:1px solid #ccc;padding:4px 8px;text-align:right}
    th{background:#f0f2f5}td:first-child,th:first-child{text-align:left}
    h2{border-bottom:2px solid #3b7dd8;padding-bottom:4px;margin-top:32px}</style>"""
    parts = [f"<title>De-SPAC study</title>{css}<h1>De-SPAC event study, 2020-present</h1>",
             f"<p>{len(df)} confirmed de-SPAC deals with completed mergers. "
             f"Flip-trade entry = last close under the old SPAC ticker; all returns in % vs that entry.</p>"]
    for title, tables in sections:
        parts.append(f"<h2>{title}</h2>")
        for name, t in tables.items():
            parts.append(f"<h3>{name}</h3>")
            if isinstance(t, pd.DataFrame):
                parts.append(t.to_html(na_rep=""))
            else:
                parts.append(f"<p>{t}</p>")
    parts.append("<h2>Charts</h2>")
    for name, b64 in imgs:
        parts.append(f"<h3>{name}</h3><img src='data:image/png;base64,{b64}'/>")
    REPORT_HTML.write_text("\n".join(parts), encoding="utf-8")


def main():
    logging.basicConfig(level=logging.INFO)
    df = load()
    # real SPACs trade near trust (~$10) pre-close and flip 1:1; exclude
    # exchange-ratio artifacts (split at close, sub-$4 shells, known reorgs)
    from despac_study.config import KNOWN_NON_SPAC_FLIPS
    tradeable = df[df["flip_date"].notna()
                   & df["last_old_close"].between(4, 120)
                   & (df.get("split_at_flip", False) != True)
                   & ~df["flip_ticker"].isin(KNOWN_NON_SPAC_FLIPS)]
    sections = [
        ("1. Deal announcement reaction", announcement_study(df)),
        ("2. Redemptions", redemption_study(tradeable)),
        ("3. Ticker-flip trade", flip_study(tradeable)),
    ]
    imgs = charts(tradeable)
    build_report(df, sections, imgs)
    print(f"deals: {len(df)} total, {len(tradeable)} with flip data")
    print(f"report -> {REPORT_HTML}")
    fs = flip_study(tradeable)
    print("\n=== flip trade overall ===")
    print(fs["overall"].T.to_string())


if __name__ == "__main__":
    main()
