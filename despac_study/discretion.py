"""Which de-SPACs pop and which flop: objective 'story stock' features.

Tags each deal with a narrative theme (finer than sector), then cuts the
flip-trade outcomes by theme, era, redemption, and pre-flip demand signals -
the objective version of "GFUZ is the only fusion play, mining SPACs are
dead money".

Run:  python -m despac_study.discretion
"""

import re

import pandas as pd

from despac_study.config import MASTER_CSV
from despac_study.polygon_enrich import get_details

# priority-ordered: first match wins (a "quantum computing AI" name = quantum)
THEMES = [
    ("fusion",        ["fusion energy", "nuclear fusion", " fusion"]),
    ("quantum",       ["quantum"]),
    ("nuclear",       ["nuclear", "uranium", "small modular reactor", " smr", "reactor"]),
    ("space",         ["space", "satellite", "launch vehicle", "orbital", "lunar", "rocket"]),
    ("evtol",         ["evtol", "air taxi", "urban air", "electric aircraft", "aerial mobility"]),
    ("defense",       ["defense", "military", "drone"]),
    ("crypto",        ["bitcoin", "crypto", "blockchain", "digital asset", "web3"]),
    ("ai",            ["artificial intelligence", " ai ", "machine learning", "data analytics"]),
    ("hydrogen",      ["hydrogen", "fuel cell"]),
    ("ev",            ["electric vehicle", "ev charging", "charging network", "battery", "lidar",
                       "autonomous driv", "self-driving"]),
    ("solar_renew",   ["solar", "renewable", "clean energy", "wind "]),
    ("betting_gaming", ["sports betting", "casino", "igaming", "esports", "gaming"]),
    ("cannabis",      ["cannabis"]),
    ("mining",        ["mining", "minerals", "metals", "gold ", "lithium", "rare earth", "coal"]),
    ("biotech",       ["therapeutic", "biopharma", "clinical", "pharma", "biotech", "oncology"]),
    ("fintech",       ["fintech", "payment", "lending", "insur", "banking"]),
    ("media",         ["media", "streaming", "entertainment", "social network", "content"]),
]

ERAS = {"2020": "2020-21", "2021": "2020-21", "2022": "2022-24",
        "2023": "2022-24", "2024": "2022-24", "2025": "2025-26", "2026": "2025-26"}


def tag_theme(row) -> str:
    t = row.get("new_ticker")
    det = get_details(t) if isinstance(t, str) and t else {}
    if not det and isinstance(t, str) and t and isinstance(row.get("flip_date"), str):
        det = get_details(t, date=row["flip_date"])
    blob = " " + " ".join(str(x) for x in [
        row.get("company_name"), row.get("polygon_name"), row.get("target_name"),
        row.get("sic_description"), row.get("sic_desc"),
        det.get("description", "") if det else ""]).lower() + " "
    blob = re.sub(r"\s+", " ", blob)
    for theme, kws in THEMES:
        if any(k in blob for k in kws):
            return theme
    return "other"


def stats(g):
    hi = g["flip_high_ret_pct"].dropna()
    ru = g["max_runup_10d_pct"].dropna()
    d5 = g["post_flip_ret_5d_pct"].dropna()
    return pd.Series({
        "n": len(g),
        "flip_high_med": round(hi.median(), 1) if len(hi) else None,
        "runup10d_med": round(ru.median(), 1) if len(ru) else None,
        "pct_run_gt25": round(100 * (ru > 25).mean(), 0) if len(ru) else None,
        "pct_run_gt50": round(100 * (ru > 50).mean(), 0) if len(ru) else None,
        "d5_med": round(d5.median(), 1) if len(d5) else None,
    })


def main():
    pd.set_option("display.width", 250)
    from despac_study.config import KNOWN_NON_SPAC_FLIPS
    df = pd.read_csv(MASTER_CSV)
    df = df[(df["is_spac"] == True) & df["flip_date"].notna()
            & df["last_old_close"].between(4, 120)
            & ~df["flip_ticker"].isin(KNOWN_NON_SPAC_FLIPS)].copy()
    if "split_at_flip" in df.columns:
        df = df[df["split_at_flip"] != True]
    df["theme"] = [tag_theme(r) for _, r in df.iterrows()]
    df["year"] = df["close_date"].astype(str).str[:4]
    df["era"] = df["year"].map(ERAS)

    print(f"tradeable flips: {len(df)}\n")
    print("=== BY THEME (all years) ===")
    t = df.groupby("theme").apply(stats, include_groups=False)
    print(t.sort_values("runup10d_med", ascending=False).to_string())

    print("\n=== THEME x ERA (median 10d max runup %, n) ===")
    pv = df.pivot_table(index="theme", columns="era", values="max_runup_10d_pct",
                        aggfunc="median").round(1)
    pvn = df.pivot_table(index="theme", columns="era", values="max_runup_10d_pct",
                         aggfunc="count")
    merged = pv.astype(str) + "  (n=" + pvn.fillna(0).astype(int).astype(str) + ")"
    print(merged.to_string())

    print("\n=== DEMAND SIGNALS ===")
    df["premium_entry"] = df["last_old_close"] > 10.5
    df["high_redemption"] = df["redemption_pct_best"] >= 90
    df["ann_popped"] = df["ann_2d_high_ret_pct"] >= 5
    for col in ("premium_entry", "high_redemption", "ann_popped"):
        print(f"\n-- {col} --")
        print(df.groupby(col).apply(stats, include_groups=False).to_string())

    print("\n=== combo: high_redemption x era ===")
    print(df[df["high_redemption"] == True].groupby("era").apply(stats, include_groups=False).to_string())

    cols = ["flip_ticker", "company_name", "theme", "close_date", "redemption_pct_best",
            "last_old_close", "flip_gap_pct", "flip_high_ret_pct", "max_runup_10d_pct",
            "post_flip_ret_5d_pct"]
    print("\n=== TOP 25 (10d max runup) ===")
    print(df.nlargest(25, "max_runup_10d_pct")[cols].to_string(index=False))
    print("\n=== BOTTOM 15 ===")
    print(df.nsmallest(15, "max_runup_10d_pct")[cols].to_string(index=False))

    out = MASTER_CSV.with_name("despac_themed.csv")
    df.to_csv(out, index=False)
    print(f"\nthemed dataset -> {out}")


if __name__ == "__main__":
    main()
