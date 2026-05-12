"""
Validate the report's RVOL threshold recommendations against reversal_data:
  - Medium: 1.5x -> 2.0x
  - Large:  1.0x -> 1.3x

Question: how many historical gold-standard A/B-grade reversals would be
filtered out by the new thresholds, and were those profitable?
"""
import pandas as pd

rev = pd.read_csv(r"C:\Users\zmbur\PycharmProjects\backtester\data\reversal_data.csv")

print(f"Total rows: {len(rev)}")
print(f"Rows missing rvol_score: {rev['rvol_score'].isna().sum()}")
print()

for cap, cur, new in [("Medium", 1.5, 2.0), ("Large", 1.0, 1.3)]:
    sub = rev[(rev["cap"] == cap) & rev["rvol_score"].notna()].copy()
    print("=" * 78)
    print(f"{cap} reversals — current threshold {cur}x -> proposed {new}x")
    print("=" * 78)
    print(f"Total {cap} reversals with rvol_score: {len(sub)}")
    print()

    # RVOL distribution
    print(f"RVOL distribution ({cap}):")
    print(sub["rvol_score"].describe().to_string())
    print()

    # Count by current vs proposed thresholds
    pass_cur = sub[sub["rvol_score"] >= cur]
    pass_new = sub[sub["rvol_score"] >= new]
    lost = sub[(sub["rvol_score"] >= cur) & (sub["rvol_score"] < new)]

    print(f"Pass current ({cur}x):  {len(pass_cur)} / {len(sub)}")
    print(f"Pass proposed ({new}x): {len(pass_new)} / {len(sub)}")
    print(f"LOST by bump:          {len(lost)} setups (in [{cur}, {new}) band)")
    print()

    # Grade breakdown of lost setups
    if len(lost) > 0:
        print(f"Grade breakdown of LOST setups:")
        print(lost["trade_grade"].value_counts().to_string())
        print()

        # A-grade specifically — these are the ones that matter
        a_lost = lost[lost["trade_grade"] == "A"]
        print(f"A-grade setups LOST by threshold bump: {len(a_lost)}")
        if len(a_lost) > 0:
            cols = ["date", "ticker", "gap_pct", "rvol_score",
                    "reversal_open_close_pct", "reversal_open_low_pct",
                    "reversal_open_post_low_pct"]
            cols = [c for c in cols if c in a_lost.columns]
            print(a_lost.sort_values("rvol_score")[cols].to_string(index=False))
            print()
            print(f"  Mean open->close of lost A-grades: "
                  f"{a_lost['reversal_open_close_pct'].mean():+.3f}")
            print(f"  Mean open->low of lost A-grades:   "
                  f"{a_lost['reversal_open_low_pct'].mean():+.3f}")
            print(f"  Mean open->post_low of lost A-grades: "
                  f"{a_lost['reversal_open_post_low_pct'].mean():+.3f}")

        b_lost = lost[lost["trade_grade"] == "B"]
        print()
        print(f"B-grade setups LOST by threshold bump: {len(b_lost)}")
        if len(b_lost) > 0:
            cols = ["date", "ticker", "gap_pct", "rvol_score",
                    "reversal_open_close_pct", "reversal_open_low_pct"]
            cols = [c for c in cols if c in b_lost.columns]
            print(b_lost.sort_values("rvol_score")[cols].to_string(index=False))
            print(f"  Mean open->close of lost B-grades: "
                  f"{b_lost['reversal_open_close_pct'].mean():+.3f}")
            print(f"  Mean open->low of lost B-grades:   "
                  f"{b_lost['reversal_open_low_pct'].mean():+.3f}")

    print()

    # RVOL-vs-outcome sanity: does higher RVOL correlate with bigger moves?
    print(f"Outcome by RVOL bucket ({cap}, A+B grade only):")
    ab = sub[sub["trade_grade"].isin(["A", "B"])].copy()
    buckets = [
        ("<1.0",     ab["rvol_score"] < 1.0),
        ("1.0-1.3",  (ab["rvol_score"] >= 1.0) & (ab["rvol_score"] < 1.3)),
        ("1.3-1.5",  (ab["rvol_score"] >= 1.3) & (ab["rvol_score"] < 1.5)),
        ("1.5-2.0",  (ab["rvol_score"] >= 1.5) & (ab["rvol_score"] < 2.0)),
        ("2.0-3.0",  (ab["rvol_score"] >= 2.0) & (ab["rvol_score"] < 3.0)),
        ("3.0-5.0",  (ab["rvol_score"] >= 3.0) & (ab["rvol_score"] < 5.0)),
        (">=5.0",    ab["rvol_score"] >= 5.0),
    ]
    for label, mask in buckets:
        s = ab[mask]
        if len(s) == 0:
            continue
        oc = s["reversal_open_close_pct"].mean()
        ol = s["reversal_open_low_pct"].mean()
        print(f"  rvol {label:10s} n={len(s):3d}  open->close={oc:+.3f}  open->low={ol:+.3f}")
    print()
    print()
