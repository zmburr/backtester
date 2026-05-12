"""
Validate the report's gap_pct inversion recommendation against the gold-standard
reversal_data and bounce_data historical databases.

Report recommends: reversal gap_pct gate should require gap <= 0 (invert current
"gap UP" scoring). This script checks how many historically-successful reversal
setups would have been FILTERED OUT by that rule.
"""
import pandas as pd
import numpy as np

REVERSAL = r"C:\Users\zmbur\PycharmProjects\backtester\data\reversal_data.csv"
BOUNCE = r"C:\Users\zmbur\PycharmProjects\backtester\data\bounce_data.csv"

rev = pd.read_csv(REVERSAL)
bnc = pd.read_csv(BOUNCE)

print("=" * 80)
print("REVERSAL_DATA — gold-standard reversal setups")
print("=" * 80)
print(f"Total rows: {len(rev)}")
print(f"Columns of interest: gap_pct, cap, trade_grade, rvol_score,")
print(f"  reversal_open_close_pct, reversal_open_low_pct, reversal_open_post_low_pct")
print()

# --- Basic gap_pct distribution ---
print("### Gap_pct distribution (all rows) ###")
print(rev["gap_pct"].describe().to_string())
print()

print("### Gap direction split ###")
gap_up = rev[rev["gap_pct"] > 0]
gap_flat = rev[rev["gap_pct"] == 0]
gap_down = rev[rev["gap_pct"] < 0]
print(f"gap_up (>0):    {len(gap_up):4d}  ({len(gap_up)/len(rev)*100:.1f}%)")
print(f"gap_flat (=0):  {len(gap_flat):4d}  ({len(gap_flat)/len(rev)*100:.1f}%)")
print(f"gap_down (<0):  {len(gap_down):4d}  ({len(gap_down)/len(rev)*100:.1f}%)")
print()

# --- By cap ---
print("### Gap direction by cap ###")
for cap in sorted(rev["cap"].dropna().unique()):
    sub = rev[rev["cap"] == cap]
    up = (sub["gap_pct"] > 0).sum()
    down = (sub["gap_pct"] <= 0).sum()
    print(f"  {cap:8s}  n={len(sub):3d}  gap_up={up:3d} ({up/len(sub)*100:4.1f}%)  "
          f"gap<=0={down:3d} ({down/len(sub)*100:4.1f}%)")
print()

# --- Trade grade breakdown (A grade = best setups) ---
print("### Gap direction by trade_grade ###")
for grade in sorted(rev["trade_grade"].dropna().unique()):
    sub = rev[rev["trade_grade"] == grade]
    up = (sub["gap_pct"] > 0).sum()
    down = (sub["gap_pct"] <= 0).sum()
    print(f"  grade={grade}  n={len(sub):3d}  gap_up={up:3d} ({up/len(sub)*100:4.1f}%)  "
          f"gap<=0={down:3d} ({down/len(sub)*100:4.1f}%)")
print()

# --- Grade x Cap cross ---
print("### Grade x Cap cross — % gap_up ###")
piv = rev.assign(gap_up=(rev["gap_pct"] > 0).astype(int)).pivot_table(
    index="trade_grade", columns="cap", values="gap_up", aggfunc=["sum", "count"]
)
print(piv.to_string())
print()

# --- Outcome comparison: do gap-up historical reversals produce real moves? ---
# reversal_open_close_pct: open->close. For short reversals, negative = profitable.
# reversal_open_low_pct: open->intraday low. For shorts, negative = MFE.
print("### Outcome comparison: gap_up vs gap_down reversals ###")
print("(Reversal setups are shorts — NEGATIVE values = profitable)")
print()
for cap in ["Medium", "Large", "Small", "Mega"]:
    sub = rev[rev["cap"] == cap]
    if len(sub) < 3:
        continue
    print(f"  --- {cap} (n={len(sub)}) ---")
    for label, mask in [
        ("gap_up (>0)", sub["gap_pct"] > 0),
        ("gap_flat_or_down (<=0)", sub["gap_pct"] <= 0),
        ("gap_up_moderate (0 to 0.05)", (sub["gap_pct"] > 0) & (sub["gap_pct"] <= 0.05)),
        ("gap_up_large (>0.05)", sub["gap_pct"] > 0.05),
    ]:
        s = sub[mask]
        if len(s) == 0:
            continue
        oc = s["reversal_open_close_pct"].mean()
        ol = s["reversal_open_low_pct"].mean()
        op = s["reversal_open_post_low_pct"].mean()
        print(f"    {label:32s}  n={len(s):3d}  "
              f"open->close={oc:+.3f}  open->low={ol:+.3f}  open->post_low={op:+.3f}")
    print()

# --- A-grade gap-up reversals (the ones we'd LOSE) ---
print("### A-grade reversals that gapped UP (these would be filtered by the invert rule) ###")
a_up = rev[(rev["trade_grade"] == "A") & (rev["gap_pct"] > 0)]
print(f"Count: {len(a_up)} out of {len(rev[rev['trade_grade']=='A'])} A-grade total")
if len(a_up) > 0:
    cols = ["date", "ticker", "cap", "gap_pct", "rvol_score",
            "reversal_open_close_pct", "reversal_open_low_pct",
            "reversal_open_post_low_pct"]
    cols = [c for c in cols if c in a_up.columns]
    # Show gap distribution within A-grade gap-ups
    print(f"  gap_pct within A-grade gap-ups: "
          f"min={a_up['gap_pct'].min():.3f}  "
          f"median={a_up['gap_pct'].median():.3f}  "
          f"max={a_up['gap_pct'].max():.3f}")
    print()
    print("Top 30 A-grade gap-up reversals by gap size:")
    print(a_up.sort_values("gap_pct", ascending=False)[cols].head(30).to_string(index=False))
print()

# --- Distribution of gap_pct specifically among A-grade by cap ---
print("### A-grade reversals — gap_pct percentiles by cap ###")
for cap in ["Medium", "Large", "Small"]:
    sub = rev[(rev["trade_grade"] == "A") & (rev["cap"] == cap)]
    if len(sub) == 0:
        continue
    print(f"  {cap} A-grade n={len(sub)}")
    print(f"    gap_pct quantiles:")
    for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
        print(f"      q{int(q*100):3d} = {sub['gap_pct'].quantile(q):+.4f}")
    up = (sub["gap_pct"] > 0).sum()
    up5 = (sub["gap_pct"] > 0.05).sum()
    print(f"    gap_pct > 0:    {up}/{len(sub)} ({up/len(sub)*100:.1f}%)")
    print(f"    gap_pct > 0.05: {up5}/{len(sub)} ({up5/len(sub)*100:.1f}%)")
print()

# --- Would the hard gate (>5% Med/Small, >3% Large) have been triggered? ---
print("### Hard suppression gate impact (report recommends: suppress if gap > 5% Med/Small, 3% Large) ###")
med_small_suppress = rev[(rev["cap"].isin(["Medium", "Small"])) & (rev["gap_pct"] > 0.05)]
large_suppress = rev[(rev["cap"] == "Large") & (rev["gap_pct"] > 0.03)]
print(f"Medium/Small with gap > 5%: {len(med_small_suppress)} rows "
      f"(of {len(rev[rev['cap'].isin(['Medium','Small'])])} Med/Small total)")
print(f"Large with gap > 3%:        {len(large_suppress)} rows "
      f"(of {len(rev[rev['cap']=='Large'])} Large total)")
print()
# Grade breakdown of those that would be suppressed
print("Grade breakdown of would-be-suppressed rows:")
sup = pd.concat([med_small_suppress, large_suppress])
if len(sup) > 0:
    print(sup["trade_grade"].value_counts().to_string())
    print()
    print("Top 20 by gap size (these would be filtered):")
    cols = ["date", "ticker", "cap", "trade_grade", "gap_pct",
            "reversal_open_close_pct", "reversal_open_low_pct"]
    cols = [c for c in cols if c in sup.columns]
    print(sup.sort_values("gap_pct", ascending=False)[cols].head(20).to_string(index=False))
print()

# =============================================================================
# BOUNCE DATA
# =============================================================================
print()
print("=" * 80)
print("BOUNCE_DATA — gold-standard bounce setups")
print("=" * 80)
print(f"Total rows: {len(bnc)}")
print()

print("### Gap_pct distribution ###")
print(bnc["gap_pct"].describe().to_string())
print()

print("### Gap direction by cap ###")
for cap in sorted(bnc["cap"].dropna().unique()):
    sub = bnc[bnc["cap"] == cap]
    up = (sub["gap_pct"] > 0).sum()
    down = (sub["gap_pct"] <= 0).sum()
    down_neg3 = (sub["gap_pct"] <= -0.03).sum()
    print(f"  {cap:8s}  n={len(sub):3d}  gap_up={up:3d} ({up/len(sub)*100:4.1f}%)  "
          f"gap<=0={down:3d} ({down/len(sub)*100:4.1f}%)  "
          f"gap<=-0.03={down_neg3:3d} ({down_neg3/len(sub)*100:4.1f}%)")
print()

# The report says: for bounces, "positive gap = warn (don't suppress)"
# Let's see how many A-grade bounces actually had positive gap
print("### A-grade bounces with positive gap (report says these warrant a WARN, not suppress) ###")
a_bnc = bnc[bnc["trade_grade"] == "A"]
a_up = a_bnc[a_bnc["gap_pct"] > 0]
print(f"A-grade bounces: {len(a_bnc)}")
print(f"A-grade bounces with gap_pct > 0: {len(a_up)} ({len(a_up)/len(a_bnc)*100:.1f}%)")
if "bounce_open_high_pct" in bnc.columns:
    a_up_clean = a_up.dropna(subset=["bounce_open_high_pct"])
    a_down_clean = a_bnc[a_bnc["gap_pct"] <= 0].dropna(subset=["bounce_open_high_pct"])
    if len(a_up_clean) > 0 and len(a_down_clean) > 0:
        print(f"  mean bounce_open_high_pct (gap_up):   {a_up_clean['bounce_open_high_pct'].mean():+.3f}")
        print(f"  mean bounce_open_high_pct (gap<=0):   {a_down_clean['bounce_open_high_pct'].mean():+.3f}")
        print(f"  mean bounce_open_close_pct (gap_up):  {a_up_clean['bounce_open_close_pct'].mean():+.3f}")
        print(f"  mean bounce_open_close_pct (gap<=0):  {a_down_clean['bounce_open_close_pct'].mean():+.3f}")
print()
