"""
Identify which features in reversal_data cleanly separate the
'parabolic-at-highs' archetype (AMC, BBBY, MRNA, NKLA, COIN, WISH)
from the 'bouncing-off-lows' profile (like recent IONQ, RKLB, CRDO failures).

Candidate downgrade-flag features:
  - breaks_ath / breaks_fifty_two_wk (boolean — did today print a new high?)
  - pct_from_200mav  (how far above the long trend?)
  - pct_from_50mav   (medium-term trend position)
  - pct_change_30, pct_change_90, pct_change_120 (recent run magnitude)
  - atr_distance_from_50mav
"""
import pandas as pd

rev = pd.read_csv(r"C:\Users\zmbur\PycharmProjects\backtester\data\reversal_data.csv")
print(f"Total rows: {len(rev)}\n")

feats = [
    "breaks_ath", "breaks_fifty_two_wk",
    "pct_from_200mav", "pct_from_50mav", "pct_from_20mav", "pct_from_10mav",
    "pct_change_30", "pct_change_90", "pct_change_120",
    "atr_distance_from_50mav", "pct_from_9ema",
]
for f in feats:
    if f not in rev.columns:
        print(f"  MISSING: {f}")
    else:
        print(f"  ok: {f}")
print()

a = rev[rev["trade_grade"] == "A"]
b = rev[rev["trade_grade"] == "B"]
c = rev[rev["trade_grade"] == "C"]
print(f"A-grade n={len(a)}  B={len(b)}  C={len(c)}\n")

# --- Boolean features ---
for col in ["breaks_ath", "breaks_fifty_two_wk"]:
    print(f"### {col} ###")
    for label, sub in [("A", a), ("B", b), ("C", c)]:
        if col in sub.columns:
            true_pct = sub[col].sum() / len(sub) * 100
            print(f"  grade={label}  True rate: {sub[col].sum()}/{len(sub)} ({true_pct:.1f}%)")
    print()

# --- Continuous features: A-grade vs B+C distribution ---
for col in ["pct_from_200mav", "pct_from_50mav", "pct_from_20mav",
            "pct_change_30", "pct_change_90", "pct_change_120",
            "atr_distance_from_50mav"]:
    if col not in rev.columns:
        continue
    print(f"### {col} — quantiles by grade ###")
    print(f"  {'quantile':>10s}  {'A':>10s}  {'B':>10s}  {'C':>10s}")
    for q in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        a_q = a[col].quantile(q) if col in a.columns else None
        b_q = b[col].quantile(q)
        c_q = c[col].quantile(q)
        print(f"  q{int(q*100):>8d}  {a_q:>+10.3f}  {b_q:>+10.3f}  {c_q:>+10.3f}")
    print()

# --- How many A-grade reversals have pct_from_200mav below key thresholds? ---
# The idea: A stock "bouncing off a low base" has small/negative pct_from_200mav
print("### A-grade reversal profile: stock position relative to trend ###")
for thresh in [0.0, 0.1, 0.2, 0.5, 1.0]:
    cnt = (a["pct_from_200mav"] >= thresh).sum()
    print(f"  A-grades with pct_from_200mav >= {thresh:+.2f}: {cnt}/{len(a)} ({cnt/len(a)*100:.1f}%)")
print()

# --- Combined archetype score ---
# Hypothesis: A-grades usually break 52wk high OR are extended above MAs with
# strong recent pct_change. Let's see.
print("### Archetype profile fingerprint ###")
print("Archetype = (breaks_52wk=True) OR (pct_from_200mav > 0.5) OR (pct_change_30 > 0.3)")
a_profile = a[
    (a["breaks_fifty_two_wk"] == True) |
    (a["pct_from_200mav"] > 0.5) |
    (a["pct_change_30"] > 0.3)
]
b_profile = b[
    (b["breaks_fifty_two_wk"] == True) |
    (b["pct_from_200mav"] > 0.5) |
    (b["pct_change_30"] > 0.3)
]
print(f"  A-grades matching archetype: {len(a_profile)}/{len(a)} ({len(a_profile)/len(a)*100:.1f}%)")
print(f"  B-grades matching archetype: {len(b_profile)}/{len(b)} ({len(b_profile)/len(b)*100:.1f}%)")
print()

# --- Show A-grades that DON'T match — are those the "different" setups? ---
print("A-grade reversals NOT matching archetype:")
non = a[
    (a["breaks_fifty_two_wk"] != True) &
    (a["pct_from_200mav"] <= 0.5) &
    (a["pct_change_30"] <= 0.3)
]
cols = ["date", "ticker", "cap", "gap_pct", "pct_from_200mav", "pct_from_50mav",
        "pct_change_30", "pct_change_90", "breaks_fifty_two_wk",
        "reversal_open_close_pct", "reversal_open_low_pct"]
cols = [c for c in cols if c in non.columns]
print(non[cols].to_string(index=False))
