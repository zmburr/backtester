"""
Verify the calculations in Signal Analysis Feedback Loop #4 (May 28, 2026).

Reconstructs the full 58-signal dataset by joining:
  - dispatcher/data/signal_log.csv      (outcomes through 5/14)
  - backtester/data/priority_signals/*  (per-signal metrics)
Then cross-validates the report's claims against the data and against
the full reversal_data.csv historical wins.

Run from project root:
    python scripts/verify_signal_analysis.py
"""
from pathlib import Path
import csv
import json
import sys
import pandas as pd
import numpy as np

LOG = Path(r"C:\Users\zmbur\PycharmProjects\dispatcher\dispatcher\data\signal_log.csv")
PRI = Path(r"C:\Users\zmbur\PycharmProjects\backtester\data\priority_signals")
REV = Path(r"C:\Users\zmbur\PycharmProjects\backtester\data\reversal_data.csv")


def load_signal_metrics():
    """Index priority_signals files by (date, ticker)."""
    out = {}
    for f in sorted(PRI.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            d = data.get("date")
            for s in data.get("signals", []):
                key = (d, s["ticker"])
                # First write wins (morning before evening if both exist)
                if key not in out:
                    out[key] = {
                        **s.get("metrics", {}),
                        "bucket": s.get("bucket"),
                        "cap": s.get("cap"),
                        "recommendation": s.get("recommendation"),
                        "score": s.get("score"),
                        "file": f.name,
                    }
        except Exception as e:
            print(f"WARN: {f.name}: {e}")
    return out


def load_log():
    """Load signal_log.csv as list of dicts."""
    if not LOG.exists():
        return []
    return list(csv.DictReader(LOG.open()))


def main():
    metrics = load_signal_metrics()
    log = load_log()

    print(f"signal_log rows: {len(log)}")
    print(f"priority_signals (date,ticker) keys: {len(metrics)}")

    # ── Join: enrich log rows with metrics ────────────────────────────────
    enriched = []
    for r in log:
        m = metrics.get((r["date"], r["ticker"]), {})
        rec = {
            **r,
            "gap_pct": m.get("gap_pct"),
            "pct_change_3": m.get("pct_change_3"),
            "pct_from_9ema": m.get("pct_from_9ema"),
            "prior_day_range_atr": m.get("prior_day_range_atr"),
            "prior_day_rvol": m.get("prior_day_rvol"),
            "atr_pct": m.get("atr_pct"),
        }
        enriched.append(rec)

    df = pd.DataFrame(enriched)
    df["tradeable_b"] = df["tradeable"].astype(str).str.lower() == "true"
    df["dir_ok"] = df["prediction_correct"].astype(str).str.lower() == "true"
    df["score_num"] = df["score"].apply(lambda s: int(str(s).split("/")[0]) if s else None)
    df["pct_change_3_f"] = pd.to_numeric(df["pct_change_3"], errors="coerce")
    df["gap_pct_f"] = pd.to_numeric(df["gap_pct"], errors="coerce")

    print("\n── Date range covered by log:")
    print(f"  {df['date'].min()} .. {df['date'].max()}")
    print(f"  Bucket counts:\n{df['bucket'].value_counts().to_string()}")

    # ── 1) Tradeable rate by score ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CHECK 1: Tradeable rate by score (report claims 5/5=2/11=18%, 4/5=5/23=22%)")
    print("=" * 70)
    rev = df[df["bucket"] == "reversal"].copy()
    bnc = df[df["bucket"] == "bounce"].copy()

    for label, sub in [("REVERSAL ONLY", rev), ("ALL SIGNALS", df), ("BOUNCE ONLY", bnc)]:
        print(f"\n{label} (n={len(sub)}):")
        for score in [6, 5, 4, 3]:
            s = sub[sub["score_num"] == score]
            if len(s) == 0:
                continue
            trade = int(s["tradeable_b"].sum())
            dir_ok = int(s["dir_ok"].sum())
            pct_trade = 100 * trade / len(s)
            print(f"  Score {score}/?: n={len(s):2d}  tradeable={trade:2d} ({pct_trade:5.1f}%)  dir_ok={dir_ok:2d}")

    # ── 2) Reversal GO vs CAUTION ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CHECK 2: Reversal GO vs CAUTION tradeable rate")
    print("=" * 70)
    for rec_lbl in ["GO", "CAUTION"]:
        s = rev[rev["recommendation"] == rec_lbl]
        if len(s):
            tr = int(s["tradeable_b"].sum())
            dr = int(s["dir_ok"].sum())
            print(f"  Reversal {rec_lbl:8s}: n={len(s):2d}  tradeable={tr} ({100*tr/len(s):.1f}%)  dir_ok={dr} ({100*dr/len(s):.1f}%)")

    # ── 3) Gap_pct claims ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CHECK 3: Gap_pct values per the report's cited examples")
    print("=" * 70)
    cited_large = [("MRVL", None), ("QCOM", None), ("AMD", "2026-03-26"), ("ARM", None), ("AAOI", None)]
    cited_med = [("PL", "2026-03-23"), ("POET", "2026-05-15"), ("CRML", None), ("RGTI", None), ("CAR", "2026-04-21")]

    def show(ticker, date_filter=None, label=""):
        sub = df[df["ticker"] == ticker]
        if date_filter:
            sub = sub[sub["date"] == date_filter]
        for _, row in sub.iterrows():
            print(f"  {label:8s} {row['ticker']:5s} {row['date']}  "
                  f"gap={row['gap_pct']!s:>10s}  mom3={row['pct_change_3']!s:>10s}  "
                  f"score={row['score']:>5s}  tradeable={row['tradeable_b']}  dir_ok={row['dir_ok']}")

    print("Large-cap reversals cited:")
    for t, d in cited_large:
        show(t, d, "Large")
    print("Medium-cap reversals cited:")
    for t, d in cited_med:
        show(t, d, "Medium")

    # ── 4) pct_change_3 caps ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CHECK 4: pct_change_3 above proposed caps (Large=0.35, Medium=0.55)")
    print("=" * 70)
    rev_l = rev[rev["cap"] == "Large"].copy()
    rev_m = rev[rev["cap"] == "Medium"].copy()

    above_large = rev_l[rev_l["pct_change_3_f"] > 0.35]
    print(f"\nLarge reversals with pct_change_3 > 0.35 (n={len(above_large)}):")
    for _, r in above_large.iterrows():
        print(f"  {r['date']} {r['ticker']:5s} mom3={r['pct_change_3_f']:.3f}  rec={r['recommendation']:7s}  tradeable={r['tradeable_b']}  dir_ok={r['dir_ok']}")

    above_med = rev_m[rev_m["pct_change_3_f"] > 0.55]
    print(f"\nMedium reversals with pct_change_3 > 0.55 (n={len(above_med)}):")
    for _, r in above_med.iterrows():
        print(f"  {r['date']} {r['ticker']:5s} mom3={r['pct_change_3_f']:.3f}  rec={r['recommendation']:7s}  tradeable={r['tradeable_b']}  dir_ok={r['dir_ok']}")

    # Tradeable wins below proposed cap
    print(f"\nLarge reversal tradeable wins (any mom3):")
    wins_l = rev_l[rev_l["tradeable_b"]]
    for _, r in wins_l.iterrows():
        print(f"  {r['date']} {r['ticker']:5s} mom3={r['pct_change_3_f']:.3f}  rec={r['recommendation']}")
    print(f"\nMedium reversal tradeable wins (any mom3):")
    wins_m = rev_m[rev_m["tradeable_b"]]
    for _, r in wins_m.iterrows():
        print(f"  {r['date']} {r['ticker']:5s} mom3={r['pct_change_3_f']:.3f}  rec={r['recommendation']}")

    # ── 5) Reversal_data.csv cross-validation ─────────────────────────────
    print("\n" + "=" * 70)
    print("CHECK 5: Historical Grade-A reversal_data.csv — would proposed caps reject wins?")
    print("=" * 70)
    rd = pd.read_csv(REV)
    # P&L for short: -reversal_open_close_pct
    rd["pnl"] = -rd["reversal_open_close_pct"] * 100
    rd["win"] = rd["pnl"] > 0
    rd_a = rd[rd["trade_grade"] == "A"].copy()
    print(f"Total Grade-A rows: {len(rd_a)} (wins: {rd_a['win'].sum()}, losses: {(~rd_a['win']).sum()})")

    for cap, cap_label, cap_threshold in [("Large", "Large", 0.35), ("Medium", "Medium", 0.55)]:
        sub = rd_a[rd_a["cap"] == cap].copy()
        n = len(sub)
        if n == 0:
            continue
        above = sub[sub["pct_change_3"] > cap_threshold]
        below = sub[sub["pct_change_3"] <= cap_threshold]
        print(f"\n  {cap} (n={n}) — proposed mom3 cap = {cap_threshold}:")
        print(f"    Above cap (would be REJECTED): n={len(above)}  win_rate={100*above['win'].mean() if len(above) else 0:.1f}%  avg_pnl={above['pnl'].mean() if len(above) else 0:+.2f}%")
        print(f"    Below cap (would be KEPT):     n={len(below)}  win_rate={100*below['win'].mean() if len(below) else 0:.1f}%  avg_pnl={below['pnl'].mean() if len(below) else 0:+.2f}%")
        if len(above):
            print(f"    Sample rejected wins:")
            wins_above = above[above["win"]].sort_values("pnl", ascending=False).head(5)
            for _, r in wins_above.iterrows():
                print(f"      {r['date']} {r['ticker']:5s} mom3={r['pct_change_3']:.3f}  pnl={r['pnl']:+.2f}%")

    # Gap_pct cross-validation
    print("\n  GAP_PCT cross-validation (proposal: drop Large floor 0.01, drop Medium floor to 0):")
    for cap in ["Large", "Medium"]:
        sub = rd_a[rd_a["cap"] == cap].copy()
        if len(sub) == 0:
            continue
        # Current scorer thresholds: Large gap_pct=0.01, Medium=0.05
        current_threshold = 0.01 if cap == "Large" else 0.05
        below = sub[sub["gap_pct"] < current_threshold]
        print(f"\n  {cap} Grade-A wins/losses with gap_pct < current threshold ({current_threshold}):")
        print(f"    n={len(below)}  win_rate={100*below['win'].mean() if len(below) else 0:.1f}%  avg_pnl={below['pnl'].mean() if len(below) else 0:+.2f}%")
        # And distribution at extremes
        big_gap = sub[sub["gap_pct"] >= 0.10]
        print(f"    Big-gap (>= 0.10): n={len(big_gap)}  win_rate={100*big_gap['win'].mean() if len(big_gap) else 0:.1f}%  avg_pnl={big_gap['pnl'].mean() if len(big_gap) else 0:+.2f}%")

    # ── 6) Same-ticker repeat analysis ────────────────────────────────────
    print("\n" + "=" * 70)
    print("CHECK 6: Same-ticker repeat signals (live log)")
    print("=" * 70)
    by_ticker = df.groupby("ticker").size().sort_values(ascending=False)
    repeats = by_ticker[by_ticker > 1]
    print(f"Tickers with >1 signal: {len(repeats)}")
    for ticker in repeats.index:
        sub = df[df["ticker"] == ticker].sort_values("date")
        first_tradeable = None
        print(f"\n  {ticker} ({len(sub)} signals):")
        for _, r in sub.iterrows():
            mark = "WIN" if r["tradeable_b"] else ("dir" if r["dir_ok"] else "XXX")
            print(f"    {r['date']} {r['bucket']:8s} {r['recommendation']:7s} {r['score']:>5s}  -> {mark}")

    # ── 7) Bounce sample ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CHECK 7: Bounce signals (report: 0/7 tradeable)")
    print("=" * 70)
    print(f"Bounce n={len(bnc)}, tradeable={bnc['tradeable_b'].sum()}, dir_ok={bnc['dir_ok'].sum()}")
    for _, r in bnc.iterrows():
        mark = "WIN" if r["tradeable_b"] else ("dir" if r["dir_ok"] else "XXX")
        print(f"  {r['date']} {r['ticker']:5s} {r['recommendation']:7s} {r['score']:>5s}  -> {mark}")

    # Save enriched dataframe
    out_path = Path(__file__).parent.parent / "data" / "signal_analysis_4_verification.csv"
    df.to_csv(out_path, index=False)
    print(f"\nEnriched data saved to: {out_path}")


if __name__ == "__main__":
    main()
