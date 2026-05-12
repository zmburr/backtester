"""Run move analysis across all crack day case studies."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzers.crack_analyzer import CrackAnalysis
from analyzers.crack_covering_rules import CoveringRules

# Original 4 + 10 new candidates from reversal_data.csv
tickers = [
    # -- Original 4 --
    ("GLD",  "2026-01-29"),
    ("MSTR", "2024-11-21"),
    ("SMCI", "2024-02-16"),
    ("NVDA", "2024-03-08"),
    # -- New candidates (sorted by crack severity) --
    ("BYND", "2025-10-22"),   # Small,  -42%
    ("CLOV", "2021-06-09"),   # Medium, -40%
    ("AMC",  "2021-01-28"),   # Medium, -28%
    ("GME",  "2021-01-28"),   # Medium, -27%
    ("BBBY", "2022-03-07"),   # Medium, -28%
    ("MRNA", "2020-02-27"),   # Medium, -27%
    ("FCEL", "2020-11-24"),   # Small,  -23%
    ("COIN", "2022-08-04"),   # Medium, -16%
    ("DJT",  "2024-10-30"),   # Medium, -16%
    ("SPCE", "2021-07-12"),   # Medium, -18%
]

# Run once, cache results
results = []
for ticker, date in tickers:
    try:
        a = CrackAnalysis(ticker, date)
        a.fetch_data()
        a.compute_vwap()
        a.detect_hod_lod()
        a.detect_legs()
        a.compute_metrics()
        a.compute_move_analysis()

        ma = a.metrics.get("move_analysis", {})
        moves = ma.get("moves", [])
        total_atrs = a.metrics.get("total_crack_atrs", 0)
        total_pct = a.metrics.get("total_crack_pct", 0)

        print(f"\n=== {ticker} {date}  (Total crack: {total_atrs:.1f}x ATR, {total_pct*100:.1f}%, ${a.metrics['total_crack_dollars']:.2f}) ===")
        print(f"  {'Move':<8} {'ATRs':<8} {'Size $':<10} {'%':<8} {'Bars':<6} {'FailPBBs':<10} {'Held PBB @'}")
        print(f"  {'-'*68}")
        for mv in moves:
            pbb_str = f"${mv['pbb_price']:.2f}" if mv["pbb_price"] > 0 else "EOD"
            print(f"  Move {mv['move_num']:<3} {mv['size_atrs']:<8.2f} ${mv['size_dollars']:<9.2f} {mv['size_pct']*100:<7.1f}% {mv['bars']:<6} {mv.get('failed_pbbs_during',0):<10} {pbb_str}")

        if ma.get("move2_to_move1_ratio") is not None:
            print(f"\n  >>> M1: {ma['move1_atrs']:.2f} ATR | M2: {ma['move2_atrs']:.2f} ATR | M2/M1: {ma['move2_to_move1_ratio']:.2f}x")
        if ma.get("move3_to_move1_ratio") is not None:
            print(f"  >>> M3: {ma['move3_atrs']:.2f} ATR | M3/M1: {ma['move3_to_move1_ratio']:.2f}x")

        results.append({
            "ticker": ticker,
            "date": date,
            "total_atrs": total_atrs,
            "total_pct": total_pct,
            "moves": moves,
            "ma": ma,
            "hod_price": a.hod_price,
            "lod_price": a.lod_price,
            "close_price": a.rth_1m.iloc[-1]["close"],
        })
    except Exception as e:
        print(f"\n*** FAILED: {ticker} {date} -- {e}")
        results.append({"ticker": ticker, "date": date, "error": str(e)})

# Summary table
print("\n\n" + "=" * 90)
print("  SUMMARY TABLE")
print("=" * 90)
print(f"  {'Ticker':<8} {'Date':<12} {'Crack%':<8} {'TotATR':<8} {'M1 ATR':<8} {'M2 ATR':<8} {'M2/M1':<8} {'M1Fails':<8} {'M1Bars':<8} {'Pattern'}")
print(f"  {'-'*88}")

for r in results:
    if "error" in r:
        print(f"  {r['ticker']:<8} {r['date']:<12} ** ERROR: {r['error'][:40]}")
        continue

    moves = r["moves"]
    ma = r["ma"]
    m1_atrs = moves[0]["size_atrs"] if len(moves) >= 1 else 0
    m2_atrs = moves[1]["size_atrs"] if len(moves) >= 2 else 0
    ratio = ma.get("move2_to_move1_ratio", 0)
    m1_fails = moves[0].get("failed_pbbs_during", 0) if len(moves) >= 1 else 0
    m1_bars = moves[0].get("bars", 0) if len(moves) >= 1 else 0
    total = r["total_atrs"]
    crack_pct = r["total_pct"] * 100

    # Classify pattern
    if m1_atrs >= 2.0:
        pattern = "BigM1"
    elif m1_atrs > 0 and ratio >= 1.5:
        pattern = "SmallM1->BigM2"
    elif m1_atrs > 0:
        pattern = "Balanced"
    else:
        pattern = "N/A"

    print(f"  {r['ticker']:<8} {r['date']:<12} {crack_pct:<8.1f} {total:<8.1f} {m1_atrs:<8.2f} {m2_atrs:<8.2f} {ratio:<8.2f} {m1_fails:<8} {m1_bars:<8} {pattern}")

# Pattern breakdown
big_m1 = [r for r in results if "error" not in r and len(r["moves"]) >= 1 and r["moves"][0]["size_atrs"] >= 2.0]
small_m1 = [r for r in results if "error" not in r and len(r["moves"]) >= 2 and r["moves"][0]["size_atrs"] < 1.0 and r["ma"].get("move2_to_move1_ratio", 0) >= 1.5]
balanced = [r for r in results if "error" not in r and r not in big_m1 and r not in small_m1 and len(r["moves"]) >= 1]

print(f"\n\n  PATTERN BREAKDOWN")
print(f"  {'-'*60}")
print(f"  BigM1 (M1 >= 2 ATR):          {len(big_m1)} tickers")
for r in big_m1:
    m1 = r["moves"][0]["size_atrs"]
    m1f = r["moves"][0].get("failed_pbbs_during", 0)
    print(f"    {r['ticker']:<8} M1={m1:.2f} ATR, {m1f} fail PBBs")

print(f"\n  SmallM1->BigM2 (M1<1, M2/M1>=1.5): {len(small_m1)} tickers")
for r in small_m1:
    m1 = r["moves"][0]["size_atrs"]
    m2 = r["moves"][1]["size_atrs"]
    ratio = r["ma"]["move2_to_move1_ratio"]
    m1f = r["moves"][0].get("failed_pbbs_during", 0)
    print(f"    {r['ticker']:<8} M1={m1:.2f}, M2={m2:.2f}, M2/M1={ratio:.2f}x, {m1f} fail PBBs")

print(f"\n  Balanced/Other:               {len(balanced)} tickers")
for r in balanced:
    m1 = r["moves"][0]["size_atrs"]
    m2 = r["moves"][1]["size_atrs"] if len(r["moves"]) >= 2 else 0
    ratio = r["ma"].get("move2_to_move1_ratio", 0)
    print(f"    {r['ticker']:<8} M1={m1:.2f}, M2={m2:.2f}, M2/M1={ratio:.2f}x")

# ======================================================================
#  COVERING RULES SIMULATION
# ======================================================================
rules = CoveringRules()
cover_results = []

print(f"\n\n{'='*110}")
print(f"  COVERING RULES SIMULATION")
print(f"{'='*110}")
print(f"  {'Ticker':<8} {'Pattern':<28} {'AvgCover':<12} {'%Captured':<11} "
      f"{'vsLOD':<9} {'vsNaive':<9} {'Fills'}")
print(f"  {'-'*108}")

for r in results:
    if "error" in r:
        print(f"  {r['ticker']:<8} ** ERROR **")
        continue

    res = rules.simulate(
        r["moves"], r["hod_price"], r["lod_price"], r["close_price"])

    fill_str = " | ".join(
        f"M{f.move_num}:{f.cover_pct*100:.0f}%@${f.cover_price:.2f}"
        if f.move_num > 0 else
        f"EOD:{f.cover_pct*100:.0f}%@${f.cover_price:.2f}"
        for f in res.fills
    )

    print(f"  {r['ticker']:<8} {res.pattern_sequence:<28} "
          f"${res.avg_cover_price:<11.2f} {res.pct_captured:<10.1f}% "
          f"{res.vs_lod_pct:<+8.1f}% {res.vs_naive_pct:<+8.1f}% "
          f"{fill_str}")

    cover_results.append(res)

if cover_results:
    avg_cap = sum(r.pct_captured for r in cover_results) / len(cover_results)
    avg_lod = sum(r.vs_lod_pct for r in cover_results) / len(cover_results)
    avg_naive = sum(r.vs_naive_pct for r in cover_results) / len(cover_results)
    print(f"\n  {'AVERAGE':<8} {'':<28} {'':12} {avg_cap:<10.1f}% "
          f"{avg_lod:<+8.1f}% {avg_naive:<+8.1f}%")
    print(f"\n  {len(cover_results)} tickers simulated. "
          f"Rules capture {avg_cap:.0f}% of crack on average, "
          f"{avg_naive:+.0f}% vs covering everything at M1 PBB.")
