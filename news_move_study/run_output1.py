"""OUTPUT #1 — is there a decision window at all?

The live feature only makes sense if a meaningful share of a news move is
still ahead of you some minutes after the gun. This script answers that from
the unbiased population (losers included) and shows what the winners-only
sample in ExitMonitor's ``lo_to_hi`` was hiding.

The kill signal: if the median move is essentially over within a couple of
minutes, there is nothing to display and the study stops here.

    python -m news_move_study.run_output1
    python -m news_move_study.run_output1 --variant distinct --refresh
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from news_move_study.events import population_summary
from news_move_study.moves import CHECKPOINTS, build_move_table

log = logging.getLogger(__name__)

PCTLS = [.1, .25, .5, .75, .9]


def _fmt_pctls(s: pd.Series, unit: str = "") -> str:
    if s.empty:
        return "  (no data)"
    q = s.quantile(PCTLS)
    return ("    " + "  ".join(f"p{int(p*100):<2}={q.loc[p]:>7.1f}{unit}" for p in PCTLS)
            + f"   mean={s.mean():.1f}{unit}  n={len(s)}")


def report(variant: str = "first", refresh: bool = False) -> pd.DataFrame:
    df = build_move_table(variant, refresh=refresh)
    if df.empty:
        print("no measured events — run `python -m news_move_study.fetch_bars` first")
        return df

    print("=" * 78)
    print("NEWS MOVE STUDY — OUTPUT #1: is there a decision window?")
    print("=" * 78)
    print("\n[population]")
    print(population_summary())
    print(f"\n  measured with usable bars           : {len(df)}")

    # ---- did the move happen at all -------------------------------------
    worked = df["worked"]
    degen = df["degenerate"]
    suspect = df["suspect"] if "suspect" in df.columns else pd.Series(False, index=df.index)
    print("\n[data quality]")
    print(f"  ref_price disagrees with its bar    : {suspect.sum():>5} "
          f"({suspect.mean()*100:.1f}%)   <- dropped (split / ticker reuse / bad date)")
    print("\n[did the gun lead anywhere]")
    print(f"  never traded through ref (total<=0) : {(~worked & ~suspect).sum():>5} "
          f"({(~worked & ~suspect).mean()*100:.1f}%)")
    print(f"  moved < 0.5% of price (degenerate)  : {(degen & ~suspect).sum():>5} "
          f"({(degen & ~suspect).mean()*100:.1f}%)   <- display needs a 'going nowhere' state")
    df = df[~suspect]
    live = df[df["worked"] & ~df["degenerate"]]
    print(f"  tradeable moves used below          : {len(live):>5}")

    print("\n[total move from the gun, % of ref price]")
    print(_fmt_pctls(df["total_move_pct"] * 100, "%"))

    # ---- the headline number --------------------------------------------
    print("\n[minutes from the gun to the extreme of the move]")
    print("  ALL events (unbiased):")
    print(_fmt_pctls(df["time_to_extreme_min"], "m"))
    print("  tradeable moves only (>0.5%):")
    print(_fmt_pctls(live["time_to_extreme_min"], "m"))

    if "gross_pnl" in df.columns and df["gross_pnl"].notna().any():
        win = df[pd.to_numeric(df["gross_pnl"], errors="coerce") > 0]
        print("  winners only — reproduces lo_to_hi's filter (the biased view):")
        print(_fmt_pctls(win["time_to_extreme_min"], "m"))

    print("\n  share of moves whose extreme has already landed by minute k:")
    for k in (1, 2, 5, 10, 15, 30, 60, 120):
        a = (df["time_to_extreme_min"] <= k).mean() * 100
        b = (live["time_to_extreme_min"] <= k).mean() * 100
        print(f"    <= {k:>3}m   all {a:5.1f}%   tradeable {b:5.1f}%")

    # ---- the number the display would actually show ----------------------
    print("\n[% of the move STILL AHEAD at minute k]  <- what the feature displays")
    print("  (tradeable moves; median across events, IQR in brackets)")
    for k in CHECKPOINTS:
        col = f"pct_complete_{k}"
        if col not in live.columns:
            continue
        s = pd.to_numeric(live[col], errors="coerce").dropna()
        s = s[(s >= 0) & (s <= 1)]
        if s.empty:
            continue
        rem = (1 - s) * 100
        print(f"    minute {k:>3}:  {rem.median():5.1f}% left   "
              f"[{rem.quantile(.25):5.1f} - {rem.quantile(.75):5.1f}]   n={len(s)}")

    # ---- read on the kill signal ----------------------------------------
    med = live["time_to_extreme_min"].median()
    rem10 = pd.to_numeric(live.get("pct_complete_10"), errors="coerce").dropna()
    rem10 = (1 - rem10[(rem10 >= 0) & (rem10 <= 1)]).median() * 100 if len(rem10) else float("nan")
    print("\n[read]")
    print(f"  median tradeable move peaks {med:.0f} min after the gun; at minute 10 the")
    print(f"  median move still has {rem10:.0f}% of its range ahead of it.")
    print("  A decision window exists if that second number is comfortably")
    print("  above zero AND the spread around it is wide — wide spread is what")
    print("  conditioning can attack. A tight spread means every event looks the")
    print("  same at minute 10 and there is nothing for odds to add.")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="News move study, output #1")
    ap.add_argument("--variant", default="first", choices=("first", "distinct"))
    ap.add_argument("--refresh", action="store_true", help="re-measure from bars")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")
    report(args.variant, args.refresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
