"""OUTPUT #2 — is the dispersion PREDICTABLE, and are the cells thick enough?

Output #1 established that at minute 10 the spread of "% of the move still
ahead" runs from 0% to 64% across events. This asks the only question that
matters next: can state observable AT minute k separate those, on events the
model never saw?

Method, deliberately dumb so the answer can't be an artifact of model choice:

  * split by time — train on pre-2025 guns, test on 2025-26;
  * bucket each candidate state feature into terciles using TRAIN edges only;
  * the prediction for a test event is the TRAIN median of "% remaining" in
    its bucket;
  * compare against the unconditional TRAIN median (the base rate) on the
    same test events, by mean absolute error.

If terciles of a single feature can't beat the base rate out-of-sample, a
fancier model on 1.5k events is unlikely to rescue it.

Cell thickness is reported alongside: a live display can only speak when its
bucket carries enough history to mean something.

    python -m news_move_study.run_output2
"""
from __future__ import annotations

import argparse
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from news_move_study.moves import build_move_table

log = logging.getLogger(__name__)

CHECKPOINTS = [2, 5, 10, 15, 30]
TRAIN_MAX_YEAR = 2024
MIN_CELL = 30  # below this a bucket can't be read out loud


def build_features(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """State at minute k + the target, for tradeable non-suspect events."""
    d = df[~df.get("suspect", False) & df["worked"] & ~df["degenerate"]].copy()
    tgt = pd.to_numeric(d[f"pct_complete_{k}"], errors="coerce")
    d["remain"] = (1 - tgt).clip(0, 1)

    mfe = pd.to_numeric(d[f"mfe_{k}"], errors="coerce")
    mae = pd.to_numeric(d[f"mae_{k}"], errors="coerce")
    imp = pd.to_numeric(d["impulse"], errors="coerce")

    # Candidate scale units for "how far has it come" (the fork you delegated).
    d["ext_impulse"] = mfe / imp                      # extension in opening thrusts
    d["ext_raw"] = mfe / d["ref_price"]               # raw % from ref
    # Shape / behaviour features, all knowable at k.
    d["pullback"] = mae / mfe.replace(0, np.nan)      # how clean the run is
    d["stall"] = pd.to_numeric(d[f"stall_{k}"], errors="coerce")
    d["volr"] = pd.to_numeric(d[f"volr_{k}"], errors="coerce")
    return d.dropna(subset=["remain"])


def tercile_test(d: pd.DataFrame, feature: str) -> Optional[Dict]:
    """Train-edge terciles -> per-bucket train median -> score on test."""
    tr = d[d["year"] <= TRAIN_MAX_YEAR]
    te = d[d["year"] > TRAIN_MAX_YEAR]
    tr = tr[tr[feature].notna()]
    te = te[te[feature].notna()]
    if len(tr) < 60 or len(te) < 60:
        return None

    try:
        edges = tr[feature].quantile([1 / 3, 2 / 3]).values
    except Exception:
        return None
    if not np.all(np.diff(edges) > 0):
        return None

    def bucket(s):
        return np.digitize(s, edges)

    tr_b, te_b = bucket(tr[feature]), bucket(te[feature])
    base = tr["remain"].median()

    cell_medians, cells = {}, {}
    for b in (0, 1, 2):
        sel = tr["remain"][tr_b == b]
        cells[b] = len(sel)
        cell_medians[b] = sel.median() if len(sel) else base

    pred = np.array([cell_medians[b] for b in te_b])
    mae_cond = np.mean(np.abs(te["remain"].values - pred))
    mae_base = np.mean(np.abs(te["remain"].values - base))
    spread = max(cell_medians.values()) - min(cell_medians.values())
    return {
        "feature": feature,
        "n_train": len(tr), "n_test": len(te),
        "min_cell": min(cells.values()),
        "base": base,
        "lo": cell_medians[0], "mid": cell_medians[1], "hi": cell_medians[2],
        "spread": spread,
        "mae_base": mae_base, "mae_cond": mae_cond,
        "improve_pct": (mae_base - mae_cond) / mae_base * 100 if mae_base else 0.0,
    }


def report(refresh: bool = False) -> None:
    df = build_move_table("first", refresh=refresh)
    print("=" * 82)
    print("NEWS MOVE STUDY — OUTPUT #2: is the dispersion predictable?")
    print("=" * 82)
    print(f"  train = guns through {TRAIN_MAX_YEAR}   test = {TRAIN_MAX_YEAR + 1}+")
    print(f"  prediction = train median of '% remaining' in the event's tercile")
    print(f"  baseline   = train median overall (the base rate)")

    features = ["ext_impulse", "ext_raw", "pullback", "stall", "volr"]
    any_win = False
    for k in CHECKPOINTS:
        d = build_features(df, k)
        tr_n = (d["year"] <= TRAIN_MAX_YEAR).sum()
        te_n = (d["year"] > TRAIN_MAX_YEAR).sum()
        print(f"\n--- minute {k}   (train {tr_n}, test {te_n}) "
              f"base rate {d[d['year'] <= TRAIN_MAX_YEAR]['remain'].median() * 100:.0f}% remaining ---")
        print(f"    {'feature':<12} {'lo':>6} {'mid':>6} {'hi':>6} {'spread':>7} "
              f"{'mincell':>8} {'MAEbase':>8} {'MAEcond':>8} {'gain':>7}")
        rows = []
        for f in features:
            r = tercile_test(d, f)
            if r is None:
                continue
            rows.append(r)
        for r in sorted(rows, key=lambda x: -x["improve_pct"]):
            flag = ""
            if r["improve_pct"] > 3 and r["min_cell"] >= MIN_CELL:
                flag = "  <-- separates"
                any_win = True
            print(f"    {r['feature']:<12} {r['lo']*100:5.0f}% {r['mid']*100:5.0f}% "
                  f"{r['hi']*100:5.0f}% {r['spread']*100:6.0f}pp {r['min_cell']:8d} "
                  f"{r['mae_base']:8.3f} {r['mae_cond']:8.3f} {r['improve_pct']:6.1f}%{flag}")

    print("\n[cell thickness]")
    d10 = build_features(df, 10)
    tr = d10[d10["year"] <= TRAIN_MAX_YEAR]
    print(f"  train events available at minute 10 : {len(tr)}")
    print(f"  1-D terciles  -> ~{len(tr)//3} per cell   (readable)")
    print(f"  2-D 3x3 grid  -> ~{len(tr)//9} per cell   "
          f"({'readable' if len(tr)//9 >= MIN_CELL else 'TOO THIN'})")
    print(f"  3-D 3x3x3     -> ~{len(tr)//27} per cell   "
          f"({'readable' if len(tr)//27 >= MIN_CELL else 'TOO THIN'})")

    print("\n[read]")
    if any_win:
        print("  At least one single feature beats the base rate out-of-sample with")
        print("  readable cells. The dispersion is partly predictable — proceed, but")
        print("  size the model to the cell counts above, not to ambition.")
    else:
        print("  No single feature beats the base rate out-of-sample by a usable")
        print("  margin. The spread in output #1 is real but looks like noise from")
        print("  where you stand at minute k. That is the kill signal.")


def main() -> int:
    ap = argparse.ArgumentParser(description="News move study, output #2")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING),
                        format="%(asctime)s %(levelname)s %(message)s")
    report(args.refresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
