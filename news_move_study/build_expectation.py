"""Build news_move_expectation.json — the lookup orderPipe reads live.

The unit is ABSOLUTE room left (as a fraction of price), not the share of the
eventual move. Share is scale-free and ranks trades backwards: a stalled event
at minute 10 shows 76% of its move remaining while holding 0.87% of actual
room, and an already-extended one shows 44% while holding 2.63% — three times
the money. Anything built on share shouts loudest at the weakest trade.

Two conditioners, on a 3x3 grid:

  * PROGRESS — favorable excursion so far vs. the median total move for the
    ticker's cap band. Stalled (<25% of typical) / partway / extended (>100%).
  * TAPE — volume since the gun vs. the 30 minutes before it, in terciles.
    Heavy means the news got priced immediately; quiet means it is still
    leaking out.

Each cell carries the median room left and P(>=1% more) so the live read can
quote a magnitude and an odds, with its own n attached.

Cap bands survive because they set the denominator for progress.

    python -m news_move_study.build_expectation
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from news_move_study.events import build_events
from news_move_study.moves import build_move_table

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "news_move_expectation.json"

CAP_BANDS = [
    ("Large", 50e9, None),
    ("Mid", 10e9, 50e9),
    ("Small", 2e9, 10e9),
    ("Micro", 0.0, 2e9),
]

CHECKPOINTS = [5, 10]
TAPE_LABELS = ["quiet", "typical", "heavy"]
# Fixed, interpretable cuts rather than terciles — "hasn't done a quarter of a
# typical move" is a sentence you can say out loud, and it keeps the buckets
# stable when the artifact is rebuilt on more data.
PROGRESS_EDGES = [0.25, 1.0]
PROGRESS_LABELS = ["stalled", "partway", "extended"]
TRAIN_MAX_YEAR = 2024
MIN_CELL = 30


def classify_cap(mkt_cap) -> str:
    if mkt_cap is None or (isinstance(mkt_cap, float) and np.isnan(mkt_cap)):
        return "Unknown"
    for name, lo, hi in CAP_BANDS:
        if mkt_cap >= lo and (hi is None or mkt_cap < hi):
            return name
    return "Unknown"


def load_panel() -> pd.DataFrame:
    """Tradeable events with cap attached."""
    moves = build_move_table("first")
    d = moves[~moves.get("suspect", False) & moves["worked"]
              & ~moves["degenerate"]].copy()
    ev = build_events("first")[["symbol", "date_iso", "mkt_cap"]]
    d = d.merge(ev, on=["symbol", "date_iso"], how="left")
    d["mkt_cap"] = pd.to_numeric(d["mkt_cap"], errors="coerce")
    d["cap"] = d["mkt_cap"].apply(classify_cap)
    return d


def add_state(d: pd.DataFrame, k: int, cap_med: Dict[str, float],
              global_med: float) -> pd.DataFrame:
    """Progress / tape / room-left at checkpoint k."""
    d = d.copy()
    d["mfe_k_pct"] = pd.to_numeric(d[f"mfe_{k}"], errors="coerce") / d["ref_price"]
    d["room"] = (d["total_move_pct"] - d["mfe_k_pct"]).clip(lower=0)
    d["volr"] = pd.to_numeric(d[f"volr_{k}"], errors="coerce")
    denom = d["cap"].map(cap_med).fillna(global_med)
    d["progress"] = d["mfe_k_pct"] / denom
    return d.dropna(subset=["room", "volr", "progress"])


def _bucket(series: pd.Series, edges) -> np.ndarray:
    return np.digitize(series, edges)


def build() -> Dict:
    d0 = load_panel()
    cap_med = {c: float(g["total_move_pct"].median())
               for c, g in d0.groupby("cap") if len(g) >= MIN_CELL}
    global_med = float(d0["total_move_pct"].median())

    cap_bands: List[Dict] = []
    for name, lo, hi in CAP_BANDS:
        sel = d0[d0["cap"] == name]["total_move_pct"]
        if len(sel) < MIN_CELL:
            continue
        cap_bands.append({
            "cap": name, "min_mkt_cap": lo, "max_mkt_cap": hi, "n": int(len(sel)),
            "med_total_move_pct": round(float(sel.median()), 5),
        })

    grid: Dict[str, Dict] = {}
    validation: Dict[str, Dict] = {}
    for k in CHECKPOINTS:
        d = add_state(d0, k, cap_med, global_med)
        volr_edges = d["volr"].quantile([1 / 3, 2 / 3]).values
        d["tape_i"] = _bucket(d["volr"], volr_edges)
        d["prog_i"] = _bucket(d["progress"], PROGRESS_EDGES)

        cells = []
        for pi, plabel in enumerate(PROGRESS_LABELS):
            for ti, tlabel in enumerate(TAPE_LABELS):
                sel = d[(d["prog_i"] == pi) & (d["tape_i"] == ti)]
                cells.append({
                    "progress": plabel, "tape": tlabel, "n": int(len(sel)),
                    "med_room_pct": round(float(sel["room"].median()), 5) if len(sel) else None,
                    "q75_room_pct": round(float(sel["room"].quantile(.75)), 5) if len(sel) else None,
                    "p_1pct": round(float((sel["room"] >= 0.01).mean()), 3) if len(sel) else None,
                    "p_2pct": round(float((sel["room"] >= 0.02).mean()), 3) if len(sel) else None,
                    "readable": bool(len(sel) >= MIN_CELL),
                })
        grid[str(k)] = {
            "volr_edges": [round(float(e), 4) for e in volr_edges],
            "progress_edges": PROGRESS_EDGES,
            "cells": cells,
        }
        validation[str(k)] = _validate(d, volr_edges)

    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source": "ExitMonitor/data/trade_data.csv, news-tagged, one event per symbol-day",
        "n_events": int(len(d0)),
        "anchor": "ref_time (news break bar) — NOT position-entry time",
        "unit": "room left = remaining favorable excursion as a fraction of ref price",
        "why_not_share": "share of the eventual move is scale-free and ranks "
                         "stalled trades above extended ones; absolute room does not",
        "useful_window_min": [2, 15],
        "min_cell": MIN_CELL,
        "global_med_total_move_pct": round(global_med, 5),
        "cap_bands": cap_bands,
        "grid": grid,
        "validation": validation,
    }


def _validate(d: pd.DataFrame, volr_edges) -> Dict:
    """Out-of-sample MAE on room-left: base vs 1-D tape vs 2-D grid.

    Train medians are applied to unseen events; cells too thin in train fall
    back to the base rate, which is what the live read does when it goes quiet.
    """
    tr, te = d[d["year"] <= TRAIN_MAX_YEAR], d[d["year"] > TRAIN_MAX_YEAR]
    if len(tr) < 90 or len(te) < 90:
        return {}
    base = float(tr["room"].median())

    def mae(pred):
        return float(np.mean(np.abs(te["room"].values - pred)))

    tape_med = {b: (tr["room"][tr["tape_i"] == b].median()
                    if (tr["tape_i"] == b).sum() >= MIN_CELL else base)
                for b in (0, 1, 2)}
    pred_1d = np.array([tape_med[b] for b in te["tape_i"]])

    cell_med = {}
    for pi in (0, 1, 2):
        for ti in (0, 1, 2):
            sel = tr["room"][(tr["prog_i"] == pi) & (tr["tape_i"] == ti)]
            cell_med[(pi, ti)] = float(sel.median()) if len(sel) >= MIN_CELL else base
    pred_2d = np.array([cell_med[(p, t)] for p, t in zip(te["prog_i"], te["tape_i"])])

    m_base, m_1d, m_2d = mae(base), mae(pred_1d), mae(pred_2d)
    return {
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "mae_base": round(m_base, 5),
        "mae_tape_1d": round(m_1d, 5),
        "mae_grid_2d": round(m_2d, 5),
        "gain_1d_pct": round((m_base - m_1d) / m_base * 100, 1),
        "gain_2d_pct": round((m_base - m_2d) / m_base * 100, 1),
        "grid_beats_1d": bool(m_2d < m_1d),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    payload = build()
    OUT.write_text(json.dumps(payload, indent=2))
    for k, v in payload["validation"].items():
        print(f"minute {k}: base {v['mae_base']:.4f}  tape-1d {v['mae_tape_1d']:.4f} "
              f"({v['gain_1d_pct']:+.1f}%)  grid-2d {v['mae_grid_2d']:.4f} "
              f"({v['gain_2d_pct']:+.1f}%)  grid_beats_1d={v['grid_beats_1d']}")
    print()
    for k, node in payload["grid"].items():
        print(f"--- minute {k} ---")
        for c in node["cells"]:
            room = f"{c['med_room_pct']*100:.2f}%" if c["med_room_pct"] is not None else "  -  "
            p1 = f"{c['p_1pct']*100:.0f}%" if c["p_1pct"] is not None else " - "
            print(f"  {c['progress']:<9} {c['tape']:<8} n={c['n']:>4}  room {room}  "
                  f"P(>=1%) {p1}  {'' if c['readable'] else '<- too thin'}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
