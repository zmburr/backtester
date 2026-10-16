"""Bounce edge by the day's market breadth: the table behind Method's "Bounce edge today".

The criteria study (2026-10-16) found the day matters more than the name. Breadth =
names market-wide scoring 4+/6 (B or better) at the open in the backscanner's universe
screen (data/bounce_population_2022_2026.csv, 218k ticker-days, 1,126 days), scored
with today's BouncePretrade thresholds; method/breadth.py counts the same thing live.

For every name-day the screen rated 4+/6 this reads the realised R under the live rules
(2-min prior-bar-break entry, 1.5-ATR-armed trail: scripts/backfill_entry_r.replay_entry_day)
from data/score_ev_study.csv, replaying any it lacks, and tabulates it by the day's state:

    normal    0-5 names     elevated  6-10     breadth day  11+

x the name's score (4, 5, 6/6; 'low' = 3 or less, from score_ev_study's random NO-GO control).
On breadth days it also replays the names that only the relaxed rules (gap and 3-day
thresholds halved) take to 4/6, for Method's "relaxed" rows.

R is per name-day (every attempt the rules take that day), counted only when the rules
entered. A cell under SHRINK_N names is pulled toward its day's average, so a tiny cell
can't headline; both the raw and the shown numbers are written.

Usage:
    python -m scripts.bounce_day_edge --population ../backtester/data/bounce_population_2022_2026.csv
Writes data/bounce_day_edge.json and data/bounce_day_edge_replays.csv (the replays it ran).
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyzers.bounce_scorer import SETUP_PROFILES  # noqa: E402
from method.breadth import BREADTH_DAY, ELEVATED, FLAG_SCORE, RELAX  # noqa: E402

DATA = ROOT / "data"
EV = DATA / "score_ev_study.csv"
OUT = DATA / "bounce_day_edge.json"
REPLAYS = DATA / "bounce_day_edge_replays.csv"
CHECKS = ["selloff_total_pct", "pct_off_30d_high", "gap_pct", "prior_day_range_atr", "pct_change_3",
          "pct_off_52wk_high"]
SHRINK_N = 30
STATES = [("normal", -1, ELEVATED - 1), ("elevated", ELEVATED - 1, BREADTH_DAY - 1), ("breadth", BREADTH_DAY - 1, 10**6)]


def score(df: pd.DataFrame, relax: dict | None = None) -> pd.Series:
    """BouncePretrade.validate's score, vectorized (matched it on 1,500 of 1,500 sampled rows)."""
    keys = df[["setup_type", "cap"]].drop_duplicates()
    total = pd.Series(0, index=df.index)
    for c in CHECKS:
        m = {(s, cap): SETUP_PROFILES.get(s, SETUP_PROFILES["GapFade_strongstock"]).get_threshold(c, cap)
             for s, cap in keys.itertuples(index=False)}
        th = pd.Series([m[(s, cap)] for s, cap in zip(df.setup_type, df.cap)], index=df.index)
        if relax and c in relax:
            th = th * relax[c]
        v = df[c]
        ok = (v >= th) if c == "prior_day_range_atr" else (v <= th)
        total += (ok & v.notna()).astype(int)
    return total


def state_of(n: int) -> str:
    return next(name for name, lo, hi in STATES if lo < n <= hi)


def replay(rows: pd.DataFrame) -> pd.DataFrame:
    from bounce_entry_study.detectors import build_2min
    from bounce_entry_study.fetch_bars import fetch_day
    from scripts.backfill_entry_r import replay_entry_day
    out = []
    for i, r in enumerate(rows.itertuples(index=False), 1):
        try:
            bars = fetch_day(r.ticker, r.date)
            if bars is None or bars.empty:
                res = {"entry_signals": 0, "note": "no bars"}      # kept, so it isn't retried
            else:
                two = build_2min(bars)
                atr_abs = float(r.atr_pct) * float(two["open"].iloc[0]) if r.atr_pct == r.atr_pct and r.atr_pct else None
                res = replay_entry_day(two, atr_abs)
        except Exception as e:  # noqa: BLE001 - one bad day must not stop the sweep
            res = {"entry_signals": 0, "note": f"error: {e}"[:80]}
        out.append({"date": r.date, "ticker": r.ticker, **res})
        if i % 50 == 0:
            print(f"  replayed {i}/{len(rows)}", flush=True)
    return pd.DataFrame(out)


def cell(r: pd.Series, days: int | None = None, dates: pd.Series | None = None) -> dict:
    """Name-weighted R, plus (with `dates`) the day-weighted view: a few big days with many
    names (4/7/25: 39 names at +5R) carry the name-weighted mean, so the card shows both."""
    keep = r.notna()
    r = r[keep]
    out = {"n": int(len(r)), "mean_r": round(float(r.mean()), 3) if len(r) else None,
           "median_r": round(float(r.median()), 3) if len(r) else None,
           "win": round(float((r > 0).mean()), 3) if len(r) else None}
    if days is not None:
        out["days"] = int(days)
    if dates is not None and len(r):
        by_day = r.groupby(dates[keep].values).mean()
        out.update(day_mean_r=round(float(by_day.mean()), 3), traded_days=int(len(by_day)),
                   pos_days=int((by_day > 0).sum()))
    return out


def shrink(c: dict, base: dict) -> dict:
    if not c["n"] or base.get("mean_r") is None:
        return {**c, "raw": dict(c), "mean_r": base.get("mean_r"), "win": base.get("win"), "shrunk": True}
    if c["n"] >= SHRINK_N:
        return c
    w = c["n"] / (c["n"] + SHRINK_N)
    return {**c, "raw": dict(c), "shrunk": True,
            "mean_r": round(w * c["mean_r"] + (1 - w) * base["mean_r"], 3),
            "win": round(w * c["win"] + (1 - w) * base["win"], 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--population", default=str(DATA / "bounce_population_2022_2026.csv"))
    args = ap.parse_args()

    pop = pd.read_csv(args.population)
    pop["ok"] = score(pop)
    pop["relaxed"] = score(pop, {"gap_pct": RELAX, "pct_change_3": RELAX})
    n_flag = pop[pop.ok >= FLAG_SCORE].groupby("date").size()
    pop["n_flag"] = pop.date.map(n_flag).fillna(0).astype(int)
    pop["state"] = pop.n_flag.map(state_of)
    days_by_state = pd.Series({d: state_of(int(n_flag.get(d, 0))) for d in pop.date.unique()}).value_counts()
    print(f"{len(pop)} name-days on {pop.date.nunique()} days; days by state {days_by_state.to_dict()}")

    ev_all = pd.read_csv(EV)
    ev = ev_all[["date", "ticker", "exit_r", "entry_signals"]]
    done = pd.read_csv(REPLAYS) if REPLAYS.exists() else pd.DataFrame(columns=["date", "ticker", "exit_r", "entry_signals"])
    known = pd.concat([ev, done.reindex(columns=["date", "ticker", "exit_r", "entry_signals"])]).drop_duplicates(["date", "ticker"])
    have = set(zip(known.date, known.ticker))

    surfaced = pop[pop.ok >= FLAG_SCORE]
    added = pop[(pop.relaxed >= FLAG_SCORE) & (pop.ok < FLAG_SCORE) & (pop.state == "breadth")]
    need = pd.concat([surfaced, added])
    need = need[[k not in have for k in zip(need.date, need.ticker)]]
    if len(need):
        print(f"replaying {len(need)} name-days not in {EV.name}")
        new = replay(need[["date", "ticker", "atr_pct"]])
        done = pd.concat([done, new]).drop_duplicates(["date", "ticker"])
        done.to_csv(REPLAYS, index=False)
        known = pd.concat([known, new.reindex(columns=["date", "ticker", "exit_r", "entry_signals"])]).drop_duplicates(["date", "ticker"])

    def with_r(df):
        m = df.merge(known, on=["date", "ticker"], how="left")
        m["exit_r"] = pd.to_numeric(m["exit_r"], errors="coerce")
        return m[pd.to_numeric(m.entry_signals, errors="coerce").fillna(0) > 0]

    s = with_r(surfaced)
    control = set(zip(*[ev_all.loc[ev_all.cohort == "nogo_control", c] for c in ("date", "ticker")]))
    low = with_r(pop[(pop.ok < FLAG_SCORE) & [k in control for k in zip(pop.date, pop.ticker)]])
    a = with_r(added)
    print(f"entries: surfaced {len(s)} of {len(surfaced)}, low-score control {len(low)}, relaxed {len(a)} of {len(added)}")

    table = {"built": datetime.date.today().isoformat(), "unit": "R per name-day under the live rules, when the rules entered",
             "population": Path(args.population).name, "shrink_n": SHRINK_N,
             "thresholds": {"elevated": ELEVATED, "breadth": BREADTH_DAY}, "states": {}, "relaxed": {}}
    for name, _, _ in STATES:
        x = s[s.state == name]
        base = cell(x.exit_r, days=int(days_by_state.get(name, 0)), dates=x.date)
        by = {}
        for k, part in (("4", x[x.ok == 4]), ("5", x[x.ok == 5]), ("6", x[x.ok >= 6]),
                        ("low", low[low.state == name])):
            by[k] = shrink(cell(part.exit_r), base)
        table["states"][name] = {**base, "by_score": by}
        print(f"{name:9} {base}  " + "  ".join(f"{k}: {v['mean_r']:+.2f}R n={v['n']}" for k, v in by.items()))
    if len(a):
        ab = a[a.state == "breadth"]
        table["relaxed"]["breadth"] = cell(ab.exit_r, dates=ab.date)
        print("relaxed on breadth days", table["relaxed"]["breadth"])
    OUT.write_text(json.dumps(table, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
