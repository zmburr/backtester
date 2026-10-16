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

R is per name-day: the first entry plus the one re-entry the rules allow after a stop-out,
counted only when the rules entered. Each cell also carries the single-entry view (entry_*):
on breadth days a name averaged 1.5 entries, so +1.2R a name was +0.8R an entry. A cell
under SHRINK_N names is pulled toward its day's average, so a tiny cell can't headline;
both the raw and the shown numbers are written.

Usage:
    python -m scripts.bounce_day_edge --population ../backtester/data/bounce_population_2022_2026.csv
Writes data/bounce_day_edge.json, data/bounce_day_edge_replays.csv (the replays it ran) and
data/bounce_day_edge_entries.csv (each name-day's entries, from cached 1-min bars).
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
ENTRIES = DATA / "bounce_day_edge_entries.csv"
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


def split_entries(rows: pd.DataFrame) -> pd.DataFrame:
    """R of each entry the rules took (r1 = first, r2 = the re-entry) for each name-day, kept in
    ENTRIES so a re-run needs no bars. Same frame and ATR as replay(), so r1 + r2 = exit_r."""
    from bounce_entry_study.detectors import build_2min
    from bounce_entry_study.fetch_bars import fetch_day
    from scripts.backfill_entry_r import rule_trades
    cols = ["date", "ticker", "trades", "r1", "r2"]
    done = pd.read_csv(ENTRIES) if ENTRIES.exists() else pd.DataFrame(columns=cols)
    have = set(zip(done.date, done.ticker))
    todo = rows.drop_duplicates(["date", "ticker"])
    todo = todo[[k not in have for k in zip(todo.date, todo.ticker)]]
    out = []
    for i, r in enumerate(todo.itertuples(index=False), 1):
        rec = {"date": r.date, "ticker": r.ticker, "trades": 0, "r1": None, "r2": None}
        try:
            bars = fetch_day(r.ticker, r.date)
            if bars is not None and not bars.empty:
                two = build_2min(bars)
                atr_abs = float(r.atr_pct) * float(two["open"].iloc[0]) if r.atr_pct == r.atr_pct and r.atr_pct else None
                taken = rule_trades(two, atr_abs)
                rec.update(trades=len(taken), r1=round(taken[0].r, 3) if taken else None,
                           r2=round(taken[1].r, 3) if len(taken) > 1 else None)
        except Exception as e:  # noqa: BLE001 - one bad day must not stop the sweep
            print(f"  {r.ticker} {r.date}: {e}")
        out.append(rec)
        if i % 200 == 0:
            print(f"  split {i}/{len(todo)}", flush=True)
    if out:
        new = pd.DataFrame(out, columns=cols)
        done = (new if done.empty else pd.concat([done, new])).drop_duplicates(["date", "ticker"])
        done.sort_values(["date", "ticker"]).to_csv(ENTRIES, index=False)
    return done


def per_entry(df: pd.DataFrame) -> pd.Series:
    return pd.concat([df.r1, df.r2]).dropna()


def cell(r: pd.Series, days: int | None = None, dates: pd.Series | None = None,
         entries: pd.Series | None = None) -> dict:
    """Name-weighted R, plus (with `dates`) the day-weighted view: a few big days with many
    names (4/7/25: 39 names at +5R) carry the name-weighted mean, so the card shows both.
    `entries` = the R of each single entry behind those names."""
    keep = r.notna()
    r = r[keep]
    out = {"n": int(len(r)), "mean_r": round(float(r.mean()), 3) if len(r) else None,
           "median_r": round(float(r.median()), 3) if len(r) else None,
           "win": round(float((r > 0).mean()), 3) if len(r) else None}
    if entries is not None and len(entries):
        out.update(entry_n=int(len(entries)), entry_mean_r=round(float(entries.mean()), 3),
                   entry_win=round(float((entries > 0).mean()), 3))
    if days is not None:
        out["days"] = int(days)
    if dates is not None and len(r):
        by_day = r.groupby(dates[keep].values).mean()
        out.update(day_mean_r=round(float(by_day.mean()), 3), traded_days=int(len(by_day)),
                   pos_days=int((by_day > 0).sum()))
    return out


SHRUNK = (("mean_r", "win"), ("entry_mean_r", "entry_win"))


def shrink(c: dict, base: dict) -> dict:
    if not c["n"] or base.get("mean_r") is None:
        return {**c, "raw": dict(c), "shrunk": True, **{k: base.get(k) for pair in SHRUNK for k in pair}}
    if c["n"] >= SHRINK_N:
        return c
    w = c["n"] / (c["n"] + SHRINK_N)
    out = {**c, "raw": dict(c), "shrunk": True}
    for pair in SHRUNK:
        for k in pair:
            if c.get(k) is not None and base.get(k) is not None:
                out[k] = round(w * c[k] + (1 - w) * base[k], 3)
    return out


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
    ent = split_entries(pd.concat([s, low, a])[["date", "ticker", "atr_pct"]])[["date", "ticker", "r1", "r2"]]
    s, low, a = (x.merge(ent, on=["date", "ticker"], how="left") for x in (s, low, a))
    for x in (s, low, a):
        x["r1"], x["r2"] = pd.to_numeric(x.r1, errors="coerce"), pd.to_numeric(x.r2, errors="coerce")
    split_ok = ((s.r1.fillna(0) + s.r2.fillna(0) - s.exit_r).abs() < 0.02).mean()
    print(f"entries split for {s.r1.notna().sum()} of {len(s)} surfaced name-days; r1 + r2 = exit_r on {split_ok:.1%}")

    table = {"built": datetime.date.today().isoformat(),
             "unit": "R per name-day under the live rules (first entry + one re-entry after a stop-out), "
                     "when the rules entered; entry_* = each single entry",
             "population": Path(args.population).name, "shrink_n": SHRINK_N,
             "thresholds": {"elevated": ELEVATED, "breadth": BREADTH_DAY}, "states": {}, "relaxed": {}}
    for name, _, _ in STATES:
        x = s[s.state == name]
        base = cell(x.exit_r, days=int(days_by_state.get(name, 0)), dates=x.date, entries=per_entry(x))
        by = {}
        for k, part in (("4", x[x.ok == 4]), ("5", x[x.ok == 5]), ("6", x[x.ok >= 6]),
                        ("low", low[low.state == name])):
            by[k] = shrink(cell(part.exit_r, entries=per_entry(part)), base)
        table["states"][name] = {**base, "by_score": by}
        print(f"{name:9} {base}  " + "  ".join(f"{k}: {v['mean_r']:+.2f}R n={v['n']}" for k, v in by.items()))
    if len(a):
        ab = a[a.state == "breadth"]
        table["relaxed"]["breadth"] = cell(ab.exit_r, dates=ab.date, entries=per_entry(ab))
        print("relaxed on breadth days", table["relaxed"]["breadth"])
    OUT.write_text(json.dumps(table, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
