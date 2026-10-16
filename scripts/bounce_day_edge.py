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

The table keeps learning. The study is frozen into two small files (--freeze, once, where
the population file lives). Every session since (method/breadth.py writes a record a day to
data/breadth_signals/) adds its counted names, replayed the same way once the day is over.
The nightly build rebuilds the table from both. Each state also gets a since_study cell (the
recorded days alone) and a watchlist cell (your priority report's GO/CAUTION bounce rows
from the signal scorecard, by the day's state). The rules (the 11-name line, the relaxed
x0.5, the checks) don't change here: that takes a review and the trader's OK.

Usage:
    python -m scripts.bounce_day_edge                # rebuild from the frozen study + the records
    python -m scripts.bounce_day_edge --freeze --population ../backtester/data/bounce_population_2022_2026.csv
Writes data/bounce_day_edge.json; --freeze also writes data/bounce_day_edge_history.csv and
data/bounce_breadth_days.csv. data/bounce_day_edge_entries.csv caches each name-day's
entries (first entry and re-entry R) from the 1-min bars; data/bounce_day_edge_replays.csv
holds the study replays the freeze ran.
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
from method.breadth import BANDS, BREADTH_DAY, ELEVATED, FLAG_SCORE, LOOP_START, RELAX  # noqa: E402

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
        try:
            bars = fetch_day(r.ticker, r.date)
            if bars is None or bars.empty:
                continue                       # not kept, so a later run tries again
            two = build_2min(bars)
            atr_abs = float(r.atr_pct) * float(two["open"].iloc[0]) if r.atr_pct == r.atr_pct and r.atr_pct else None
            taken = rule_trades(two, atr_abs)
        except Exception as e:  # noqa: BLE001 - one bad day must not stop the sweep
            print(f"  {r.ticker} {r.date}: {e}")
            continue
        out.append({"date": r.date, "ticker": r.ticker, "trades": len(taken),
                    "r1": round(taken[0].r, 3) if taken else None,
                    "r2": round(taken[1].r, 3) if len(taken) > 1 else None})
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


HISTORY = DATA / "bounce_day_edge_history.csv"      # the study's entered name-days, frozen
STUDY_DAYS = DATA / "bounce_breadth_days.csv"       # the study's days and their counts, frozen
SIGNALS = DATA / "breadth_signals"                  # one record a session since (method/breadth.py)
OUTCOMES = DATA / "signal_outcomes.csv"             # the scorecard: your watchlist's GO/CAUTION R
NAME_COLS = ["date", "ticker", "kind", "score", "n_flag", "state", "exit_r", "r1", "r2"]
DRIFT_DAYS = 5                                      # recorded breadth days before the card can warn


def freeze(population: str) -> None:
    """The study's name-days and days, from the population file and the replays already run,
    written to two small files so the table can be rebuilt anywhere (the mini has no
    population file). Run once; the study doesn't change."""
    pop = pd.read_csv(population)
    pop["ok"] = score(pop)
    pop["relaxed"] = score(pop, {"gap_pct": RELAX, "pct_change_3": RELAX})
    n_flag = pop[pop.ok >= FLAG_SCORE].groupby("date").size()
    pop["n_flag"] = pop.date.map(n_flag).fillna(0).astype(int)
    pop["state"] = pop.n_flag.map(state_of)

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

    control = set(zip(*[ev_all.loc[ev_all.cohort == "nogo_control", c] for c in ("date", "ticker")]))
    parts = [with_r(surfaced).assign(kind="strict", score=lambda x: x.ok),
             with_r(pop[(pop.ok < FLAG_SCORE) & [k in control for k in zip(pop.date, pop.ticker)]])
             .assign(kind="low", score=lambda x: x.ok),
             with_r(added).assign(kind="relaxed", score=lambda x: x.relaxed)]
    names = pd.concat(parts, ignore_index=True)
    ent = split_entries(names[["date", "ticker", "atr_pct"]])[["date", "ticker", "r1", "r2"]]
    names = names.merge(ent, on=["date", "ticker"], how="left")[NAME_COLS]
    names.sort_values(["date", "ticker", "kind"]).to_csv(HISTORY, index=False)
    days = pd.DataFrame({"date": sorted(pop.date.unique())})
    days["n_flag"] = days.date.map(n_flag).fillna(0).astype(int)
    days["state"] = days.n_flag.map(state_of)
    days.to_csv(STUDY_DAYS, index=False)
    print(f"froze {len(names)} entered name-days ({names.kind.value_counts().to_dict()}) and {len(days)} days")


def recorded(today: str | None = None, signals_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(name-days, days) from the session records. The names in the study's universe only
    (counted), so the cells mean what the study's did; each replayed once its day is over."""
    days, names = [], []
    for p in sorted((signals_dir or SIGNALS).glob("*_morning.json")):
        rec = json.loads(p.read_text())
        d = rec["date"]
        days.append({"date": d, "n_flag": int(rec["n"]), "state": rec["state"], "n_etf": rec.get("n_etf")})
        for s in rec.get("signals") or []:
            if s.get("counted"):
                names.append({"date": d, "ticker": s["ticker"], "kind": str(s["recommendation"]).lower(),
                              "score": int(s["score"]), "n_flag": int(rec["n"]), "state": rec["state"],
                              "atr_pct": (s.get("metrics") or {}).get("atr_pct")})
    days = pd.DataFrame(days, columns=["date", "n_flag", "state", "n_etf"])
    names = pd.DataFrame(names, columns=["date", "ticker", "kind", "score", "n_flag", "state", "atr_pct"])
    today = today or datetime.date.today().isoformat()
    done = names[names.date < today]                # bars are final once the day is over
    if len(done):
        ent = split_entries(done[["date", "ticker", "atr_pct"]])
        done = done.merge(ent[["date", "ticker", "trades", "r1", "r2"]], on=["date", "ticker"], how="inner")
        done = done[pd.to_numeric(done.trades, errors="coerce").fillna(0) > 0].copy()
        for c in ("r1", "r2"):
            done[c] = pd.to_numeric(done[c], errors="coerce")
        done["exit_r"] = done.r1.fillna(0) + done.r2.fillna(0)
    return done.reindex(columns=NAME_COLS), days


def watchlist_rows(day_state: dict, outcomes: Path = OUTCOMES) -> pd.DataFrame:
    """Your watchlist's GO/CAUTION bounce signals the rules entered (the scorecard's surfaced
    cohort), with the day's market-wide state."""
    if not outcomes.exists():
        return pd.DataFrame(columns=["date", "exit_r", "state"])
    o = pd.read_csv(outcomes, dtype=str, keep_default_na=False)
    o = o[(o.bucket == "bounce") & (o.cohort.isin(["", "surfaced"])) & (o.recommendation.isin(["GO", "CAUTION"]))
          & (pd.to_numeric(o.entry_signals, errors="coerce").fillna(0) > 0)]
    out = pd.DataFrame({"date": o.target_date, "exit_r": pd.to_numeric(o.exit_r, errors="coerce")})
    out["state"] = out.date.map(day_state)
    return out[out.state.notna() & out.exit_r.notna()]


def build_table(hist: pd.DataFrame, study_days: pd.DataFrame, rec: pd.DataFrame, rec_days: pd.DataFrame,
                watch: pd.DataFrame | None = None) -> dict:
    """The card's table: every cell from the study plus the recorded days, and each state's
    since-the-study cell from the recorded days alone."""
    names = (hist if rec.empty else pd.concat([hist, rec], ignore_index=True)).copy()
    for c in ("exit_r", "r1", "r2", "score", "n_flag"):
        names[c] = pd.to_numeric(names[c], errors="coerce")
    days = study_days[["date", "n_flag", "state"]]
    if len(rec_days):
        days = pd.concat([days, rec_days[["date", "n_flag", "state"]]]).drop_duplicates("date", keep="last")
    by_state = days.state.value_counts()
    rec_by_state = rec_days.state.value_counts()
    strict, low, relaxed = (names[names.kind == k] for k in ("strict", "low", "relaxed"))
    last = rec_days.date.max() if len(rec_days) else None
    table = {"built": datetime.date.today().isoformat(),
             "unit": "R per name-day under the live rules (first entry + one re-entry after a stop-out), "
                     "when the rules entered; entry_* = each single entry",
             "sources": {"study": f"{study_days.date.min()}..{study_days.date.max()}",
                         "recorded": {"since": LOOP_START, "sessions": int(len(rec_days)), "last": last}},
             "shrink_n": SHRINK_N, "thresholds": {"elevated": ELEVATED, "breadth": BREADTH_DAY},
             "states": {}, "bands": {}, "relaxed": {}, "watchlist": {}, "drift": None}
    for name, _, _ in STATES:
        x = strict[strict.state == name]
        base = cell(x.exit_r, days=int(by_state.get(name, 0)), dates=x.date, entries=per_entry(x))
        by = {}
        for k, part in (("4", x[x.score == 4]), ("5", x[x.score == 5]), ("6", x[x.score >= 6]),
                        ("low", low[low.state == name])):
            by[k] = shrink(cell(part.exit_r, entries=per_entry(part)), base)
        y = rec[(rec.kind == "strict") & (rec.state == name)] if len(rec) else rec
        since = cell(pd.to_numeric(y.exit_r, errors="coerce"), days=int(rec_by_state.get(name, 0)), dates=y.date,
                     entries=per_entry(y) if len(y) else None)
        table["states"][name] = {**base, "by_score": by, "since_study": {**since, "since": LOOP_START}}
        if watch is not None:
            w = watch[watch.state == name]
            table["watchlist"][name] = cell(w.exit_r, dates=w.date)
    # a breadth day's strength: the edge sat at 16+ names (11-15 was close to a coin flip)
    rec_n = pd.to_numeric(rec.n_flag, errors="coerce") if len(rec) else pd.Series(dtype=float)
    for band, lo, hi in BANDS:
        x = strict[strict.n_flag.between(lo, hi)]
        base = cell(x.exit_r, days=int(days.n_flag.between(lo, hi).sum()), dates=x.date, entries=per_entry(x))
        y = rec[(rec.kind == "strict") & rec_n.between(lo, hi)] if len(rec) else rec
        since = cell(pd.to_numeric(y.exit_r, errors="coerce"), dates=y.date, entries=per_entry(y) if len(y) else None,
                     days=int(rec_days.n_flag.between(lo, hi).sum()) if len(rec_days) else 0)
        table["bands"][band] = {**base, "since_study": {**since, "since": LOOP_START}}
    rb = relaxed[relaxed.state == "breadth"]
    if len(rb):
        table["relaxed"]["breadth"] = cell(rb.exit_r, dates=rb.date, entries=per_entry(rb))
    s = table["states"]["breadth"]["since_study"]
    if s.get("days", 0) >= DRIFT_DAYS and s.get("mean_r") is not None and s["mean_r"] < 0:
        table["drift"] = {"state": "breadth", "days": s["days"], "n": s["n"], "mean_r": s["mean_r"],
                          "history_mean_r": table["states"]["breadth"]["mean_r"]}
    return table


def rebuild(today: str | None = None, out: Path = OUT, signals_dir: Path | None = None,
            outcomes: Path = OUTCOMES) -> dict:
    """Nightly (method/breadth.py --build): the frozen study + every recorded session."""
    hist = pd.read_csv(HISTORY)
    study_days = pd.read_csv(STUDY_DAYS)
    rec, rec_days = recorded(today, signals_dir)
    day_state = dict(zip(study_days.date, study_days.state)) | dict(zip(rec_days.date, rec_days.state))
    table = build_table(hist, study_days, rec, rec_days, watchlist_rows(day_state, outcomes))
    out.write_text(json.dumps(table, indent=1))
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", action="store_true", help="re-freeze the study from the population file first")
    ap.add_argument("--population", default=str(DATA / "bounce_population_2022_2026.csv"))
    args = ap.parse_args()
    if args.freeze:
        freeze(args.population)
    table = rebuild()
    for name, st in table["states"].items():
        s = st["since_study"]
        print(f"{name:9} {st['mean_r']:+.2f}R n={st['n']} on {st['days']} days | since the study: "
              f"{s.get('days', 0)} days, n={s['n']}" + (f" {s['mean_r']:+.2f}R" if s.get("mean_r") is not None else ""))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
