"""Comps from here: how bounce days in the same state traded after this moment.

The watcher's analog band drew 5 hand-logged winners as % from the open, on the clock, as a
min-max envelope; on DRAM 10/15 its comps came from 3 market crashes and it called a working
bounce dead at 10:03. This tests a replacement: a state match on every bounce-candidate day
we can replay, failed ones included.

Verdict 2026-10-17 (walk-forward, train before 2025 or 2026): NOT shown in Method. Pooled it
ranks (+0.5 ATR first: top fifth 29% vs bottom 5%), but a plain clock + bounce-off-the-low
baseline does nearly as well; it is worse than that baseline on breadth days (9-20 test
dates) and on new-low-first odds everywhere; the +15/30/60 cone is about +-0.1 ATR wide; and
on DRAM 10/15 it also said give up (10:00: 14% up first, 75% new low first; DRAM then made
+1.2 ATR). The give-up rule stays the rules' own: the stop at the low. Re-run --study once
more breadth days are recorded.

Pieces:

    population   score_ev_study (surfaced + NO-GO controls), the scorecard's bounce rows,
                 the breadth study's name-days and the book, one row per (date, ticker)
    nodes        every 2 min 9:40-14:00, from the 1-min bars up to that moment only:
                 running low of day, minutes since it, the bounce off it and the distance to
                 VWAP in ATR, whether the low came before 10:00, the flush (open to low) in
                 ATR, the day type (normal / elevated / 11-15 / 16+) and the kind
                 (stock / ETF / leveraged ETF)
    labels       +0.5 ATR above the node close before a new low (up), a new low first
                 (low), or neither by 15:55; the move at +15/+30/+60 min and to the close,
                 in ATR from the node close
    match        the K nearest nodes in state, clock within +-20 min, one node a comp day
                 and at most 3 days a date (correlated breadth days are what sank the band)

ATR in dollars is the day's atr_pct (prior-day ATR / price, as every source carries it) times
the 9:30 open, the convention of scripts/backfill_entry_r and scripts/bounce_day_edge.

    python -m method.comps --build [--bars DIR ...] [--population CSV] [--fetch N]
    python -m method.comps --study
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
METHOD_DIR = Path(os.environ.get("METHOD_DIR") or DATA / "method")
NODES_PATH = METHOD_DIR / "comps_nodes.npz"
POP_PATH = METHOD_DIR / "comps_population.csv"
KINDS_PATH = METHOD_DIR / "comps_kinds.json"
BAR_CACHE = DATA / "bounce_entry_study"          # fetch_day's cache: {TICKER}_{date}.pkl
POPULATION_CSV = DATA / "bounce_population_2022_2026.csv"   # atr_pct for the breadth study's names

NODE_FROM, NODE_TO, STEP = 10, 270, 2             # minutes after 9:30: 9:40 .. 14:00
LAST_M = 385                                      # labels run through the 15:54 bar (15:55)
UP_ATR = 0.5
HORIZONS = (15, 30, 60)
K = 200
CLOCK_WIN = 20
PER_DATE = 3
MIN_COMPS = 30                                    # fewer than this and the match says nothing
ATR_PCT_OK = (0.005, 1.0)                         # outside it the source row is junk (splits)

DAY_TYPES = ("normal", "elevated", "11-15", "16+")
KINDS = ("stock", "etf", "lev")
FEATS = ("mins_since_low", "bounce_atr", "vwap_atr", "low_pre10", "flush_atr")
WEIGHTS = {"mins_since_low": 1.0, "bounce_atr": 1.5, "vwap_atr": 1.0, "low_pre10": 0.75, "flush_atr": 0.75}
DAY_TYPE_W = 0.75                                 # per step on the day-type ladder
KIND_W = 1.0
CLOCK_W = 0.5                                     # at the edge of the clock window
OUTCOME = {"up": 1, "low": -1, "neither": 0}


# --------------------------------------------------------------------------- population

def day_type_of(n: int | None) -> str:
    from method.breadth import BREADTH_DAY, ELEVATED, STRONG
    n = int(n or 0)
    return "16+" if n >= STRONG else "11-15" if n >= BREADTH_DAY else "elevated" if n >= ELEVATED else "normal"


def day_types() -> dict[str, str]:
    """{date: day type}: the frozen study's counts (2022 to 6/30/26), then the recorded
    mornings (data/breadth_signals). Dates neither covers are normal."""
    out = {}
    study = DATA / "bounce_breadth_days.csv"
    if study.exists():
        d = pd.read_csv(study)
        out.update({r.date: day_type_of(r.n_flag) for r in d.itertuples()})
    for p in sorted((DATA / "breadth_signals").glob("*_morning.json")):
        try:
            rec = json.loads(p.read_text())
            out[rec["date"]] = day_type_of(rec.get("n"))
        except Exception as e:
            log.warning(f"{p.name}: {e}")
    return out


def _atr_ok(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if ATR_PCT_OK[0] < x <= ATR_PCT_OK[1] else None


def population(population_csv: Path | None = POPULATION_CSV) -> pd.DataFrame:
    """One row per (date, ticker): date, ticker, atr_pct, cap, src. The first source with a
    sane atr_pct wins, in the order below."""
    parts = []
    ev = pd.read_csv(DATA / "score_ev_study.csv")
    parts.append(pd.DataFrame({"date": ev.date, "ticker": ev.ticker, "atr_pct": ev.atr_pct,
                               "cap": ev.cap, "src": "ev:" + ev.cohort.astype(str)}))
    so = pd.read_csv(DATA / "signal_outcomes.csv")
    so = so[so.bucket == "bounce"]
    parts.append(pd.DataFrame({"date": so.target_date, "ticker": so.ticker, "atr_pct": so.atr_pct,
                               "cap": so.cap, "src": "scorecard:" + so.cohort.astype(str)}))
    eh = pd.read_csv(DATA / "bounce_day_edge_history.csv")
    eh = eh.assign(atr_pct=np.nan, cap=None, src="breadth:" + eh.kind.astype(str))[
        ["date", "ticker", "atr_pct", "cap", "src"]]
    if population_csv is not None and Path(population_csv).exists():
        keys = set(zip(eh.date, eh.ticker))
        atr = {}
        for chunk in pd.read_csv(population_csv, usecols=["date", "ticker", "cap", "atr_pct"], chunksize=200_000):
            hit = chunk[[k in keys for k in zip(chunk.date, chunk.ticker)]]
            atr.update({(r.date, r.ticker): (r.atr_pct, r.cap) for r in hit.itertuples()})
        eh["atr_pct"] = [atr.get(k, (np.nan, None))[0] for k in zip(eh.date, eh.ticker)]
        eh["cap"] = [atr.get(k, (np.nan, None))[1] for k in zip(eh.date, eh.ticker)]
    parts.append(eh)
    bd = pd.read_csv(DATA / "bounce_data.csv", encoding="utf-8-sig")
    parts.append(pd.DataFrame({"date": pd.to_datetime(bd.date, format="%m/%d/%Y").dt.strftime("%Y-%m-%d"),
                               "ticker": bd.ticker.astype(str).str.strip(), "atr_pct": bd.atr_pct,
                               "cap": bd.cap, "src": "book"}))
    pop = pd.concat(parts, ignore_index=True)
    pop["ticker"] = pop.ticker.astype(str).str.upper()
    pop = pop[~pop.ticker.str.endswith(".T")]
    pop["atr_pct"] = pop.atr_pct.map(_atr_ok)
    pop["has_atr"] = pop.atr_pct.notna()
    # the first source with an ATR for the key, else the first source
    pop = pop.sort_values("has_atr", ascending=False, kind="stable").drop_duplicates(["date", "ticker"])
    return pop.drop(columns="has_atr").sort_values(["date", "ticker"]).reset_index(drop=True)


LEV_NAME = re.compile(r"(\d(\.\d+)?\s*x\b)|\bultra|leveraged", re.I)


def fetch_kinds(tickers: set[str]) -> dict[str, str]:
    """{ticker: etf | lev} for the exchange-traded products among `tickers`, from Polygon's
    reference list (active and delisted). Leveraged = 2X/3X/Ultra/Leveraged in the name."""
    from method.breadth import _client
    client = _client()
    out = {}
    for typ in ("ETF", "ETN", "ETV", "ETS"):
        for active in (True, False):
            try:
                for t in client.list_tickers(market="stocks", type=typ, active=active, limit=1000):
                    if t.ticker in tickers:
                        out[t.ticker] = "lev" if LEV_NAME.search(t.name or "") else "etf"
            except Exception as e:
                log.warning(f"ticker types {typ} active={active}: {e}")
    return out


def kinds_for(pop: pd.DataFrame, refresh: bool = False) -> dict[str, str]:
    """{ticker: stock | etf | lev}, cached in comps_kinds.json; the playbook email's
    leveraged list and cap == ETF fill gaps in Polygon's list."""
    from scripts.playbook_email import PRINCIPALS
    tickers = set(pop.ticker)
    cache = {}
    if KINDS_PATH.exists() and not refresh:
        cache = json.loads(KINDS_PATH.read_text())
    if refresh or not tickers <= set(cache):
        got = fetch_kinds(tickers - set(cache) if not refresh else tickers)
        cache.update({t: got.get(t, "stock") for t in (tickers - set(cache) if not refresh else tickers)})
        KINDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        KINDS_PATH.write_text(json.dumps(cache, sort_keys=True))
    etf_cap = set(pop.loc[pop.cap == "ETF", "ticker"])
    out = {}
    for t in tickers:
        k = cache.get(t, "stock")
        if t in PRINCIPALS:
            k = "lev"
        elif k == "stock" and t in etf_cap:
            k = "etf"
        out[t] = k
    return out


# --------------------------------------------------------------------------- bars

def bar_dirs(extra: list[str] | None = None) -> list[Path]:
    dirs = [BAR_CACHE] + [Path(p) for p in (extra or [])]
    dirs += [Path(p) for p in os.environ.get("METHOD_COMPS_BARS", "").split(os.pathsep) if p]
    return [d for d in dict.fromkeys(dirs) if d.exists()]


def load_bars(ticker: str, date: str, dirs: list[Path]) -> pd.DataFrame | None:
    for d in dirs:
        p = d / f"{ticker.upper()}_{date}.pkl"
        if p.exists():
            try:
                return pd.read_pickle(p)
            except Exception as e:
                log.warning(f"{p.name}: {e}")
    return None


def rth_arrays(bars: pd.DataFrame) -> dict | None:
    """The regular session as arrays: m = minutes after 9:30 of each bar's start (bars are
    indexed by their start, Polygon's and the PC pusher's convention), o h l c v, and the
    session VWAP through each bar (bar vwap x volume when the bars carry it, else the
    typical price; no volume at all = the typical price's running mean)."""
    if bars is None or bars.empty:
        return None
    idx = bars.index
    if idx.tz is None:
        idx = idx.tz_localize("US/Eastern")
    else:
        idx = idx.tz_convert("US/Eastern")
    m = (idx.hour * 60 + idx.minute - 570).to_numpy()
    keep = (m >= 0) & (m < 390)
    if not keep.any():
        return None
    b = bars[keep]
    m = m[keep]
    o, h, l, c = (b[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    v = b["volume"].to_numpy(dtype=float) if "volume" in b else np.ones(len(b))
    v = np.where(np.isfinite(v) & (v > 0), v, 0.0)
    px = b["vwap"].to_numpy(dtype=float) if "vwap" in b else (h + l + c) / 3
    px = np.where(np.isfinite(px), px, (h + l + c) / 3)
    if v.sum() <= 0:
        v = np.ones(len(b))
    cv = np.cumsum(v)
    vw = np.where(cv > 0, np.cumsum(px * v) / np.where(cv > 0, cv, 1), np.cumsum((h + l + c) / 3) / np.arange(1, len(b) + 1))
    return {"m": m, "o": o, "h": h, "l": l, "c": c, "vw": vw}


def _running_low(l: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    run = np.minimum.accumulate(l)
    prev = np.r_[np.inf, run[:-1]]
    pos = np.maximum.accumulate(np.where(l < prev, np.arange(len(l)), 0))
    return run, pos


def state_at(a: dict, atr: float, t: int) -> dict | None:
    """The state at minute `t` after 9:30 (a node closes at 9:30 + t), from bars that
    started before t only. None before the first bar or without an ATR."""
    if not a or not atr or atr <= 0:
        return None
    j = int(np.searchsorted(a["m"], t, side="left")) - 1
    if j < 0:
        return None
    run, pos = _running_low(a["l"][:j + 1])
    lod, lod_m, c = float(run[j]), int(a["m"][pos[j]]), float(a["c"][j])
    return {"t": int(t), "j": j, "close": c, "lod": lod, "lod_m": lod_m,
            "mins_since_low": float(t - lod_m),
            "bounce_atr": (c - lod) / atr,
            "vwap_atr": (c - float(a["vw"][j])) / atr,
            "low_pre10": float(lod_m < 30),
            "flush_atr": (float(a["o"][0]) - lod) / atr}


def labels_at(a: dict, atr: float, s: dict) -> dict:
    """What happened after the node: +0.5 ATR above its close before a new low ('up'), a new
    low first ('low'; a bar that does both counts as low), or neither by 15:55; the move in
    ATR at +15/+30/+60 min and to the close."""
    j, c, lod, t = s["j"], s["close"], s["lod"], s["t"]
    m, h, l, cl = a["m"], a["h"], a["l"], a["c"]
    after = slice(j + 1, int(np.searchsorted(m, LAST_M, side="left")))
    hh, ll = h[after], l[after]
    up_i = np.flatnonzero(hh >= c + UP_ATR * atr)
    lo_i = np.flatnonzero(ll < lod)
    up_first = up_i[0] if len(up_i) else None
    lo_first = lo_i[0] if len(lo_i) else None
    if lo_first is not None and (up_first is None or lo_first <= up_first):
        out = "low"
    elif up_first is not None:
        out = "up"
    else:
        out = "neither"
    res = {"outcome": out}
    for hz in HORIZONS:
        k = int(np.searchsorted(m, t + hz, side="left")) - 1
        res[f"fwd{hz}"] = (float(cl[max(k, j)]) - c) / atr
    res["close_vs"] = (float(cl[-1]) - c) / atr
    return res


def day_nodes(bars: pd.DataFrame, atr_pct: float, times=None) -> list[dict]:
    """Every node of one day (state + labels). ATR = atr_pct x the 9:30 open."""
    a = rth_arrays(bars)
    if a is None or not atr_pct:
        return []
    atr = float(atr_pct) * float(a["o"][0])
    out = []
    for t in (range(NODE_FROM, NODE_TO + 1, STEP) if times is None else times):
        s = state_at(a, atr, t)
        if s is None:
            continue
        out.append({**s, **labels_at(a, atr, s)})
    return out


# --------------------------------------------------------------------------- the table

class Comps:
    """The node table in arrays. Days: date, ticker, day type, kind. Nodes: day id, minute,
    the state features, outcome (+1 up / -1 low first / 0 neither), fwd15/30/60, close_vs."""

    def __init__(self, days: pd.DataFrame, nodes: dict[str, np.ndarray]):
        self.days = days.reset_index(drop=True)
        self.n = nodes
        self.day_date = self.days.date.to_numpy()
        self.day_type = self.days.day_type.map(DAY_TYPES.index).to_numpy()
        self.day_kind = self.days.kind.map(KINDS.index).to_numpy()
        dates = np.unique(self.day_date)
        self.day_date_code = np.searchsorted(dates, self.day_date)
        X = np.column_stack([self.n[f] for f in FEATS]).astype(float)
        self.X = _transform(X)
        self.sd = np.nanstd(self.X, axis=0)
        self.sd[self.sd == 0] = 1.0

    def save(self, path: Path = NODES_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f"n_{k}": v for k, v in self.n.items()},
                            d_date=self.days.date.to_numpy().astype(str), d_ticker=self.days.ticker.to_numpy().astype(str),
                            d_day_type=self.days.day_type.to_numpy().astype(str), d_kind=self.days.kind.to_numpy().astype(str),
                            d_atr_pct=self.days.atr_pct.to_numpy(dtype=float), d_src=self.days.src.to_numpy().astype(str))

    @classmethod
    def load(cls, path: Path = NODES_PATH) -> "Comps":
        z = np.load(path, allow_pickle=False)
        days = pd.DataFrame({k[2:]: z[k] for k in z.files if k.startswith("d_")})
        nodes = {k[2:]: z[k] for k in z.files if k.startswith("n_")}
        return cls(days, nodes)

    def subset(self, day_mask: np.ndarray) -> "Comps":
        keep_days = np.flatnonzero(day_mask)
        remap = -np.ones(len(self.days), dtype=np.int64)
        remap[keep_days] = np.arange(len(keep_days))
        nm = day_mask[self.n["day"]]
        nodes = {k: v[nm] for k, v in self.n.items()}
        nodes["day"] = remap[nodes["day"]].astype(np.int32)
        return Comps(self.days.iloc[keep_days], nodes)


def _transform(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    X[:, 0] = np.log1p(np.clip(X[:, 0], 0, 390))        # minutes since the low
    X[:, 1] = np.clip(X[:, 1], 0, 4)                      # bounce off the low, ATR
    X[:, 2] = np.clip(X[:, 2], -3, 3)                     # to VWAP, ATR
    X[:, 4] = np.clip(X[:, 4], 0, 6)                      # flush, ATR
    return X


def build_table(pop: pd.DataFrame, kinds: dict[str, str], types: dict[str, str], dirs: list[Path],
                fetch: int = 0) -> tuple[Comps, dict]:
    t0 = time.time()
    rows, node_cols = [], {k: [] for k in ("day", "t", *FEATS, "outcome", *[f"fwd{h}" for h in HORIZONS], "close_vs")}
    stats = {"population": len(pop), "no_atr": 0, "no_bars": 0, "fetched": 0, "no_nodes": 0}
    for r in pop.itertuples():
        if not r.atr_pct or r.atr_pct != r.atr_pct:
            stats["no_atr"] += 1
            continue
        bars = load_bars(r.ticker, r.date, dirs)
        if bars is None and stats["fetched"] < fetch:
            from bounce_entry_study.fetch_bars import fetch_day
            bars = fetch_day(r.ticker, r.date)
            stats["fetched"] += 1
        if bars is None:
            stats["no_bars"] += 1
            continue
        nodes = day_nodes(bars, r.atr_pct)
        if not nodes:
            stats["no_nodes"] += 1
            continue
        d = len(rows)
        rows.append({"date": r.date, "ticker": r.ticker, "day_type": types.get(r.date, "normal"),
                     "kind": kinds.get(r.ticker, "stock"), "atr_pct": float(r.atr_pct), "src": r.src})
        for n in nodes:
            node_cols["day"].append(d)
            node_cols["t"].append(n["t"])
            for f in FEATS:
                node_cols[f].append(n[f])
            node_cols["outcome"].append(OUTCOME[n["outcome"]])
            for h in HORIZONS:
                node_cols[f"fwd{h}"].append(n[f"fwd{h}"])
            node_cols["close_vs"].append(n["close_vs"])
    nodes = {"day": np.asarray(node_cols["day"], dtype=np.int32), "t": np.asarray(node_cols["t"], dtype=np.int16),
             "outcome": np.asarray(node_cols["outcome"], dtype=np.int8)}
    for k in (*FEATS, *[f"fwd{h}" for h in HORIZONS], "close_vs"):
        nodes[k] = np.asarray(node_cols[k], dtype=np.float32)
    stats.update(days=len(rows), nodes=len(nodes["t"]), seconds=round(time.time() - t0, 1))
    return Comps(pd.DataFrame(rows), nodes), stats


def build(dirs: list[Path] | None = None, population_csv: Path | None = POPULATION_CSV,
          fetch: int = 0, refresh_kinds: bool = False) -> dict:
    pop = population(population_csv)
    POP_PATH.parent.mkdir(parents=True, exist_ok=True)
    pop.to_csv(POP_PATH, index=False)
    comps, stats = build_table(pop, kinds_for(pop, refresh_kinds), day_types(), dirs or bar_dirs(), fetch)
    comps.save()
    stats["file_mb"] = round(NODES_PATH.stat().st_size / 1e6, 2)
    stats["kinds"] = comps.days.kind.value_counts().to_dict()
    stats["day_types"] = comps.days.day_type.value_counts().to_dict()
    return stats


# --------------------------------------------------------------------------- the match

def state_now(bars: pd.DataFrame, atr: float, t: int, day_type: str = "normal", kind: str = "stock") -> dict | None:
    """The live state of one name at minute `t` after 9:30 (bars = 1-min, indexed by start;
    atr in dollars)."""
    s = state_at(rth_arrays(bars), atr, t)
    if s is None:
        return None
    return {**s, "day_type": day_type, "kind": kind}


def match(state: dict, comps: Comps, k: int = K, clock_win: int = CLOCK_WIN, per_date: int = PER_DATE,
          before: str | None = None, exclude_date: str | None = None) -> dict | None:
    """The K nearest comp nodes to `state`: clock within +-clock_win, one node a comp day (its
    nearest), at most `per_date` days a date; their outcomes, odds and cone."""
    n = comps.n
    t = int(state["t"])
    mask = np.abs(n["t"].astype(int) - t) <= clock_win
    if before is not None or exclude_date is not None:
        okd = np.ones(len(comps.days), dtype=bool)
        if before is not None:
            okd &= comps.day_date < before
        if exclude_date is not None:
            okd &= comps.day_date != exclude_date
        mask &= okd[n["day"]]
    idx = np.flatnonzero(mask)
    if not len(idx):
        return None
    x = _transform(np.array([[state[f] for f in FEATS]], dtype=float))[0]
    w = np.array([WEIGHTS[f] for f in FEATS])
    z = (comps.X[idx] - x) / comps.sd
    d2 = (w * z * z).sum(axis=1)
    day = n["day"][idx]
    st_type = DAY_TYPES.index(state.get("day_type", "normal"))
    st_kind = KINDS.index(state.get("kind", "stock"))
    d2 = d2 + DAY_TYPE_W * np.abs(comps.day_type[day] - st_type) + KIND_W * (comps.day_kind[day] != st_kind)
    d2 = d2 + CLOCK_W * ((n["t"][idx].astype(int) - t) / max(clock_win, 1)) ** 2
    # one node a day: its nearest
    order = np.lexsort((d2, day))
    first = np.r_[True, day[order][1:] != day[order][:-1]]
    pick = order[first]
    pick = pick[np.argsort(d2[pick], kind="stable")]
    # at most per_date days a date
    dc = comps.day_date_code[day[pick]]
    seen: dict[int, int] = {}
    keep = []
    for i, c in zip(pick, dc):
        if seen.get(c, 0) < per_date:
            seen[c] = seen.get(c, 0) + 1
            keep.append(i)
            if len(keep) >= k:
                break
    sel = idx[np.asarray(keep, dtype=np.int64)]
    return summarize(comps, sel, state)


def summarize(comps: Comps, sel: np.ndarray, state: dict | None = None) -> dict:
    n = comps.n
    out = n["outcome"][sel]
    res = {"n": int(len(sel)), "dates": int(len(np.unique(comps.day_date_code[n["day"][sel]]))),
           "p_up": float(np.mean(out == 1)) if len(sel) else None,
           "p_low_first": float(np.mean(out == -1)) if len(sel) else None,
           "close_med": float(np.median(n["close_vs"][sel])) if len(sel) else None,
           "cone": [[h, *[float(v) for v in np.percentile(n[f"fwd{h}"][sel], [25, 50, 75])]] for h in HORIZONS]
           if len(sel) else []}
    if state is not None:
        res["state"] = {k: (round(float(state[k]), 3) if isinstance(state[k], (int, float, np.floating)) else state[k])
                        for k in ("t", "mins_since_low", "bounce_atr", "vwap_atr", "low_pre10", "flush_atr",
                                  "day_type", "kind") if k in state}
    res["enough"] = res["n"] >= MIN_COMPS
    return res


# --------------------------------------------------------------------------- the study

CHECKPOINTS = (30, 60, 90, 150)                   # 10:00, 10:30, 11:00, 12:00
SPLIT = "2025-01-01"


def _node_state(comps: Comps, i: int) -> dict:
    d = comps.n["day"][i]
    return {"t": int(comps.n["t"][i]), **{f: float(comps.n[f][i]) for f in FEATS},
            "day_type": comps.days.day_type.iat[d], "kind": comps.days.kind.iat[d]}


def _flagged(src: str) -> bool:
    """A name the scans flagged (GO/CAUTION, the book, the breadth count) vs a NO-GO control."""
    return src in ("ev:surfaced", "ev:book_missed", "scorecard:surfaced", "book", "breadth:strict", "breadth:relaxed")


def calib(pred: np.ndarray, act: np.ndarray, bins: int) -> tuple[list, float]:
    q = pd.qcut(pd.Series(pred).rank(method="first"), bins, labels=False)
    g = pd.DataFrame({"q": q, "p": pred, "a": act}).groupby("q").agg(p=("p", "mean"), a=("a", "mean"), n=("a", "size"))
    return g.reset_index().to_dict("records"), float((g.p - g.a).abs().mean())


def study(comps: Comps | None = None, split: str = SPLIT, boots: int = 500, seed: int = 7) -> dict:
    """Walk-forward: comps from days before `split`, scored on the days from it. Per
    checkpoint and pooled: the up-rate spread from the top to the bottom fifth of predicted
    p_up (bar >= 15 pts), the decile calibration error (bar <= 5 pts), Brier vs the
    clock-only base rate, and the same by day type."""
    comps = comps or Comps.load()
    train_days = comps.day_date < split
    tr = comps.subset(train_days)
    rows = []
    for t in CHECKPOINTS:
        base_up = float(np.mean(tr.n["outcome"][tr.n["t"] == t] == 1))
        test_idx = np.flatnonzero((comps.n["t"] == t) & ~train_days[comps.n["day"]])
        for i in test_idx:
            s = _node_state(comps, i)
            m = match(s, tr)
            if not m:
                continue
            d = comps.n["day"][i]
            rows.append({"t": t, "date": comps.day_date[d], "ticker": comps.days.ticker.iat[d],
                         "flagged": _flagged(comps.days.src.iat[d]),
                         "day_type": s["day_type"], "kind": s["kind"], "pred": m["p_up"], "pred_low": m["p_low_first"],
                         "base": base_up, "up": int(comps.n["outcome"][i] == 1),
                         "low": int(comps.n["outcome"][i] == -1), "close_vs": float(comps.n["close_vs"][i]),
                         "pred_close": m["close_med"], "n": m["n"], "dates": m["dates"],
                         **{f"cone{h}_{q}": c[i_q] for h, *c in m["cone"] for i_q, q in ((0, "p25"), (2, "p75"))},
                         **{f"fwd{h}": float(comps.n[f"fwd{h}"][i]) for h in HORIZONS}})
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)

    def gate(sub: pd.DataFrame) -> dict:
        if len(sub) < 50:
            return {"n": len(sub)}
        q = pd.qcut(sub.pred.rank(method="first"), 5, labels=False)
        top, bot = sub.up[q == 4].mean(), sub.up[q == 0].mean()
        lq = pd.qcut(sub.pred_low.rank(method="first"), 5, labels=False)
        cal, mae = calib(sub.pred.to_numpy(), sub.up.to_numpy(), 10)
        b_model = float(np.mean((sub.pred - sub.up) ** 2))
        b_base = float(np.mean((sub.base - sub.up) ** 2))
        # date-clustered bootstrap of the spread
        dates = sub.date.unique()
        by = {d: g for d, g in sub.assign(q=q).groupby("date")}
        spreads = []
        for _ in range(boots):
            s2 = pd.concat([by[d] for d in rng.choice(dates, len(dates))])
            t2, b2 = s2.up[s2.q == 4].mean(), s2.up[s2.q == 0].mean()
            if t2 == t2 and b2 == b2:
                spreads.append(t2 - b2)
        cvq = pd.qcut(sub.pred_close.rank(method="first"), 5, labels=False)
        cover = {h: round(float(np.mean((sub[f"fwd{h}"] >= sub[f"cone{h}_p25"]) & (sub[f"fwd{h}"] <= sub[f"cone{h}_p75"]))), 3)
                 for h in HORIZONS}
        return {"n": len(sub), "dates": len(dates), "up_rate": round(sub.up.mean(), 3),
                "spread_pts": round(100 * (top - bot), 1),
                "spread_ci90": [round(100 * float(np.percentile(spreads, 5)), 1),
                                round(100 * float(np.percentile(spreads, 95)), 1)] if spreads else None,
                "top_fifth_up": round(top, 3), "bottom_fifth_up": round(bot, 3),
                "low_first_spread_pts": round(100 * (sub.low[lq == 4].mean() - sub.low[lq == 0].mean()), 1),
                "close_spread_atr": round(float(sub.close_vs[cvq == 4].median() - sub.close_vs[cvq == 0].median()), 2),
                "calib_mae_pts": round(100 * mae, 1), "brier_model": round(b_model, 4), "brier_clock": round(b_base, 4),
                "brier_skill": round(1 - b_model / b_base, 3) if b_base else None,
                "cone_cover_p25_p75": cover,
                "pass": bool(100 * (top - bot) >= 15 and 100 * mae <= 5 and b_model < b_base),
                "deciles": [{"p": round(r["p"], 3), "a": round(r["a"], 3), "n": int(r["n"])} for r in cal]}

    rep = {"split": split, "train_days": int(train_days.sum()), "test_days": int((~train_days).sum()),
           "test_rows": len(df), "pooled": gate(df), "by_checkpoint": {}, "by_day_type": {}}
    for t in CHECKPOINTS:
        rep["by_checkpoint"][f"{9 + (30 + t) // 60}:{(30 + t) % 60:02d}"] = gate(df[df.t == t])
    rep["by_day_type"]["breadth (11+)"] = gate(df[df.day_type.isin(["11-15", "16+"])])
    rep["by_day_type"]["normal/elevated"] = gate(df[df.day_type.isin(["normal", "elevated"])])
    rep["by_cohort"] = {"flagged (GO/CAUTION, book)": gate(df[df.flagged]),
                        "controls (NO-GO)": gate(df[~df.flagged])}
    rep["_rows"] = df
    return rep


def readout(ticker: str, date: str, atr_pct: float, times=(30, 33, 60), comps: Comps | None = None,
            dirs: list[Path] | None = None, day_type: str | None = None, kind: str | None = None) -> list[dict]:
    """What the comps said for one past name-day at a few moments, from comps before that
    date only, next to what the name then did."""
    comps = comps or Comps.load()
    bars = load_bars(ticker, date, dirs or bar_dirs())
    if bars is None:
        from bounce_entry_study.fetch_bars import fetch_day
        bars = fetch_day(ticker, date)
    a = rth_arrays(bars)
    atr = float(atr_pct) * float(a["o"][0])
    day_type = day_type or day_types().get(date, "normal")
    if kind is None:
        hit = comps.days[comps.days.ticker == ticker.upper()]
        kind = hit.kind.iat[0] if len(hit) else "stock"
    out = []
    for t in times:
        s = state_at(a, atr, t)
        if s is None:
            continue
        lab = labels_at(a, atr, s)
        m = match({**s, "day_type": day_type, "kind": kind}, comps, before=date)
        out.append({"time": f"{9 + (30 + t) // 60}:{(30 + t) % 60:02d}", "close": round(s["close"], 2),
                    "lod": round(s["lod"], 2), "comps": m, "actual": lab})
    return out


def _print_study(rep: dict) -> None:
    def line(name, g):
        if "spread_pts" not in g:
            print(f"  {name:<18} n={g['n']} (too few)")
            return
        print(f"  {name:<18} n={g['n']:>5} dates={g['dates']:>4} up {g['up_rate']:.0%}  spread {g['spread_pts']:+.1f} pts "
              f"(90% {g['spread_ci90']})  calib {g['calib_mae_pts']:.1f} pts  Brier {g['brier_model']:.4f} vs clock "
              f"{g['brier_clock']:.4f} (skill {g['brier_skill']:+.3f})  low-first spread {g['low_first_spread_pts']:+.1f}  "
              f"close spread {g['close_spread_atr']:+.2f} ATR  cone cover {g['cone_cover_p25_p75']}  {'PASS' if g['pass'] else 'fail'}")
    print(f"train days {rep['train_days']} (< {rep['split']}), test days {rep['test_days']}, test rows {rep['test_rows']}")
    line("pooled", rep["pooled"])
    for k, g in rep["by_checkpoint"].items():
        line(k, g)
    for k, g in rep["by_day_type"].items():
        line(k, g)
    for k, g in rep.get("by_cohort", {}).items():
        line(k[:18], g)
    print("  deciles (pooled):", " ".join(f"{d['p']:.2f}/{d['a']:.2f}" for d in rep["pooled"].get("deciles", [])))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--study", action="store_true")
    ap.add_argument("--bars", action="append", help="another bar cache dir ({TICKER}_{date}.pkl)")
    ap.add_argument("--population", default=str(POPULATION_CSV))
    ap.add_argument("--fetch", type=int, default=0, help="fetch up to N missing days from Polygon")
    ap.add_argument("--refresh-kinds", action="store_true")
    ap.add_argument("--out", help="study: write the JSON report here")
    ap.add_argument("--split", default=SPLIT, help="study: comps from days before this date")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.build:
        print(json.dumps(build(bar_dirs(args.bars), Path(args.population), args.fetch, args.refresh_kinds), indent=1))
    if args.study:
        rep = study(split=args.split)
        _print_study(rep)
        if args.out:
            rows = rep.pop("_rows")
            Path(args.out).write_text(json.dumps(rep, indent=1, default=str))
            rows.to_csv(Path(args.out).with_suffix(".rows.csv"), index=False)


if __name__ == "__main__":
    main()
