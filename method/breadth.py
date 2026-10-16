"""
Market breadth for Method: how many names market-wide are B or better (4+/6) at the open.

The bounce criteria study (2026-10-16) found the day matters more than the name. On
days when 11+ names market-wide scored 4+/6 at the open, those names averaged +1.2R
under the live rules (2-min entry, 1.5-ATR trail); on other days about +0.1R. The
count here is the study's own, so its 11 means what the study measured:

  universe   the bounce backscanner's (scanners/bounce_backscanner.py): price > $5,
             20-day average volume >= 500k shares, no '.', '-' or '/' in the ticker,
             estimated cap Large/ETF/Medium/Small
  pre-filter 2+ straight down closes, a gap down, 5%+ off the 30-day high
  score      compute_bounce_metrics at the prior close, BouncePretrade.validate

Only the gap moves with the price, so each name has one price at or below which it
scores 4+/6 (its flag price). A nightly build lists those names for the next
session; a pass is then one all-tickers snapshot and a comparison per name. Names
under $5 or 500k shares that would flag with $9M+ a day of volume are listed too,
marked not counted (no price floor on what the trader sees).

On breadth days the study also relaxed the gap and 3-day thresholds by half: the
names that adds averaged +1.05R (on other days the same relaxation had no edge), so
relaxed names are listed on breadth days only, and never counted.

Polygon's grouped daily (one call a day for every ticker) is kept per day in
data/method/grouped/, the last STORE_DAYS sessions. A split after a day was stored
leaves it unadjusted, so names that split inside the window get their split-adjusted
history from Polygon per name instead.

Usage:
    python -m method.breadth --build                 # store + candidates for the next session
    python -m method.breadth --build --session 2026-10-19
    python -m method.breadth --now                   # count at the current snapshot
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd

from method import core

logger = logging.getLogger("method.breadth")

TZ = "US/Eastern"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
METHOD_DIR = Path(os.environ.get("METHOD_DIR") or DATA_DIR / "method")
STORE = METHOD_DIR / "grouped"
EDGE_FILE = DATA_DIR / "bounce_day_edge.json"
STORE_DAYS = 310                  # the backscanner's history window (52-week high, 200-day SMA)

MIN_PRICE = 5.0                   # study universe (bounce_backscanner.MIN_PRICE)
MIN_AVG_VOLUME = 500_000          # study universe (bounce_backscanner.MIN_AVG_VOLUME)
LIST_MIN_DOLLAR_ADV = 9_000_000   # listed but not counted (feedback: no price floor, $9M ADV ok)
COUNT_CAPS = ("Large", "ETF", "Medium", "Small")   # the population's --cap
FLAG_SCORE = 4
BREADTH_DAY = 11                  # the study's trigger
ELEVATED = 6
RELAX = 0.5                       # gap + 3-day thresholds on breadth days
NEAR = 0.05                       # candidates this close to flagging get live prices from the PC
NEAR_MAX = 40


def state_for(n: int | None) -> str | None:
    if n is None:
        return None
    return "breadth" if n >= BREADTH_DAY else "elevated" if n >= ELEVATED else "normal"


# ---------------------------------------------------------------------------
# Store: one grouped-daily frame per session
# ---------------------------------------------------------------------------

def _client():
    from polygon import RESTClient
    return RESTClient(api_key=os.environ["POLYGON_API_KEY"])


def trading_days(start: datetime.date, end: datetime.date) -> list[str]:
    import pandas_market_calendars as mcal
    return [d.strftime("%Y-%m-%d") for d in mcal.get_calendar("NYSE").valid_days(start_date=start, end_date=end)]


def window(session: str, n: int = STORE_DAYS) -> list[str]:
    """The n sessions before `session`."""
    s = datetime.date.fromisoformat(session)
    days = [d for d in trading_days(s - datetime.timedelta(days=int(n * 1.6) + 10), s) if d < session]
    return days[-n:]


def fetch_grouped(day: str, client=None) -> pd.DataFrame:
    client = client or _client()
    aggs = client.get_grouped_daily_aggs(day) or []
    return pd.DataFrame([{"ticker": a.ticker, "open": a.open, "high": a.high, "low": a.low,
                          "close": a.close, "volume": a.volume} for a in aggs])


def update_store(days: list[str], fetch=fetch_grouped, store: Path = STORE) -> list[str]:
    """Fetch the days not stored yet; drop files older than the window. Returns the days
    still missing (Polygon had nothing, e.g. today's bar before it's published)."""
    store.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in store.glob("*.pkl")}
    missing = []
    for d in days:
        if d in have:
            continue
        try:
            df = fetch(d)
        except Exception as e:
            logger.warning(f"grouped daily {d}: {e}")
            df = None
        if df is None or df.empty:
            missing.append(d)
            continue
        df.to_pickle(store / f"{d}.pkl")
        time.sleep(0.12)
    for p in store.glob("*.pkl"):
        if days and p.stem < days[0]:
            p.unlink()
    return missing


def load_store(days: list[str], store: Path = STORE) -> pd.DataFrame:
    """Long frame (date, ticker, open, high, low, close, volume) for the stored days."""
    frames = []
    for d in days:
        p = store / f"{d}.pkl"
        if p.exists():
            frames.append(pd.read_pickle(p).assign(date=d))
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Candidates: who could flag at the next open, and at what price
# ---------------------------------------------------------------------------

def _thresholds(setup: str, cap: str) -> dict:
    from analyzers.bounce_scorer import SETUP_PROFILES
    prof = SETUP_PROFILES[setup]
    return {c: prof.get_threshold(c, cap) for c in ("gap_pct", "pct_change_3")}


_CHECKER = None


def _checker():
    global _CHECKER
    if _CHECKER is None:
        from analyzers.bounce_scorer import BouncePretrade
        _CHECKER = BouncePretrade()
    return _CHECKER


def prefilter(long: pd.DataFrame) -> list[str]:
    """Cheap pass over every ticker: 2+ straight down closes, 5%+ off the 30-day high,
    and enough volume for either list. compute_candidate() redoes these exactly."""
    if long.empty:
        return []
    df = long[~long["ticker"].str.contains(r"[.\-/]", regex=True, na=False)]
    df = df.sort_values(["ticker", "date"], kind="stable")
    g = df.groupby("ticker", sort=False)
    n = g.size()
    c = [g.nth(-k).set_index("ticker")["close"].reindex(n.index) for k in (1, 2, 3)]
    down2 = (c[0] < c[1]) & (c[1] < c[2])            # NaN (too short) compares False
    t30, t20 = g.tail(30), g.tail(20)
    hi30 = t30.groupby("ticker")["high"].max().reindex(n.index)
    adv = t20.groupby("ticker")["volume"].mean().reindex(n.index)
    dadv = (t20["close"] * t20["volume"]).groupby(t20["ticker"]).mean().reindex(n.index)
    ok = (down2 & (c[0] / hi30 - 1 <= -0.05) & (n >= 14)
          & ((adv >= MIN_AVG_VOLUME) | (dadv >= LIST_MIN_DOLLAR_ADV)))
    return sorted(ok[ok.fillna(False).astype(bool)].index)


def compute_candidate(ticker: str, session: str, hist: pd.DataFrame) -> dict | None:
    """One name, exactly as the backscanner would score it on `session`, gap aside.
    `hist` = completed daily bars before the session, oldest first, date-string index."""
    from scanners.bounce_backscanner import BounceBackscanner

    hist = hist.iloc[-(STORE_DAYS - 1):]
    if len(hist) < 14:
        return None
    pc = float(hist["close"].iloc[-1])
    adv = float(hist["volume"].iloc[-min(20, len(hist)):].mean())
    counted_universe = pc > MIN_PRICE and adv >= MIN_AVG_VOLUME
    if not counted_universe and pc * adv < LIST_MIN_DOLLAR_ADV:
        return None
    c = hist["close"].tolist()
    down = 0
    for i in range(len(c) - 1, 0, -1):
        if c[i] < c[i - 1]:
            down += 1
        else:
            break
    hi30 = float(hist["high"].iloc[-30:].max())
    if down < 2 or hi30 <= 0 or pc / hi30 - 1 > -0.05:
        return None

    # The backscanner's history ends with the scan day's bar; a flat placeholder at the
    # prior close gives the same metrics with a zero gap (the gap is the price's job).
    today = pd.DataFrame([{"open": pc, "high": pc, "low": pc, "close": pc, "volume": 0.0}], index=[session])
    m = BounceBackscanner.compute_bounce_metrics(None, ticker, session, pd.concat([hist, today]))
    if m is None:
        return None
    cap = BounceBackscanner.estimate_cap(ticker, m.get("current_price", 0), m.get("avg_daily_vol", 0))
    if cap not in COUNT_CAPS:
        counted_universe = False
        if pc * adv < LIST_MIN_DOLLAR_ADV:
            return None
    m["gap_pct"] = 0.0
    res = _checker().validate(ticker=ticker, metrics=m, cap=cap)
    passed = {i.name: i.passed for i in res.items}
    th = _thresholds(res.setup_type, cap)
    base = sum(v for k, v in passed.items() if k != "gap_pct")
    p3 = m.get("pct_change_3")
    relaxed3 = p3 is not None and p3 <= th["pct_change_3"] * RELAX
    base_relaxed = base - int(passed.get("pct_change_3", False)) + int(relaxed3)

    def flag_price(n_static: int, gap_th: float) -> float | None:
        if n_static >= FLAG_SCORE:
            return pc                      # any gap down (strictly below the close)
        if n_static == FLAG_SCORE - 1:
            return pc * (1 + min(gap_th, 0.0))
        return None

    strict = flag_price(base, th["gap_pct"])
    relaxed = flag_price(base_relaxed, th["gap_pct"] * RELAX)
    if strict is None and relaxed is None:
        return None
    return {
        "ticker": ticker, "prior_close": pc, "cap": cap, "setup": res.setup_type,
        "counted": bool(counted_universe), "static_score": int(base), "static_relaxed": int(base_relaxed),
        "gap_th": float(th["gap_pct"]), "flag_price": strict, "relaxed_price": relaxed,
        "failing": [i.description or i.name for i in res.items if not i.passed and i.name != "gap_pct"],
        "adv_dollars": pc * adv,
    }


def ticker_hists(long: pd.DataFrame, tickers) -> dict:
    """{ticker: daily bars, oldest first, date-string index} for the given names."""
    sub = long[long["ticker"].isin(set(tickers))].sort_values(["ticker", "date"], kind="stable")
    return {t: h.set_index("date")[["open", "high", "low", "close", "volume"]]
            for t, h in sub.groupby("ticker", sort=False)}


def split_tickers(since: str, client=None) -> set[str]:
    """Names with a split executed after `since` (their stored days predate the adjustment)."""
    client = client or _client()
    out = set()
    try:
        for s in client.list_splits(execution_date_gte=since, limit=1000):
            if s.ticker:
                out.add(s.ticker)
    except Exception as e:
        logger.warning(f"splits since {since}: {e}")
    return out


ETF_TYPES = ("ETF", "ETN", "ETV", "ETS")


def etf_tickers(client=None) -> set[str]:
    """Polygon's exchange-traded products (a few calls). Leveraged single-stock ETFs were 13%
    of flagged names in 2022 and 51% in 2026, so the card says how many of the count are ETFs."""
    client = client or _client()
    out = set()
    for typ in ETF_TYPES:
        try:
            for t in client.list_tickers(market="stocks", type=typ, active=True, limit=1000):
                out.add(t.ticker)
        except Exception as e:
            logger.warning(f"ticker types {typ}: {e}")
    return out


def adjusted_hist(ticker: str, session: str) -> pd.DataFrame | None:
    """Split-adjusted completed daily bars from Polygon, per name."""
    from scanners.premarket_bounce_scanner import completed_daily
    h = completed_daily(ticker, session)
    if h is None or h.empty:
        return None
    h = h.copy()
    h.index = [pd.Timestamp(i).strftime("%Y-%m-%d") for i in h.index]
    return h[["open", "high", "low", "close", "volume"]]


def build_candidates(session: str, long: pd.DataFrame, split_names: set[str] | None = None,
                     adjusted=adjusted_hist) -> dict:
    """{ticker: candidate} for `session` from the stored days before it."""
    long = long[long["date"] < session]
    names = prefilter(long)
    split_names = split_names or set()
    # a split inside the window distorts the stored days, so the cheap pass can't judge
    # those names: re-check every liquid one on adjusted bars
    last_day = long["date"].max() if not long.empty else None
    if split_names and last_day:
        recent = long[(long["date"] == last_day) & long["ticker"].isin(split_names)]
        liquid = recent[recent["close"] * recent["volume"] >= LIST_MIN_DOLLAR_ADV / 3]["ticker"]
        names = sorted(set(names) | set(liquid))
    hists = ticker_hists(long, names)
    out = {}
    for t in names:
        h = None
        if t in split_names:
            try:
                h = adjusted(t, session)
            except Exception as e:
                logger.warning(f"{t}: adjusted history failed: {e}")
        if h is None:
            h = hists.get(t)
        if h is None:
            continue
        try:
            c = compute_candidate(t, session, h)
        except Exception as e:
            logger.warning(f"{t}: candidate failed: {e}")
            continue
        if c:
            out[t] = c
    return out


def candidates_path(session: str) -> Path:
    return METHOD_DIR / f"breadth_candidates_{session}.json"


def load_candidates(session: str) -> dict | None:
    return core.read_json(candidates_path(session))


def build(session: str, force: bool = False) -> dict | None:
    """Store + candidates for `session`. Skips when built with the full window."""
    path = candidates_path(session)
    have = core.read_json(path)
    if have and not force and not have.get("missing_days"):
        return have
    days = window(session)
    missing = update_store(days)
    long = load_store(days)
    t0 = time.time()
    splits = split_tickers(days[0]) if days else set()
    cands = build_candidates(session, long, splits)
    etfs = etf_tickers()
    for t, c in cands.items():
        c["etf"] = t in etfs
    out = {"session": session, "built_at": pd.Timestamp.now(tz=TZ).isoformat(timespec="seconds"),
           "days": len(days) - len(missing), "missing_days": missing,
           "last_day": max((d for d in days if d not in missing), default=None),
           "splits": len(splits), "candidates": cands}
    core.write_json(path, out)
    for p in METHOD_DIR.glob("breadth_candidates_*.json"):     # keep a week
        if p.stem.rsplit("_", 1)[-1] < (datetime.date.fromisoformat(session) - datetime.timedelta(days=7)).isoformat():
            p.unlink()
    logger.info(f"breadth candidates for {session}: {len(cands)} names "
                f"({sum(c['counted'] for c in cands.values())} counted universe) from {out['days']} days, "
                f"{len(splits)} splits, missing {missing or 'none'}, {time.time() - t0:.0f} s")
    return out


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

def flag_of(c: dict, price: float | None) -> str | None:
    """'strict' when the price makes the name 4+/6, 'relaxed' when only the relaxed rules do."""
    if price is None or not price > 0 or not price < c["prior_close"]:
        return None
    if c.get("flag_price") is not None and price <= c["flag_price"]:
        return "strict"
    if c.get("relaxed_price") is not None and price <= c["relaxed_price"]:
        return "relaxed"
    return None


def count(cands: dict, prices: dict) -> dict:
    """prices = {ticker: price}. The breadth number counts strict flags in the study's
    universe; `names` lists every strict flag, `relaxed` the relaxed-only ones, `near`
    the closest names still above their flag price."""
    flagged, relaxed, near = [], [], []
    for t, c in cands.items():
        p = prices.get(t)
        f = flag_of(c, p)
        if f == "strict":
            flagged.append(t)
        elif f == "relaxed":
            relaxed.append(t)
        if f != "strict" and c.get("flag_price") is not None and p:
            near.append((p / c["flag_price"] - 1, t))
    n = sum(cands[t]["counted"] for t in flagged)
    n_etf = sum(bool(cands[t]["counted"] and cands[t].get("etf")) for t in flagged)
    near.sort()
    return {"n": n, "n_etf": n_etf, "state": state_for(n), "names": flagged, "relaxed": relaxed,
            "near": [{"ticker": t, "away": a, "flag_price": cands[t]["flag_price"]}
                     for a, t in near if a <= NEAR][:NEAR_MAX],
            "priced": sum(t in prices for t in cands), "candidates": len(cands)}


# ---------------------------------------------------------------------------
# Prices: Polygon's all-tickers snapshot (15-min delayed), Trillium where fresh
# ---------------------------------------------------------------------------

def snapshot_prices(day: str, client=None) -> dict:
    """{ticker: {"price", "ts", "open"}} from one all-tickers call. `open` is the official
    9:30 open, set once the delayed minute bar is in the session (day.open equals the
    9:30 bar's open, checked 2026-10-16)."""
    client = client or _client()
    open_ms = pd.Timestamp(f"{day} 09:30", tz=TZ).value // 1_000_000
    day_ms = pd.Timestamp(f"{day} 04:00", tz=TZ).value // 1_000_000
    out = {}
    for s in client.get_snapshot_all("stocks"):
        lt, mn, d = getattr(s, "last_trade", None), getattr(s, "min", None), getattr(s, "day", None)
        price = ts = None
        if lt is not None and lt.price and lt.sip_timestamp and lt.sip_timestamp // 1_000_000 >= day_ms:
            price, ts = float(lt.price), lt.sip_timestamp // 1_000_000
        elif mn is not None and mn.close and mn.timestamp and mn.timestamp >= day_ms:
            price, ts = float(mn.close), mn.timestamp
        if price is None:
            continue
        rec = {"price": price, "ts": ts}
        if d is not None and d.open and mn is not None and mn.timestamp and mn.timestamp >= open_ms:
            rec["open"] = float(d.open)
        out[s.ticker] = rec
    return out


def edge_table() -> dict:
    return core.read_json(EDGE_FILE, {}) or {}


def edge_for(table: dict, state: str | None, score: int | None, relaxed: bool = False) -> dict | None:
    """The replayed edge for a name: its day's state x its score (relaxed: the relaxed group)."""
    if not table or not state:
        return None
    st = (table.get("states") or {}).get(state)
    if not st:
        return None
    if relaxed:
        return (table.get("relaxed") or {}).get(state)
    if score is None:
        return st
    key = "6" if score >= 6 else str(score) if score >= FLAG_SCORE else "low"
    return (st.get("by_score") or {}).get(key) or st


def evaluate(session: str, mode: str, now: datetime.datetime, feed: dict | None,
             prev: dict | None = None, snap_prices: dict | None = None) -> dict:
    """The market_breadth block for a snapshot. `feed` = fresh Trillium summaries (they win);
    `snap_prices` = snapshot_prices() for this pass, or None to reuse the last pass's
    Polygon prices (the server's re-count on a PC post)."""
    built = load_candidates(session)
    if not built:
        return {"session": session, "status": "not built"}
    cands = built["candidates"]
    feed = feed or {}
    prev_poly = (prev or {}).get("poly") if (prev or {}).get("session") == session else None
    poly = {}
    if snap_prices is not None:
        for t in cands:
            if t in snap_prices:
                poly[t] = snap_prices[t]
    else:
        poly = dict(prev_poly or {})
    at_open = mode == "market"
    prices, src, opened = {}, {}, 0
    for t in cands:
        b, p = feed.get(t), poly.get(t)
        o = ((b or {}).get("open") or (p or {}).get("open")) if at_open else None
        if o:                                  # the session's open, once a feed has it
            prices[t], src[t] = o, "T" if (b or {}).get("open") else "P"
            opened += 1
        elif b and b.get("price"):
            prices[t], src[t] = b["price"], "T"
        elif p:
            prices[t], src[t] = p["price"], "P"
    res = count(cands, prices)
    table = edge_table()
    locked = at_open and opened > 0 and opened >= 0.9 * len(prices)
    res.update({
        "session": session, "status": "ok", "mode": mode,
        "basis": "open" if locked else "now",
        "locked": locked,
        "asof": now.isoformat(timespec="seconds"),
        "trillium": sum(v == "T" for v in src.values()),
        "built_at": built.get("built_at"), "last_day": built.get("last_day"),
        "edge": edge_for(table, res["state"], None),
        "poly": poly,
        "detail": {t: {**{k: cands[t].get(k) for k in ("prior_close", "cap", "setup", "counted", "etf",
                                                          "flag_price", "relaxed_price", "failing")},
                       "price": prices.get(t), "src": src.get(t)}
                   for t in res["names"] + res["relaxed"] + [x["ticker"] for x in res["near"]]},
    })
    done = core.read_json(METHOD_DIR / f"breadth_push_{session}.json")
    if done:
        res["pushed"] = done.get("at")
    if at_open and prev and prev.get("session") == session and prev.get("locked") and prev.get("basis") == "open":
        keep = {k: prev[k] for k in ("n", "n_etf", "state", "names", "relaxed", "edge", "detail") if k in prev}
        res.update(keep)                       # the count at the open stays fixed for the day
        res["locked"] = True
    return res


def board_names(mb: dict | None) -> list[str]:
    """Names to add to the bounce board: every strict flag, and relaxed flags on breadth days."""
    if not mb or mb.get("status") != "ok":
        return []
    names = list(mb.get("names") or [])
    if mb.get("state") == "breadth":
        names += list(mb.get("relaxed") or [])
    return names


def watch_names(mb: dict | None) -> list[str]:
    """Candidates the PC should price in real time: near names and the flagged ones."""
    if not mb or mb.get("status") != "ok" or mb.get("locked"):
        return []
    return list(dict.fromkeys(board_names(mb) + [x["ticker"] for x in mb.get("near") or []]))


def breadth_push(mb: dict, send, state_path: Path | None = None) -> bool:
    """One push a session, the first time the count reaches BREADTH_DAY. The cron pass and
    the server's live re-count can both get here, so the record is its own file, written
    only when a phone got the push. Returns True when sent now."""
    if mb.get("status") != "ok" or (mb.get("n") or 0) < BREADTH_DAY:
        return False
    state_path = state_path or METHOD_DIR / f"breadth_push_{mb['session']}.json"
    done = core.read_json(state_path)
    if done:
        mb["pushed"] = done.get("at")
        return False
    e = mb.get("edge") or {}
    top = ", ".join(mb["names"][:6])
    when = "at the open" if mb.get("basis") == "open" else "premarket"
    etfs = f" ({mb['n_etf']} ETFs)" if mb.get("n_etf") else ""
    body = (f"{mb['n']} names B or better {when}{etfs}: {top}."
            + (f" Days like this: {e['mean_r']:+.1f}R a name, {e['win'] * 100:.0f}% win."
               if e.get("mean_r") is not None else ""))
    try:
        n = send(f"Breadth day · {mb['n']} names", body, url="/#/bounce", tag=f"breadth-{mb['session']}")
    except Exception as e:
        logger.warning(f"breadth push failed: {e}")
        n = 0
    if not n:
        return False
    mb["pushed"] = mb.get("asof")
    core.write_json(state_path, {"at": mb["pushed"], "n": mb["n"], "basis": mb.get("basis"), "names": mb["names"]})
    return True


def public(mb: dict | None) -> dict | None:
    """The block without the per-name Polygon prices (kept server-side for re-counts)."""
    if not mb:
        return mb
    return {k: v for k, v in mb.items() if k != "poly"}


# ---------------------------------------------------------------------------

def next_session(now: datetime.datetime) -> str:
    """Before 9:30 on a session day: today. Otherwise the next session."""
    from scripts.evening_signal_log import next_trading_day
    today = now.date()
    days = trading_days(today, today)
    if days and now.time() < datetime.time(9, 30):
        return today.isoformat()
    return next_trading_day(today).isoformat()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(description="Method market breadth")
    ap.add_argument("--build", action="store_true", help="update the store and build candidates")
    ap.add_argument("--session", help="YYYY-MM-DD (default: the next open)")
    ap.add_argument("--force", action="store_true", help="rebuild even if built")
    ap.add_argument("--now", action="store_true", help="print the count at the current snapshot")
    args = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(DATA_DIR.parent / ".env")
    now = pd.Timestamp.now(tz=TZ).to_pydatetime()
    session = args.session or next_session(now)
    if args.build:
        build(session, force=args.force)
    if args.now:
        built = load_candidates(session) or build(session)
        res = count(built["candidates"], {t: v["price"] for t, v in snapshot_prices(now.date().isoformat()).items()})
        print(json.dumps({k: res[k] for k in ("n", "state", "names", "relaxed", "near", "priced", "candidates")},
                         indent=1, default=str))


if __name__ == "__main__":
    main()
