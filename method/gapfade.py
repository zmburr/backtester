"""
The 3-day gap fade: Method's check for the one price-action setup the user trades
(11/09/2026: "focus on gap fades ... needs 2 down days plus a 3rd gap down day to
trigger, and same for shorts, then focus on news").

  long   2+ straight down closes (each close below the one before, no up day allowed),
         then the session opens below the prior close
  short  2+ straight up closes, then the session opens above the prior close

Any gap triggers; the grade at the open sets the size (the bounce ladder's A+/A/B/C, the
reversal open's A at +4%). The bounce scorer's consecutive_down_days forgives one up day
(SOXL 11/05/26 reads 5 there and 1 here: 11/03 closed +7.3%), so this module counts its own.

The night before, the card lists every name with the structure in and the prices that
matter: the trigger (any gap past the prior close), B and A. At the open each name is
ON (the structure plus a gap the right way), NO GAP (it opened the other way) or OFF
(no structure). Bounce and reversal rows carry the same status, and the breadth-day entry
alerts say OFF-PLAYBOOK on a name that isn't ON (method/watch.py).

Universe: the long side is the market breadth build's candidates (method/breadth.py: 2+
strict down closes, 5% off the 30-day high, B or better on some gap) plus watchlist and
pinned names with the structure and a B on the ladder. The short side is the market-wide
up-side build (breadth.build_up: 2+ strict up closes, 10%+ over the 9EMA) plus reversal
candidates with the structure and 5%+ over the 9EMA.

Pushes, off until METHOD_DIR/gapfade_push.on exists (or GAPFADE_PUSH=1), so the volume can
be measured first (--replay):
  evening  one a session, the names that can reach A on a gap within 1 ATR and are liquid
           ($200M+ a day) or yours (watchlist, pinned, the priority report)
  open     one a name, when such a name opens ON with a B or better

Usage:
    python -m method.gapfade --replay 2026-05-01 2026-11-06     # push volume over the store
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
from pathlib import Path

from method import core

logger = logging.getLogger("method.gapfade")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MIN_STREAK = 2
READY_ADV = 200e6              # liquid enough to push for without being on your lists
A_WITHIN_ATR = 1.0             # A reachable on a gap of at most this many ATRs
STATICS_MAX = 30               # liquid long candidates given statics (ladder A price) a session
SHORT_ROWS_MAX = 15            # market short names added to the reversal pass
SHORT_ROWS_ADV = 50e6
SHORT_EXT_WATCH = 0.05         # your names: 5%+ over the 9EMA to be listed short
EVENING_PUSH_BY = datetime.time(19, 30)    # push even if a build is missing by then
RANK = {"-": 0, "C": 1, "B": 2, "A": 3, "A+": 4}
YOURS = ("watchlist", "pin", "report")


def _method_dir() -> Path:
    return Path(os.environ.get("METHOD_DIR") or DATA_DIR / "method")


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

def streaks(closes) -> tuple[int, int]:
    """(down, up): straight closes below / above the one before, ending at the last close."""
    c = [float(x) for x in closes if x is not None]
    down = up = 0
    for i in range(len(c) - 1, 0, -1):
        if c[i] < c[i - 1]:
            down += 1
        else:
            break
    for i in range(len(c) - 1, 0, -1):
        if c[i] > c[i - 1]:
            up += 1
        else:
            break
    return down, up


def status(side: str, streak: int | None, prior_close: float | None, open_: float | None = None) -> str:
    """'setting_up' (structure in, no open yet), 'on', 'no_gap', 'off', or 'unknown'."""
    if streak is None or not prior_close:
        return "unknown"
    if streak < MIN_STREAK:
        return "off"
    if open_ is None:
        return "setting_up"
    faded = open_ < prior_close if side == "long" else open_ > prior_close
    return "on" if faded else "no_gap"


def would_trigger(side: str, prior_close: float | None, price: float | None) -> bool | None:
    if not prior_close or price is None:
        return None
    return price < prior_close if side == "long" else price > prior_close


def ladder_prices(static: dict) -> tuple[float | None, float | None]:
    """(B price, A price) on the long side: the highest price at which a gap down grades B or
    better / A or better, capped at the prior close (any gap down = that grade)."""
    prior = static.get("prior_close")
    zones = ((static.get("ladder") or {}).get("zones")) or []
    if not prior:
        return None, None

    def top(min_rank):
        tops = [z["top"] for z in zones if RANK.get(z.get("grade"), 0) >= min_rank and z.get("top")]
        return min(max(tops), prior) if tops else None
    return top(RANK["B"]), top(RANK["A"])


# ---------------------------------------------------------------------------
# The lists
# ---------------------------------------------------------------------------

def _src(t: str, watchlist, pinned, report) -> str:
    return "pin" if t in pinned else "watchlist" if t in watchlist else "report" if t in report else "market"


def _ready_long(item: dict) -> bool:
    liquid = (item.get("adv") or 0) >= READY_ADV or item["src"] in YOURS
    a, atr, prior = item.get("a_price"), item.get("atr"), item["prior_close"]
    return bool(liquid and a is not None and atr and prior - a <= A_WITHIN_ATR * atr)


def _ready_short(item: dict) -> bool:
    liquid = (item.get("adv") or 0) >= READY_ADV or item["src"] in YOURS
    atr = item.get("atr")
    return bool(liquid and atr and item["a_price"] - item["prior_close"] <= A_WITHIN_ATR * atr)


def long_items(cands: dict, statics: dict, watchlist, pinned, report=()) -> list[dict]:
    """Every long name setting up for the session, ranked: ready first, then the nearest A."""
    watchlist, pinned, report = set(watchlist), set(pinned), set(report)
    out = {}
    for t, c in (cands or {}).items():
        if c.get("flag_price") is None:          # B only under the relaxed rules: breadth days only
            continue
        out[t] = {"ticker": t, "side": "long", "streak": int(c.get("down") or MIN_STREAK),
                  "prior_close": float(c["prior_close"]), "b_price": float(c["flag_price"]),
                  "a_price": None, "atr": None, "adv": c.get("adv_dollars"), "etf": bool(c.get("etf")),
                  "cap": c.get("cap"), "src": _src(t, watchlist, pinned, report)}
    for t, s in (statics or {}).items():
        down = s.get("down_streak")
        prior = s.get("prior_close")
        if down is None or down < MIN_STREAK or not prior:
            continue
        b, a = ladder_prices(s)
        if t not in out and (b is None or (t not in watchlist and t not in pinned and t not in report)):
            continue                               # no B on any gap, or a name only here for its A price
        it = out.get(t) or {"ticker": t, "side": "long", "adv": None, "etf": False, "cap": None,
                            "src": _src(t, watchlist, pinned, report)}
        it.update(streak=int(down), prior_close=float(prior), atr=s.get("atr"))
        it["b_price"] = b if b is not None else it.get("b_price")
        it["a_price"] = a
        out[t] = it
    items = list(out.values())
    for it in items:
        it["a_ready"] = _ready_long(it)
    return rank(items)


def short_items(cands: dict, statics: dict, rows: list[dict], streak_of, watchlist, pinned,
                report=()) -> list[dict]:
    """Every short name setting up: the market up-side build plus your reversal names with the
    structure. `streak_of(ticker)` = (down, up) for a reversal row with no statics."""
    watchlist, pinned, report = set(watchlist), set(pinned), set(report)
    out = {}
    for t, c in (cands or {}).items():
        out[t] = {"ticker": t, "side": "short", "streak": int(c.get("up") or MIN_STREAK),
                  "prior_close": float(c["prior_close"]), "ext9": c.get("ext9"),
                  "atr": c.get("atr"), "adv": c.get("adv_dollars"), "etf": bool(c.get("etf")),
                  "src": _src(t, watchlist, pinned, report)}
    for r in rows or []:
        t = r["ticker"]
        s = (statics or {}).get(t) or {}
        up = s.get("up_streak")
        if up is None:
            up = (streak_of(t) or (None, None))[1] if streak_of else None
        prior = r.get("prior_close") or s.get("prior_close")
        ext = r.get("pct_from_9ema") if r.get("pct_from_9ema") is not None else s.get("pct_from_9ema")
        if up is None or up < MIN_STREAK or not prior:
            continue
        if t not in out and (ext is None or ext < SHORT_EXT_WATCH):
            continue
        it = out.get(t) or {"ticker": t, "side": "short", "adv": None, "etf": False,
                            "src": _src(t, watchlist, pinned, report)}
        atr = s.get("atr") or ((r.get("atr_pct") or 0) * (r.get("price") or prior) or None)
        it.update(streak=int(up), prior_close=float(prior), ext9=ext, atr=it.get("atr") or atr)
        out[t] = it
    items = list(out.values())
    for it in items:
        it["b_price"] = it["prior_close"]                    # flat or small gap: B (VWAP fail)
        it["a_price"] = it["prior_close"] * (1 + core.REVERSAL_GAP_GATE)
        it["a_ready"] = _ready_short(it)
    return rank(items)


def rank(items: list[dict]) -> list[dict]:
    def key(it):
        atr, a = it.get("atr"), it.get("a_price")
        to_a = abs(it["prior_close"] - a) / atr if atr and a is not None else 9e9
        return (not it.get("a_ready"), not (it["src"] in YOURS or (it.get("adv") or 0) >= READY_ADV),
                to_a, -(it.get("adv") or 0))
    return sorted(items, key=key)


def statics_names(cands: dict, skip=()) -> list[str]:
    """Liquid long candidates that need statics for their A price (the ladder), biggest first."""
    skip = set(skip)
    liquid = [(c.get("adv_dollars") or 0, t) for t, c in (cands or {}).items()
              if t not in skip and c.get("flag_price") is not None and (c.get("adv_dollars") or 0) >= READY_ADV]
    return [t for _, t in sorted(liquid, reverse=True)[:STATICS_MAX]]


def short_row_names(cands: dict, skip=()) -> list[str]:
    """Market short names worth a reversal score (and live prices), biggest first."""
    skip = set(skip)
    liquid = [(c.get("adv_dollars") or 0, t) for t, c in (cands or {}).items()
              if t not in skip and (c.get("adv_dollars") or 0) >= SHORT_ROWS_ADV]
    return [t for _, t in sorted(liquid, reverse=True)[:SHORT_ROWS_MAX]]


# ---------------------------------------------------------------------------
# Prices, opens and status
# ---------------------------------------------------------------------------

def annotate(block: dict, mode: str, bounce_rows: list[dict], reversal_rows: list[dict],
             feed: dict | None = None, poly: dict | None = None, prev: dict | None = None,
             grade_of=None) -> dict:
    """Each item's live price, its open once the session has one, and its status.
    Opens come from the rows (Trillium or Polygon minute bars), the PC feed, Polygon's
    all-tickers snapshot (`poly`), else the last pass's block for the same session.
    `grade_of(item, open)` = the long side's grade at the open (statics' ladder)."""
    feed, poly = feed or {}, poly or {}
    by_side = {"long": {r["ticker"]: r for r in bounce_rows or []},
               "short": {r["ticker"]: r for r in reversal_rows or []}}
    carried = {}
    if prev and prev.get("session") == block.get("session"):
        for side in ("long", "short"):
            for it in prev.get(side) or []:
                if it.get("open") is not None:
                    carried[(side, it["ticker"])] = it
    for side in ("long", "short"):
        for it in block.get(side) or []:
            t = it["ticker"]
            row, b, p = by_side[side].get(t) or {}, feed.get(t) or {}, poly.get(t) or {}
            price = row.get("price") or b.get("price") or p.get("price")
            if price is not None:
                it["price"] = float(price)
                it["gap"] = it["price"] / it["prior_close"] - 1
                it["would_trigger"] = would_trigger(side, it["prior_close"], it["price"])
            o = None
            if mode == "market":
                o = row.get("open") or b.get("open") or p.get("open")
                if o is None and (side, t) in carried:
                    o = carried[(side, t)]["open"]
            if o is not None:
                it["open"] = float(o)
                it["open_gap"] = it["open"] / it["prior_close"] - 1
                if side == "long":
                    g = row.get("grade_open") or (grade_of(it, it["open"]) if grade_of else None)
                    if g is None and it.get("b_price") is not None:
                        g = "B" if it["open"] <= it["b_price"] else "C"
                else:
                    g = core.reversal_scenario(it["open_gap"])
                it["grade_open"] = g
            it["status"] = status(side, it["streak"], it["prior_close"], it.get("open"))
    return block


def tag_rows(rows: list[dict], side: str, block: dict, streak_of, mode: str) -> list[dict]:
    """Every row says where it stands: on / no_gap / off in the session, setting_up / off
    before it. `streak_of(ticker)` = (down, up), or None when unknown."""
    listed = {it["ticker"]: it for it in block.get(side) or []}
    for r in rows:
        it = listed.get(r["ticker"])
        if it:
            r["gapfade"] = it.get("status")
            r["gapfade_streak"] = it.get("streak")
            continue
        if streak_of is None:                      # a re-grade between passes: the pass's own count stands
            streak = r.get("gapfade_streak")
        else:
            st = streak_of(r["ticker"])
            streak = None if st is None else (st[0] if side == "long" else st[1])
        o = r.get("open") if mode == "market" else None
        r["gapfade"] = status(side, streak, r.get("prior_close"), o)
        r["gapfade_streak"] = streak
    return rows


def missing_opens(block: dict) -> bool:
    return any(it.get("open") is None for side in ("long", "short") for it in block.get(side) or [])


# ---------------------------------------------------------------------------
# Pushes
# ---------------------------------------------------------------------------

def push_on() -> bool:
    return os.environ.get("GAPFADE_PUSH") == "1" or (_method_dir() / "gapfade_push.on").exists()


def push_path(session: str) -> Path:
    return _method_dir() / f"gapfade_push_{session}.json"


def _px(p: float) -> str:
    return f"{p:,.2f}"


def _day(session: str) -> str:
    d = datetime.date.fromisoformat(session)
    return f"{d:%a} {d.month}/{d.day}"


def evening_line(it: dict) -> str:
    if it["side"] == "long":
        a = f", A {_px(it['a_price'])}" if it.get("a_price") is not None and it["a_price"] < it["prior_close"] \
            else ", A on any gap" if it.get("a_price") is not None else ""
        return f"{it['ticker']} long: below {_px(it['prior_close'])} triggers{a}"
    return f"{it['ticker']} short: above {_px(it['prior_close'])}, A {_px(it['a_price'])}"


def evening_push(block: dict, session: str, mode: str, now: datetime.datetime, send,
                 built: bool, enabled: bool | None = None) -> bool:
    """One push a session for the A-ready names, once tomorrow's lists are built (or by
    EVENING_PUSH_BY). Recorded only when a phone got it."""
    if mode != "evening" or not (push_on() if enabled is None else enabled):
        return False
    if not built and now.time() < EVENING_PUSH_BY:
        return False
    path = push_path(session)
    state = core.read_json(path, {}) or {}
    if state.get("evening"):
        block["pushed"] = state["evening"]
        return False
    ready = [it for side in ("long", "short") for it in block.get(side) or [] if it.get("a_ready")]
    if not ready:
        return False
    total = len(block.get("long") or []) + len(block.get("short") or [])
    from method.watch import fit
    body = fit([evening_line(it) for it in ready])
    if total > len(ready):
        body = body + f" ({total} setting up on the board)"
    title = f"Gap fade setting up · {_day(session)}"
    try:
        n = send(title, body[:240], url="/#/bounce", tag=f"gapfade-eve-{session}")
    except Exception as e:
        logger.warning(f"gap fade evening push failed: {e}")
        n = 0
    if not n:
        return False
    state["evening"] = now.isoformat(timespec="seconds")
    state["evening_names"] = [it["ticker"] for it in ready]
    core.write_json(path, state)
    block["pushed"] = state["evening"]
    return True


def open_line(it: dict) -> tuple[str, str]:
    t, side = it["ticker"], it["side"]
    g = it.get("grade_open")
    title = f"{t} gap fade ON{f' · {g}' if g and g != '-' else ''}"
    how = "down closes, gapped down" if side == "long" else "up closes, gapped up"
    body = (f"{t} opened {_px(it['open'])} ({it['open_gap'] * 100:+.1f}% vs {_px(it['prior_close'])}). "
            f"{it['streak']} straight {how}: 3-day gap fade {'long' if side == 'long' else 'short'}"
            f"{f', {g} at the open' if g and g != '-' else ''}.")
    return title, body


def open_pushes(block: dict, session: str, mode: str, send, pinned=(), enabled: bool | None = None) -> list[str]:
    """One push a name when a ready (or pinned) name opens ON with a B or better
    (reversal: any gap up; +4% is the A). Recorded only when a phone got it."""
    if mode != "market" or not (push_on() if enabled is None else enabled):
        return []
    path = push_path(session)
    state = core.read_json(path, {}) or {}
    done = set(state.get("open") or [])
    sent = []
    for side in ("long", "short"):
        for it in block.get(side) or []:
            t = it["ticker"]
            if t in done or it.get("status") != "on" or not (it.get("a_ready") or t in pinned):
                continue
            if side == "long" and RANK.get(it.get("grade_open") or "-", 0) < RANK["B"]:
                continue
            title, body = open_line(it)
            try:
                n = send(title, body, url=f"/#/{'bounce' if side == 'long' else 'reversal'}/{t}",
                         tag=f"gapfade-{t}-{session}")
            except Exception as e:
                logger.warning(f"gap fade open push failed for {t}: {e}")
                n = 0
            if n:
                sent.append(t)
    if sent:
        state["open"] = sorted(done | set(sent))
        core.write_json(path, state)
        logger.info(f"gap fade open pushes: {', '.join(sent)}")
    return sent


# ---------------------------------------------------------------------------
# Streaks for names without statics (reversal rows from the report)
# ---------------------------------------------------------------------------

def streak_cache_path(session: str) -> Path:
    return _method_dir() / f"gapfade_streaks_{session}.json"


class StreakLookup:
    """(down, up) for a name: statics, a per-session cache, else one daily fetch
    (`daily(ticker, session)` = completed bars). None when nothing has it."""

    def __init__(self, session: str, statics: dict, daily):
        self.session, self.statics, self.daily = session, statics or {}, daily
        self.path = streak_cache_path(session)
        self.cache = core.read_json(self.path, {}) or {}
        self.dirty = False

    def __call__(self, t: str):
        s = self.statics.get(t) or {}
        if s.get("down_streak") is not None:
            return s["down_streak"], s.get("up_streak") or 0
        if t in self.cache:
            v = self.cache[t]
            return None if v is None else tuple(v)
        val = None
        try:
            h = self.daily(t, self.session)
            if h is not None and len(h):
                val = streaks(h["close"].tolist())
        except Exception as e:
            logger.warning(f"{t}: streak fetch failed: {e}")
        self.cache[t] = list(val) if val is not None else None
        self.dirty = True
        return val

    def flush(self) -> None:
        if self.dirty:
            core.write_json(self.path, self.cache)
            self.dirty = False


# ---------------------------------------------------------------------------
# Replay: how often the pushes would have fired (before switching them on)
# ---------------------------------------------------------------------------

def replay(start: str, end: str, ladder_for=None) -> dict:
    """Rebuild each session's lists from the grouped store (method/breadth.py) as the
    evening build would have, then read the next open. Market names only (the watchlist's
    history isn't in the store as a list), so this is a floor on the evening pushes.
    `ladder_for(ticker, hist, cap)` = static with ladder (default: the real scanner + ladder)."""
    from method import breadth
    stored = sorted(p.stem for p in breadth.STORE.glob("*.pkl"))
    sessions = [d for d in stored if start <= d <= end]
    if not sessions:
        return {"sessions": 0}
    long_df = breadth.load_store(stored)
    if ladder_for is None:
        ladder_for = _real_static
    nights = []
    for s in sessions:
        before = long_df[long_df["date"] < s]
        day = long_df[long_df["date"] == s]
        open_of = dict(zip(day["ticker"], day["open"].astype(float)))
        cands = breadth.build_candidates(s, before)
        ups = breadth.build_up_candidates(s, before)
        hists = breadth.ticker_hists(before, statics_names(cands))
        statics = {}
        for t, h in hists.items():
            try:
                st = ladder_for(t, h, cands[t].get("cap") or "Medium")
            except Exception as e:
                logger.warning(f"{t} {s}: ladder failed: {e}")
                st = None
            if st:
                statics[t] = st
        block = {"session": s, "long": long_items(cands, statics, (), ()),
                 "short": short_items(ups, {}, [], None, (), ())}
        ready = [it["ticker"] for side in ("long", "short") for it in block[side] if it["a_ready"]]
        on = []
        for side in ("long", "short"):
            for it in block[side]:
                o = open_of.get(it["ticker"])
                if o is None or not it["a_ready"]:
                    continue
                if status(side, it["streak"], it["prior_close"], o) == "on":
                    if side == "long" and it.get("b_price") is not None and o > it["b_price"]:
                        continue                                 # opened ON but under B: no push
                    on.append(f"{it['ticker']}{'' if side == 'long' else '(S)'} {(o / it['prior_close'] - 1) * 100:+.1f}%")
        nights.append({"session": s, "long": len(block["long"]), "short": len(block["short"]),
                       "ready": ready, "opened_on": on})
        logger.info(f"{s}: {len(block['long'])} long, {len(block['short'])} short setting up; "
                    f"ready {', '.join(ready) or '-'}; opened ON {', '.join(on) or '-'}")
    n = len(nights)
    eve = sum(bool(x["ready"]) for x in nights)
    opn = sum(len(x["opened_on"]) for x in nights)
    return {"sessions": n, "evening_push_nights": eve, "open_pushes": opn,
            "evening_per_month": round(eve / n * 21, 1), "open_per_month": round(opn / n * 21, 1),
            "nights": nights}


def _real_static(t: str, hist, cap: str) -> dict | None:
    from method.ladder import ladder_payload
    from scanners.premarket_bounce_scanner import static_from_hist
    s = static_from_hist(hist)
    s["ladder"] = ladder_payload(s, cap, t)
    s["down_streak"], s["up_streak"] = streaks(hist["close"].tolist())
    return s


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(description="Method 3-day gap fade")
    ap.add_argument("--replay", nargs=2, metavar=("START", "END"), help="push volume over the stored sessions")
    ap.add_argument("--out", help="write every night's lists to this JSON file")
    args = ap.parse_args()
    if args.replay:
        from dotenv import load_dotenv
        load_dotenv(DATA_DIR.parent / ".env")
        res = replay(*args.replay)
        if args.out:
            core.write_json(Path(args.out), res)
        print({k: v for k, v in res.items() if k != "nights"})


if __name__ == "__main__":
    main()
