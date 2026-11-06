"""
Method's "Watching" list: names you pin so Method watches them closely and pushes when
their setup forms (user, 2026-11-05: "i want SNDK MU SOXL to be closely monitored
especially in the overnight session").

A pinned name is always on the board (refresh adds it to the tickers, so its statics and
grade ladder get built even when it isn't on the stock_screener watchlist), and it gets
its own pushes overnight, premarket and in the evening, whatever the priority report said:

  grade     each upgrade to B, A or A+ on tomorrow's ladder
  gap       the price crossing -3 / -5 / -8 / -12% (and +3 / +5%) vs the prior close
  near      within 1% of the next better grade's price

Each fires once per name per session, and only off a fresh print: overnight that means a
Blue Ocean (or hand-entered) quote, never the evening price carried forward; otherwise a
bar no older than MAX_AGE_MIN. Market hours stay with the entry watcher. Nothing is
recorded unless a phone got the push, so a push with no phone to reach retries later.

Pins live in data/method/pins.json until you remove them.

Usage:
    python -m method.pins ls
    python -m method.pins add MU SNDK SOXL
    python -m method.pins rm SOXL
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
from pathlib import Path

from analyzers.bounce_ladder import GRADE_RANK
from method import core

logger = logging.getLogger("method.pins")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _method_dir() -> Path:
    return Path(os.environ.get("METHOD_DIR") or DATA_DIR / "method")


def pins_path() -> Path:
    return _method_dir() / "pins.json"


def alerts_path(session: str) -> Path:
    return _method_dir() / f"pin_alerts_{session}.json"


ALERT_MODES = ("overnight", "premarket", "evening")
GAP_STEPS = (-0.03, -0.05, -0.08, -0.12, 0.03, 0.05)
NEAR = 0.01                    # within 1% of the next better grade
ALERT_GRADES = ("B", "A", "A+")
MAX_AGE_MIN = 30               # Polygon premarket bars run ~15 min behind
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")


# ---------------------------------------------------------------------------
# The list
# ---------------------------------------------------------------------------

def load() -> dict:
    return core.read_json(pins_path(), {}) or {}


def tickers() -> list[str]:
    return sorted(load())


def clean_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    if not TICKER_RE.match(t):
        raise ValueError(f"Not a ticker: {t!r}")
    return t


def set_pin(ticker: str, on: bool, now: datetime.datetime | None = None) -> list[str]:
    t = clean_ticker(ticker)
    pins = load()
    if on:
        pins.setdefault(t, {"added_at": (now or datetime.datetime.now()).isoformat(timespec="seconds")})
    else:
        pins.pop(t, None)
    core.write_json(pins_path(), pins)
    return sorted(pins)


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

def fresh(row: dict, mode: str, now: datetime.datetime) -> bool:
    """Only a new print may alert: overnight, a Blue Ocean or hand-entered quote (src P there
    is the evening price carried forward); otherwise a bar no older than MAX_AGE_MIN."""
    if mode == "overnight" and row.get("src") not in ("Y", "M"):
        return False
    asof = row.get("asof")
    if not asof:
        return False
    try:
        ts = datetime.datetime.fromisoformat(asof)
    except ValueError:
        return False
    if ts.tzinfo is None and now.tzinfo is not None:
        ts = ts.replace(tzinfo=now.tzinfo)
    elif ts.tzinfo is not None and now.tzinfo is None:
        ts = ts.replace(tzinfo=None)
    return (now - ts).total_seconds() <= MAX_AGE_MIN * 60


def _step_key(step: float) -> str:
    return f"gap{step * 100:+.0f}"


def events(row: dict, state: dict) -> list[tuple[str, str]]:
    """(key, why) for every alert this row earns that `state` hasn't had yet."""
    out = []
    g = row.get("grade")
    best = state.get("grade", "-")
    if g in ALERT_GRADES and GRADE_RANK.get(g, 0) > GRADE_RANK.get(best, 0):
        out.append(("grade", f"now an {g}" if g.startswith("A") else f"now a {g}"))
    gap = row.get("gap")
    if gap is not None:
        for step in GAP_STEPS:
            k = _step_key(step)
            crossed = gap <= step if step < 0 else gap >= step
            if crossed and k not in state:
                out.append((k, f"{step * 100:+.0f}% vs the close"))
        out = _deepest_gap_only(out)
    nxt = row.get("next")
    if nxt and nxt.get("price") and row.get("price"):
        k = f"near-{nxt['grade']}"
        if k not in state and row["price"] <= nxt["price"] * (1 + NEAR):
            out.append((k, f"{(row['price'] / nxt['price'] - 1) * 100:.1f}% from {nxt['grade']}"))
    return out


def _deepest_gap_only(evs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """A jump through several steps at once is one alert: keep the furthest step each way
    (the shallower ones are still marked sent by the caller)."""
    downs = [e for e in evs if e[0].startswith("gap-")]
    ups = [e for e in evs if e[0].startswith("gap+")]
    rest = [e for e in evs if not e[0].startswith("gap")]
    keep = []
    if downs:
        keep.append(min(downs, key=lambda e: float(e[0][3:])))
    if ups:
        keep.append(max(ups, key=lambda e: float(e[0][3:])))
    return rest + keep


def message(row: dict, whys: list[str]) -> tuple[str, str]:
    t, price = row["ticker"], row.get("price") or 0
    gap = (row.get("gap") or 0) * 100
    where = row.get("src_label") or row.get("src") or ""
    when = row.get("bar_time") or ""
    title = f"{t} {price:,.2f} ({gap:+.1f}%)"
    parts = [f"{t} {price:,.2f} ({gap:+.1f}% vs the close{', ' + where if where else ''}{' ' + when if when else ''})",
             f"{row.get('grade') or '-'} {row.get('score')}/{row.get('max_score', 6)}"]
    nxt = row.get("next")
    if nxt:
        parts.append(f"{nxt['grade']} at {nxt['price']:,.2f}")
    why = "; ".join(whys)
    body = " · ".join(parts) + ". " + why[:1].upper() + why[1:] + "."
    return title, body


def pin_alerts(rows: list[dict], session: str, mode: str, send, now: datetime.datetime,
               pinned: list[str] | None = None) -> list[str]:
    """Push each pinned name's new events. Returns the tickers pushed."""
    if mode not in ALERT_MODES:
        return []
    watched = set(tickers() if pinned is None else pinned)
    if not watched:
        return []
    path = alerts_path(session)
    state = core.read_json(path, {}) or {}
    sent = []
    for r in rows:
        t = r.get("ticker")
        if t not in watched or not fresh(r, mode, now):
            continue
        mine = state.get(t, {})
        evs = events(r, mine)
        if not evs:
            continue
        title, body = message(r, [why for _, why in evs])
        try:
            n = send(title, body, url=f"/#/bounce/{t}", tag=f"pin-{t}-{session}")
        except Exception as e:
            logger.warning(f"pin push failed for {t}: {e}")
            n = 0
        if not n:
            continue
        for k, _ in evs:
            if k == "grade":
                mine["grade"] = r["grade"]
            else:
                mine[k] = r.get("bar_time") or True
        gap = r.get("gap")
        if gap is not None:                       # every step already passed counts as told
            for step in GAP_STEPS:
                if (gap <= step if step < 0 else gap >= step):
                    mine.setdefault(_step_key(step), r.get("bar_time") or True)
        state[t] = mine
        sent.append(t)
    if sent:
        core.write_json(path, state)
        logger.info(f"pin alerts pushed: {', '.join(sent)}")
    return sent


def board(rows: list[dict], pinned: list[str] | None = None) -> list[dict]:
    """The Watching card: each pinned name's row (or a placeholder while its statics build)."""
    pinned = tickers() if pinned is None else pinned
    by = {r["ticker"]: r for r in rows}
    out = []
    for t in pinned:
        r = by.get(t)
        if r:
            out.append({k: r.get(k) for k in ("ticker", "grade", "score", "max_score", "price", "gap",
                                               "next", "src", "src_label", "bar_time")})
        else:
            out.append({"ticker": t, "pending": True})
    return out


def main():
    ap = argparse.ArgumentParser(description="Method's Watching list")
    ap.add_argument("cmd", choices=["ls", "add", "rm"])
    ap.add_argument("tickers", nargs="*")
    a = ap.parse_args()
    if a.cmd != "ls":
        for t in a.tickers:
            set_pin(t, a.cmd == "add")
    print(json.dumps(load(), indent=1))


if __name__ == "__main__":
    main()
