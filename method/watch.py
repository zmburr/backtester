"""
Method's entry watcher: the 2-min entry rule, live, on a breadth day's names.

The rule is the replay's (scripts/backfill_entry_r.rule_trades): the first 2-min bar that
closes above the prior bar's high right after a fresh low, stop at the low of day, then the
trail (arms at +1.5 ATR, exits under the prior 2-min low), one re-entry after a losing exit.
So an alert is the trade the breadth card's edge numbers measure.

The PC runs rule_state() on the Trillium 1-min bars it already pulls for each name and posts
it with the name's summary (method/pc_pusher.py). The server pushes the entries it hasn't
sent (entry_alerts) and the page shows each name's state. Breadth days (11+) only, once the
open's count locks, and only names in the counted universe ($5+, 500k shares a day): the
edge was measured there. Other days the priority report's names belong to orderPipe's
morning watcher.
"""

from __future__ import annotations

import datetime

import pandas as pd

from method import core

TZ = "US/Eastern"
STALE_MIN = 10                         # an entry first seen later than this shows but doesn't push
FLOOR_ATR = 0.5                        # size off at least half an ATR of risk (the study's floor)
FEED_DOWN_S = 180                      # no live prices this long on a breadth morning: say so
WATCH_FROM, WATCH_UNTIL = datetime.time(9, 31), datetime.time(15, 30)   # no fresh entry after 15:30
ONE = pd.Timedelta(minutes=1)


def hhmm(ts) -> str:
    return pd.Timestamp(ts).strftime("%H:%M")


def clock12(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.hour % 12 or 12}:{ts.minute:02d}"


# ---------------------------------------------------------------- the rule on one name (PC)

def two_min(bars: pd.DataFrame | None, now=None) -> pd.DataFrame:
    """Completed 2-min regular-session bars from 1-min bars indexed by bar START (ET). A minute
    counts once it has closed; a 2-min bar once its second minute is in, or a minute after its
    end (a minute with no trades sends no bar)."""
    from bounce_entry_study.detectors import build_2min
    if bars is None or bars.empty:
        return pd.DataFrame()
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz=TZ)
    b = bars[bars.index + ONE <= now].copy()
    if b.empty:
        return pd.DataFrame()
    if "volume" not in b:
        b["volume"] = 1.0                   # Trillium summaries carry OHLC only; VWAP isn't used here
    if "vwap" not in b:
        b["vwap"] = b["close"]
    two = build_2min(b)
    if two.empty:
        return two
    done = (two.index + ONE <= b.index[-1]) | (two.index + 3 * ONE <= now)
    return two[done]


def rule_state(bars: pd.DataFrame | None, atr: float | None, now=None) -> dict | None:
    """The rule's trades so far on one name: each entry (its 2-min bar's close time, fill,
    stop, state, R) and what the next entry needs. None before the session's second 2-min bar."""
    from bounce_entry_study.exits import ExitSpec, simulate_day_exits
    from scripts.backfill_entry_r import ARM_ATR, LOD_RECENCY_WIN, MAX_ENTRY_SIGNALS, entry_signals
    two = two_min(bars, now)
    if len(two) < 2:
        return None
    atr = float(atr) if isinstance(atr, (int, float)) and atr == atr and atr > 0 else None
    fires, _ = entry_signals(two, atr)
    res = simulate_day_exits(two, fires, ExitSpec(kind="trail", arm_atr=ARM_ATR), atr)
    taken = res.attempts[:MAX_ENTRY_SIGNALS]
    live = res.open and len(res.attempts) <= MAX_ENTRY_SIGNALS      # the open one is the last
    ends = two.index + 2 * ONE
    highs, lows = two["high"].to_numpy(), two["low"].to_numpy()
    last_close = float(two["close"].iloc[-1])
    entries = []
    for k, a in enumerate(taken, 1):
        e = {"n": k, "t": hhmm(a.entry_time), "entry": round(a.entry, 4), "stop": round(a.stop, 4),
             "risk": round(a.risk, 4)}
        if live and k == len(res.attempts):
            after = highs[ends > a.entry_time]
            peak = max([a.entry, *after])
            e.update(state="open", r=round((last_close - a.entry) / a.risk, 2))
            if atr:
                e["arm_at"] = round(a.entry + ARM_ATR * atr, 4)
                e["armed"] = bool(peak >= e["arm_at"])
                if e["armed"]:
                    e["trail"] = round(float(lows[-1]), 4)    # out on a trade under the last 2-min low
        else:
            e.update(state="stopped" if a.stopped else "out", r=round(a.r, 2),
                     exit=round(float(a.exit), 4), exit_t=hhmm(a.exit_time))
        entries.append(e)

    # what the next entry needs: flat, no winner banked yet (a profitable exit ends the day),
    # an entry left, before 15:30; then a 2-min close over this bar's high right after a low
    banked = any(not a.stopped and a.r > 0 for a in (res.attempts[:-1] if live else res.attempts))
    nxt = None
    if not live and not banked and len(res.attempts) < MAX_ENTRY_SIGNALS and ends[-1].time() < WATCH_UNTIL:
        low, low_i = lows[0], 0
        for i in range(1, len(lows)):        # the bar that set the day's low, as entry_signals tracks it
            if lows[i] < low:
                low, low_i = lows[i], i
        nxt = {"over": round(float(highs[-1]), 4), "fresh_low": bool(len(lows) - low_i <= LOD_RECENCY_WIN)}
    return {"asof": hhmm(ends[-1]), "lod": round(float(lows.min()), 4), "entries": entries, "next": nxt}


# ---------------------------------------------------------------- alerts (server)

def watch_names(mb: dict | None) -> list[str]:
    """A breadth day's flagged names the alerts cover: strict and relaxed, counted universe only."""
    if not mb or mb.get("status") != "ok" or mb.get("state") != "breadth":
        return []
    detail = mb.get("detail") or {}
    names = list(mb.get("names") or []) + list(mb.get("relaxed") or [])
    return [t for t in dict.fromkeys(names) if (detail.get(t) or {}).get("counted")]


def shares(entry: float, stop: float, risk: float | None, atr: float | None, dollars: float) -> int:
    """Shares for `dollars` of risk off the larger of the stop distance and half an ATR, rounded down."""
    per = max(entry - stop, risk or 0.0, FLOOR_ATR * atr if atr else 0.0)
    if per <= 0:
        return 0
    n = int(round(dollars / per, 6))      # 9000 / 0.6000000000000014 is 15,000 shares
    step = 100 if n >= 1000 else 10 if n >= 100 else 1
    return n // step * step


def size_rows(rows: list[dict], mb: dict | None) -> list[dict]:
    """On a breadth day, each watched row's rule state gets the day's risk an attempt and the
    shares for it: every entry, and the next one off today's low (an estimate until it fires)."""
    band = (mb or {}).get("band")
    names = set(watch_names(mb))
    if band not in core.ENTRY_RISK_R or not names:
        return rows
    risk_r = core.ENTRY_RISK_R[band]
    dollars = risk_r * core.R_DOLLARS
    for r in rows:
        ru = r.get("rules")
        if r["ticker"] not in names or not ru:
            continue
        atr = r.get("atr")
        nxt = ru.get("next")
        if nxt and ru.get("lod"):
            nxt = dict(nxt, shares=shares(nxt["over"], ru["lod"], None, atr, dollars))
        r["rules"] = {**ru, "risk_r": risk_r, "alerted": core.ENTRIES_ALERTED[band], "next": nxt,
                      "entries": [dict(e, shares=shares(e["entry"], e["stop"], e.get("risk"), atr, dollars))
                                  for e in ru.get("entries") or []]}
    return rows


def px(p: float) -> str:
    return f"{p:,.2f}"


def entry_line(t: str, e: dict, row: dict, band: str) -> str:
    score = row.get("score_open") if row.get("score_open") is not None else row.get("score")
    if row.get("breadth_flag") == "relaxed":      # flagged under the relaxed rules: its strict score reads low
        score = None
        t = f"{t} relaxed" if e["n"] == 1 else t
    head = f"{t} re-entry" if e["n"] > 1 else f"{t} {score}/6" if score is not None else t
    r = core.ENTRY_RISK_R[band]
    n = shares(e["entry"], e["stop"], e.get("risk"), row.get("atr"), r * core.R_DOLLARS)
    size = f" · {n:,} sh ({r}R)" if n else ""
    return f"{head} buy {px(e['entry'])} stop {px(e['stop'])}{size}"


def fit(lines: list[str], limit: int = 240) -> str:
    """As many lines as fit in a push body, then how many more are on the board."""
    out = []
    for i, line in enumerate(lines):
        more = len(lines) - i - 1
        tail = f"\n+{more} more on the board" if more else ""
        if len("\n".join(out + [line]) + tail) > limit:
            left = len(lines) - i
            return "\n".join(out + [f"+{left} more on the board"])
        out.append(line)
    return "\n".join(out)


def _state(path) -> dict:
    return core.read_json(path, {}) or {}


def _prune(path, days: int = 14) -> None:
    """entry_alerts_<session>.json files older than two weeks go."""
    try:
        session = path.stem.rsplit("_", 1)[-1]
        cutoff = (datetime.date.fromisoformat(session) - datetime.timedelta(days=days)).isoformat()
        for p in path.parent.glob("entry_alerts_*.json"):
            if p.stem.rsplit("_", 1)[-1] < cutoff:
                p.unlink()
    except (OSError, ValueError):
        pass


def entry_alerts(session: str, mb: dict | None, rows: list[dict], names: dict, now: datetime.datetime,
                 send, path) -> list[str]:
    """Push the entries the rule took on a breadth day's names that no phone has had yet: one
    push a batch, best score first. Returns the keys pushed ("MUU#1"). A key is kept only once
    a phone got it; an entry first seen more than STALE_MIN after its bar shows on the board
    but never pushes. 11-15 days alert the first entry only (core.ENTRIES_ALERTED)."""
    if not mb or mb.get("session") != session or not mb.get("locked"):
        return []
    band = mb.get("band")
    if band not in core.ENTRIES_ALERTED:
        return []
    allowed = core.ENTRIES_ALERTED[band]
    state = _state(path)
    done = set(state.get("sent") or []) | set(state.get("late") or [])
    by_t = {r["ticker"]: r for r in rows}
    fresh, late = [], []
    for t in watch_names(mb):
        for e in (((names.get(t) or {}).get("rules") or {}).get("entries") or [])[:allowed]:
            key = f"{t}#{e['n']}"
            if key in done:
                continue
            fired = pd.Timestamp(f"{session} {e['t']}", tz=TZ)
            if (pd.Timestamp(now) - fired).total_seconds() > STALE_MIN * 60:
                late.append(key)
            else:
                fresh.append((t, e, by_t.get(t) or {}))
    if late:
        state["late"] = sorted(set(state.get("late") or []) | set(late))
        core.write_json(path, state)
    if not fresh:
        return []

    def score(row):
        s = row.get("score_open") if row.get("score_open") is not None else row.get("score")
        return s if isinstance(s, (int, float)) else 0
    fresh.sort(key=lambda x: (-score(x[2]), x[1]["n"], x[1]["t"], x[0]))
    word = "Strong breadth day" if band == "16+" else "Breadth day"
    title = f"{word} · {len(fresh)} {'entry' if len(fresh) == 1 else 'entries'}"
    body = fit([entry_line(t, e, row, band) for t, e, row in fresh])
    url = f"/#/bounce/{fresh[0][0]}" if len(fresh) == 1 else "/#/bounce"
    if not send(title, body, url=url, tag=f"entry-{pd.Timestamp(now):%H%M%S}"):
        return []
    keys = [f"{t}#{e['n']}" for t, e, _ in fresh]
    first = not path.exists()
    state["sent"] = sorted(set(state.get("sent") or []) | set(keys))
    core.write_json(path, state)
    if first:
        _prune(path)
    return keys


def feed_down_alert(session: str, mb: dict | None, feed: dict | None, now: datetime.datetime,
                    send, path) -> bool:
    """One push when the PC's prices for a breadth day's names stop during the watch window,
    so a quiet phone never passes for 'no entries'. It re-arms once prices come back."""
    if (not mb or mb.get("session") != session or not mb.get("locked") or not watch_names(mb)
            or not (WATCH_FROM <= now.time() < WATCH_UNTIL)):
        return False
    names = (feed or {}).get("names") or {} if (feed or {}).get("day") == session else {}
    stamps = []
    for t in watch_names(mb):
        try:
            stamps.append(datetime.datetime.fromisoformat(names[t]["received_at"]))
        except (KeyError, TypeError, ValueError):
            continue
    newest = max(stamps) if stamps else None
    state = _state(path)
    if newest is not None and (now - newest).total_seconds() <= FEED_DOWN_S:
        if state.pop("feed_down", None):
            core.write_json(path, state)
        return False
    if state.get("feed_down"):
        return False
    since = f"since {clock12(newest)}" if newest else "this morning"
    if not send("Entry alerts paused", f"No live prices from the PC {since}. Breadth-day entry alerts "
                f"need the Trillium feed: check the PC.", url="/#/bounce", tag="feed-down"):
        return False
    state["feed_down"] = now.isoformat(timespec="seconds")
    core.write_json(path, state)
    return True
