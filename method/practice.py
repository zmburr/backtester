"""
Method's Practice arena: old bounce and reversal days, played blind, scored against
your rules and your real fills.

The deck mixes your graded book (data/bounce_data.csv, data/reversal_data.csv), days
nobody picked (method/practice_seed.csv: B+ bounce candidates from the 2022-26
population, parabolic GO names from the 2020-26 reversal universe, intraday flushes
from 2019-26) and your own price-action trade days from the Intrinsic feed, with your
fills. Each day is one method/review.review() replay (cached with History and Best),
plus what the arena needs on top of it:

  * triggers: every 2-min bar the live entry rule fires on (bounce:
    scripts/backfill_entry_r.entry_signals; reversal: a close below the prior bar's
    low, 9:32-10:30), as 2-min bar START epochs, like the replay's trades;
  * the rules' trades with R on the live sizing risk (stop distance floored at
    0.5 ATR bounce / 0.25 ATR reversal, as the morning watcher sizes a try);
  * your fills from the trade feed, when you traded it;
  * quick-read cards: a chart frozen at a decision point and one question whose
    answer comes from your rules or the data (setup, size for the day, trigger,
    stop, hold or sell, and "will it work?" scored only against the base rate).

Play and quick-read results land in data/method/practice/log.jsonl, so a session
can be reviewed later.

Usage:
    python -m method.practice seed --population P --universe U [U ...] --flush F   # on the PC
    python -m method.practice build [--warm] [--limit N]                           # on the mini
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import logging
import os
import random
import threading
import time
import zlib
from pathlib import Path

import pandas as pd

from method import core
from method.refresh import DATA_DIR, METHOD_DIR

logger = logging.getLogger("method.practice")

TZ = "US/Eastern"
HOME = METHOD_DIR / "practice"
DECK = HOME / "deck.json"
STATE = HOME / "state.json"
LOG = HOME / "log.jsonl"
REVIEW_DIR = METHOD_DIR / "review"
SEED = Path(__file__).with_name("practice_seed.csv")
FEED = Path(os.environ.get("METHOD_TRADE_FEED") or "~/IntrinsicSync/feed/trade_data.csv").expanduser()

FLOOR_ATR = {"bounce": 0.5, "reversal": 0.25}       # the morning watcher's try sizing (try_size.FLOOR_ATR)
ARM_ATR = 1.5                                        # no defensive bounce exit before +1.5 ATR
FLUSH_ATR = 4.0                                      # reversal: cover half into a 4-ATR flush
REV_GATE = (datetime.time(9, 32), datetime.time(10, 30))
LAST_ENTRY = datetime.time(15, 30)
BAR = pd.Timedelta(minutes=2)

# Your trade tags that make a day a price-action day (news trades stay out)
PA_TAGS = {"capitulation", "bounce", "intradaycapitch", "capitchpaniclong", "3dgapfade", "3dgapfade_weakstock",
           "2dgapfade", "gapdowntrendbreak", "rightsidepopfade", "2dbreakoutib", "priordaylowbreak",
           "consolidationbreakdown", "athfail", "parabolic", "gapfade"}

SETUP_CHOICES = {
    "bounce": ["Gap-fade", "Intraday flush"],
    "reversal": ["3-day gap-fade", "2-day gap-fade", "Gap-down trend break", "Other"],
}
SOURCE_WORD = {"book": "your graded book", "pool": "a candidate nobody picked",
               "flush": "an intraday flush", "yours": "a day you traded"}


# ---------------------------------------------------------------------------- labels

def setup_class(side: str, setup: str | None, source: str = "book") -> str | None:
    """The answer to "which setup is this?" from a day's label."""
    s = (setup or "").lower()
    if side == "bounce":
        if source == "flush" or "intradaycapitch" in s or "capitchpanic" in s:
            return "Intraday flush"
        if "gapfade" in s or "capitulation" in s or source == "pool":
            return "Gap-fade"
        return None
    if "generic" in s:
        return None                                     # the universe scan's fallback label: not your taxonomy
    if "3dgapfade" in s:
        return "3-day gap-fade"
    if "2dgapfade" in s:
        return "2-day gap-fade"
    if "gapdowntrendbreak" in s:
        return "Gap-down trend break"
    return "Other" if s else None


def day_worth(n: int | None) -> str | None:
    """Bounce Lessons: 0-10 names ≈ 0R, 11-15 = 1R first entry only, 16+ = 3R a try."""
    if n is None:
        return None
    if n >= 16:
        return f"{core.ENTRY_RISK_R['16+']}R a try"
    if n >= 11:
        return f"{core.ENTRY_RISK_R['11-15']}R"
    return "Pass"


# ---------------------------------------------------------------------------- the deck

def _key(side: str, ticker: str, date: str) -> str:
    return f"{side}:{ticker}:{date}"


def book_rows(library_rows: list[dict]) -> list[dict]:
    out = []
    for r in library_rows:
        if r.get("ok") is False:
            continue
        out.append({"key": r["key"], "side": r["side"], "ticker": r["ticker"], "date": r["date"],
                    "cap": r.get("cap"), "setup": r.get("setup"), "grade": r.get("grade"), "source": "book"})
    return out


def seed_rows(path: Path | None = None) -> list[dict]:
    path = path or SEED
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [{"key": _key(r["side"], r["ticker"], r["date"]), "side": r["side"], "ticker": r["ticker"],
             "date": r["date"], "cap": r.get("cap") or None, "setup": r.get("setup") or None,
             "grade": None, "source": r["source"]} for r in rows]


def _feed_time(v) -> int | None:
    """Feed Start/End ('2026-11-05 09:33:41-05:00' or naive ET) -> ET-shifted epoch."""
    try:
        ts = pd.Timestamp(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert(TZ).tz_localize(None)
    return int(ts.timestamp())


def _tags(v) -> set[str]:
    return {t.strip().lower() for t in str(v or "").split(",") if t.strip()}


def feed_days(path: Path | None = None) -> tuple[list[dict], dict]:
    """Your price-action trade days ({deck rows}, {key: [fills]}). A long day plays as a
    bounce, a short day as a reversal (SOXS shorted into a semis flush is a short-the-spike day).
    The tags decide which days join the deck; fills are kept for every stock day you traded, so a
    book or pool day shows what you did even when you didn't tag it."""
    path = path or FEED
    if not path.exists():
        return [], {}
    df = pd.read_csv(path, low_memory=False, usecols=lambda c: c in {
        "Date", "Symbol", "Start", "End", "Net P&L", "Max Size", "Avg Price at Max", "Tags", "Avg Exit Price", "Side"})
    rows, fills = [], {}
    df["_d"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    for (d, sym), g in df.dropna(subset=["_d"]).groupby(["_d", "Symbol"], sort=False):
        if not isinstance(sym, str) or not sym.isalpha() or len(sym) > 6:
            continue                                    # option symbols and odd rows
        tags = set().union(*(_tags(t) for t in g["Tags"]))
        sides = pd.to_numeric(g["Side"], errors="coerce").dropna()
        if sides.empty:
            continue
        side = "bounce" if sides.sum() >= 0 else "reversal"
        date = d.date().isoformat()
        key = _key(side, sym.upper(), date)
        fills[key] = [{
            "start": _feed_time(r["Start"]), "end": _feed_time(r["End"]),
            "entry": core.to_num(r["Avg Price at Max"]), "exit": core.to_num(r["Avg Exit Price"]),
            "size": core.to_num(r["Max Size"]), "side": core.to_num(r["Side"]),
            "pnl": core.to_num(r["Net P&L"]),
        } for _, r in g.iterrows()]
        if not tags & PA_TAGS:
            continue
        setup = next((t for t in ("intradaycapitch", "capitchpaniclong", "3dgapfade", "2dgapfade",
                                  "gapdowntrendbreak", "capitulation") if t in tags), None)
        rows.append({"key": key, "side": side, "ticker": sym.upper(), "date": date, "cap": None,
                     "setup": setup, "grade": None, "source": "yours"})
    return rows, fills


def build_deck(library_rows: list[dict], seed: list[dict], yours: list[dict]) -> list[dict]:
    """One row a day: your book first, then the seed, then your trade days. A day in more than
    one source keeps the first label and notes that you traded it."""
    seen: dict[str, dict] = {}
    for r in [*book_rows(library_rows), *seed, *yours]:
        if r["key"] in seen:
            if r["source"] == "yours":
                seen[r["key"]]["traded"] = True
            continue
        r = dict(r)
        r["year"] = int(r["date"][:4])
        r["setup_class"] = setup_class(r["side"], r.get("setup"), r["source"])
        r["traded"] = r["source"] == "yours"
        seen[r["key"]] = r
    return list(seen.values())


# ---------------------------------------------------------------------------- dealing

def _weight(row: dict, this_year: int) -> float:
    """Older days come up more often: those are the ones you won't remember."""
    return 1.0 + 0.35 * max(0, this_year - row["year"])


def cycle_order(rows: list[dict], seed: int, this_year: int) -> list[str]:
    """A weighted shuffle (Efraimidis-Spirakis): every day once a cycle, old days early."""
    rng = random.Random(seed)
    keyed = [(rng.random() ** (1.0 / _weight(r, this_year)), r["key"]) for r in rows]
    return [k for _, k in sorted(keyed, reverse=True)]


_state_lock = threading.Lock()


def deal(mode: str, n: int, deck: list[dict], state_path: Path | None = None, today: datetime.date | None = None) -> list[str]:
    """The next n days for a mode. Nothing repeats until the whole deck has been dealt."""
    state_path = state_path or STATE
    today = today or datetime.date.today()
    keys = {r["key"] for r in deck}
    if not keys:
        return []
    with _state_lock:
        state = core.read_json(state_path, {}) or {}
        st = state.get(mode) or {}
        order = [k for k in st.get("order", []) if k in keys]
        pos = int(st.get("pos", 0))
        cycle = int(st.get("cycle", 0))
        missing = keys - set(order)
        if missing:                                     # a new day joins the end of this cycle
            order += cycle_order([r for r in deck if r["key"] in missing], cycle * 7919 + len(order), today.year)
        out = []
        while len(out) < min(n, len(keys)):
            if pos >= len(order):
                cycle += 1
                order, pos = cycle_order(deck, cycle * 7919 + zlib.crc32(mode.encode()) % 1000, today.year), 0
            out.append(order[pos])
            pos += 1
        state[mode] = {"order": order, "pos": pos, "cycle": cycle}
        core.write_json(state_path, state)
    return out


# ---------------------------------------------------------------------------- one day

def candles_frame(candles: list[dict]) -> pd.DataFrame:
    """The replay's 1-min candles (ET-shifted epochs) back to a frame on ET wall-clock time."""
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles)
    idx = pd.to_datetime(df["time"], unit="s")
    out = pd.DataFrame({"open": df["open"].values, "high": df["high"].values, "low": df["low"].values,
                        "close": df["close"].values, "volume": df["value"].values}, index=idx)
    out["vwap"] = df["vw"].values if "vw" in df else out["close"]
    return out


def _epoch(ts) -> int:
    return int(pd.Timestamp(ts).timestamp())


def two_min(bars: pd.DataFrame) -> pd.DataFrame:
    from bounce_entry_study.detectors import build_2min
    return build_2min(bars) if not bars.empty else pd.DataFrame()


def bounce_triggers(two: pd.DataFrame, atr: float | None) -> list[int]:
    """Every 2-min bar the live entry rule fires on (bar START epochs)."""
    from scripts.backfill_entry_r import entry_signals
    fires, _ = entry_signals(two, atr)
    return [_epoch(t - BAR) for t in fires]


def reversal_triggers(two: pd.DataFrame) -> list[int]:
    out = []
    for i in range(1, len(two)):
        end = (two.index[i] + BAR).time()
        if REV_GATE[0] <= end <= REV_GATE[1] and two["close"].iloc[i] < two["low"].iloc[i - 1]:
            out.append(_epoch(two.index[i]))
    return out


def live_r(side: str, trades: list[dict], atr: float | None) -> list[dict]:
    """The rules' trades with R on the live sizing risk: |entry - stop| floored at
    0.5 ATR (bounce) or 0.25 ATR (reversal), so a tiny stop can't inflate R."""
    out = []
    floor = FLOOR_ATR[side] * (atr or 0)
    for t in trades or []:
        sign = 1 if side == "bounce" else -1
        risk = max(abs(t["entry"] - t["stop"]), floor) or abs(t["entry"]) * 0.0025
        if t.get("half"):
            r = 0.5 * sign * (t["half"]["price"] - t["entry"]) / risk + 0.5 * sign * (t["exit"] - t["entry"]) / risk
        else:
            r = sign * (t["exit"] - t["entry"]) / risk
        out.append({**t, "r_live": float(r), "risk_live": float(risk)})
    return out


def _bar(two: pd.DataFrame, i: int) -> dict:
    r = two.iloc[i]
    return {"t": _epoch(two.index[i]), "end": _epoch(two.index[i] + BAR), "o": float(r.open), "h": float(r.high),
            "l": float(r.low), "c": float(r.close)}


def setup_card(side: str, cls: str | None, triggers: list[int], two: pd.DataFrame, source: str) -> dict | None:
    """At the first trigger (the moment it matters: take this one?), which setup is it?"""
    if not cls or not triggers:
        return None
    t = triggers[0] + 120
    why = {"Gap-fade": "It gapped down into a capitulation low: your gap-fade book (+3.2R a day on the 2-min rule).",
           "Intraday flush": "It opened and sold off through the day: an intraday flush. Your study: about −0.9R a day "
                             "buying these, unless the low is in before 10:00 or 11+ names are flushing.",
           "3-day gap-fade": "Up days in a row, then a gap up that fades: the 3-day gap-fade.",
           "2-day gap-fade": "Two up days, then a gap up that fades: the 2-day gap-fade.",
           "Gap-down trend break": "A gap down that breaks the uptrend. Flat by the close at every cap.",
           "Other": "Not one of the three main reversal setups."}.get(cls, "")
    return {"type": "setup", "cut": t, "tf": "5", "q": "Which setup is this?", "choices": SETUP_CHOICES[side],
            "answer": cls, "why": f"{why} Labelled from {SOURCE_WORD.get(source, source)}."}


def size_card(n: int | None, state: str | None, open_t: int | None) -> dict | None:
    ans = day_worth(n)
    if ans is None or open_t is None:
        return None
    return {"type": "size", "cut": open_t + 60, "tf": "5",
            "q": f"{n} names were B or better market-wide at the open ({state or '–'} day). How much is this day worth?",
            "choices": ["Pass", "1R", "3R a try"], "answer": ans,
            "why": "Your rule, from 2022-26: 0-10 names ≈ 0R a name, 11-15 = +0.1R (a coin flip, 1R first entry only), "
                   "16+ = +1.6R (3R an attempt, spread across names)."}


def trigger_cards(side: str, two: pd.DataFrame, triggers: list[int], rng: random.Random) -> list[dict]:
    """One real trigger and one near miss, each frozen at that bar's close."""
    if len(two) < 3:
        return []
    fire = set(triggers)
    long = side == "bounce"
    lows, highs, closes = two["low"].values, two["high"].values, two["close"].values
    misses = []
    day_low_i = 0
    for i in range(1, len(two)):
        if lows[i] < lows[day_low_i]:
            day_low_i = i
        t = _epoch(two.index[i])
        if t in fire:
            continue
        end = (two.index[i] + BAR).time()
        if long:
            if end > LAST_ENTRY or i - day_low_i > 3:
                continue
            if highs[i] > highs[i - 1] and closes[i] <= highs[i - 1]:
                misses.append((i, "It broke the prior bar's high but closed back under it. The rule needs the close."))
            elif closes[i] > two["open"].values[i] and closes[i] <= highs[i - 1]:
                misses.append((i, "A green bar, but it didn't close above the prior bar's high."))
        else:
            if not REV_GATE[0] <= end <= REV_GATE[1]:
                if closes[i] < lows[i - 1] and end > REV_GATE[1] and end.hour < 12:
                    misses.append((i, "It closed below the prior bar's low, but after 10:30: outside the window."))
                continue
            if lows[i] < lows[i - 1] and closes[i] >= lows[i - 1]:
                misses.append((i, "It broke the prior bar's low but closed back above it. The rule needs the close."))
    out = []
    first = sorted(fire)[:1]
    for t in first:
        i = next((k for k in range(len(two)) if _epoch(two.index[k]) == t), None)
        if i is None:
            continue
        why = ("It closed above the prior bar's high, right after a fresh low: that's your entry."
               if long else "It closed below the prior bar's low inside 9:32-10:30: that's your entry.")
        out.append({"type": "trigger", "cut": _epoch(two.index[i] + BAR), "tf": "2", "mark": t,
                    "q": "Is this bar your trigger?", "choices": ["Yes", "No"], "answer": "Yes", "why": why})
    if misses:
        i, why = rng.choice(misses[:6])
        out.append({"type": "trigger", "cut": _epoch(two.index[i] + BAR), "tf": "2", "mark": _epoch(two.index[i]),
                    "q": "Is this bar your trigger?", "choices": ["Yes", "No"], "answer": "No", "why": why})
    return out


def stop_card(side: str, two: pd.DataFrame, triggers: list[int], atr: float | None) -> dict | None:
    """At the first trigger: which price is the stop? (The low of day, or the high for a short.)"""
    if not triggers:
        return None
    t = triggers[0]
    i = next((k for k in range(len(two)) if _epoch(two.index[k]) == t), None)
    if i is None or i < 1:
        return None
    long = side == "bounce"
    upto = two.iloc[: i + 1]
    stop = float(upto["low"].min() if long else upto["high"].max())
    prior = float(two["low"].iloc[i - 1] if long else two["high"].iloc[i - 1])
    vw = two["vwap_at_end"].iloc[i] if "vwap_at_end" in two else None
    opn = float(two["open"].iloc[0])
    pool = [("the low of day" if long else "the high of day", stop), ("the prior bar's low" if long else "the prior bar's high", prior),
            ("VWAP", float(vw) if vw is not None and pd.notna(vw) else None), ("the open", opn)]
    if atr:
        pool.append(("half an ATR past the low" if long else "half an ATR past the high", stop - 0.5 * atr if long else stop + 0.5 * atr))
    seen, choices = [], []
    tol = 0.05 * (atr or abs(stop) * 0.01)
    for name, p in pool:
        if p is None or any(abs(p - q) <= tol for q in seen):
            continue
        seen.append(p)
        choices.append({"label": name, "price": p})
        if len(choices) == 4:
            break
    if len(choices) < 2:
        return None
    floor = f" Size it off {FLOOR_ATR[side]} ATR if the stop is closer than that." if atr else ""
    return {"type": "stop", "cut": _epoch(two.index[i] + BAR), "tf": "2", "mark": t,
            "q": "You're taking this trigger. Where's the stop?", "choices": [c["label"] for c in choices],
            "prices": {c["label"]: c["price"] for c in choices}, "answer": choices[0]["label"],
            "why": f"The stop is the {'low' if long else 'high'} of day: if it goes, the setup is wrong.{floor}"}


def manage_cards(side: str, two: pd.DataFrame, trades: list[dict], atr: float | None, rng: random.Random) -> list[dict]:
    """Inside the rules' first trade: a shake before +1.5 ATR (hold), and the trail break after it (sell)."""
    if not trades or not atr or side != "bounce":
        return []
    tr = trades[0]
    t0 = tr["entry_t"]
    idx = [k for k in range(len(two)) if _epoch(two.index[k]) > t0]
    lows, highs, closes, opens = two["low"].values, two["high"].values, two["close"].values, two["open"].values
    out, peak, holds = [], tr["entry"], []
    for k in idx:
        if lows[k] < tr["stop"]:
            break
        armed = peak >= tr["entry"] + ARM_ATR * atr
        if not armed and closes[k] < opens[k] and lows[k] < lows[k - 1] and closes[k] > tr["stop"]:
            holds.append(k)
        if armed and lows[k] < lows[k - 1]:
            out.append({"type": "manage", "cut": _epoch(two.index[k] + BAR), "tf": "2", "mark": _epoch(two.index[k]),
                        "entry": tr["entry"], "stop": tr["stop"],
                        "q": "You're long from the entry marked. Hold or sell?", "choices": ["Hold", "Sell"], "answer": "Sell",
                        "why": "It's past +1.5 ATR and just broke the prior 2-min low: the trail is hit. Take it."})
            break
        peak = max(peak, highs[k])
    if holds:
        k = rng.choice(holds[:4])
        out.insert(0, {"type": "manage", "cut": _epoch(two.index[k] + BAR), "tf": "2", "mark": _epoch(two.index[k]),
                       "entry": tr["entry"], "stop": tr["stop"],
                       "q": "You're long from the entry marked. Hold or sell?", "choices": ["Hold", "Sell"], "answer": "Hold",
                       "why": "Not yet +1.5 ATR and the stop at the low holds: no defensive exit. "
                              "Selling here is the early exit your capture numbers show (66% kept in 2026)."})
    return out


def works_card(side: str, two: pd.DataFrame, triggers: list[int], atr: float | None) -> dict | None:
    """At the first trigger: does it reach +1.5 ATR before the stop? Scored only against the base rate."""
    if not triggers or not atr:
        return None
    t = triggers[0]
    i = next((k for k in range(len(two)) if _epoch(two.index[k]) == t), None)
    if i is None:
        return None
    long = side == "bounce"
    entry = float(two["close"].iloc[i])
    stop = float(two["low"].iloc[: i + 1].min() if long else two["high"].iloc[: i + 1].max())
    target = entry + ARM_ATR * atr if long else entry - ARM_ATR * atr
    hit = False
    for k in range(i + 1, len(two)):
        if (two["low"].iloc[k] < stop) if long else (two["high"].iloc[k] > stop):
            break
        if (two["high"].iloc[k] >= target) if long else (two["low"].iloc[k] <= target):
            hit = True
            break
    return {"type": "works", "cut": _epoch(two.index[i] + BAR), "tf": "2", "mark": t,
            "q": f"Does this entry reach +{ARM_ATR} ATR before the stop?", "choices": ["Yes", "No"],
            "answer": "Yes" if hit else "No",
            "why": "A guess, not a rule: your score here is compared with always saying the same thing."}


def cards_for(side: str, cls: str | None, source: str, two: pd.DataFrame, triggers: list[int],
              trades: list[dict], atr: float | None, n: int | None, state: str | None,
              open_t: int | None, seed: int) -> list[dict]:
    rng = random.Random(seed)
    cards = [setup_card(side, cls, triggers, two, source),
             size_card(n, state, open_t) if side == "bounce" else None,
             *trigger_cards(side, two, triggers, rng),
             stop_card(side, two, triggers, atr),
             *manage_cards(side, two, trades, atr, rng),
             works_card(side, two, triggers, atr)]
    return [c for c in cards if c]


def day_payload(row: dict, review: dict, fills: list[dict] | None, states: dict) -> dict:
    """Everything the arena needs for one day, on top of the replay."""
    side = row["side"]
    out = {k: review.get(k) for k in ("ticker", "date", "side", "cap", "candles", "daily", "prior_close", "atr",
                                       "session", "at_open", "mid_bb", "ema9", "gap_level", "ladder", "error")}
    out.update(key=row["key"], source=row["source"], source_word=SOURCE_WORD.get(row["source"], row["source"]),
               setup=row.get("setup"), setup_class=row.get("setup_class"), your_grade=row.get("grade"),
               year=row["year"], traded=bool(row.get("traded") or fills), floor_atr=FLOOR_ATR[side], arm_atr=ARM_ATR,
               flush_atr=FLUSH_ATR, one_r=core.R_DOLLARS)
    n, state = states.get(row["date"], (None, None))
    out.update(day_n=n, day_state=state, day_worth=day_worth(n) if side == "bounce" else None)
    bars = candles_frame(review.get("candles") or [])
    atr = review.get("atr")
    if bars.empty:
        out.update(triggers=[], rules=[], fills=fills or [], cards=[], ok=False)
        return out
    two = two_min(bars)
    if side == "bounce":
        trig = bounce_triggers(two, atr)
    else:
        from reversal_entry_study.run_study import _two_min
        trig = reversal_triggers(_two_min(bars.between_time("09:30", "15:59")))
    rules = live_r(side, review.get("trades") or [], atr)
    rth = bars.between_time("09:30", "15:59")
    open_t = _epoch(rth.index[0]) if not rth.empty else None
    seed = int(hashlib.md5(row["key"].encode()).hexdigest()[:8], 16)
    cards = cards_for(side, row.get("setup_class"), row["source"], two, trig, rules, atr, n, state, open_t, seed)
    out.update(triggers=trig, rules=rules, rules_r=sum(t["r_live"] for t in rules) if rules else 0.0,
               fills=fills or [], cards=cards, ok=True)
    return core.clean(out)


# ---------------------------------------------------------------------------- the log

def append_log(entry: dict, path: Path | None = None) -> dict:
    path = path or LOG
    entry = {"at": pd.Timestamp.now(tz=TZ).isoformat(timespec="seconds"), **entry}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(core.clean(entry)) + "\n")
    return entry


def read_log(path: Path | None = None) -> list[dict]:
    path = path or LOG
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


FLAG_WORDS = {
    "early": "Bought before the trigger", "add_no_break": "Added without a fresh break", "added_to_loser": "Added to a loser",
    "third_try": "A 3rd try after 2 stops", "early_exit": "Sold before +1.5 ATR", "early_cover": "Covered before the plan said",
    "over_budget": "Risk above your budget", "budget_vs_breadth": "Budget above what the day supports",
}


def stats(rows: list[dict], last: int = 20) -> dict:
    """The stats card: play flags and R over your last `last` days, quick reads by question."""
    plays = [r for r in rows if r.get("mode") == "play"]
    recent = plays[-last:]
    flags = {}
    for k in FLAG_WORDS:
        n = sum(1 for r in recent if any(f.get("type") == k for f in r.get("flags") or []))
        flags[k] = {"label": FLAG_WORDS[k], "days": n, "rate": n / len(recent) if recent else None}
    prev = plays[-2 * last:-last]
    trend = {}
    for k in FLAG_WORDS:
        if prev and recent:
            a = sum(1 for r in prev if any(f.get("type") == k for f in r.get("flags") or [])) / len(prev)
            trend[k] = flags[k]["rate"] - a
    reads = [r for r in rows if r.get("mode") == "read"]
    by_type = {}
    for r in reads[-200:]:
        t = by_type.setdefault(r.get("card"), {"n": 0, "right": 0})
        t["n"] += 1
        t["right"] += int(bool(r.get("correct")))
    works = [r for r in reads if r.get("card") == "works"]
    base = None
    if works:
        yes = sum(1 for r in works if r.get("answer") == "Yes") / len(works)
        base = {"n": len(works), "you": sum(1 for r in works if r.get("correct")) / len(works),
                "always_yes": yes, "always_no": 1 - yes}
    return {
        "plays": len(plays), "recent": len(recent),
        "you_r": sum(r.get("you_r") or 0 for r in recent), "rules_r": sum(r.get("rules_r") or 0 for r in recent),
        "flags": flags, "trend": trend, "reads": len(reads), "by_type": by_type, "works": base,
        "clean_days": sum(1 for r in recent if not r.get("flags")),
    }


# ---------------------------------------------------------------------------- seed (PC) and build (mini)

def make_seed(population: Path, universes: list[Path], flush: Path, out: Path | None = None, seed: int = 11) -> int:
    """Days nobody picked, sampled once on the PC (the study files live there) and committed."""
    rng = random.Random(seed)
    rows = []
    book = {(s, t["ticker"], t["date"]) for s in ("bounce", "reversal") for t in _book_dates(s)}

    pop = pd.read_csv(population, usecols=["date", "ticker", "cap", "setup_type", "recommendation", "bounce_day_return"])
    pop = pop[pop.recommendation.isin(["GO", "CAUTION"])]
    pop["year"] = pop.date.str[:4]
    for _, g in pop.groupby("year"):
        g = g[[("bounce", t, d) not in book for t, d in zip(g.ticker, g.date)]]
        win, lose = g[g.bounce_day_return > 0], g[g.bounce_day_return <= 0]
        for part in (win, lose):
            for _, r in part.sample(min(12, len(part)), random_state=rng.randint(0, 10**6)).iterrows():
                rows.append({"source": "pool", "side": "bounce", "ticker": r.ticker, "date": r.date, "cap": r.cap,
                             "setup": r.setup_type})

    uni = pd.concat([pd.read_csv(p, usecols=lambda c: c in {"date", "ticker", "cap", "setup_type", "recommendation",
                                                             "in_reversal_csv", "fade_day_return"}) for p in universes])
    uni = uni[(uni.recommendation == "GO") & (uni.get("in_reversal_csv", False) != True)]   # noqa: E712
    uni = uni.drop_duplicates(["date", "ticker"])
    uni["year"] = uni.date.str[:4]
    for _, g in uni.groupby("year"):
        g = g[[("reversal", t, d) not in book for t, d in zip(g.ticker, g.date)]]
        typed, generic = g[g.setup_type != "generic"], g[g.setup_type == "generic"]
        for part in (typed, generic):                   # half typed 3-day gap-fades, half the scan's generic GOs
            for _, r in part.sample(min(6, len(part)), random_state=rng.randint(0, 10**6)).iterrows():
                rows.append({"source": "pool", "side": "reversal", "ticker": r.ticker, "date": r.date, "cap": r.cap,
                             "setup": r.setup_type})

    fl = pd.read_csv(flush, usecols=["date", "ticker", "lev3x"]).drop_duplicates(["date", "ticker"])
    fl = fl[fl.lev3x.astype(str).str.lower() != "true"]
    fl["year"] = fl.date.astype(str).str[:4]
    for _, g in fl.groupby("year"):
        for _, r in g.sample(min(8, len(g)), random_state=rng.randint(0, 10**6)).iterrows():
            rows.append({"source": "flush", "side": "bounce", "ticker": r.ticker, "date": str(r.date)[:10], "cap": "",
                         "setup": "IntradayCapitch"})

    out = out or SEED
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "side", "ticker", "date", "cap", "setup"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def _book_dates(side: str) -> list[dict]:
    from method import library as lib
    return lib.load_trades(side)


def load_deck(path: Path | None = None) -> dict:
    return core.read_json(path or DECK, {}) or {}


def build(warm: bool = False, limit: int | None = None, pause: float = 0.2) -> dict:
    """Write the deck; with warm=True replay every day into the review cache (Polygon, once)."""
    from method import library as lib
    lib_rows = (core.read_json(lib.LIBRARY) or {}).get("rows") or []
    yours, fills = feed_days()
    deck = build_deck(lib_rows, seed_rows(), yours)
    old = load_deck()
    bad = set(old.get("bad") or [])
    if warm:
        index = lib.load_outcome_index()
        done = 0
        for r in deck:
            if r["key"] in bad:
                continue
            if limit is not None and done >= limit:
                break
            path = REVIEW_DIR / f"{r['side']}_{r['ticker']}_{r['date']}.json"
            if path.exists():
                continue
            try:
                data = lib.cached_review(r["side"], r["ticker"], r["date"], REVIEW_DIR, cap=r.get("cap"),
                                         row=index.get((r["date"], r["side"], r["ticker"])))
                if not data.get("candles"):
                    bad.add(r["key"])
            except Exception as e:
                logger.warning(f"{r['key']}: {e}")
                bad.add(r["key"])
            done += 1
            time.sleep(pause)
    rows = [r for r in deck if r["key"] not in bad]
    counts = {}
    for r in rows:
        counts[f"{r['side']}/{r['source']}"] = counts.get(f"{r['side']}/{r['source']}", 0) + 1
    core.write_json(DECK, {"built_at": pd.Timestamp.now(tz=TZ).isoformat(timespec="seconds"), "rows": rows,
                           "bad": sorted(bad), "counts": counts})
    core.write_json(HOME / "fills.json", fills)
    logger.info(f"practice deck: {len(rows)} days {counts}, {len(bad)} without bars")
    return {"rows": len(rows), "bad": len(bad), "counts": counts}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(description="Method's Practice arena: the deck")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed", help="sample days nobody picked (on the PC)")
    s.add_argument("--population", type=Path, required=True)
    s.add_argument("--universe", type=Path, nargs="+", required=True)
    s.add_argument("--flush", type=Path, required=True)
    b = sub.add_parser("build", help="write the deck (on the mini)")
    b.add_argument("--warm", action="store_true", help="replay every day into the review cache")
    b.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.cmd == "seed":
        print(make_seed(args.population, args.universe, args.flush), "seed days")
    else:
        print(build(warm=args.warm, limit=args.limit))


if __name__ == "__main__":
    main()
