"""
Hyperliquid watch: push to Method when the 24/7 equity perps make a big move while the
US market is shut (user, 2026-10-17, after AI stocks fell 6-8% in an hour on a Saturday
evening and nothing said so).

One process polls trade.xyz's HIP-3 dex on Hyperliquid every 30 s; one free, keyless
request returns every perp's mark and 24h volume. Each name keeps the marks since the
later of an hour ago and the last regular-hours close. A name's move is its mark against
that window's high (a drop) or low (a rip). Outside 9:30-4:00 on NYSE session days:

  market  the AI basket's equal-weight hour is 2%+, or 6+ names move 4%+ the same way.
          One push lists the leaders; the names in it don't push again on their own.
  index   XYZ100 or SP500 moves 1.25%+.
  name    a US-listed, unleveraged name with $1M+ of 24h volume moves 5%+.

Replayed over 5-min bars, 10/02-10/17 (`--replay 360`), it pushes 5 times: the market
alert at 10/17 17:25 (basket -3.3%, 19 names down 4%+; no other off-hours hour had more
than 3 names or a basket worse than -1.9%) and at 18:05 on the rebound; RKLB -7.7% after
the 10/14 close, CRWV -5.0% premarket 10/15 and EWZ +6.3% at the 10/05 4 AM open; the
index alert never (largest off-hours hour -0.9%). Leveraged ETFs and Asia-listed names
swing too much on their own to push singly (KORU, MINIMAX and ZHIPU alone made ~4 a day
at 5%), and names under $5M of 24h volume were most of the rest; leveraged ETFs are left
out of the market alert's count, Asia-listed names are in it.

State (marks, cooldowns, the alert log) is in METHOD_HOME/hl_watch, outside the repo the
mini auto-commits. With METHOD_AUTORESTART=1 (launchd, method/deploy/com.zach.hlwatch.plist)
it exits when its own code changes on disk, so a git pull goes live on the KeepAlive
restart; the mini's frequent data commits don't restart it.
Run: python -m method.hl_watch [--replay HOURS | --test].
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

from method import push

log = logging.getLogger("method.hl_watch")

TZ = ZoneInfo("America/New_York")
API = "https://api.hyperliquid.xyz/info"
DEX = "xyz"
POLL_S = 30
WINDOW_S = 3600
SAVE_EVERY_S = 300
COOLDOWN_S = 3 * 3600
RTH = (datetime.time(9, 30), datetime.time(16, 0))

BASKET_MOVE = 0.02
BREADTH_MOVE = 0.04
BREADTH_N = 6
INDEX_MOVE = 0.0125
NAME_MOVE = 0.05
NAME_MIN_VOL = 5_000_000
EXTEND = {"market": 0.02, "index": 0.01, "name": 0.03}   # a further move this big re-pushes inside the cooldown
RETRY_S = 300                   # a push that reached no phone is tried again after this

INDEXES = ("XYZ100", "SP500")
NON_EQUITY = set("GOLD SILVER CL BRENTOIL NATGAS COPPER PLATINUM PALLADIUM HO URANIUM ALUMINIUM CORN WHEAT "
                 "TTF EUR GBP JPY KRW DXY VIX VOL H100 JP225 KR200 NIFTY IBOV".split())
LEVERAGED = {"SOXL", "KORU", "SNXX"}
ASIA_LISTED = set("ZHIPU MINIMAX CXMT UNITREE CAMBRICON GIGADEV KIOXIA SKHX SMSN HYUNDAI SOFTBANK IBIDEN "
                  "INNOLIGHT AMEC CATL YMTC SHEIN RELIANCE JIO".split())
AI_BASKET = ("NVDA AMD MU SNDK WDC SKHY SKHX SMSN TSM AVGO MRVL ARM INTC AMAT LRCX ASML CRWV NBIS IREN "
             "ORCL DELL LITE AAOI MSFT META GOOGL AMZN PLTR SMH QCOM KIOXIA DRAM").split()
BASKET_MIN_NAMES = 10


def home() -> Path:
    return push.home() / "hl_watch"


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------

_session_days: dict[datetime.date, bool] = {}


def is_session_day(day: datetime.date) -> bool:
    if day not in _session_days:
        try:
            import pandas_market_calendars as mcal
            _session_days[day] = not mcal.get_calendar("NYSE").schedule(start_date=day, end_date=day).empty
        except Exception:                               # no calendar: weekdays
            _session_days[day] = day.weekday() < 5
    return _session_days[day]


def in_rth(now: datetime.datetime) -> bool:
    """Regular hours, when the user is watching the real tape and nothing pushes."""
    return is_session_day(now.date()) and RTH[0] <= now.time() < RTH[1]


# ---------------------------------------------------------------------------
# Detection (pure: marks in, alerts out)
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    kind: str                   # market / index / name
    key: str                    # cooldown key, e.g. "name:RKLB:down"
    size: float                 # the move that triggered it (signed)
    title: str
    body: str
    names: list = field(default_factory=list)      # [(name, move)] the push lists


def window_move(samples: list[tuple[float, float]]) -> float | None:
    """The last mark against the window's high (a drop) or low (a rip), whichever is larger."""
    if len(samples) < 2:
        return None
    px = [p for _, p in samples]
    hi, lo, cur = max(px), min(px), px[-1]
    down, up = cur / hi - 1, cur / lo - 1
    return down if -down >= up else up


def hour_return(samples: list[tuple[float, float]]) -> float | None:
    if len(samples) < 2 or samples[0][1] <= 0:
        return None
    return samples[-1][1] / samples[0][1] - 1


def breadth_eligible(name: str) -> bool:
    return name not in NON_EQUITY and name not in LEVERAGED and name not in INDEXES


def single_eligible(name: str) -> bool:
    return breadth_eligible(name) and name not in ASIA_LISTED


def pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _stamp(now: float) -> str:
    t = datetime.datetime.fromtimestamp(now, TZ)
    return t.strftime("%-I:%M %p ET %a")


def detect(marks: dict[str, list], vols: dict[str, float], sent: dict[str, dict], now: float) -> list[Alert]:
    """Alerts due now. `marks` holds each name's (ts, px) in the window, oldest first;
    `sent` is the cooldown record (key -> {ts, size}), read here and written by the caller
    once a push lands."""
    moves = {n: m for n, s in marks.items() if (m := window_move(s)) is not None}
    rets = {n: r for n, s in marks.items() if (r := hour_return(s)) is not None}

    def due(key: str, kind: str, size: float) -> bool:
        prev = sent.get(key)
        if not prev or now - prev["ts"] >= COOLDOWN_S:
            return True
        return abs(size) >= abs(prev["size"]) + EXTEND[kind]

    alerts: list[Alert] = []
    idx = "  ".join(f"{n} {pct(rets[n])}" for n in INDEXES if n in rets)

    # market: the AI basket's hour, or breadth
    basket = [rets[n] for n in AI_BASKET if n in rets]
    basket_ret = sum(basket) / len(basket) if len(basket) >= BASKET_MIN_NAMES else 0.0
    covered: dict[str, set] = {"down": set(), "up": set()}
    for sign, word in ((-1, "down"), (1, "up")):
        movers = sorted(((n, m) for n, m in moves.items() if breadth_eligible(n) and m * sign >= BREADTH_MOVE),
                        key=lambda x: -abs(x[1]))
        basket_hit = basket_ret * sign >= BASKET_MOVE
        if not basket_hit and len(movers) < BREADTH_N:
            continue
        covered[word] = {n for n, _ in movers}
        size = basket_ret               # either trigger; a further 2% on the basket re-pushes
        key = f"market:{word}"
        if not due(key, "market", size):
            continue
        count = f"{len(movers)} names {word} 4%+"
        title = f"AI stocks {pct(basket_ret)} in an hour" if basket_hit else f"{count} in an hour"
        shown = sorted(movers, key=lambda x: (x[0] not in AI_BASKET, -abs(x[1])))[:6]   # AI names lead
        lead = ", ".join(f"{n} {pct(m)}" for n, m in shown) or "no single name 4%+"
        lev = ", ".join(f"{n} {pct(moves[n])}" for n in sorted(LEVERAGED)
                        if n in moves and moves[n] * sign >= BREADTH_MOVE)
        body = (f"{lead}. {count}, AI basket {pct(basket_ret)}. {lev + '. ' if lev else ''}{idx}. "
                f"Hyperliquid, {_stamp(now)}.")
        alerts.append(Alert("market", key, size, title, body, movers))

    # index
    for n in INDEXES:
        m = moves.get(n)
        if m is None or abs(m) < INDEX_MOVE:
            continue
        word = "down" if m < 0 else "up"
        key = f"index:{n}:{word}"
        if due(key, "index", m):
            others = "  ".join(f"{o} {pct(moves[o])}" for o in INDEXES if o != n and o in moves)
            body = f"{others + '. ' if others else ''}AI basket {pct(basket_ret)}. Hyperliquid, {_stamp(now)}."
            alerts.append(Alert("index", key, m, f"{n} {pct(m)} in an hour", body, [(n, m)]))

    # single names, unless the market is the story: a market alert in the last hour, or
    # the name is in the market move this poll
    if any(k.startswith("market:") and now - v["ts"] < WINDOW_S for k, v in sent.items()):
        return alerts
    for n, m in sorted(moves.items(), key=lambda x: -abs(x[1])):
        if abs(m) < NAME_MOVE or not single_eligible(n) or vols.get(n, 0) < NAME_MIN_VOL:
            continue
        word = "down" if m < 0 else "up"
        if n in covered[word]:
            continue
        key = f"name:{n}:{word}"
        if not due(key, "name", m):
            continue
        px = [p for _, p in marks[n]]
        ref = max(px) if m < 0 else min(px)
        body = (f"{ref:,.2f} to {px[-1]:,.2f}. ${vols.get(n, 0) / 1e6:,.0f}M traded in 24h. "
                f"{idx}. Hyperliquid, {_stamp(now)}.")
        alerts.append(Alert("name", key, m, f"{n} {pct(m)} in an hour", body, [(n, m)]))
    return alerts


def record(alert: Alert, sent: dict, now: float) -> None:
    """Start the cooldown; a market alert also covers each name it listed at 5%+."""
    sent[alert.key] = {"ts": now, "size": alert.size}
    if alert.kind == "market":
        word = alert.key.split(":")[1]
        for n, m in alert.names:
            if abs(m) >= NAME_MOVE and single_eligible(n):
                sent[f"name:{n}:{word}"] = {"ts": now, "size": m}


# ---------------------------------------------------------------------------
# Hyperliquid
# ---------------------------------------------------------------------------

def _post(body: dict, timeout: float = 15) -> object:
    req = urllib.request.Request(API, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def fetch(dex: str = DEX) -> dict[str, tuple[float, float]]:
    """name -> (mark, 24h notional volume) for every live perp on the dex."""
    meta, ctxs = _post({"type": "metaAndAssetCtxs", "dex": dex})
    out = {}
    for asset, ctx in zip(meta["universe"], ctxs):
        if asset.get("isDelisted"):
            continue
        try:
            px = float(ctx["markPx"])
        except (KeyError, TypeError, ValueError):
            continue
        if px > 0:
            out[asset["name"].split(":", 1)[-1]] = (px, float(ctx.get("dayNtlVlm") or 0))
    return out


# ---------------------------------------------------------------------------
# The live loop
# ---------------------------------------------------------------------------

class Watch:
    """The poll-to-poll state. With no state_dir (the replay) nothing touches disk."""

    def __init__(self, state_dir: Path | None, sender=push.send):
        self.dir = state_dir
        self.send = sender
        self.marks: dict[str, list] = {}
        self.vols: dict[str, float] = {}
        self.sent: dict[str, dict] = {}
        self.tried: dict[str, float] = {}
        self.saved_at = 0.0
        self._load()

    def _load(self) -> None:
        if self.dir is None:
            return
        try:
            s = json.loads((self.dir / "state.json").read_text())
        except (OSError, ValueError):
            return
        self.marks = {n: [tuple(x) for x in v] for n, v in s.get("marks", {}).items()}
        self.sent = s.get("sent", {})

    def save(self) -> None:
        if self.dir is None:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / "state.tmp"
        tmp.write_text(json.dumps({"marks": self.marks, "sent": self.sent}))
        os.replace(tmp, self.dir / "state.json")

    def add(self, quotes: dict[str, tuple[float, float]], now: float) -> None:
        cut = now - WINDOW_S
        for n, (px, vol) in quotes.items():
            self.marks.setdefault(n, []).append((now, px))
            self.vols[n] = vol
        for n in list(self.marks):
            self.marks[n] = [x for x in self.marks[n] if x[0] >= cut]
            if not self.marks[n]:
                del self.marks[n]

    def step(self, quotes: dict[str, tuple[float, float]], now: float) -> list[Alert]:
        """One poll: quiet (and forget the window) in regular hours, else detect and push."""
        if in_rth(datetime.datetime.fromtimestamp(now, TZ)):
            self.marks.clear()          # after the close, moves count from the close
            return []
        self.add(quotes, now)
        fired = []
        for a in detect(self.marks, self.vols, self.sent, now):
            if now - self.tried.get(a.key, 0) < RETRY_S:
                continue
            self.tried[a.key] = now
            try:
                n = self.send(a.title, a.body, url="/#/bounce", tag=a.key)
            except Exception as e:
                log.warning(f"push failed: {e}")
                n = 0
            self._log(a, now, n)
            if n:
                record(a, self.sent, now)
                fired.append(a)
        for k in [k for k, v in self.sent.items() if now - v["ts"] > 2 * COOLDOWN_S]:
            del self.sent[k]
        if fired or now - self.saved_at >= SAVE_EVERY_S:
            self.save()
            self.saved_at = now
        return fired

    def _log(self, a: Alert, now: float, sent: int) -> None:
        if self.dir is None:
            return
        log.info(f"ALERT {a.key} {pct(a.size)} pushed={sent} | {a.title} | {a.body}")
        self.dir.mkdir(parents=True, exist_ok=True)
        with (self.dir / "alerts.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": round(now), "kind": a.kind, "key": a.key, "size": round(a.size, 4),
                                 "title": a.title, "body": a.body, "pushed": sent}) + "\n")


def _code_stamp() -> tuple[float, ...]:
    return tuple(Path(f).stat().st_mtime for f in (__file__, push.__file__))


def serve(poll_s: int = POLL_S) -> None:
    w = Watch(home())
    failures, last_note = 0, 0.0
    code, restart = _code_stamp(), os.environ.get("METHOD_AUTORESTART") == "1"
    log.info(f"Hyperliquid watch up: dex {DEX}, every {poll_s}s, state {w.dir}")
    while True:
        if restart and _code_stamp() != code:
            w.save()
            log.info("code changed on disk; exiting so launchd restarts it")
            return
        now = time.time()
        quiet = in_rth(datetime.datetime.fromtimestamp(now, TZ))
        try:
            w.step({} if quiet else fetch(), now)
            failures = 0
        except Exception as e:                          # a bad poll never ends the watch
            failures += 1
            log.warning(f"poll failed ({failures} in a row): {e!r}")
        if now - last_note >= 600:
            moves = sorted(((n, m) for n, s in w.marks.items() if (m := window_move(s)) is not None),
                           key=lambda x: -abs(x[1]))[:5]
            log.info("quiet (regular hours)" if quiet else
                     f"{len(w.marks)} names; biggest hour: " + ", ".join(f"{n} {pct(m)}" for n, m in moves))
            last_note = now
        time.sleep(max(1.0, poll_s - (time.time() - now)) if failures < 5 else min(300, poll_s * failures))


# ---------------------------------------------------------------------------
# Replay: the same detector over 5-min bars, no pushes
# ---------------------------------------------------------------------------

def rolling_volume(candles: list[dict]) -> dict[int, tuple[float, float]]:
    """bar start -> (close, notional traded in the 24h to that bar), from 5-min candles."""
    out, window, total = {}, [], 0.0
    for b in sorted(candles, key=lambda b: b["t"]):
        notional = float(b["v"]) * float(b["c"])
        window.append((b["t"], notional))
        total += notional
        while window[0][0] <= b["t"] - 24 * 3600 * 1000:
            total -= window.pop(0)[1]
        out[b["t"]] = (float(b["c"]), total)
    return out


def replay(hours: float, dex: str = DEX) -> list[tuple[float, Alert]]:
    meta, ctxs = _post({"type": "metaAndAssetCtxs", "dex": dex})
    names = [a["name"] for a, c in zip(meta["universe"], ctxs)
             if not a.get("isDelisted") and a["name"].split(":", 1)[-1] not in NON_EQUITY
             and (float(c.get("dayNtlVlm") or 0) > 0 or float(c.get("openInterest") or 0) > 0)]
    end = int(time.time() * 1000)
    start = end - int((hours + 24) * 3600 * 1000)          # a day of lead-in for 24h volume
    bars: dict[str, dict[int, tuple[float, float]]] = {}
    for full in names:
        for attempt in range(4):
            try:
                c = _post({"type": "candleSnapshot",
                           "req": {"coin": full, "interval": "5m", "startTime": start, "endTime": end}}, timeout=30)
                break
            except urllib.error.HTTPError as e:
                if e.code != 429:
                    raise
                time.sleep(10 * (attempt + 1))
        else:
            continue
        bars[full.split(":", 1)[-1]] = rolling_volume(c)
        time.sleep(0.8)
    stamps = sorted({t for s in bars.values() for t in s if t >= end - hours * 3600 * 1000})
    log.info(f"replaying {len(stamps)} bars across {len(bars)} names")
    fired = []
    w = Watch(None, sender=lambda *a, **k: 1)
    for t in stamps:
        quotes = {n: s[t] for n, s in bars.items() if t in s}
        for a in w.step(quotes, t / 1000):
            fired.append((t / 1000, a))
    return fired


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--replay", type=float, metavar="HOURS", help="run the detector over recent 5-min bars, no pushes")
    ap.add_argument("--test", action="store_true", help="send one sample push to the phone")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.test:
        n = push.send("AI stocks -3.4% in an hour",
                      "TEST from the Hyperliquid watch. SKHX -6.9%, MU -6.9%, SNDK -6.4%. XYZ100 -0.8%.",
                      url="/#/bounce", tag="hl-test")
        print(f"sent to {n} device(s)")
    elif args.replay:
        for ts, a in replay(args.replay):
            print(f"{datetime.datetime.fromtimestamp(ts, TZ):%a %m-%d %H:%M}  [{a.kind}] {a.title}\n    {a.body}")
    else:
        serve()


if __name__ == "__main__":
    main()
