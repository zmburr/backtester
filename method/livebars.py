"""
Today's 1-min bars from the PC (Trillium), kept for Method's live chart.

The pusher already pulls each name's whole session from Trillium every pass; it now
also posts the bars the server doesn't have yet (method/pc_pusher.py). The server
keeps them per name, in memory and appended to data/method/live_bars/<day>/<TK>.jsonl,
so an auto-restart doesn't lose the morning. Every post's reply says how far each
name goes (`have`); the pusher sends from that bar on, so the bar that was still
forming last time is replaced, and a restart on either side heals itself.

Bars are [epoch, open, high, low, close, volume], epoch = the bar's START in UTC
seconds (bars_frame already moved SHEL's close-time stamps back a minute).
/api/bars serves them in the chart's shape (chart_rows) while the name is fresh.
"""

from __future__ import annotations

import datetime
import json
import threading
import time
from pathlib import Path

import pandas as pd

TZ = "US/Eastern"
FRESH_S = 300                    # the chart stays on Trillium while a name was posted this recently
KEEP_DAYS = 14


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.day: str | None = None
        self.bars: dict[str, dict[int, list]] = {}
        self.seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def _open(self, day: str) -> None:
        """Switch to `day`, reading what's on disk (later lines of a file win)."""
        if self.day == day:
            return
        self.day, self.bars, self.seen = day, {}, {}
        for p in sorted((self.root / day).glob("*.jsonl")):
            rows: dict[int, list] = {}
            try:
                for line in p.read_text().splitlines():
                    r = json.loads(line)
                    rows[int(r[0])] = r
                self.seen[p.stem] = p.stat().st_mtime
            except (OSError, ValueError, TypeError, IndexError):
                continue
            self.bars[p.stem] = rows
        self._prune(day)

    def _prune(self, day: str) -> None:
        try:
            cutoff = (datetime.date.fromisoformat(day) - datetime.timedelta(days=KEEP_DAYS)).isoformat()
            for d in self.root.glob("????-??-??"):
                if d.name < cutoff:
                    for p in d.glob("*"):
                        p.unlink()
                    d.rmdir()
        except (OSError, ValueError):
            pass

    def merge(self, day: str, posted: dict[str, list], now_s: float | None = None) -> dict:
        """Upsert each name's posted bars by bar time; append only what changed. Returns `have`."""
        now_s = time.time() if now_s is None else now_s
        with self._lock:
            self._open(day)
            for t, rows in posted.items():
                cur = self.bars.setdefault(t, {})
                new = [r for r in rows if cur.get(int(r[0])) != r]
                for r in new:
                    cur[int(r[0])] = r
                if new:
                    try:
                        (self.root / day).mkdir(parents=True, exist_ok=True)
                        with (self.root / day / f"{t}.jsonl").open("a") as fh:
                            fh.write("".join(json.dumps(r) + "\n" for r in new))
                    except OSError:
                        pass                 # memory still has them; the chart works until a restart
                self.seen[t] = now_s
            return self._have()

    def _have(self) -> dict:
        return {t: max(rows) for t, rows in self.bars.items() if rows}

    def have(self, day: str) -> dict:
        with self._lock:
            self._open(day)
            return self._have()

    def rows(self, day: str, ticker: str) -> list[list]:
        with self._lock:
            self._open(day)
            cur = self.bars.get(ticker) or {}
            return [cur[k] for k in sorted(cur)]

    def fresh(self, day: str, ticker: str, now_s: float | None = None) -> bool:
        now_s = time.time() if now_s is None else now_s
        with self._lock:
            self._open(day)
            return bool(self.bars.get(ticker)) and 0 <= now_s - self.seen.get(ticker, 0) <= FRESH_S


def chart_rows(rows: list[list]) -> list[dict]:
    """Stored bars in /api/bars's intraday shape: times ET-shifted, as the chart's UTC axis
    reads ET wall-clock time (server._et_epoch)."""
    if not rows:
        return []
    idx = pd.to_datetime([r[0] for r in rows], unit="s", utc=True).tz_convert(TZ).tz_localize(None)
    shift = (idx.asi8 // 10**9).tolist()
    return [{"time": int(e), "open": r[1], "high": r[2], "low": r[3], "close": r[4], "value": r[5]}
            for e, r in zip(shift, rows)]


def frame(rows: list[list], now=None) -> pd.DataFrame | None:
    """Stored bars as 1-min bars indexed by bar START (ET), for the rules' replay (method/review.py).
    A minute still forming is left out, and so is the first minute of a 2-min bar that hasn't
    closed (method/watch.two_min's rule), so the replay never trades a half-built bar."""
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume"])
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("t"), unit="s", utc=True).dt.tz_convert(TZ))
    df = df.astype(float)
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
    now = pd.Timestamp.now(tz=TZ) if now is None else pd.Timestamp(now)
    df = df[df.index + pd.Timedelta(minutes=1) <= now]
    if df.empty:
        return None
    last = df.index[-1]
    m = int((last - last.normalize()).total_seconds() // 60) - 570
    if 0 <= m < 390 and m % 2 == 0 and last + pd.Timedelta(minutes=3) > now:
        df = df.iloc[:-1]
    return df if not df.empty else None


def with_today(daily: list[dict], intraday: list[dict], day: str) -> list[dict]:
    """The daily bars with today's candle made from the live regular-session bars."""
    rth = [b for b in intraday if 570 <= (b["time"] // 60) % 1440 < 960]
    if not rth:
        return daily
    today = {"time": day, "open": rth[0]["open"], "high": max(b["high"] for b in rth),
             "low": min(b["low"] for b in rth), "close": rth[-1]["close"],
             "value": float(sum(b["value"] or 0 for b in rth))}
    return [b for b in daily if b.get("time") != day] + [today]
