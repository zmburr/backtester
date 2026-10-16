"""
Real-time prices from the PC (Trillium), laid over Method's snapshot.

The mini can't reach Trillium; the PC can. method/pc_pusher.py summarizes each
name's 1-min bars with summarize_bars() and posts them to /api/live every 15 s
(active names) or 2 min (the whole board). The server re-grades those rows on
the spot (refresh.apply_live), and the cron refresh prefers a name's feed while
it's under FRESH_S old. When the PC sleeps, rows fall back to Polygon's
15-min-delayed bars on the next refresh.

Polygon and Trillium bars go through the same summarize_bars(), so a row looks
the same whichever feed it came from. Bars are indexed by bar START time (ET);
Trillium stamps the close, so the pusher shifts its bars back a minute.
"""

from __future__ import annotations

import datetime
import hmac
import secrets

import pandas as pd

TZ = "US/Eastern"
FRESH_S = 120                    # a name's feed older than this falls back to Polygon
SRC_LABEL = {"T": "Trillium", "P": "Polygon, delayed"}


def summarize_bars(bars: pd.DataFrame, day: str, asof=None) -> dict | None:
    """What a Method row needs from one day of 1-min bars: the last price and bar,
    premarket low/high, the regular-session open, low and close, and the after-hours low."""
    if bars is None or bars.empty:
        return None
    if asof is not None:
        bars = bars[bars.index <= pd.Timestamp(asof)]
        if bars.empty:
            return None
    open_ts = pd.Timestamp(f"{day} 09:30", tz=TZ)
    close_ts = pd.Timestamp(f"{day} 16:00", tz=TZ)
    pm = bars[bars.index < open_ts]
    rth = bars[(bars.index >= open_ts) & (bars.index < close_ts)]
    ah = bars[bars.index >= close_ts]
    last = bars.index[-1]
    out = {"price": float(bars["close"].iloc[-1]), "bar_time": last.strftime("%H:%M"),
           "bar_ts": last.isoformat()}
    if not pm.empty:
        out.update(pm_low=float(pm["low"].min()), pm_low_time=pm["low"].idxmin().strftime("%H:%M"),
                   pm_high=float(pm["high"].max()))
    if not rth.empty:
        out.update(open=float(rth["open"].iloc[0]), day_low=float(rth["low"].min()),
                   day_low_time=rth["low"].idxmin().strftime("%H:%M"),
                   rth_close=float(rth["close"].iloc[-1]))
    if not ah.empty:
        out["ah_low"] = float(ah["low"].min())
    return out


def merge_feed(feed: dict | None, day: str, names: dict, received_at: datetime.datetime) -> dict:
    """Fold one post into the stored feed. Posts carry a subset of names (the active
    ones every 15 s), so each name keeps its own receive time; a new day starts clean."""
    feed = feed if feed and feed.get("day") == day else {"day": day, "names": {}}
    stamp = received_at.isoformat(timespec="seconds")
    for t, summ in names.items():
        feed["names"][t] = dict(summ, received_at=stamp)
    feed["received_at"] = stamp
    return feed


def fresh_names(feed: dict | None, day: str, now: datetime.datetime) -> dict:
    """{ticker: summary} for the names posted within FRESH_S, today only."""
    if not feed or feed.get("day") != day:
        return {}
    out = {}
    for t, summ in (feed.get("names") or {}).items():
        try:
            age = (now - datetime.datetime.fromisoformat(summ["received_at"])).total_seconds()
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= age <= FRESH_S:
            out[t] = summ
    return out


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and 0 < v < 1e7


def clean_post(body: dict) -> tuple[str, dict]:
    """Validate a pusher post: {"day": "YYYY-MM-DD", "names": {ticker: summary}}.
    Raises ValueError on anything malformed; drops names with a bad price."""
    day = str(body.get("day") or "")
    datetime.date.fromisoformat(day)
    names = body.get("names")
    if not isinstance(names, dict) or len(names) > 500:
        raise ValueError("names must be a dict of at most 500 tickers")
    keep = ("price", "bar_time", "bar_ts", "pm_low", "pm_low_time", "pm_high", "open",
            "day_low", "day_low_time", "rth_close", "ah_low")
    out = {}
    for t, summ in names.items():
        t = str(t).upper()
        if not (t.isascii() and t.replace(".", "").isalnum() and len(t) <= 10):
            continue
        if not isinstance(summ, dict) or not _num(summ.get("price")):
            continue
        s = {k: summ[k] for k in keep if summ.get(k) is not None}
        if any(k in s and not _num(s[k]) for k in ("pm_low", "pm_high", "open", "day_low", "rth_close", "ah_low")):
            continue
        s["price"] = float(s["price"])
        out[t] = s
    return day, out


# ---------------------------------------------------------------- token
# The login gate already limits Method to the owner's devices; the token also
# limits price posts to the PC that holds a copy of it.

def ensure_token(path) -> str:
    try:
        tok = path.read_text().strip()
        if tok:
            return tok
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tok = secrets.token_urlsafe(32)
    path.write_text(tok)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return tok


def token_ok(path, given: str | None) -> bool:
    try:
        want = path.read_text().strip()
    except OSError:
        return False
    return bool(want) and bool(given) and hmac.compare_digest(want, given)
