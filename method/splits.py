"""Stock splits between a name's last completed daily bar and the session Method grades.

Method builds a name's daily context (statics) once per session, the evening before.
Polygon lists a split days ahead but adjusts its bars only once the split has executed,
so a name splitting at the session's open was graded on old-share history against
new-share prints: SNDK's 8-for-1 on 11/20/2026 read as a -89% gap and an A+ (prior
close 755.16 = 94.40 in new shares; it opened near 90, a -5% gap). Everything here
puts prices in the session's share basis.

- `pending(session)`: the splits executing in the days up to the session (one Polygon
  call, cached in splits_<session>.json for TTL).
- `adjust(ticker, hist, session)`: completed daily bars in new shares when a split falls
  between their last bar and the session and Polygon hasn't applied it yet, plus a note
  for the static (`s["split"]`).
- `to_session` / `nearest`: evening and overnight prints from before the split executes,
  in the session's basis.
"""

from __future__ import annotations

import datetime
import logging
import math
import os
from pathlib import Path

import pandas as pd

from method import core

logger = logging.getLogger("method.splits")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TZ = "US/Eastern"
TTL = datetime.timedelta(minutes=30)    # a split Polygon lists late still lands before the next session
LOOKBACK_DAYS = 10                      # the last completed bar can sit a long weekend before the session
APPLIED_TOL = 0.005                     # Polygon's adjusted close this far off the raw one: it applied the split
PRICE_COLS = ("open", "high", "low", "close", "vwap")
SUMMARY_PRICES = ("price", "rth_close", "ah_low", "pm_low", "pm_high", "open", "day_low")


def _method_dir() -> Path:
    return Path(os.environ.get("METHOD_DIR") or DATA_DIR / "method")


def _client():
    from data_queries.polygon_queries import poly_client
    return poly_client


def _num(x: float) -> str:
    return f"{x:g}"


def ratio_label(split_from: float, split_to: float) -> str:
    """Polygon's 1 -> 8 is an 8-for-1 split; 10 -> 1 is a 1-for-10 reverse split."""
    return f"{_num(split_to)}-for-{_num(split_from)}"


def pending(session: str, client=None, now: datetime.datetime | None = None) -> dict[str, list[dict]]:
    """{ticker: [{"date", "factor", "ratio"}]}: splits executing in the LOOKBACK_DAYS up to
    and including `session`. `factor` turns an old-share price into a new-share one
    (8-for-1 = 0.125). On a failed call the last list stands (none beats a wrong one only
    when there was never a list)."""
    path = _method_dir() / f"splits_{session}.json"
    at = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz=TZ)
    cached = core.read_json(path)
    if cached and at - pd.Timestamp(cached["fetched_at"]) < TTL:
        return cached["splits"]
    lo = (datetime.date.fromisoformat(session) - datetime.timedelta(days=LOOKBACK_DAYS)).isoformat()
    out: dict[str, list[dict]] = {}
    try:
        for s in (client or _client()).list_splits(execution_date_gte=lo, execution_date_lte=session, limit=1000):
            if not (s.ticker and s.split_from and s.split_to and s.execution_date):
                continue
            out.setdefault(s.ticker, []).append({
                "date": s.execution_date, "factor": float(s.split_from) / float(s.split_to),
                "ratio": ratio_label(s.split_from, s.split_to)})
    except Exception as e:
        logger.warning(f"splits up to {session}: {e}")
        return (cached or {}).get("splits") or {}
    for v in out.values():
        v.sort(key=lambda x: x["date"])
    core.write_json(path, {"session": session, "fetched_at": at.isoformat(), "splits": out})
    return out


def bar_date(hist: pd.DataFrame | None) -> str | None:
    """The last completed bar's date, when the frame has a date index."""
    if hist is None or hist.empty or not isinstance(hist.index, pd.DatetimeIndex):
        return None
    return hist.index[-1].date().isoformat()


def between(ticker: str, after: str | None, session: str, splits: dict | None = None) -> list[dict]:
    """The ticker's splits executing after `after` (a bar date) up to the session."""
    if after is None:
        return []
    found = (pending(session) if splits is None else splits).get(ticker) or []
    return [s for s in found if after < s["date"] <= session]


def note_for(hits: list[dict], by: str) -> dict:
    return {"date": hits[-1]["date"], "factor": math.prod(h["factor"] for h in hits),
            "ratio": " then ".join(h["ratio"] for h in hits), "by": by}


def raw_close(ticker: str, day: str, client=None) -> float | None:
    """Polygon's unadjusted close for one day."""
    aggs = list((client or _client()).list_aggs(ticker, 1, "day", day, day, adjusted=False))
    return float(aggs[-1].close) if aggs else None


def adjust(ticker: str, hist: pd.DataFrame | None, session: str, splits: dict | None = None,
           raw=None) -> tuple[pd.DataFrame | None, dict | None]:
    """Completed daily bars before `session` in the session's shares, and the split note
    (None when no split falls between the last bar and the session). Polygon adjusts only
    once a split executes, so the evening before it the bars are still in old shares: this
    checks the raw close and scales prices (and volume the other way) itself. Raises when
    the raw close can't be read, so the static is retried rather than built wrong."""
    hits = between(ticker, bar_date(hist), session, splits)
    if not hits:
        return hist, None
    last = bar_date(hist)
    r = (raw or raw_close)(ticker, last)
    if r is None:
        raise RuntimeError(f"{ticker}: no raw close for {last} to check a {hits[-1]['ratio']} split")
    if abs(float(hist["close"].iloc[-1]) / r - 1) > APPLIED_TOL:
        return hist, note_for(hits, "polygon")          # Polygon's bars are already in new shares
    note = note_for(hits, "method")
    f = note["factor"]
    out = hist.copy()
    for c in PRICE_COLS:
        if c in out.columns:
            out[c] = out[c] * f
    if "volume" in out.columns:
        out["volume"] = out["volume"] / f
    logger.info(f"{ticker}: {note['ratio']} split {note['date']}: daily bars put in new shares (x{f:g})")
    return out, note


def nearest(price: float | None, ref: float | None, factor: float | None) -> float | None:
    """`price` or `price * factor`, whichever sits nearer `ref` (the session's prior close).
    For prints from before the split executes, whose share basis isn't certain (Blue Ocean
    the night before, an evening fallback): a real move would have to be about
    sqrt(ratio) times to be misread, which a 2-for-1 night doesn't see."""
    if not price or not ref or not factor or factor == 1:
        return price
    alt = price * factor
    return alt if abs(math.log(alt / ref)) < abs(math.log(price / ref)) else price


def to_session(b: dict | None, s: dict) -> dict | None:
    """An evening bar summary (today's after hours, the day before the split executes) in
    the session's basis: every price scaled when the split brings it nearer the static's
    prior close."""
    sp = s.get("split")
    if not b or not sp or b.get("price") is None:
        return b
    probe = b.get("rth_close") or b["price"]
    if nearest(probe, s.get("prior_close"), sp["factor"]) == probe:
        return b
    f = sp["factor"]
    return {**b, **{k: b[k] * f for k in SUMMARY_PRICES if b.get(k) is not None}}


def needs_rebuild(ticker: str, s: dict, session: str, splits: dict) -> bool:
    """A cached static built before a split it should have adjusted for (statics cached
    before this check carry no `last_bar`: rebuilt once if the name splits)."""
    if not splits.get(ticker):
        return False
    if "last_bar" not in s:
        return True
    hits = between(ticker, s["last_bar"], session, splits)
    return bool(hits) and (s.get("split") or {}).get("date") != hits[-1]["date"]


def warning(s: dict) -> str | None:
    sp = s.get("split")
    if not sp:
        return None
    day = datetime.date.fromisoformat(sp["date"])
    return f"{sp['ratio']} split {day.month}/{day.day}: prior prices in new shares"
