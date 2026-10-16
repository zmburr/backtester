"""
Method — pure pieces shared by the refresh job and the server: the clock
(which mode, which session), the open scenarios, the history episodes and
JSON cleaning. No network, no heavy imports.
"""

from __future__ import annotations

import datetime
import json
import math
import os
from pathlib import Path

# Mode windows (ET) and how often a pass is worth running in each.
PREMARKET = (datetime.time(4, 0), datetime.time(9, 30))
MARKET = (datetime.time(9, 30), datetime.time(16, 0))
EVENING = (datetime.time(16, 30), datetime.time(20, 0))   # daily bars settle after the close
OVERNIGHT = (datetime.time(20, 0), datetime.time(4, 0))     # Blue Ocean: Sun-Thu nights
CADENCE_MIN = {"premarket": 5, "market": 15, "evening": 15, "overnight": 15}

MIN_GO = 2                      # same episode gate as the playbook email
REVERSAL_GAP_GATE = 0.04        # 3DGapFade gap gate (analyzers/reversal_pretrade.py)

SCENARIO_TEXT = {       # bounce names use the game plan's grades instead (method/ladder.py)
    "reversal": {
        "A": "exhaustion gap (+4% or more)",
        "B": "flat or small gap: wait for the VWAP fail",
        "C": "gapping down: GapDownTrendBreak rules",
    },
}


def mode_for(now: datetime.datetime, session_day: bool) -> str:
    """premarket / market / evening on a trading day, else closed."""
    if not session_day:
        return "closed"
    t = now.time()
    for name, (start, end) in (("premarket", PREMARKET), ("market", MARKET),
                               ("evening", EVENING)):
        if start <= t < end:
            return name
    return "closed"


def is_overnight(now: datetime.datetime) -> bool:
    """8 PM-4 AM on the nights Blue Ocean trades: Sunday through Thursday evenings and
    the early hours of Monday through Friday."""
    t, wd = now.time(), now.weekday()           # Monday = 0 ... Sunday = 6
    if t >= OVERNIGHT[0]:
        return wd in (6, 0, 1, 2, 3)
    if t < OVERNIGHT[1]:
        return wd in (0, 1, 2, 3, 4)
    return False


def session_for(mode: str, today: datetime.date, next_trading_day) -> datetime.date:
    """The session Method is planning: tomorrow's in the evening, else today's."""
    return next_trading_day(today) if mode == "evening" else today


def due(mode: str, last_generated: datetime.datetime | None, now: datetime.datetime) -> bool:
    """Cron fires every 5 min; a mode with a slower cadence skips the passes
    in between. A minute of slack absorbs cron jitter."""
    if last_generated is None:
        return True
    gap = (now - last_generated).total_seconds() / 60
    return gap >= CADENCE_MIN.get(mode, 5) - 1 or gap < 0


def reversal_scenario(gap: float | None) -> str | None:
    if gap is None:
        return None
    if gap >= REVERSAL_GAP_GATE:
        return "A"
    return "B" if gap >= 0 else "C"


def clean(obj):
    """JSON-safe: numpy scalars -> float/int, NaN/inf -> None, recursively."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            obj = obj.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    return obj


def write_json(path: Path, data) -> None:
    """Atomic: the server never reads a half-written snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(data), indent=1))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


# ---------------------------------------------------------------------------
# History: past setup days and how they played out
# ---------------------------------------------------------------------------

def to_num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def index_outcomes(outcomes: list[dict]) -> dict:
    """Scorecard rows keyed (target_date, bucket, ticker); morning beats evening
    when the scorecard has both for one target."""
    by_key = {}
    for row in outcomes:
        key = (row.get("target_date"), row.get("bucket"), row.get("ticker"))
        if key not in by_key or row.get("session") == "morning":
            by_key[key] = row
    return by_key


def build_episodes(reports: dict, outcomes: list[dict], next_trading_day,
                   min_go: int = MIN_GO) -> list[dict]:
    """reports: {(date_str, session): report json}. A report with 2+ GO on a
    side opens an episode for the session it plans (morning -> same day,
    evening -> next session); reports planning the same session merge.
    Outcomes are the scorecard rows (data/signal_outcomes.csv) for that
    target session, side and ticker."""
    episodes: dict[tuple[str, str], dict] = {}
    for (date_str, session), report in sorted(reports.items()):
        for side in ("bounce", "reversal"):
            sigs = [s for s in report.get("signals", []) if s.get("bucket") == side]
            go = [s["ticker"] for s in sigs if s.get("recommendation") == "GO"]
            if len(go) < min_go:
                continue
            day = datetime.date.fromisoformat(date_str)
            target = (next_trading_day(day) if session == "evening" else day).isoformat()
            ep = episodes.setdefault((target, side), {
                "target": target, "side": side, "reports": [], "names": {}})
            ep["reports"].append(f"{date_str} {session}")
            for s in sigs:
                rec = s.get("recommendation")
                if rec not in ("GO", "CAUTION"):
                    continue
                prev = ep["names"].get(s["ticker"])
                if prev is None or (prev["rec"] != "GO" and rec == "GO"):
                    ep["names"][s["ticker"]] = {"ticker": s["ticker"], "rec": rec,
                                                "score": s.get("score"), "cap": s.get("cap")}

    by_key = index_outcomes(outcomes)

    out = []
    for (target, side), ep in episodes.items():
        names = []
        for n in sorted(ep["names"].values(), key=lambda n: (n["rec"] != "GO", n["ticker"])):
            o = by_key.get((target, side, n["ticker"]), {})
            names.append({**n,
                          "d0_pct": to_num(o.get("d0_pct")),
                          "d1_pct": to_num(o.get("d1_pct")),
                          "mfe_atr_d0": to_num(o.get("mfe_atr_d0")),
                          "exit_r": to_num(o.get("exit_r")),
                          "tradeable_d0": str(o.get("tradeable_d0")) == "True" if o else None})
        n_go = sum(1 for n in names if n["rec"] == "GO")
        out.append({"target": target, "side": side, "reports": ep["reports"],
                    "n_go": n_go, "names": names})
    out.sort(key=lambda e: (e["target"], e["side"]), reverse=True)
    return out
