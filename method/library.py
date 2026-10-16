"""
Method's Best trades: your graded bounce and reversal trades (data/bounce_data.csv,
data/reversal_data.csv) replayed with your systems on them.

Each row runs once through method/review.review(): the bounce grade at the open
(the game plan ladder), the rules' entries and exits and their R. The list
(data/method/library.json) carries what the tab sorts and filters on; the
replays land in the review cache, so opening a trade is instant.

cached_review() is the one cache rule for every replay (History and Best):
past days only, the current review VERSION, complete (the next day is over),
and either a scorecard row (its ATR sets the R) or old enough that none is coming.

Usage:
    python -m method.library            # replay rows not built yet (nightly cron)
    python -m method.library --full     # rebuild every row
    python -m method.library --limit 5  # a few, for a check
"""

from __future__ import annotations

import argparse
import csv
import datetime
import logging
import math
import time
from pathlib import Path

import pandas as pd

from method import core
from method import review as rv
from method.refresh import DATA_DIR, METHOD_DIR

logger = logging.getLogger("method.library")

TZ = "US/Eastern"
LIBRARY = METHOD_DIR / "library.json"
REVIEW_DIR = METHOD_DIR / "review"
OUTCOMES = DATA_DIR / "signal_outcomes.csv"
SOURCES = {"bounce": DATA_DIR / "bounce_data.csv", "reversal": DATA_DIR / "reversal_data.csv"}
REVIEW_CACHE_AFTER_DAYS = 5      # past days without a scorecard row are cached once they're this old
OUTCOME_KEYS = ("d0_pct", "d1_pct", "mfe_atr_d0", "exit_r", "entry_r", "atr_pct", "entry_open")

# "Your log": the CSV row's own record of the day, (column, label, format)
LOG_FIELDS = {
    "bounce": [
        ("Setup", "Setup", "text"), ("gap_pct", "Gap", "pct"),
        ("bounce_open_high_pct", "Open → high", "pct"), ("bounce_open_low_pct", "Open → low", "pct"),
        ("bounce_open_close_pct", "Open → close", "pct"), ("bounce_open_to_day_after_open_pct", "Next open", "pct"),
        ("time_of_low_price", "Low at", "time"), ("time_of_bounce", "Bounce high at", "time"),
        ("consecutive_down_days", "Down days", "int"), ("selloff_total_pct", "Selloff", "pct"),
        ("pct_off_30d_high", "Off 30d high", "pct"), ("atr_pct", "ATR", "upct"),
        ("spy_open_close_pct", "SPY that day", "pct"),
    ],
    "reversal": [
        ("setup", "Setup", "text"), ("intraday_setup", "Intraday", "text"), ("gap_pct", "Gap", "pct"),
        ("reversal_open_low_pct", "Open → low", "pct"), ("reversal_open_close_pct", "Open → close", "pct"),
        ("reversal_open_post_low_pct", "Low → close", "pct"),
        ("reversal_open_to_day_after_open_pct", "Next open", "pct"),
        ("time_of_high_price", "High at", "time"), ("time_of_reversal", "Turned at", "time"),
        ("time_of_low", "Low at", "time"), ("consecutive_up_days", "Up days", "int"),
        ("pct_from_9ema", "Above 9EMA", "pct"), ("atr_pct", "ATR", "upct"),
        ("spy_open_close_pct", "SPY that day", "pct"),
    ],
}


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _fmt(v, kind: str) -> str | None:
    if v is None or (isinstance(v, float) and math.isnan(v)) or v == "":
        return None
    if kind == "text":
        return str(v)
    if kind == "time":
        try:
            return pd.Timestamp(v).strftime("%H:%M")
        except (TypeError, ValueError):
            return None
    f = _num(v)
    if f is None:
        return None
    if kind == "pct":
        return f"{'+' if f >= 0 else '−'}{abs(f) * 100:.1f}%"
    if kind == "upct":
        return f"{f * 100:.1f}%"
    return str(int(round(f)))


# ---------------------------------------------------------------- the shared replay cache

def outcome_fields(row: dict | None) -> dict | None:
    return None if row is None else {k: core.to_num(row.get(k)) for k in OUTCOME_KEYS}


def load_outcome_index(path: Path = OUTCOMES) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("cohort") != "control"]
    return core.index_outcomes(rows)


def cached_review(side: str, ticker: str, date: str, review_dir: Path, cap: str | None = None,
                  row: dict | None = None, today: datetime.date | None = None) -> dict:
    """One day replayed, from the cache when it holds a current, complete copy."""
    day = datetime.date.fromisoformat(date)
    today = today or pd.Timestamp.now(tz=TZ).date()
    path = review_dir / f"{side}_{ticker}_{date}.json"
    if day < today:
        data = core.read_json(path)
        if data and data.get("v") == rv.VERSION:
            return data
    if cap is None:
        from scanners.premarket_bounce_scanner import CAP_CACHE_FILE, _load_json, get_cap
        cap = get_cap(ticker, _load_json(CAP_CACHE_FILE))
    data = core.clean(rv.review(ticker, date, side, cap, outcome=outcome_fields(row), live=day >= today))
    old_enough = (today - day).days > REVIEW_CACHE_AFTER_DAYS
    if day < today and data.get("candles") and data.get("complete") and (row is not None or old_enough):
        core.write_json(path, data)
    return data


# ---------------------------------------------------------------- the trades

def load_trades(side: str, path: Path | None = None) -> list[dict]:
    """The CSV's rows as list entries: who, when, your grade, the move, your log."""
    df = pd.read_csv(path or SOURCES[side])
    dates = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    out, seen = [], set()
    for (_, r), d in zip(df.iterrows(), dates):
        if pd.isna(d) or not isinstance(r.get("ticker"), str):
            continue
        ticker, date = r["ticker"].strip().upper(), d.date().isoformat()
        key = f"{side}:{ticker}:{date}"
        if key in seen:
            continue
        seen.add(key)
        atr = _num(r.get("atr_pct"))
        if side == "bounce":
            move, close = _num(r.get("bounce_open_high_pct")), _num(r.get("bounce_open_close_pct"))
            nxt, setup = _num(r.get("bounce_open_to_day_after_open_pct")), r.get("Setup")
        else:
            low = _num(r.get("reversal_open_low_pct"))
            move = -low if low is not None else None        # how far it fell: bigger is better
            close, nxt, setup = _num(r.get("reversal_open_close_pct")), _num(r.get("reversal_open_to_day_after_open_pct")), r.get("setup")
        log = [[label, v] for col, label, kind in LOG_FIELDS[side] if (v := _fmt(r.get(col), kind))]
        out.append({
            "key": key, "side": side, "ticker": ticker, "date": date,
            "grade": str(r.get("trade_grade") or "").strip() or None, "cap": r.get("cap") or None,
            "setup": setup if isinstance(setup, str) else None,
            "move": move, "move_atr": move / atr if move is not None and atr else None,
            "open_close": close, "next_open": nxt, "atr_pct": atr, "log": log,
        })
    return out


def system_fields(side: str, data: dict) -> dict:
    """What your systems said that day, from the replay."""
    o = data.get("at_open") or {}
    trades = data.get("trades") or []
    sysd = {"gap": o.get("gap"), "rules_r": data.get("rules_r"), "n_trades": len(trades),
            "ok": bool(data.get("candles")), "error": data.get("error")}
    if side == "bounce":
        sysd.update(sys_grade=o.get("grade"), sys_score=o.get("score"), sys_rec=o.get("rec"),
                    sys_intensity=o.get("intensity"), sys_setup=(data.get("ladder") or {}).get("setup_type"))
    elif o.get("gap") is not None:
        sysd["sys_scenario"] = core.reversal_scenario(o["gap"])
    return sysd


def build(full: bool = False, limit: int | None = None, pause: float = 0.2,
          sources: dict | None = None) -> dict:
    """Replay what isn't built yet (or everything with full=True) and write the list."""
    old = {r["key"]: r for r in ((core.read_json(LIBRARY) or {}).get("rows") or [])}
    index = load_outcome_index()
    rows, built, failed = [], 0, []
    for side, path in (sources or SOURCES).items():
        for base in load_trades(side, path):
            prev = old.get(base["key"])
            if not full and prev and prev.get("v") == rv.VERSION and prev.get("ok"):
                rows.append({**prev, **base})                # CSV edits show up without a replay
                continue
            if limit is not None and built >= limit:
                if prev:
                    rows.append({**prev, **base})
                continue
            try:
                data = cached_review(side, base["ticker"], base["date"], REVIEW_DIR, cap=base["cap"],
                                     row=index.get((base["date"], side, base["ticker"])))
                entry = {**base, **system_fields(side, data), "v": rv.VERSION}
            except Exception as e:
                logger.warning(f"{base['key']}: {e}")
                entry = {**base, "ok": False, "error": str(e), "v": rv.VERSION}
            if not entry.get("ok"):
                failed.append(base["key"])
            rows.append(entry)
            built += 1
            time.sleep(pause)
    core.write_json(LIBRARY, {"built_at": pd.Timestamp.now(tz=TZ).isoformat(timespec="seconds"),
                              "v": rv.VERSION, "rows": rows})
    logger.info(f"library: {len(rows)} trades, {built} replayed, {len(failed)} without bars"
                + (f": {', '.join(failed)}" if failed else ""))
    return {"rows": len(rows), "built": built, "failed": failed}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(description="Replay your graded trades for Method's Best tab")
    ap.add_argument("--full", action="store_true", help="rebuild every row")
    ap.add_argument("--limit", type=int, help="replay at most this many rows")
    args = ap.parse_args()
    build(full=args.full, limit=args.limit)


if __name__ == "__main__":
    main()
