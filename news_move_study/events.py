"""The news-event population — one row per starting gun.

Source is ExitMonitor/data/trade_data.csv (the traded log). An event is a
(symbol, date, reference bar): ``ref_time`` is when the news broke (or close
to it) and ``ref_price`` is the price there. Rows sharing a ref_time are
partial fills of one entry and collapse to a single event.

Two variants are built so the diagnostics can show sensitivity to how
re-entries are treated:

  * ``first``    — one event per symbol-day, earliest ref_time. Conservative:
                   three re-entries into one MANU move can't count as three.
  * ``distinct`` — one event per distinct ref_time. A genuinely separate
                   headline later the same day keeps its own gun, at the cost
                   of overlapping windows when it was really a re-entry.

``first`` is the primary. Everything downstream takes a DataFrame with
columns: symbol, date_iso, ref_ts (tz-aware ET), ref_price, direction
(+1 long / -1 short), plus the static conditioners already in the log
(cap-ish: mkt_cap, ADTV_$, spread_%, LQA_Score, is_ETF).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from pytz import timezone

log = logging.getLogger(__name__)

ET = timezone("US/Eastern")

TRADE_LOG = Path(r"C:\Users\zmbur\PycharmProjects\ExitMonitor\data\trade_data.csv")

STATIC_COLS = ["mkt_cap", "ADTV_$", "spread_%", "LQA_Score", "is_ETF"]


def _parse_ref_ts(ref_time, date_iso: str):
    """ref_time comes in two shapes in the log:
        '2023-06-06 13:50:00-04:00'  (full, tz-aware)
        '11:56:00'                   (time only — join to the trade date)
    Returns a tz-aware ET Timestamp, or None."""
    if ref_time is None or (isinstance(ref_time, float) and pd.isna(ref_time)):
        return None
    s = str(ref_time).strip()
    if not s:
        return None
    try:
        if "-" in s or ":" in s and len(s) > 8:
            ts = pd.Timestamp(s)
        else:
            ts = pd.Timestamp(f"{date_iso} {s}")
    except (ValueError, TypeError):
        try:
            ts = pd.Timestamp(f"{date_iso} {s}")
        except (ValueError, TypeError):
            return None
    if ts.tzinfo is None:
        try:
            ts = ts.tz_localize(ET)
        except Exception:
            return None
    else:
        ts = ts.tz_convert(ET)
    # A time-only ref_time joined to the wrong date shows up as a date mismatch.
    if ts.strftime("%Y-%m-%d") != date_iso:
        ts = pd.Timestamp(f"{date_iso} {ts.strftime('%H:%M:%S')}").tz_localize(ET)
    return ts


def load_news_rows(path: Path = TRADE_LOG) -> pd.DataFrame:
    """Every news-tagged row with a usable gun (ref_time + ref_price + side)."""
    df = pd.read_csv(path, low_memory=False)
    tags = df["Tags"].fillna("").str.lower()
    df = df[tags.str.contains(r"\bnews\b", regex=True)].copy()

    df["date_iso"] = pd.to_datetime(df["Date"], format="%m/%d/%Y",
                                    errors="coerce").dt.strftime("%Y-%m-%d")
    df["ref_price"] = pd.to_numeric(df["ref_price"], errors="coerce")
    df["side"] = pd.to_numeric(df["Side"], errors="coerce")
    df["symbol"] = df["Symbol"].astype(str).str.strip().str.upper()

    df = df.dropna(subset=["date_iso", "ref_price", "side"])
    df = df[df["ref_price"] > 0]
    df["ref_ts"] = [
        _parse_ref_ts(rt, d) for rt, d in zip(df["ref_time"], df["date_iso"])
    ]
    df = df[df["ref_ts"].notna()].copy()
    df["direction"] = df["side"].apply(lambda s: 1 if s > 0 else -1)
    return df


def build_events(variant: str = "first", path: Path = TRADE_LOG) -> pd.DataFrame:
    """Collapse the news rows to events. variant: 'first' | 'distinct'."""
    rows = load_news_rows(path)
    if rows.empty:
        return rows

    keep = ["symbol", "date_iso", "ref_ts", "ref_price", "direction"] + [
        c for c in STATIC_COLS if c in rows.columns
    ]
    rows = rows.sort_values(["symbol", "date_iso", "ref_ts"])

    # Realized P&L for the whole symbol-day. Carried ONLY so the diagnostics can
    # reproduce the winners-only bias in ExitMonitor's lo_to_hi (which skips
    # gpnl <= 0) and show what correcting it does. Never a study feature — it
    # describes your exits, not the move.
    pnl_col = "Gross P&L" if "Gross P&L" in rows.columns else None
    if pnl_col:
        day_pnl = (pd.to_numeric(rows[pnl_col], errors="coerce")
                   .groupby([rows["symbol"], rows["date_iso"]]).sum()
                   .rename("gross_pnl"))
    else:
        day_pnl = None

    if variant == "distinct":
        # One event per distinct reference bar.
        ev = rows.drop_duplicates(subset=["symbol", "date_iso", "ref_ts"], keep="first")
    else:
        # One event per symbol-day, at the earliest gun.
        ev = rows.drop_duplicates(subset=["symbol", "date_iso"], keep="first")

    ev = ev[keep].reset_index(drop=True)
    ev["year"] = ev["date_iso"].str.slice(0, 4).astype(int)
    if day_pnl is not None:
        ev = ev.merge(day_pnl, left_on=["symbol", "date_iso"],
                      right_index=True, how="left")
    return ev


def population_summary(path: Path = TRADE_LOG) -> str:
    """Human-readable collapse stats — how much re-entry is in the log."""
    rows = load_news_rows(path)
    first = build_events("first", path)
    distinct = build_events("distinct", path)
    per_day = rows.groupby(["symbol", "date_iso"])["ref_ts"].nunique()
    lines = [
        f"news-tagged rows with a usable gun : {len(rows):>5}",
        f"events, one per symbol-day (first) : {len(first):>5}   <- primary",
        f"events, one per distinct ref bar   : {len(distinct):>5}",
        f"symbol-days with >1 distinct gun   : {(per_day > 1).sum():>5} "
        f"({(per_day > 1).mean() * 100:.1f}%)",
        f"by year (first): " + ", ".join(
            f"{y}:{n}" for y, n in first.groupby("year").size().items()),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(population_summary())
