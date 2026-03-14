"""
Load and normalize bounce_data.csv and reversal_data.csv for options replay.

Generates entry offset variants for each trade to test multiple entry timings.
"""

import logging
from pathlib import Path
from datetime import timedelta

import pandas as pd

logger = logging.getLogger(__name__)

BOUNCE_CSV = Path(__file__).resolve().parent.parent / "data" / "bounce_data.csv"
REVERSAL_CSV = Path(__file__).resolve().parent.parent / "data" / "reversal_data.csv"

# Entry offsets in minutes from the base time
BOUNCE_OFFSETS = {"low+0": 0, "low+2": 2, "low+5": 5, "low+10": 10}
REVERSAL_OFFSETS = {"high+0": 0, "open+0": "open", "open+15": "open+15", "open+30": "open+30"}


def load_bounce_trades(expand_offsets: bool = False) -> pd.DataFrame:
    """Load bounce_data.csv and normalize to replay-compatible format.

    Returns DataFrame with columns:
        symbol, date, entry_time, side, setup_type, trade_grade, cap,
        atr_pct, gap_pct, source, entry_offset
    """
    if not BOUNCE_CSV.exists():
        logger.error("bounce_data.csv not found at %s", BOUNCE_CSV)
        return pd.DataFrame()

    raw = pd.read_csv(BOUNCE_CSV)
    logger.info("Loaded %d bounce trades from %s", len(raw), BOUNCE_CSV)

    df = pd.DataFrame()
    df["symbol"] = raw["ticker"].astype(str).str.strip().str.upper()
    df["date"] = pd.to_datetime(raw["date"], format="mixed").dt.date
    df["side"] = 1  # bounces are longs → calls

    # Parse entry time from time_of_low_price
    entry_ts = pd.to_datetime(raw["time_of_low_price"], format="mixed", utc=True, errors="coerce")
    valid = entry_ts.notna()
    entry_ts = entry_ts.copy()
    entry_ts[valid] = entry_ts[valid].dt.tz_convert("US/Eastern").dt.tz_localize(None)
    df["entry_time"] = entry_ts

    df["setup_type"] = raw.get("Setup", "").fillna("unknown")
    df["trade_grade"] = raw.get("trade_grade", "").fillna("")
    df["cap"] = raw.get("cap", "").fillna("")
    df["atr_pct"] = pd.to_numeric(raw.get("atr_pct", 0), errors="coerce").fillna(0)
    df["gap_pct"] = pd.to_numeric(raw.get("gap_pct", 0), errors="coerce").fillna(0)
    df["source"] = "bounce"
    df["entry_offset"] = "low+0"

    # Drop rows without valid entry time
    df = df[df["entry_time"].notna()].copy()

    if not expand_offsets:
        return df.reset_index(drop=True)

    # Generate offset variants
    all_rows = [df.copy()]
    for offset_name, offset_min in BOUNCE_OFFSETS.items():
        if offset_name == "low+0":
            continue
        variant = df.copy()
        variant["entry_time"] = variant["entry_time"] + timedelta(minutes=offset_min)
        variant["entry_offset"] = offset_name
        all_rows.append(variant)

    return pd.concat(all_rows, ignore_index=True)


def load_reversal_trades(expand_offsets: bool = False) -> pd.DataFrame:
    """Load reversal_data.csv and normalize to replay-compatible format.

    Returns DataFrame with same columns as load_bounce_trades().
    """
    if not REVERSAL_CSV.exists():
        logger.error("reversal_data.csv not found at %s", REVERSAL_CSV)
        return pd.DataFrame()

    raw = pd.read_csv(REVERSAL_CSV)
    logger.info("Loaded %d reversal trades from %s", len(raw), REVERSAL_CSV)

    df = pd.DataFrame()
    df["symbol"] = raw["ticker"].astype(str).str.strip().str.upper()
    df["date"] = pd.to_datetime(raw["date"], format="mixed").dt.date
    df["side"] = -1  # reversals are shorts → puts

    # Parse entry time from time_of_high_price
    entry_ts = pd.to_datetime(raw["time_of_high_price"], format="mixed", utc=True, errors="coerce")
    valid = entry_ts.notna()
    entry_ts = entry_ts.copy()
    entry_ts[valid] = entry_ts[valid].dt.tz_convert("US/Eastern").dt.tz_localize(None)
    df["entry_time"] = entry_ts

    df["setup_type"] = raw.get("setup", "").fillna("unknown")
    df["trade_grade"] = raw.get("trade_grade", "").fillna("")
    df["cap"] = raw.get("cap", "").fillna("")
    df["atr_pct"] = pd.to_numeric(raw.get("atr_pct", 0), errors="coerce").fillna(0)
    df["gap_pct"] = pd.to_numeric(raw.get("gap_pct", 0), errors="coerce").fillna(0)
    df["source"] = "reversal"
    df["entry_offset"] = "high+0"

    # Drop rows without valid entry time
    df = df[df["entry_time"].notna()].copy()

    if not expand_offsets:
        return df.reset_index(drop=True)

    # Generate offset variants
    # For reversals: high+0 (already set), open+0, open+15, open+30
    all_rows = [df.copy()]

    # Open-based offsets: use 9:30 ET as market open
    for offset_name, minutes_after_open in [("open+0", 0), ("open+15", 15), ("open+30", 30)]:
        variant = df.copy()
        # Set entry time to market open + offset on the trade date
        variant["entry_time"] = variant["date"].apply(
            lambda d: pd.Timestamp(f"{d} 09:30:00") + timedelta(minutes=minutes_after_open)
        )
        variant["entry_offset"] = offset_name
        all_rows.append(variant)

    return pd.concat(all_rows, ignore_index=True)


def load_all_cap_trades(expand_offsets: bool = False) -> pd.DataFrame:
    """Load both bounce and reversal trades into a single DataFrame."""
    bounce = load_bounce_trades(expand_offsets=expand_offsets)
    reversal = load_reversal_trades(expand_offsets=expand_offsets)

    combined = pd.concat([bounce, reversal], ignore_index=True)
    combined = combined.sort_values(["date", "entry_time"], ascending=[False, False]).reset_index(drop=True)

    logger.info("Combined cap trades: %d bounce + %d reversal = %d total",
                len(bounce), len(reversal), len(combined))
    return combined


def get_cap_trade_options(df: pd.DataFrame) -> list:
    """Build dropdown options for the batch tab."""
    options = []
    for idx, row in df.iterrows():
        direction = "LONG" if row["side"] == 1 else "SHORT"
        source = row.get("source", "?").upper()
        setup = row.get("setup_type", "")[:20]
        offset = row.get("entry_offset", "")
        label = f"{row['date']}  {row['symbol']}  {direction}  [{source}]  {setup}  ({offset})"
        options.append({"label": label, "value": int(idx)})
    return options
