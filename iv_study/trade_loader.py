"""Load reversal_data.csv into trade records for the IV study.

time_of_high_price is authoritative for t=0 (100% populated). The
time_of_high_bucket column is dirty (code 1 mixes premarket and RTH highs) --
buckets are re-derived here from the parsed timestamp. Timestamps carry mixed
EST/EDT offsets (-05:00 / -04:00), so parse via utc=True then convert.
"""

import logging

import pandas as pd

from iv_study import config

logger = logging.getLogger(__name__)


def _parse_et(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert("US/Eastern")


def _bucket(t) -> str:
    if pd.isna(t):
        return "unknown"
    minutes = t.hour * 60 + t.minute
    if minutes < 9 * 60 + 30:
        return "premarket"
    if minutes < 10 * 60:
        return "open30"
    return "post10"


def load_trades(csv_path=None) -> pd.DataFrame:
    """All trades with parsed timestamps, top bucket, and optionability flag."""
    path = csv_path or config.REVERSAL_CSV
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df["date_iso"] = df["date"].dt.strftime("%Y-%m-%d")

    df["t_top"] = _parse_et(df["time_of_high_price"])
    df["t_reversal"] = _parse_et(df["time_of_reversal"])
    df["top_bucket"] = df["t_top"].apply(_bucket)

    df["optionable"] = ~df["ticker"].isin(config.MICROCAP_DROPLIST)

    logger.info(
        "Loaded %d trades (%d optionable) | top buckets: %s",
        len(df), int(df["optionable"].sum()),
        df["top_bucket"].value_counts().to_dict(),
    )
    return df


def optionable_trades(df: pd.DataFrame = None) -> pd.DataFrame:
    df = df if df is not None else load_trades()
    return df[df["optionable"]].copy()


def pilot_trades(df: pd.DataFrame = None) -> pd.DataFrame:
    """The Phase 1 pilot rows, in config.PILOT_TRADES order."""
    df = df if df is not None else load_trades()
    keys = {(t, d) for t, d in config.PILOT_TRADES}
    mask = df.apply(lambda r: (r["ticker"], r["date_iso"]) in keys, axis=1)
    out = df[mask].copy()
    order = {(t, d): i for i, (t, d) in enumerate(config.PILOT_TRADES)}
    out["_ord"] = out.apply(lambda r: order[(r["ticker"], r["date_iso"])], axis=1)
    out = out.sort_values("_ord").drop(columns="_ord")
    missing = keys - set(zip(out["ticker"], out["date_iso"]))
    if missing:
        logger.warning("Pilot trades not found in CSV: %s", missing)
    return out
