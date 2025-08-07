
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal      # --- NEW -------------
from transformers.models.kosmos2.processing_kosmos2 import extract_entities_with_patch_indices

# ---> your existing helper files
import data_queries.polygon_queries as pq  # noqa: E402, F401
import data_queries.trillium_queries as trlm  # noqa: E402, F401

# ------------- CONFIG --------------------------------------------------------

# IPO blotter: (ticker, ipo_date 'M/D/YY', ipo_price)
IPOS: List[Tuple[str, str, float]] = [
    ("COIN", "4/14/21", 250),
    ("RBLX", "3/10/21", 45),
    ("RIVN", "11/10/21", 78),
    ("U", "9/18/20", 52),
    ("SNOW", "9/16/20", 120),
    ("ABNB", "12/10/20", 68),
    ("DASH", "12/9/20", 102),
    ("HOOD", "7/29/21", 38),
    ("BIGC", "8/5/20", 24),
    ("DOCS", "6/24/21", 26),
    ("GDRX", "9/23/20", 33),
    ("ARM", "9/14/23", 52),
    ("CRCL", "6/5/25", 31),
    ("FIG", "7/31/25", 33),
    ("CPNG", "3/11/21", 35),
    ("LYFT", "3/29/19", 72),
    ("SPOT", "4/03/18", 132),
    ("UBER", "5/10/19", 45),
    ("FLY", "8/7/35", 45),
]

OUTPUT_PATH = Path(r"C:\Users\zmbur\PycharmProjects\backtester\data\ipo_analysis_results.csv")
SPY = "SPY"
EARLY_WINDOWS_MIN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 30]
CORR_WINDOW_MIN = 30

# ------------- DATA LAYER ----------------------------------------------------


def _normalize_date(date_str: str) -> str:
    """Turn M/D/YY -> YYYY‑MM‑DD (Polygon & Trillium friendly)."""
    return datetime.strptime(date_str, "%m/%d/%y").strftime("%Y-%m-%d")


def _polygon_intraday(ticker: str, date: str) -> Optional[pd.DataFrame]:
    """Return 1‑min bars from Polygon or None on failure."""
    try:
        df = pq.get_intraday(ticker, date, multiplier=1, timespan="minute")
        if df is None or (hasattr(df, "empty") and df.empty):
            return None
        if isinstance(df, dict) and df.get("status") == "NOT_AUTHORIZED":
            return None
        return df
    except Exception:
        return None


def _trillium_intraday(ticker: str, date: str) -> Optional[pd.DataFrame]:
    """Return 1‑min bars from Trillium (bar‑1min) or None on failure."""
    try:
        return trlm.get_intraday(ticker, date, "bar-1min")
    except Exception:
        return None


def _get_intraday_any(ticker: str, date: str) -> pd.DataFrame:
    """
    Try Polygon first, then Trillium. Raise on total failure.
    Ensures the dataframe uses standard columns [open, high, low, close, volume].
    """
    df = _polygon_intraday(ticker, date)
    source = "Polygon"
    if df is None or df.empty:
        df = _trillium_intraday(ticker, date)
        source = "Trillium"

    if df is None or df.empty:
        raise ValueError(f"No intraday data found for {ticker} on {date} from either source.")

    needed_cols = {"open", "high", "low", "close"}
    if not needed_cols.issubset(df.columns):
        raise ValueError(f"{source} dataframe for {ticker} lacks OHLC columns.")

    df = df.sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("US/Eastern").tz_localize(None)

    return df



def _first_print(df: pd.DataFrame) -> pd.Series:
    return df.iloc[0]


def _calc_window_pct(df: pd.DataFrame, window_min: int, ref_price: float) -> float:
    end_time = df.index[0] + timedelta(minutes=window_min)
    sub = df[df.index <= end_time]
    return np.nan if sub.empty else (sub.iloc[-1].close - ref_price) / ref_price


def _drawdown_from_high(df: pd.DataFrame, high_price: float) -> float:
    hi_idx = df[df.high == high_price].index[0]
    tail = df[df.index >= hi_idx]
    return 0.0 if tail.empty else (high_price - tail.low.min()) / high_price


def _corr_with_spy(ipo_df: pd.DataFrame, spy_df: pd.DataFrame) -> float:
    end_time = ipo_df.index[0] + timedelta(minutes=CORR_WINDOW_MIN)
    ipo_sub = ipo_df[ipo_df.index <= end_time]
    spy_sub = spy_df[spy_df.index <= end_time]
    if ipo_sub.empty or spy_sub.empty:
        return np.nan
    joined = (
        pd.concat(
            [np.log(ipo_sub.close).diff(), np.log(spy_sub.close).diff()],
            axis=1,
            join="inner",
        )
        .dropna()
    )
    return joined.corr().iloc[0, 1] if len(joined) >= 2 else np.nan


def analyze_single_ipo(ticker: str, date_raw: str, ipo_price: float) -> Dict:
    print(f"Processing {ticker} IPO on {date_raw} at ${ipo_price:.2f}")
    date = _normalize_date(date_raw)

    # Try to get intraday data; if it fails, continue with daily data for gap metrics
    ipo_df: Optional[pd.DataFrame]
    spy_df: Optional[pd.DataFrame]
    try:
        ipo_df = _get_intraday_any(ticker, date)
    except Exception:
        ipo_df = None
    try:
        spy_df = _get_intraday_any(SPY, date)
    except Exception:
        spy_df = None

    # Use first print for IPO day reference (can be later than 9:30 on IPOs)
    first_price = _first_print(ipo_df).open if ipo_df is not None else np.nan
    hi_price = ipo_df.high.max() if ipo_df is not None else np.nan
    print(first_price, hi_price)
    # Restrict to regular trading session for close/open calculations
    def _regular_session(df: pd.DataFrame) -> pd.DataFrame:
        try:
            return df.between_time("09:30", "16:00")
        except Exception:
            return df

    day_close_price = np.nan
    # Prefer official daily close; fallback to intraday regular session close
    try:
        day_close_price = pq.get_daily(ticker, date).close
    except Exception:
        if ipo_df is not None:
            ipo_regular = _regular_session(ipo_df)
            day_close_price = (
                ipo_regular.iloc[-1].close if not ipo_regular.empty else ipo_df.iloc[-1].close
            )

    # --- NEW ------------- compute next‑day open gap -------------------------
    next_open_price = np.nan
    try:
        next_date = pq.adjust_date_forward(date,1)
        print(date, "→", next_date)  # Debug output to see date transitions
    # Prefer official daily open for the next trading day
    except Exception:
        next_date = pq.adjust_date_forward(date,3)

    try:
        next_open_price = pq.get_daily(ticker, next_date).open
        print("Using Polygon daily open for next day:", next_open_price)
    except Exception:
        next_day_df = _get_intraday_any(ticker, next_date)
        next_regular = _regular_session(next_day_df)
        next_open_price = (
            next_regular.iloc[0].open if not next_regular.empty else _first_print(next_day_df).open
        )
    pct_move_d1close_to_d2open = (next_open_price - day_close_price) / day_close_price


    # -------------------------------------------------------------------------

    record: Dict = {
        "ticker": ticker,
        "ipo_date": date,
        "ipo_price": ipo_price,
        "first_print_price": first_price,
        "pct_open_vs_ipo": (first_price - ipo_price) / ipo_price,
        "day_high": hi_price,
        "time_to_day_high_min": (
            ((ipo_df[ipo_df.high == hi_price].index[0] - ipo_df.index[0]).seconds // 60)
            if ipo_df is not None and not np.isnan(hi_price)
            else np.nan
        ),
        "drawdown_from_high_pct": (
            _drawdown_from_high(ipo_df, hi_price) if ipo_df is not None and not np.isnan(hi_price) else np.nan
        ),
        "ipo_session_close": day_close_price,
        "next_day_regular_open": next_open_price,
        "pct_move_d1close_to_d2open": pct_move_d1close_to_d2open,      # --- NEW ---
    }

    # Early‑window stats
    for w in EARLY_WINDOWS_MIN:
        record[f"pct_move_{w}m"] = (
            _calc_window_pct(ipo_df, w, first_price) if ipo_df is not None and not np.isnan(first_price) else np.nan
        )
        record[f"spy_pct_move_{w}m"] = (
            _calc_window_pct(spy_df, w, spy_df.iloc[0].open) if spy_df is not None else np.nan
        )

    record["corr_ipo_spy_30m"] = (
        _corr_with_spy(ipo_df, spy_df) if ipo_df is not None and spy_df is not None else np.nan
    )
    return record

# ------------- CLI -----------------------------------------------------------


def main() -> None:
    rows: List[Dict] = []
    for ticker, date_raw, ipo_price in IPOS:
        try:
            rows.append(analyze_single_ipo(ticker, date_raw, ipo_price))
            print(f"✓ {ticker} processed")
        except Exception as exc:
            print(f"✗ {ticker} skipped – {exc}")

    pd.DataFrame(rows).sort_values("ipo_date").to_csv(OUTPUT_PATH, index=False)
    print(f"\nResults written to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()