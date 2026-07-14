"""Stage 05: event-window price data + per-deal trade metrics -> despac_data.csv.

Per deal, daily bars for:
  - old SPAC ticker: announcement-20d .. flip+5d  (announcement pop, drift,
    vote reaction, last close before the ticker flip)
  - post-flip ticker: flip-5d .. flip+25d         (flip-day gap/run, follow-through)

Key trade metrics (all vs last old-ticker close = the "get in before the
ticker changes" entry):
  flip_gap_pct, flip_day_ret_pct, flip_high_ret_pct, post_flip_ret_{1,3,5,10}d,
  max_runup_10d_pct

Run:  python -m despac_study.fetch_prices
"""

import argparse
import logging

import numpy as np
import pandas as pd

from despac_study.config import CLASSIFIED_CSV, MASTER_CSV
from despac_study.polygon_enrich import _get_json

logger = logging.getLogger(__name__)


def get_daily_bars(ticker: str, start: str, end: str) -> pd.DataFrame:
    # UNadjusted: the old and new symbols are separate series in Polygon, and
    # later reverse splits get baked into the new symbol's adjusted history
    # but not the old one's - which fabricates 30-50x "flip returns". Raw
    # traded prices are continuous across the flip.
    d = _get_json(f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
                  ("aggs", ticker, start, end), limit=5000, adjusted="false")
    res = (d or {}).get("results") or []
    if not res:
        return pd.DataFrame()
    df = pd.DataFrame(res)
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("US/Eastern").dt.strftime("%Y-%m-%d")
    return df.set_index("date")[["o", "h", "l", "c", "v"]]


def split_near_flip(old_t, flip_t, close_d: str, flip_d: str) -> bool:
    """Any recorded split on either symbol around the close/flip => the two
    price series are not 1:1 continuous (reverse-merger ratio, not a SPAC)."""
    lo = (pd.Timestamp(close_d) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    hi = (pd.Timestamp(flip_d) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    for t in {old_t, flip_t}:
        if not isinstance(t, str) or not t:
            continue
        d = _get_json("/v3/reference/splits", ("splits", t, lo, hi),
                      ticker=t, **{"execution_date.gte": lo, "execution_date.lte": hi})
        if (d or {}).get("results"):
            return True
    return False


def _ret(a, b):
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return None
    return round(100.0 * (a / b - 1.0), 2)


def deal_metrics(row) -> dict:
    out = {"cik": row["cik"]}
    old_t = row.get("old_ticker")
    flip_t = row.get("flip_ticker")
    flip_d = row.get("flip_date")
    close_d = str(row.get("close_date") or "")
    ann_d = row.get("ann_date")
    vote_d = row.get("vote_date")

    have_old = isinstance(old_t, str) and old_t
    have_flip = isinstance(flip_t, str) and flip_t and isinstance(flip_d, str) and flip_d

    old = pd.DataFrame()
    if have_old:
        start = (pd.Timestamp(ann_d) - pd.Timedelta(days=30)).strftime("%Y-%m-%d") \
            if isinstance(ann_d, str) and ann_d else \
            (pd.Timestamp(close_d) - pd.Timedelta(days=240)).strftime("%Y-%m-%d")
        end = flip_d if have_flip else (pd.Timestamp(close_d) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        old = get_daily_bars(old_t, start, end)

    # ---- announcement reaction (old ticker) ----
    if not old.empty and isinstance(ann_d, str) and ann_d:
        after = old[old.index >= ann_d]
        before = old[old.index < ann_d]
        if len(after) >= 1 and len(before) >= 1:
            d0 = after.iloc[0]
            prev_c = before.iloc[-1]["c"]
            out["ann_day0_date"] = after.index[0]
            out["ann_prev_close"] = prev_c
            out["ann_gap_pct"] = _ret(d0["o"], prev_c)
            out["ann_day0_ret_pct"] = _ret(d0["c"], prev_c)
            hi2 = after.head(2)["h"].max()
            out["ann_2d_high_ret_pct"] = _ret(hi2, prev_c)
            vol_base = before.tail(20)["v"].median()
            if vol_base and vol_base > 0:
                out["ann_day0_vol_mult"] = round(float(d0["v"] / vol_base), 1)
            # drift into the vote
            if isinstance(vote_d, str) and vote_d:
                upto = old[(old.index > out["ann_day0_date"]) & (old.index <= vote_d)]
                if len(upto):
                    out["ann_to_vote_ret_pct"] = _ret(upto.iloc[-1]["c"], d0["c"])
                    out["vote_day_close"] = upto.iloc[-1]["c"]

    # ---- last tradeable close under the old ticker ----
    last_old_close, last_old_date = None, None
    if not old.empty:
        pre_flip = old[old.index < flip_d] if have_flip else old
        if len(pre_flip):
            last_old_close = float(pre_flip.iloc[-1]["c"])
            last_old_date = pre_flip.index[-1]
    out["last_old_date"] = last_old_date
    out["last_old_close"] = last_old_close
    if isinstance(vote_d, str) and vote_d and out.get("vote_day_close") and last_old_close:
        out["vote_to_lastold_ret_pct"] = _ret(last_old_close, out["vote_day_close"])

    # ---- flip day and follow-through ----
    if have_flip:
        f_start = (pd.Timestamp(flip_d) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        f_end = (pd.Timestamp(flip_d) + pd.Timedelta(days=25)).strftime("%Y-%m-%d")
        flip = get_daily_bars(flip_t, f_start, f_end)
        fbars = flip[flip.index >= flip_d] if not flip.empty else flip
        if len(fbars):
            f0 = fbars.iloc[0]
            out["flip_day0_date"] = fbars.index[0]
            out["flip_open"] = f0["o"]
            out["flip_close"] = f0["c"]
            out["flip_high"] = f0["h"]
            out["flip_volume"] = f0["v"]
            if last_old_close:
                out["flip_gap_pct"] = _ret(f0["o"], last_old_close)
                out["flip_day_ret_pct"] = _ret(f0["c"], last_old_close)
                out["flip_high_ret_pct"] = _ret(f0["h"], last_old_close)
                for k in (1, 3, 5, 10):
                    if len(fbars) > k:
                        out[f"post_flip_ret_{k}d_pct"] = _ret(fbars.iloc[k]["c"], last_old_close)
                out["max_runup_10d_pct"] = _ret(fbars.head(11)["h"].max(), last_old_close)
                out["max_drawdown_10d_pct"] = _ret(fbars.head(11)["l"].min(), last_old_close)
            # trading-day gap between last old bar and first new bar
            if last_old_date:
                out["untraded_gap_days"] = int(
                    np.busday_count(last_old_date, fbars.index[0])) - 1
        out["flip_lag_cal_days"] = (pd.Timestamp(flip_d) - pd.Timestamp(close_d)).days if close_d else None
        if close_d:
            out["split_at_flip"] = split_near_flip(old_t, flip_t, close_d, flip_d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    df = pd.read_csv(CLASSIFIED_CSV)
    if args.limit:
        df = df.head(args.limit)
    rows = []
    for i, row in df.iterrows():
        try:
            rows.append(deal_metrics(row))
        except Exception as e:
            logger.warning("metrics failed cik=%s: %s", row["cik"], e)
            rows.append({"cik": row["cik"], "price_error": str(e)})
        if (i + 1) % 50 == 0:
            logger.info("prices %d/%d", i + 1, len(df))
    met = pd.DataFrame(rows)
    master = df.merge(met, on="cik", how="left")
    master.to_csv(MASTER_CSV, index=False)
    n = len(master)
    print(f"\nWrote {n} deals -> {MASTER_CSV}")
    for col in ("ann_day0_ret_pct", "flip_gap_pct", "flip_day_ret_pct", "post_flip_ret_5d_pct"):
        if col in master:
            print(f"  {col:24s} coverage {master[col].notna().sum()}/{n}  median {master[col].median()}")


if __name__ == "__main__":
    main()
