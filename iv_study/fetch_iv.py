"""Phase 01 -- fetch per-trade intraday ATM IV series into data/iv_study/.

Usage (from project root):
    venv/Scripts/python.exe -m iv_study.fetch_iv --pilot
    venv/Scripts/python.exe -m iv_study.fetch_iv --all
    venv/Scripts/python.exe -m iv_study.fetch_iv --ticker GME --date 2021-03-10

Every attempted trade gets a manifest.csv row (status: ok/thin/no_data/no_exp/
no_chain/error) so attrition is visible. Theta responses are pickle-cached by
options_replay, so re-runs are near-free.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from options_replay import theta_client
from iv_study import config, trade_loader
from iv_study.iv_fetch import fetch_trade_iv

logger = logging.getLogger(__name__)


def _save_series(series: pd.DataFrame, ticker: str, date_iso: str) -> str:
    """Parquet if pyarrow is available, else pickle. Returns the saved path."""
    base = config.DATA_DIR / f"{ticker}_{date_iso.replace('-', '')}"
    fixed_strike = series.attrs.get("fixed_strike")
    series = series.copy()
    series["fixed_strike"] = fixed_strike  # attrs don't survive parquet round-trips
    try:
        path = base.with_suffix(".parquet")
        series.to_parquet(path)
    except ImportError:
        path = base.with_suffix(".pkl")
        series.to_pickle(path)
    return str(path)


def run(trades: pd.DataFrame) -> pd.DataFrame:
    if not theta_client.check_terminal_running():
        raise SystemExit(
            "Theta Terminal is not reachable on localhost:25503. "
            "Start it (external Java process) and re-run."
        )

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, trade in enumerate(trades.itertuples(), 1):
        ticker, date_iso = trade.ticker, trade.date_iso
        logger.info("[%d/%d] %s %s (top %s, %s)",
                    i, len(trades), ticker, date_iso, trade.t_top, trade.top_bucket)
        try:
            series, meta = fetch_trade_iv(ticker, date_iso)
        except theta_client.ThetaTerminalOfflineError:
            raise SystemExit("Theta Terminal went offline mid-run. Restart it and re-run "
                             "(completed trades are cached).")
        except Exception as e:
            logger.exception("%s %s failed", ticker, date_iso)
            meta = {"ticker": ticker, "date": date_iso, "status": "error", "err": str(e)[:200]}
            series = None

        meta["top_bucket"] = trade.top_bucket
        meta["t_top"] = str(trade.t_top)
        meta["path"] = _save_series(series, ticker, date_iso) if series is not None else ""
        rows.append(meta)
        logger.info("  -> %s (valid_frac=%s, exp=%s, strikes=%s)",
                    meta["status"], meta.get("valid_frac"), meta.get("exp"),
                    meta.get("n_strikes"))

    manifest = pd.DataFrame(rows)
    manifest.to_csv(config.MANIFEST_CSV, index=False)
    logger.info("Manifest -> %s", config.MANIFEST_CSV)
    logger.info("Status counts: %s", manifest["status"].value_counts().to_dict())
    return manifest


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true", help="Phase 1 pilot trades")
    group.add_argument("--all", action="store_true", help="all optionable trades")
    group.add_argument("--ticker", help="single trade (with --date)")
    ap.add_argument("--date", help="YYYY-MM-DD, required with --ticker")
    args = ap.parse_args()

    df = trade_loader.load_trades()
    if args.pilot:
        trades = trade_loader.pilot_trades(df)
    elif args.all:
        trades = trade_loader.optionable_trades(df)
    else:
        if not args.date:
            ap.error("--ticker requires --date")
        trades = df[(df["ticker"] == args.ticker.upper()) & (df["date_iso"] == args.date)]
        if trades.empty:
            raise SystemExit(f"No trade found for {args.ticker} {args.date}")

    run(trades)


if __name__ == "__main__":
    main()
