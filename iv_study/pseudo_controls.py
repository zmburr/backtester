"""Track B within-ticker pseudo-controls: daily open/close ATM IV marks for the
N trading days before each trade.

Every CSV row is a reversal that happened, so no non-reversal control days
exist by construction. The cheap remedy: compare the trade day's open-IV
behavior against the SAME ticker's run-up days -- a matched within-name
comparison immune to cross-name IV-level differences.

Usage (from project root, after fetch_iv):
    venv/Scripts/python.exe -m iv_study.pseudo_controls
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from options_replay import theta_client
from iv_study import config, trade_loader
from iv_study.iv_fetch import _discover

logger = logging.getLogger(__name__)


def _trading_days_before(date_iso: str, n: int) -> list:
    import pandas_market_calendars as mcal

    end = pd.Timestamp(date_iso)
    sched = mcal.get_calendar("NYSE").schedule(
        start_date=(end - pd.Timedelta(days=n * 3 + 10)).strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
    )
    days = [d.strftime("%Y-%m-%d") for d in sched.index if d < end]
    return days[-n:]


def fetch_control_day(symbol: str, date_iso: str) -> dict:
    """One daily mark: ATM IV near the open and near the close (1h greeks bars).

    Returns row dict with open_iv/close_iv (may be NaN) or None if no chain.
    """
    for exp, dte, chain, spot in _discover(symbol, date_iso):
        strikes = sorted(chain["strike"].dropna().unique())
        strike = min(strikes, key=lambda k: abs(k - spot))

        ivs = []
        for right in ("C", "P"):
            try:
                g = theta_client.get_option_greeks(symbol, exp, strike, right,
                                                   date_iso, interval="1h")
            except theta_client.ThetaTerminalOfflineError:
                raise
            except Exception as e:
                logger.info("control greeks failed %s %s %s%s: %s",
                            symbol, date_iso, strike, right, e)
                continue
            if g.empty or "implied_vol" not in g.columns:
                continue
            g = g[pd.to_numeric(g["implied_vol"], errors="coerce") > 0]
            if "iv_error" in g.columns:
                g = g[pd.to_numeric(g["iv_error"], errors="coerce").fillna(0) < config.IV_ERR_MAX]
            if g.empty:
                continue
            ivs.append(g["implied_vol"].astype(float))

        if not ivs:
            continue
        iv = pd.concat(ivs, axis=1).mean(axis=1).sort_index()
        return {
            "ticker": symbol, "date": date_iso, "exp": exp, "dte": dte,
            "strike": strike, "spot": round(spot, 2),
            "open_iv": float(iv.iloc[0]), "close_iv": float(iv.iloc[-1]),
            "open_ts": str(iv.index[0]), "close_ts": str(iv.index[-1]),
        }
    return None


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    if not theta_client.check_terminal_running():
        raise SystemExit("Theta Terminal is not reachable on localhost:25503.")

    trades = trade_loader.optionable_trades()
    wanted = {}  # (ticker, control_date) -> trade date, deduped (repeat tickers overlap)
    for t in trades.itertuples():
        for d in _trading_days_before(t.date_iso, config.PSEUDO_CONTROL_DAYS):
            wanted.setdefault((t.ticker, d), t.date_iso)

    logger.info("Fetching %d control ticker-days for %d trades", len(wanted), len(trades))
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one(key):
        ticker, day = key
        try:
            return fetch_control_day(ticker, day)
        except theta_client.ThetaTerminalOfflineError:
            raise
        except Exception:
            logger.exception("control %s %s failed", ticker, day)
            return None

    rows, failed, done = [], 0, 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(_one, key) for key in sorted(wanted)]
        for fut in as_completed(futures):
            row = fut.result()  # ThetaTerminalOfflineError propagates and aborts
            done += 1
            if row:
                rows.append(row)
            else:
                failed += 1
            if done % 50 == 0:
                logger.info("  %d/%d (%d no-data)", done, len(wanted), failed)

    config.CONTROLS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.CONTROLS_DIR / "control_marks.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("%d/%d control marks -> %s", len(rows), len(wanted), out)


if __name__ == "__main__":
    main()
