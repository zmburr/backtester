"""Phase 1 pilot overlay plots: price vs ATM IV with the top marked.

Throwaway inspection tool for the go/no-go gate -- NOT the phase 03 event study.

Usage (from project root, after `-m iv_study.fetch_iv --pilot`):
    venv/Scripts/python.exe -m iv_study.pilot_plot
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from iv_study import config, trade_loader

logger = logging.getLogger(__name__)


def _load_series(ticker: str, date_iso: str):
    base = config.DATA_DIR / f"{ticker}_{date_iso.replace('-', '')}"
    for path in (base.with_suffix(".parquet"), base.with_suffix(".pkl")):
        if path.exists():
            df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_pickle(path)
            return df
    return None


def plot_trade(series: pd.DataFrame, trade) -> Path:
    t_top = trade.t_top
    fig, (ax_px, ax_iv) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2]},
    )

    ax_px.plot(series.index, series["spot"], color="black", lw=1.2)
    ax_px.set_ylabel("underlying")
    ax_px.set_title(f"{trade.ticker} {trade.date_iso} — top {t_top.strftime('%H:%M')} "
                    f"({trade.top_bucket}) | fixed strike {series['fixed_strike'].iloc[0]}")

    ax_iv.plot(series.index, series["atm_iv"], color="tab:blue", lw=1.4,
               label="ATM IV (constant moneyness)")
    ax_iv.plot(series.index, series["fixed_iv"], color="tab:orange", lw=1.0, alpha=0.7,
               label="fixed-strike IV (cross-check)")
    ax_iv.set_ylabel("implied vol")
    ax_iv.legend(loc="best", fontsize=8)

    for ax in (ax_px, ax_iv):
        if series.index[0] <= t_top <= series.index[-1]:
            ax.axvline(t_top, color="red", ls="--", lw=1, alpha=0.8)
        ax.grid(alpha=0.25)

    fig.autofmt_xdate()
    out = config.REPORTS_DIR / f"pilot_{trade.ticker}_{trade.date_iso.replace('-', '')}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    trades = trade_loader.pilot_trades()
    summary = []
    for trade in trades.itertuples():
        series = _load_series(trade.ticker, trade.date_iso)
        if series is None or series.empty:
            logger.warning("No series for %s %s — run fetch_iv --pilot first",
                           trade.ticker, trade.date_iso)
            continue
        out = plot_trade(series, trade)
        valid = series["atm_iv"].dropna()
        corr = float(pd.Series(valid).corr(series.loc[valid.index, "fixed_iv"]))
        summary.append({
            "ticker": trade.ticker, "date": trade.date_iso, "bucket": trade.top_bucket,
            "n_bars": len(series), "n_valid": len(valid),
            "iv_min": round(valid.min(), 3), "iv_max": round(valid.max(), 3),
            "ladder_vs_fixed_corr": round(corr, 3),
            "png": out.name,
        })
        logger.info("%s %s -> %s", trade.ticker, trade.date_iso, out.name)

    if summary:
        print("\n" + pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
