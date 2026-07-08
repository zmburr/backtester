"""Build data/iv_study/iv_reference.json -- historical IV-profile distributions.

Quantiles of the run-up IV features across the 105 optionable reversal trades,
used by live_profile.py to place a live candidate against the historical top
profile. The JSON is committed so machines without the study cache (the
MacBook) never need to re-run the study.

Usage (from project root):
    venv/Scripts/python.exe -m iv_study.build_reference
"""

import json
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from iv_study import config

logger = logging.getLogger(__name__)

QUANTILES = {"q10": 0.10, "q25": 0.25, "q50": 0.50, "q75": 0.75, "q90": 0.90}
FEATURES = ["iv_runup_chg", "iv_ramp_slope", "iv_ramp_final2d",
            "iv_ramp_final2d_pctile", "iv_gap_open", "open_iv_pctile"]


def prior_close_iv_pctile(ticker: str, trade_date: str, controls: pd.DataFrame):
    """Percentile of the final pre-trade close IV vs the earlier run-up closes.

    The premarket-knowable analog of open_iv_pctile: computable the night
    before the reversal day. Mirrors the windowing in track_b_features.
    """
    ctrl = controls[(controls["ticker"] == ticker) & (controls["date"] < trade_date)]
    closes = ctrl.sort_values("date").tail(config.PSEUDO_CONTROL_DAYS)["close_iv"].to_numpy(float)
    closes = closes[np.isfinite(closes)]
    if len(closes) < 4:
        return np.nan
    return float(100.0 * (closes[:-1] < closes[-1]).mean())


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    feats = pd.read_csv(config.DATA_DIR / "iv_features.csv")
    controls = pd.read_csv(config.CONTROLS_DIR / "control_marks.csv")

    usable = feats[feats["n_controls"].fillna(0) >= 4].copy()
    usable["prior_close_iv_pctile"] = [
        prior_close_iv_pctile(t.ticker, t.date, controls) for t in usable.itertuples()
    ]

    ref = {
        "generated": date.today().isoformat(),
        "source": "iv_study run 2026-07-07 (121 reversal trades, 106 with IV series)",
        "n": int(len(usable)),
        "features": {},
    }
    for col in FEATURES + ["prior_close_iv_pctile"]:
        v = pd.to_numeric(usable.get(col), errors="coerce").dropna()
        if len(v) < 20:
            logger.warning("Skipping %s: only %d values", col, len(v))
            continue
        ref["features"][col] = {
            "n": int(len(v)),
            **{q: round(float(v.quantile(f)), 4) for q, f in QUANTILES.items()},
            "frac_positive": round(float((v > 0).mean()), 3),
        }

    out = config.DATA_DIR / "iv_reference.json"
    out.write_text(json.dumps(ref, indent=2))
    logger.info("Reference -> %s", out)
    for k, d in ref["features"].items():
        logger.info("  %-24s n=%d  q25=%+.3f  q50=%+.3f  q75=%+.3f",
                    k, d["n"], d["q25"], d["q50"], d["q75"])


if __name__ == "__main__":
    main()
