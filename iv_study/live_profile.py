"""Live IV top-profile for a candidate ticker, scored against historical tops.

Computes the run-up IV features from the study (iv_runup_chg, ramp slope,
final-2-day change + steepening, prior-close IV percentile) for the trading
days before `as_of`, then places each against the distributions in
data/iv_study/iv_reference.json (n=105 historical reversal trades).

Premarket-safe: the run-up features use prior-day CLOSE marks only, so a
morning priority-report run needs nothing from the (not yet traded) session.
Same-day open-IV features are added only when as_of is a PAST date or the
session is complete (after ~16:15 ET) -- fetching a partial current day would
poison the infinite pickle cache with an incomplete frame.

CLI (from project root):
    venv/Scripts/python.exe -m iv_study.live_profile NVDA
    venv/Scripts/python.exe -m iv_study.live_profile NVDA 2024-02-12
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from options_replay import theta_client
from iv_study import config
from iv_study.pseudo_controls import fetch_control_day, _trading_days_before

logger = logging.getLogger(__name__)

_REFERENCE = None
_Q_KEYS = ["q10", "q25", "q50", "q75", "q90"]
_Q_PCTS = [10.0, 25.0, 50.0, 75.0, 90.0]


def load_reference() -> dict:
    global _REFERENCE
    if _REFERENCE is None:
        path = config.DATA_DIR / "iv_reference.json"
        _REFERENCE = json.loads(path.read_text())
    return _REFERENCE


def hist_pctile(feature: str, value: float):
    """Approximate percentile of `value` within the historical-top distribution."""
    ref = load_reference()["features"].get(feature)
    if ref is None or value is None or not np.isfinite(value):
        return None
    qs = [ref[k] for k in _Q_KEYS]
    return float(np.clip(np.interp(value, qs, _Q_PCTS), 5.0, 95.0))


def _now_et():
    return pd.Timestamp.now(tz="US/Eastern")


def _session_complete(as_of: str) -> bool:
    now = _now_et()
    day = pd.Timestamp(as_of, tz="US/Eastern")
    return day.date() < now.date() or (day.date() == now.date()
                                       and now.hour * 60 + now.minute >= 16 * 60 + 15)


def get_iv_profile(ticker: str, as_of: str = None) -> dict:
    """IV run-up profile for `ticker` as of `as_of` (default: today, ET).

    Returns dict with features, hist_* percentiles vs the 105 historical tops,
    and `conditions` for the match rule -- or None (reason logged) when Theta
    is down, the name has no options, or fewer than 4 usable daily marks.
    """
    ticker = ticker.upper()
    as_of = as_of or _now_et().strftime("%Y-%m-%d")

    if not theta_client.check_terminal_running():
        logger.info("IV profile skipped for %s: Theta Terminal not reachable", ticker)
        return None

    marks = []
    for day in _trading_days_before(as_of, config.PSEUDO_CONTROL_DAYS):
        try:
            row = fetch_control_day(ticker, day)
        except theta_client.ThetaTerminalOfflineError:
            logger.warning("Theta Terminal went offline during %s IV profile", ticker)
            return None
        except Exception as e:
            logger.info("IV mark failed %s %s: %s", ticker, day, e)
            row = None
        if row:
            marks.append(row)

    closes = np.array([m["close_iv"] for m in marks], dtype=float)
    closes = closes[np.isfinite(closes)]
    if len(closes) < 4:
        logger.info("IV profile skipped for %s: only %d usable marks", ticker, len(closes))
        return None

    profile = {
        "ticker": ticker, "as_of": as_of, "n_marks": int(len(closes)),
        "prior_close_iv": float(closes[-1]),
        "iv_runup_chg": float(closes[-1] / closes[0] - 1),
        "iv_ramp_slope": float(np.polyfit(np.arange(len(closes)),
                                          closes / closes.mean(), 1)[0]),
        "prior_close_iv_pctile": float(100.0 * (closes[:-1] < closes[-1]).mean()),
    }
    two_day = closes[2:] / closes[:-2] - 1
    profile["iv_ramp_final2d"] = float(two_day[-1])
    if len(two_day) >= 3:
        profile["iv_ramp_final2d_pctile"] = float(100.0 * (two_day[:-1] < two_day[-1]).mean())

    # Same-day open IV only when the session can no longer change (see docstring).
    if _session_complete(as_of):
        try:
            today_row = fetch_control_day(ticker, as_of)
        except Exception:
            today_row = None
        if today_row and np.isfinite(today_row.get("open_iv", np.nan)):
            profile["open_iv"] = float(today_row["open_iv"])
            if closes[-1] > 0:
                profile["iv_gap_open"] = profile["open_iv"] / closes[-1] - 1
            profile["open_iv_pctile"] = float(
                100.0 * (np.array([m["open_iv"] for m in marks]) < profile["open_iv"]).mean())

    for feat in ("iv_runup_chg", "iv_ramp_final2d", "iv_gap_open"):
        if feat in profile:
            profile[f"hist_{feat}"] = hist_pctile(feat, profile[feat])

    ref = load_reference()["features"]
    profile["conditions"] = {
        "runup_ge_q25": profile["iv_runup_chg"] >= ref["iv_runup_chg"]["q25"],
        "final2d_ge_q25": profile["iv_ramp_final2d"] >= ref["iv_ramp_final2d"]["q25"],
        "prior_close_near_top": profile["prior_close_iv_pctile"] >= 87.5,
    }
    profile["reference_n"] = load_reference()["n"]
    return profile


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m iv_study.live_profile TICKER [YYYY-MM-DD]")
    profile = get_iv_profile(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    if profile is None:
        raise SystemExit("no profile (see log)")
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
