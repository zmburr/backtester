"""Characterization tests for the two reversal pre-trade validators.

Locks in the CURRENT behavior of:
  - analyzers/pretrade_checklist.py  -> PreTradeChecklist (6 cap-adjusted criteria)
  - analyzers/reversal_pretrade.py   -> ReversalPretrade (5 per-cap criteria + readiness gate)

Inputs are FROZEN hardcoded metric dicts sourced from real rows of
data/reversal_data.csv (MRNA/Medium, CODX/Micro, VIR/Small) plus realistic
ETF/Large magnitudes. Results are converted via dataclasses.asdict and the
non-deterministic 'timestamp' field is stripped before snapshotting.
"""
import dataclasses

import pandas as pd
import pytest

from analyzers.pretrade_checklist import PreTradeChecklist
from analyzers.reversal_pretrade import ReversalPretrade


def _strip_ts(result):
    """asdict the ChecklistResult and drop the wall-clock timestamp."""
    d = dataclasses.asdict(result)
    d.pop("timestamp", None)
    return d


# ---------------------------------------------------------------------------
# Frozen metric dicts (real reversal_data.csv rows)
# ---------------------------------------------------------------------------

# MRNA 2/27/2020 — Medium cap, Grade A 3DGapFade
METRICS_MRNA_MEDIUM = {
    "pct_from_9ema": 0.582786261,
    "prior_day_range_atr": 1.723592776,
    "rvol_score": 4.703756871,
    "consecutive_up_days": 3,
    "gap_pct": 0.230452675,
    "reversal_open_close_pct": -0.27090301,
    "pct_change_3": 0.93006993,
    "pct_from_50mav": 0.713286297,
    "atr_pct": 0.0577,
    "spy_5day_return": -0.037811746,
    "closed_outside_upper_band": True,
    "close_green_red": True,
}

# CODX 2/28/2020 — Micro cap, Grade A 3DGapFade
METRICS_CODX_MICRO = {
    "pct_from_9ema": 1.681372029,
    "prior_day_range_atr": 9.538105513,
    "rvol_score": 1.645657432,
    "consecutive_up_days": 4,
    "gap_pct": 0.357769424,
    "reversal_open_close_pct": -0.389478542,
    "pct_change_3": 4.022016222,
    "pct_from_50mav": 7.197431848,
    "atr_pct": 0.0561,
    "spy_5day_return": -0.069158076,
    "closed_outside_upper_band": True,
    "close_green_red": True,
}

# VIR 2/28/2020 — Small cap, Grade C GapDownTrendBreak
METRICS_VIR_SMALL = {
    "pct_from_9ema": 1.180801906,
    "prior_day_range_atr": 9.369888344,
    "rvol_score": 2.825791178,
    "consecutive_up_days": 4,
    "gap_pct": 0.181893688,
    "reversal_open_close_pct": -0.34645116,
    "pct_change_3": 2.274275196,
    "pct_from_50mav": 2.899383338,
    "atr_pct": 0.0654,
    "spy_5day_return": -0.15297619,
    "closed_outside_upper_band": True,
    "close_green_red": True,
}

# AVGO-like — Large cap (realistic large-cap reversal magnitudes)
METRICS_LARGE = {
    "pct_from_9ema": 0.11,
    "prior_day_range_atr": 4.288174677,
    "rvol_score": 3.752348934,
    "consecutive_up_days": 2,
    "gap_pct": 0.027549125,
    "reversal_open_close_pct": -0.038054411,
    "pct_change_3": 0.22896708,
    "pct_from_50mav": 0.432251689,
    "atr_pct": 0.035738293,
    "spy_5day_return": 0.002937033,
    "closed_outside_upper_band": True,
    "close_green_red": True,
}

# SMH-like — ETF cap (realistic ETF reversal magnitudes)
METRICS_ETF = {
    "pct_from_9ema": 0.06,
    "prior_day_range_atr": 2.428905383,
    "rvol_score": 1.886474361,
    "consecutive_up_days": 2,
    "gap_pct": 0.022326276,
    "reversal_open_close_pct": -0.017125788,
    "pct_change_3": 0.164759548,
    "pct_from_50mav": 0.190290319,
    "atr_pct": 0.025370626,
    "spy_5day_return": 0.002937033,
    "closed_outside_upper_band": True,
    "close_green_red": False,
}


# ---------------------------------------------------------------------------
# PreTradeChecklist.validate — one snapshot per cap tier
# ---------------------------------------------------------------------------

@pytest.mark.characterization
class TestPreTradeChecklist:
    """Lock in PreTradeChecklist.validate output across all 5 cap tiers."""

    def test_medium_mrna(self, assert_golden):
        """Medium-cap MRNA 3DGapFade — strong setup."""
        checker = PreTradeChecklist()
        result = checker.validate("MRNA", "Medium", METRICS_MRNA_MEDIUM)
        assert_golden("pretrade_checklist_mrna_medium", _strip_ts(result))

    def test_micro_codx(self, assert_golden):
        """Micro-cap CODX 3DGapFade — extreme parabolic."""
        checker = PreTradeChecklist()
        result = checker.validate("CODX", "Micro", METRICS_CODX_MICRO)
        assert_golden("pretrade_checklist_codx_micro", _strip_ts(result))

    def test_small_vir(self, assert_golden):
        """Small-cap VIR GapDownTrendBreak."""
        checker = PreTradeChecklist()
        result = checker.validate("VIR", "Small", METRICS_VIR_SMALL)
        assert_golden("pretrade_checklist_vir_small", _strip_ts(result))

    def test_large(self, assert_golden):
        """Large-cap reversal with lenient thresholds."""
        checker = PreTradeChecklist()
        result = checker.validate("AVGO", "Large", METRICS_LARGE)
        assert_golden("pretrade_checklist_avgo_large", _strip_ts(result))

    def test_etf(self, assert_golden):
        """ETF reversal with the most lenient thresholds."""
        checker = PreTradeChecklist()
        result = checker.validate("SMH", "ETF", METRICS_ETF)
        assert_golden("pretrade_checklist_smh_etf", _strip_ts(result))

    def test_validate_from_row(self, assert_golden):
        """validate_from_row reads ticker/cap/metrics off a frozen pd.Series."""
        row = pd.Series({"ticker": "MRNA", "cap": "Medium", **METRICS_MRNA_MEDIUM})
        checker = PreTradeChecklist()
        result = checker.validate_from_row(row)
        assert_golden("pretrade_checklist_from_row_mrna", _strip_ts(result))


# ---------------------------------------------------------------------------
# ReversalPretrade.validate — 3DGapFade per-cap + readiness gate
# ---------------------------------------------------------------------------

@pytest.mark.characterization
class TestReversalPretrade:
    """Lock in ReversalPretrade.validate output (5 criteria + readiness gate)."""

    def test_medium_mrna(self, assert_golden):
        """Medium-cap MRNA 3DGapFade — passes readiness (pct_change_3 high)."""
        pretrade = ReversalPretrade()
        result = pretrade.validate("MRNA", METRICS_MRNA_MEDIUM,
                                   setup_type="3DGapFade", cap="Medium")
        assert_golden("reversal_pretrade_mrna_medium", _strip_ts(result))

    def test_micro_codx(self, assert_golden):
        """Micro-cap CODX 3DGapFade — rvol just under Micro threshold."""
        pretrade = ReversalPretrade()
        result = pretrade.validate("CODX", METRICS_CODX_MICRO,
                                   setup_type="3DGapFade", cap="Micro")
        assert_golden("reversal_pretrade_codx_micro", _strip_ts(result))

    def test_readiness_gate_trips(self, assert_golden):
        """Medium setup that scores GO on criteria but trips the 3D-momentum readiness gate.

        pct_change_3 = 0.07 < Medium readiness threshold (0.10) flips GO -> NO-GO.
        """
        metrics = {
            "pct_from_9ema": 0.50,        # >= 0.44 pass
            "prior_day_range_atr": 1.50,  # >= 1.15 pass
            "rvol_score": 3.00,           # >= 2.80 pass
            "gap_pct": 0.10,              # >= 0.07 pass
            "pct_change_3": 0.07,         # >= 0.09 criterion FAIL, and < 0.10 readiness
        }
        pretrade = ReversalPretrade()
        result = pretrade.validate("TEST", metrics, setup_type="3DGapFade", cap="Medium")
        assert_golden("reversal_pretrade_readiness_gate", _strip_ts(result))

    def test_etf(self, assert_golden):
        """ETF-cap 3DGapFade — lenient per-cap thresholds."""
        pretrade = ReversalPretrade()
        result = pretrade.validate("SMH", METRICS_ETF,
                                   setup_type="3DGapFade", cap="ETF")
        assert_golden("reversal_pretrade_smh_etf", _strip_ts(result))
