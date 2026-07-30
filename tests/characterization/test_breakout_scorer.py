"""Characterization tests for analyzers/breakout_scorer.py.

Locks in CURRENT behavior of the breakout setup scorer, the live pre-trade
checklist, and the ATR-adjusted intensity composite.

All inputs are FROZEN/hardcoded here (realistic values lifted from
data/breakout_data.csv) so the score_setup / validate goldens are hermetic.

IMPORTANT COUPLING NOTE: compute_breakout_intensity (and therefore the
`intensity`, `intensity_tier`, `tier_thresholds`, and `recommendation` fields
of score_setup / validate) percentile-ranks against the A+B reference
distribution loaded from data/breakout_data.csv into module globals
(_REF_BY_PROFILE / _INTENSITY_THRESHOLDS) at import time. There is no ref
parameter to inject. If breakout_data.csv changes, these intensity-dependent
goldens must be re-blessed with REGEN_GOLDEN=1.
"""
import dataclasses

import pytest

from analyzers.breakout_scorer import (
    BreakoutScorer,
    BreakoutPretrade,
    compute_breakout_intensity,
)


# ---------------------------------------------------------------------------
# Frozen, realistic inputs (sourced from data/breakout_data.csv)
# ---------------------------------------------------------------------------

# D1 (t=0) — PLTR 2/6/2024, Grade A, 52wk_breakout/gap_n_go. Outcome just below
# the D1 outcome threshold (0.0926 < 0.1002), so the outcome criterion fails.
PLTR_D1 = {
    't': 0.0,
    'ticker': 'PLTR',
    'setup_type': '52wk_breakout,gap_n_go,news_day1',
    'pct_from_9ema': 0.252066,
    'pct_from_50mav': 0.267313,
    'range_contraction_atr': 1.231207,
    'prior_day_close_vs_high_pct': 0.172662,
    'gap_from_pm_high': -0.001476,
    'atr_pct': 0.039587,
    'gap_pct': 0.214115,
    'breakout_open_high_pct': 0.092611,
}

# D1 (t=0) — DWAC 1/22/2024, Grade B, extremely extended (gap_n_go). High intensity.
DWAC_D1 = {
    't': 0.0,
    'ticker': 'DWAC',
    'setup_type': '52wk_breakout,gap_n_go,news_day1',
    'pct_from_9ema': 0.840244,
    'pct_from_50mav': 1.731497,
    'range_contraction_atr': 3.933200,
    'prior_day_close_vs_high_pct': 0.431973,
    'gap_from_pm_high': -0.021681,
    'atr_pct': 0.056870,
    'gap_pct': 0.111827,
    'breakout_open_high_pct': 0.711558,
}

# D1 (t=0) — TSLA 1/7/2021, Grade B, moderate extension. Mid-tier intensity.
TSLA_D1 = {
    't': 0.0,
    'ticker': 'TSLA',
    'setup_type': 'ATH_breakout,news_day1',
    'pct_from_9ema': 0.123108,
    'pct_from_50mav': 0.457207,
    'range_contraction_atr': 0.891181,
    'prior_day_close_vs_high_pct': 0.276301,
    'gap_from_pm_high': -0.001759,
    'atr_pct': 0.050803,
    'gap_pct': 0.028638,
    'breakout_open_high_pct': 0.050615,
}

# D2 (t=1) — ARM 2/8/2024, Grade A, post-earnings IPO/ATH continuation. Strong.
ARM_D2 = {
    't': 1.0,
    'ticker': 'ARM',
    'setup_type': 'ATH_breakout,IPO_high_breakout,news_day2',
    'pct_from_9ema': 0.554454776,
    'pct_to_52wk_high': -0.036652489,
    'atr_pct': 0.018192968,
    'gap_pct': 0.225944683,
    'percent_of_premarket_vol': 0.545401674,
    'overnight_gap_d1_to_d2_pct': 0.871794872,
    'breakout_open_high_pct': 0.726591075,
}

# D2 (t=1) — SOXL 2/9/2024, Grade B, ATH continuation. Weaker proximity metrics.
SOXL_D2 = {
    't': 1.0,
    'ticker': 'SOXL',
    'setup_type': 'ATH_breakout,news_day2',
    'pct_from_9ema': 0.10937408,
    'pct_to_52wk_high': -0.060207437,
    'atr_pct': 0.04437401,
    'gap_pct': 0.019111709,
    'percent_of_premarket_vol': 0.080496129,
    'overnight_gap_d1_to_d2_pct': 0.019111709,
    'breakout_open_high_pct': 0.830754352,
}

# D2 (t=1) — synthetic "floor" case with realistically-low metrics, used to
# exercise the AVOID tier (intensity below the historical tradable floor).
WEAK_D2 = {
    't': 1.0,
    'ticker': 'WEAK',
    'setup_type': 'news_day2',
    'pct_from_9ema': 0.01,
    'pct_to_52wk_high': -0.60,
    'atr_pct': 0.02,
    'gap_pct': -0.02,
    'percent_of_premarket_vol': 0.005,
    'overnight_gap_d1_to_d2_pct': -0.01,
}


def _checklist_to_golden(result):
    """asdict a ChecklistResult and strip the non-deterministic timestamp."""
    d = dataclasses.asdict(result)
    d.pop('timestamp', None)  # datetime.now() — not reproducible
    return d


# ---------------------------------------------------------------------------
# BreakoutScorer.score_setup — historical scoring (pretrade + outcome criterion)
# ---------------------------------------------------------------------------

@pytest.mark.characterization
class TestScoreSetup:
    """Golden snapshots of BreakoutScorer.score_setup for D1 and D2 setups."""

    def test_score_d1_pltr(self, assert_golden):
        """D1 news-break (PLTR): outcome criterion fails -> grade A, FULL_SIZE."""
        scorer = BreakoutScorer()
        result = scorer.score_setup('PLTR', '2024-02-06', 'D1_news_break', PLTR_D1)
        assert_golden('breakout_score_pltr_d1', result)

    def test_score_d1_dwac(self, assert_golden):
        """D1 news-break (DWAC): extremely extended, high intensity, full pass."""
        scorer = BreakoutScorer()
        result = scorer.score_setup('DWAC', '2024-01-22', 'D1_news_break', DWAC_D1)
        assert_golden('breakout_score_dwac_d1', result)

    def test_score_d2_arm(self, assert_golden):
        """D2 continuation (ARM): strong post-earnings continuation."""
        scorer = BreakoutScorer()
        result = scorer.score_setup('ARM', '2024-02-08', 'D2_continuation', ARM_D2)
        assert_golden('breakout_score_arm_d2', result)

    def test_score_d2_soxl(self, assert_golden):
        """D2 continuation (SOXL): weaker proximity metrics, mid intensity."""
        scorer = BreakoutScorer()
        result = scorer.score_setup('SOXL', '2024-02-09', 'D2_continuation', SOXL_D2)
        assert_golden('breakout_score_soxl_d2', result)


# ---------------------------------------------------------------------------
# BreakoutPretrade.validate — live checklist (no outcome leakage)
# ---------------------------------------------------------------------------

@pytest.mark.characterization
class TestPretradeValidate:
    """Golden snapshots of BreakoutPretrade.validate across recommendation tiers."""

    def test_validate_full_size_dwac(self, assert_golden):
        """FULL_SIZE: DWAC D1 intensity above the A+B median."""
        checker = BreakoutPretrade()
        result = checker.validate('DWAC', DWAC_D1, force_setup='D1_news_break')
        assert_golden('breakout_pretrade_dwac_full_size', _checklist_to_golden(result))

    def test_validate_reduced_size_tsla(self, assert_golden):
        """REDUCED_SIZE: TSLA D1 intensity above floor but below median."""
        checker = BreakoutPretrade()
        result = checker.validate('TSLA', TSLA_D1, force_setup='D1_news_break')
        assert_golden('breakout_pretrade_tsla_reduced_size', _checklist_to_golden(result))

    def test_validate_avoid_weak(self, assert_golden):
        """AVOID: weak D2 intensity below the historical tradable floor."""
        checker = BreakoutPretrade()
        result = checker.validate('WEAK', WEAK_D2)
        assert_golden('breakout_pretrade_weak_avoid', _checklist_to_golden(result))


# ---------------------------------------------------------------------------
# compute_breakout_intensity — ATR-adjusted percentile composite
# ---------------------------------------------------------------------------

@pytest.mark.characterization
class TestComputeIntensity:
    """Golden snapshots of compute_breakout_intensity.

    Coupled to data/breakout_data.csv (no ref param). Re-bless if the CSV changes.
    """

    def test_intensity_d1_dwac(self, assert_golden):
        """D1 intensity composite for the heavily-extended DWAC setup."""
        result = compute_breakout_intensity(DWAC_D1, 'D1_news_break')
        assert_golden('breakout_intensity_dwac_d1', result)

    def test_intensity_d2_arm(self, assert_golden):
        """D2 intensity composite for the strong ARM continuation setup."""
        result = compute_breakout_intensity(ARM_D2, 'D2_continuation')
        assert_golden('breakout_intensity_arm_d2', result)
