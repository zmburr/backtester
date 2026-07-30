"""Characterization + unit tests for analyzers/reversal_scorer.py.

Golden tests lock in the CURRENT behavior of ReversalScorer.score_setup and
compute_reversal_intensity using FROZEN hardcoded inputs (realistic values
copied from data/reversal_data.csv). Pure-math unit tests verify the small
deterministic helpers independently.

NOTE on coupling:
- compute_reversal_intensity goldens here pass an explicit, hand-built ref_df,
  so they are fully hermetic / decoupled from the on-disk CSV.
- score_setup, however, has NO ref_df parameter. Its momentum-percentile gate
  and its intensity sub-score percentile-rank against module-level reference
  data loaded from data/reversal_data.csv at import time. Those score_setup
  goldens are therefore COUPLED to the current contents of reversal_data.csv
  (the Grade-A subset). If that CSV changes, momentum_pctile_3 / intensity /
  possibly pretrade_recommendation may shift and these goldens must be re-blessed.
"""

import dataclasses

import pandas as pd
import pytest

from analyzers.reversal_scorer import (
    ReversalScorer,
    compute_reversal_intensity,
    CriteriaThresholds,
    check_archetype,
    CAP_THRESHOLDS,
)


# ---------------------------------------------------------------------------
# Frozen, realistic metric fixtures (values lifted from data/reversal_data.csv)
# Each dict carries every key score_setup / compute_reversal_intensity read.
# ---------------------------------------------------------------------------
METRICS_MICRO_CODX = {  # CODX 2/28/2020, Micro, grade A, 3DGapFade
    'pct_from_9ema': 1.681372029,
    'prior_day_range_atr': 9.538105513,
    'rvol_score': 1.645657433,
    'pct_change_3': 4.022016222,
    'gap_pct': 0.357769424,
    'reversal_open_close_pct': -0.389478542,
    'atr_pct': 0.0561,
    'pct_from_50mav': 7.197431848,
    'pct_from_200mav': 14.36056381,
    'pct_change_30': 19.44339623,
    'breaks_fifty_two_wk': True,
}

METRICS_MEDIUM_AMD = {  # AMD 9/5/2018, Medium, grade A, 2DBreakoutIB
    'pct_from_9ema': 0.177610333,
    'prior_day_range_atr': 2.603718632,
    'rvol_score': 2.275649339,
    'pct_change_3': 0.181599036,
    'gap_pct': 0.04811119,
    'reversal_open_close_pct': -0.030601836,
    'atr_pct': 0.0348,
    'pct_from_50mav': 0.570449944,
    'pct_from_200mav': 1.186941897,
    'pct_change_30': 0.816553428,
    'breaks_fifty_two_wk': True,
}

METRICS_LARGE_MRNA = {  # MRNA 8/10/2021, Large, grade B, 2DBreakoutIB
    'pct_from_9ema': 0.208600273,
    'prior_day_range_atr': 3.616344493,
    'rvol_score': 1.398952311,
    'pct_change_3': 0.166338346,
    'gap_pct': 0.002126035,
    'reversal_open_close_pct': -0.059196704,
    'atr_pct': 0.048,
    'pct_from_50mav': 0.835192356,
    'pct_from_200mav': 1.829272628,
    'pct_change_30': 1.177715977,
    'breaks_fifty_two_wk': True,
}

METRICS_ETF_SMH = {  # SMH 5/30/2023, ETF, grade A, 3DGapFade
    'pct_from_9ema': 0.102991377,
    'prior_day_range_atr': 2.428905383,
    'rvol_score': 1.886474361,
    'pct_change_3': 0.164759548,
    'gap_pct': 0.022326276,
    'reversal_open_close_pct': -0.017125788,
    'atr_pct': 0.025370626,
    'pct_from_50mav': 0.190290319,
    'pct_from_200mav': 0.338156229,
    'pct_change_30': 0.196109567,
    'breaks_fifty_two_wk': True,
}


# ---------------------------------------------------------------------------
# A small FIXED reference DataFrame for compute_reversal_intensity.
# Columns are exactly the ATR-adjusted / raw metrics the spec percentile-ranks
# against. Hand-made so the intensity goldens are hermetic (no CSV dependency).
# ---------------------------------------------------------------------------
def _fixed_ref_df():
    return pd.DataFrame({
        'ema9_atr':            [1.0, 3.0, 5.0, 10.0, 30.0],
        'mom3_atr':            [2.0, 5.0, 8.0, 20.0, 70.0],
        'gap_atr':             [0.1, 0.5, 1.0, 3.0, 6.0],
        'prior_day_range_atr': [0.8, 1.5, 2.5, 5.0, 9.5],
        'rvol_score':          [1.0, 1.4, 1.9, 2.5, 3.0],
        'pct_from_50mav':      [0.2, 0.6, 1.0, 3.0, 7.0],
    })


@pytest.fixture
def scorer():
    return ReversalScorer()


# ===========================================================================
# Characterization (golden) tests — ReversalScorer.score_setup
# ===========================================================================
@pytest.mark.characterization
class TestScoreSetupGolden:
    """Lock in score_setup output across caps and typed/generic paths."""

    def test_micro_codx_typed_3dgapfade(self, scorer, assert_golden):
        """Micro cap, euphoric 3DGapFade setup (typed path, readiness gate active)."""
        result = scorer.score_setup(
            ticker='CODX', date='2/28/2020', cap='Micro',
            metrics=METRICS_MICRO_CODX, setup='3DGapFade',
        )
        assert_golden('reversal_score_micro_codx_typed', result)

    def test_medium_amd_typed_euphoric(self, scorer, assert_golden):
        """Medium cap, euphoric 2DBreakoutIB setup (typed path)."""
        result = scorer.score_setup(
            ticker='AMD', date='9/5/2018', cap='Medium',
            metrics=METRICS_MEDIUM_AMD, setup='2DBreakoutIB',
        )
        assert_golden('reversal_score_medium_amd_typed', result)

    def test_medium_amd_generic_no_setup(self, scorer, assert_golden):
        """Medium cap, generic path (setup=None) — no readiness gate."""
        result = scorer.score_setup(
            ticker='AMD', date='9/5/2018', cap='Medium',
            metrics=METRICS_MEDIUM_AMD, setup=None,
        )
        assert_golden('reversal_score_medium_amd_generic', result)

    def test_large_mrna_generic(self, scorer, assert_golden):
        """Large cap, generic path (setup=None)."""
        result = scorer.score_setup(
            ticker='MRNA', date='8/10/2021', cap='Large',
            metrics=METRICS_LARGE_MRNA, setup=None,
        )
        assert_golden('reversal_score_large_mrna_generic', result)

    def test_etf_smh_typed_3dgapfade(self, scorer, assert_golden):
        """ETF cap, typed 3DGapFade setup."""
        result = scorer.score_setup(
            ticker='SMH', date='5/30/2023', cap='ETF',
            metrics=METRICS_ETF_SMH, setup='3DGapFade',
        )
        assert_golden('reversal_score_etf_smh_typed', result)


# ===========================================================================
# Characterization (golden) tests — compute_reversal_intensity (hermetic)
# ===========================================================================
@pytest.mark.characterization
class TestIntensityGolden:
    """Lock in intensity composites against a fixed, explicit ref_df."""

    def test_intensity_high_setup(self, assert_golden):
        """Strong setup vs fixed ref — high composite expected."""
        result = compute_reversal_intensity(
            METRICS_MICRO_CODX, cap='Micro', ref_df=_fixed_ref_df())
        assert_golden('reversal_intensity_high_codx', result)

    def test_intensity_mid_setup(self, assert_golden):
        """Mid-strength setup vs fixed ref."""
        result = compute_reversal_intensity(
            METRICS_MEDIUM_AMD, cap='Medium', ref_df=_fixed_ref_df())
        assert_golden('reversal_intensity_mid_amd', result)

    def test_intensity_low_setup(self, assert_golden):
        """Weak setup (small ETF metrics) vs fixed ref — low composite."""
        result = compute_reversal_intensity(
            METRICS_ETF_SMH, cap='ETF', ref_df=_fixed_ref_df())
        assert_golden('reversal_intensity_low_smh', result)

    def test_intensity_missing_atr_returns_none(self, assert_golden):
        """No atr_pct -> composite None, empty details (guard path)."""
        m = dict(METRICS_MEDIUM_AMD)
        m['atr_pct'] = 0  # falsy / zero ATR triggers the early return
        result = compute_reversal_intensity(m, cap='Medium', ref_df=_fixed_ref_df())
        assert_golden('reversal_intensity_no_atr', result)


# ===========================================================================
# Pure-math UNIT tests — independently hand-computed assertions
# ===========================================================================
class TestGradeMapping:
    """Hand-verified score->grade and recommendation maps."""

    def test_full_score_to_grade(self, scorer):
        """6->A+, 5->A, 4->B, 3->C, else F (per _score_to_grade)."""
        assert scorer._score_to_grade(6) == 'A+'
        assert scorer._score_to_grade(5) == 'A'
        assert scorer._score_to_grade(4) == 'B'
        assert scorer._score_to_grade(3) == 'C'
        assert scorer._score_to_grade(2) == 'F'
        assert scorer._score_to_grade(0) == 'F'

    def test_pretrade_grade(self):
        """5->A+, 4->A, 3->B, 2->C, else F (per _pretrade_grade)."""
        assert ReversalScorer._pretrade_grade(5) == 'A+'
        assert ReversalScorer._pretrade_grade(4) == 'A'
        assert ReversalScorer._pretrade_grade(3) == 'B'
        assert ReversalScorer._pretrade_grade(2) == 'C'
        assert ReversalScorer._pretrade_grade(1) == 'F'

    def test_full_recommendation(self, scorer):
        """>=5 GO, ==4 CAUTION, else NO-GO."""
        assert scorer._get_recommendation(6) == 'GO'
        assert scorer._get_recommendation(5) == 'GO'
        assert scorer._get_recommendation(4) == 'CAUTION'
        assert scorer._get_recommendation(3) == 'NO-GO'

    def test_pretrade_recommendation(self):
        """>=4 GO, ==3 CAUTION, else NO-GO."""
        assert ReversalScorer._pretrade_recommendation(5) == 'GO'
        assert ReversalScorer._pretrade_recommendation(4) == 'GO'
        assert ReversalScorer._pretrade_recommendation(3) == 'CAUTION'
        assert ReversalScorer._pretrade_recommendation(2) == 'NO-GO'


class TestCriterionCheck:
    """Hand-verified single-criterion comparisons."""

    def test_normal_ge_threshold(self, scorer):
        """value >= threshold passes; below fails."""
        assert scorer._check_criterion(2.0, 2.0) is True
        assert scorer._check_criterion(2.5, 2.0) is True
        assert scorer._check_criterion(1.99, 2.0) is False

    def test_reversal_le_negative_threshold(self, scorer):
        """Reversal stored negative: passes when value <= threshold."""
        # -0.20 <= -0.10 -> True; -0.05 <= -0.10 -> False
        assert scorer._check_criterion(-0.20, -0.10, is_reversal=True) is True
        assert scorer._check_criterion(-0.10, -0.10, is_reversal=True) is True
        assert scorer._check_criterion(-0.05, -0.10, is_reversal=True) is False

    def test_nan_and_none_fail(self, scorer):
        """NaN / None always fail."""
        assert scorer._check_criterion(float('nan'), 2.0) is False
        assert scorer._check_criterion(None, 2.0) is False


class TestEvaluateCriteriaCount:
    """Hand-counted criteria pass tally for a known Medium setup."""

    def test_medium_amd_passes_all_six(self, scorer):
        """AMD/Medium clears every threshold -> score 6, no failures.

        Medium thresholds: 9ema>=0.15, range>=1.2, rvol>=2.0, mom3>=0.10,
        gap>=0.05? (gap 0.0481 < 0.05 -> FAIL), reversal<=-0.05 (-0.0306 -> FAIL).
        So expect 4 passes (9ema, range, rvol, mom3) and 2 fails (gap, reversal).
        """
        thr = CAP_THRESHOLDS['Medium']
        score, passed, failed = scorer._evaluate_criteria(METRICS_MEDIUM_AMD, thr)
        assert score == 4
        assert set(passed) == {'pct_from_9ema', 'prior_day_range_atr',
                               'rvol_score', 'pct_change_3'}
        assert set(failed) == {'gap_pct', 'reversal_pct'}


class TestArchetypeGate:
    """Hand-verified check_archetype boolean logic."""

    def test_passes_by_52wk(self):
        """breaks_fifty_two_wk True alone passes."""
        passed, detail = check_archetype({'breaks_fifty_two_wk': True})
        assert passed is True
        assert detail['passed_by_52wk'] is True

    def test_passes_by_200mav(self):
        """pct_from_200mav 0.60 > 0.50 threshold passes."""
        passed, detail = check_archetype({'pct_from_200mav': 0.60})
        assert passed is True
        assert detail['passed_by_200mav'] is True
        assert detail['passed_by_30d'] is False

    def test_passes_by_30d(self):
        """pct_change_30 0.35 > 0.30 threshold passes."""
        passed, detail = check_archetype({'pct_change_30': 0.35})
        assert passed is True
        assert detail['passed_by_30d'] is True

    def test_empty_metrics_fails(self):
        """No metrics -> safe default fail."""
        passed, detail = check_archetype({})
        assert passed is False

    def test_boundary_not_strictly_greater(self):
        """Exactly at threshold does NOT pass (strict >)."""
        passed, _ = check_archetype({'pct_from_200mav': 0.50, 'pct_change_30': 0.30})
        assert passed is False


class TestComputeIntensityMath:
    """Hand-computed composite against a trivial single-row ref."""

    def test_single_row_ref_all_above(self):
        """With one ref row below every metric, each pctile = 50 (tie-broken),
        actually score>ref so strict-rank logic gives 100; verify composite=100.

        _pctrank(rank) = 100*(n_lt + 0.5*n_eq)/n. With 1 ref row strictly below
        each derived value, n_lt=1, n_eq=0 -> pctile=100 for all 6 -> composite 100.
        """
        ref = pd.DataFrame({
            'ema9_atr':            [0.0],
            'mom3_atr':            [0.0],
            'gap_atr':             [0.0],
            'prior_day_range_atr': [0.0],
            'rvol_score':          [0.0],
            'pct_from_50mav':      [0.0],
        })
        metrics = {
            'atr_pct': 0.05,
            'pct_from_9ema': 1.0,   # ema9_atr = 20
            'pct_change_3': 1.0,    # mom3_atr = 20
            'gap_pct': 0.05,        # gap_atr = 1
            'prior_day_range_atr': 5.0,
            'rvol_score': 3.0,
            'pct_from_50mav': 2.0,
        }
        result = compute_reversal_intensity(metrics, ref_df=ref)
        assert result['composite'] == 100.0

    def test_midpoint_no_tie_composite_50(self):
        """Each derived value lands strictly between the 2nd and 3rd of 4 ref
        rows -> 2 of 4 strictly below, no ties -> pctile 50 for all 6 metrics ->
        weighted composite = 50.0 (tie-free so independent of rank semantics).

        atr_pct=0.1 so: ema9_atr=1.5/0.1=15 (ref 0,10,20,30 -> 50),
        mom3_atr=15 (50), gap_atr=0.15/0.1=1.5 (ref 0,1,2,3 -> 50),
        prior_day_range_atr=1.5 (50), rvol_score=1.5 (50), pct_from_50mav=1.5 (50).
        """
        ref = pd.DataFrame({
            'ema9_atr':            [0.0, 10.0, 20.0, 30.0],
            'mom3_atr':            [0.0, 10.0, 20.0, 30.0],
            'gap_atr':             [0.0, 1.0, 2.0, 3.0],
            'prior_day_range_atr': [0.0, 1.0, 2.0, 3.0],
            'rvol_score':          [0.0, 1.0, 2.0, 3.0],
            'pct_from_50mav':      [0.0, 1.0, 2.0, 3.0],
        })
        metrics = {
            'atr_pct': 0.1,
            'pct_from_9ema': 1.5,
            'pct_change_3': 1.5,
            'gap_pct': 0.15,
            'prior_day_range_atr': 1.5,
            'rvol_score': 1.5,
            'pct_from_50mav': 1.5,
        }
        result = compute_reversal_intensity(metrics, ref_df=ref)
        assert result['composite'] == 50.0


class TestThresholdsDataclass:
    """Sanity on the dataclass-based thresholds and unknown-cap fallback."""

    def test_unknown_cap_falls_back_to_medium(self, scorer):
        """Unknown cap returns the Medium threshold object."""
        assert scorer._get_thresholds('Nonsense') is CAP_THRESHOLDS['Medium']

    def test_known_caps_present(self):
        """All five cap tiers exist and are CriteriaThresholds."""
        for cap in ['Micro', 'Small', 'Medium', 'Large', 'ETF']:
            assert isinstance(CAP_THRESHOLDS[cap], CriteriaThresholds)
