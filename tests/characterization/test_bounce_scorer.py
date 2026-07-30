"""Characterization (golden) tests for analyzers/bounce_scorer.py.

Locks in the CURRENT behavior of the two public entry points:
  - BounceScorer().score_setup(...)  -> dict (7-criteria historical scoring)
  - BouncePretrade().validate(...)   -> ChecklistResult (6-criteria pre-trade)

All inputs are FROZEN hardcoded dicts sourced from real rows in
data/bounce_data.csv (so the tests are hermetic and do not read the CSV at
runtime). A future refactor that changes a score/grade/recommendation will
fail loudly against these snapshots.
"""
import dataclasses

import pytest

from analyzers.bounce_scorer import BounceScorer, BouncePretrade


pytestmark = pytest.mark.characterization


# ---------------------------------------------------------------------------
# Frozen inputs — real metric values pulled from data/bounce_data.csv rows.
# Each dict carries the 7 scoring metrics plus the classification/bonus/warning
# fields that BouncePretrade.validate() reads.
# ---------------------------------------------------------------------------

# COIN 5/12/2022 — 3DGapFade_weakstock, Medium cap (deep capitulation)
COIN_WEAK_MEDIUM = {
    'selloff_total_pct': -0.587245,
    'pct_off_30d_high': -0.764746,
    'gap_pct': -0.096426,
    'one_day_before_range_pct': 1.601724,
    'prior_day_range_atr': 1.601724,
    'pct_change_3': -0.418752,
    'pct_off_52wk_high': -0.868420,
    'bounce_open_close_pct': 0.205192,
    'pct_from_200mav': -0.736904,
    'pct_change_30': -0.753228,
    'consecutive_down_days': 5.0,
    'day_of_range_pct': 1.303244,
    'spy_open_close_pct': 0.007628,
    'closed_outside_lower_band': True,
    'prior_day_close_vs_low_pct': 0.320755,
    'bollinger_width': 0.870108,
}

# ARKK 5/12/2022 — 3DGapFade_weakstock, ETF cap (scores mid -> CAUTION pretrade)
ARKK_WEAK_ETF = {
    'selloff_total_pct': -0.295632,
    'pct_off_30d_high': -0.500417,
    'gap_pct': -0.027349,
    'one_day_before_range_pct': 1.134726,
    'prior_day_range_atr': 1.134726,
    'pct_change_3': -0.126672,
    'pct_off_52wk_high': -0.728906,
    'bounce_open_close_pct': 0.085468,
    'pct_from_200mav': -0.557163,
    'pct_change_30': -0.476995,
    'consecutive_down_days': 5.0,
    'day_of_range_pct': 1.324681,
    'spy_open_close_pct': 0.007628,
    'closed_outside_lower_band': True,
    'prior_day_close_vs_low_pct': 0.068650,
    'bollinger_width': 0.510506,
}

# TSM 8/5/2024 — 3DGapFade_strongstock, Large cap (above 200MA)
TSM_STRONG_LARGE = {
    'selloff_total_pct': -0.096140,
    'pct_off_30d_high': -0.308110,
    'gap_pct': -0.153749,
    'one_day_before_range_pct': 1.275106,
    'prior_day_range_atr': 1.275106,
    'pct_change_3': -0.192642,
    'pct_off_52wk_high': -0.308110,
    'bounce_open_close_pct': 0.105259,
    'pct_from_200mav': 0.138337,
    'pct_change_30': -0.230513,
    'consecutive_down_days': 2.0,
    'day_of_range_pct': 1.693586,
    'spy_open_close_pct': 0.011219,
    'closed_outside_lower_band': False,
    'prior_day_close_vs_low_pct': 0.426112,
    'bollinger_width': 0.298036,
}

# XPEV 12/2/2020 — 3DGapFade_strongstock, Medium cap
XPEV_STRONG_MEDIUM = {
    'selloff_total_pct': -0.185439,
    'pct_off_30d_high': -0.368909,
    'gap_pct': -0.102177,
    'one_day_before_range_pct': 1.308260,
    'prior_day_range_atr': 1.308260,
    'pct_change_3': -0.268668,
    'pct_off_52wk_high': -0.368909,
    'bounce_open_close_pct': 0.191236,
    'pct_from_200mav': 1.885043,
    'pct_change_30': 1.334161,
    'consecutive_down_days': 2.0,
    'day_of_range_pct': 1.160713,
    'spy_open_close_pct': 0.005400,
    'closed_outside_lower_band': False,
    'prior_day_close_vs_low_pct': 0.087967,
    'bollinger_width': 1.241854,
}

# SOXL 8/5/2024 — 3DGapFade_strongstock, ETF cap (deep ETF selloff)
SOXL_STRONG_ETF = {
    'selloff_total_pct': -0.339040,
    'pct_off_30d_high': -0.664384,
    'gap_pct': -0.328000,
    'one_day_before_range_pct': 1.069755,
    'prior_day_range_atr': 1.069755,
    'pct_change_3': -0.470151,
    'pct_off_52wk_high': -0.664384,
    'bounce_open_close_pct': 0.185799,
    'pct_from_200mav': -0.218066,
    'pct_change_30': -0.595876,
    'consecutive_down_days': 2.0,
    'day_of_range_pct': 1.000348,
    'spy_open_close_pct': 0.011219,
    'closed_outside_lower_band': False,
    'prior_day_close_vs_low_pct': 0.347826,
    'bollinger_width': 0.930130,
}

# GME 3/10/2021 — 2DGapFade_strongstock, Medium cap. Shallow selloff + gap UP
# + positive 3-day momentum => fails most criteria => NO-GO pretrade.
GME_STRONG_MEDIUM_NOGO = {
    'selloff_total_pct': 0.000000,
    'pct_off_30d_high': -0.442174,
    'gap_pct': 0.091252,
    'one_day_before_range_pct': 1.616497,
    'prior_day_range_atr': 1.616497,
    'pct_change_3': 0.956077,
    'pct_off_52wk_high': -0.442174,
    'bounce_open_close_pct': -0.016442,
    'pct_from_200mav': 11.488734,
    'pct_change_30': 0.820719,
    'consecutive_down_days': 0.0,
    'day_of_range_pct': 3.815332,
    'spy_open_close_pct': -0.000282,
    'closed_outside_lower_band': False,
    'prior_day_close_vs_low_pct': 0.928641,
    'bollinger_width': 2.457243,
}


def _result_to_golden(result):
    """Convert a ChecklistResult to a JSON-able dict, scrubbing the
    non-deterministic timestamp so snapshots are stable across runs."""
    d = dataclasses.asdict(result)
    d['timestamp'] = '<scrubbed>'
    return d


# ---------------------------------------------------------------------------
# BounceScorer.score_setup — historical 7-criteria scoring
# ---------------------------------------------------------------------------

class TestScoreSetup:
    """Golden snapshots of BounceScorer.score_setup across profiles and caps."""

    def setup_method(self):
        self.scorer = BounceScorer()

    def test_weakstock_medium(self, assert_golden):
        """Deep weak-stock capitulation (COIN) at Medium cap — expect a top score."""
        result = self.scorer.score_setup(
            'COIN', '2022-05-12', 'GapFade_weakstock', COIN_WEAK_MEDIUM, cap='Medium')
        assert_golden('bounce_score_weakstock_medium', result)

    def test_weakstock_etf(self, assert_golden):
        """Weak-stock bounce (ARKK) at ETF cap — mid-tier score."""
        result = self.scorer.score_setup(
            'ARKK', '2022-05-12', 'GapFade_weakstock', ARKK_WEAK_ETF, cap='ETF')
        assert_golden('bounce_score_weakstock_etf', result)

    def test_strongstock_large(self, assert_golden):
        """Strong-stock pullback (TSM) at Large cap."""
        result = self.scorer.score_setup(
            'TSM', '2024-08-05', 'GapFade_strongstock', TSM_STRONG_LARGE, cap='Large')
        assert_golden('bounce_score_strongstock_large', result)

    def test_strongstock_medium(self, assert_golden):
        """Strong-stock pullback (XPEV) at Medium cap."""
        result = self.scorer.score_setup(
            'XPEV', '2020-12-02', 'GapFade_strongstock', XPEV_STRONG_MEDIUM, cap='Medium')
        assert_golden('bounce_score_strongstock_medium', result)

    def test_strongstock_etf(self, assert_golden):
        """Strong-stock pullback (SOXL) at ETF cap — deep leveraged-ETF selloff."""
        result = self.scorer.score_setup(
            'SOXL', '2024-08-05', 'GapFade_strongstock', SOXL_STRONG_ETF, cap='ETF')
        assert_golden('bounce_score_strongstock_etf', result)


# ---------------------------------------------------------------------------
# BouncePretrade.validate — live 6-criteria pre-trade checklist
# ---------------------------------------------------------------------------

class TestValidate:
    """Golden snapshots of BouncePretrade.validate covering GO/CAUTION/NO-GO."""

    def setup_method(self):
        self.checker = BouncePretrade()

    def test_go_weakstock_medium(self, assert_golden):
        """COIN classifies weakstock (below 200MA) and passes -> GO."""
        result = self.checker.validate('COIN', COIN_WEAK_MEDIUM, cap='Medium')
        assert result.recommendation == 'GO'
        assert_golden('bounce_validate_go_weakstock_medium', _result_to_golden(result))

    def test_caution_weakstock_etf(self, assert_golden):
        """ARKK weakstock at ETF cap lands at the CAUTION boundary (4/6)."""
        result = self.checker.validate('ARKK', ARKK_WEAK_ETF, cap='ETF')
        assert result.recommendation == 'CAUTION'
        assert_golden('bounce_validate_caution_weakstock_etf', _result_to_golden(result))

    def test_nogo_strongstock_medium(self, assert_golden):
        """GME (above 200MA, gap up, positive momentum) fails -> NO-GO."""
        result = self.checker.validate('GME', GME_STRONG_MEDIUM_NOGO, cap='Medium')
        assert result.recommendation == 'NO-GO'
        assert_golden('bounce_validate_nogo_strongstock_medium', _result_to_golden(result))
