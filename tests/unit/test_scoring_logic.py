"""Pure branch-logic unit tests for the scorer classification / criteria helpers.

These are NOT snapshots. Every expected value is reasoned out by hand directly
from the threshold constants in the modules under test. Each test constructs a
minimal metrics dict that lands on a known side of a classification boundary and
asserts the exact label / bool the branch logic must return.

Functions covered:
  - analyzers.bounce_scorer.classify_stock
  - analyzers.bounce_scorer.classify_from_setup_column
  - analyzers.breakout_scorer.classify_breakout_setup
  - analyzers.breakout_scorer.get_intensity_threshold
  - analyzers.reversal_scorer.check_archetype
  - analyzers.reversal_pretrade.classify_reversal_setup

Importing reversal_scorer / breakout_scorer triggers a one-time read of the
reference CSVs at import time (reference data only); the classify_* helpers
themselves are pure and need no CSV.
"""
import numpy as np
import pytest

from analyzers.bounce_scorer import classify_stock, classify_from_setup_column
from analyzers import breakout_scorer
from analyzers.breakout_scorer import classify_breakout_setup, get_intensity_threshold
from analyzers.reversal_scorer import (
    check_archetype,
    ARCHETYPE_PCT_FROM_200MAV,
    ARCHETYPE_PCT_CHANGE_30,
)
from analyzers.reversal_pretrade import classify_reversal_setup


# ===========================================================================
# bounce_scorer.classify_stock
#   pct_from_200mav present & not NaN:
#       < 0  -> 'GapFade_weakstock'
#       >= 0 -> 'GapFade_strongstock'
#   else fallback to pct_change_30:
#       present & not NaN & <= -0.25 -> 'GapFade_weakstock'
#       otherwise                    -> 'GapFade_strongstock'
# ===========================================================================
class TestClassifyStock:
    """200MA-primary / 30d-fallback weak vs strong classification."""

    def test_below_200ma_is_weak(self):
        """pct_from_200mav < 0 -> weakstock."""
        setup, _ = classify_stock({'pct_from_200mav': -0.10})
        assert setup == 'GapFade_weakstock'

    def test_above_200ma_is_strong(self):
        """pct_from_200mav > 0 -> strongstock."""
        setup, _ = classify_stock({'pct_from_200mav': 0.10})
        assert setup == 'GapFade_strongstock'

    def test_exactly_zero_200ma_is_strong(self):
        """Boundary: pct_from_200mav == 0 is NOT < 0, so strongstock."""
        setup, _ = classify_stock({'pct_from_200mav': 0.0})
        assert setup == 'GapFade_strongstock'

    def test_tiny_negative_200ma_is_weak(self):
        """Just below zero flips to weakstock."""
        setup, _ = classify_stock({'pct_from_200mav': -1e-6})
        assert setup == 'GapFade_weakstock'

    def test_fallback_30d_at_threshold_is_weak(self):
        """No 200MA; pct_change_30 == -0.25 satisfies <= -0.25 -> weakstock."""
        setup, details = classify_stock({'pct_change_30': -0.25})
        assert setup == 'GapFade_weakstock'
        assert 'fallback' in details['signals'][0].lower()

    def test_fallback_30d_just_above_threshold_is_strong(self):
        """No 200MA; pct_change_30 == -0.24 fails <= -0.25 -> strongstock."""
        setup, _ = classify_stock({'pct_change_30': -0.24})
        assert setup == 'GapFade_strongstock'

    def test_fallback_no_metrics_defaults_strong(self):
        """No 200MA and no 30d data -> strongstock default."""
        setup, _ = classify_stock({})
        assert setup == 'GapFade_strongstock'

    def test_nan_200ma_uses_fallback(self):
        """NaN 200MA is ignored; falls back to (weak) 30d signal."""
        setup, _ = classify_stock({'pct_from_200mav': np.nan, 'pct_change_30': -0.40})
        assert setup == 'GapFade_weakstock'


# ===========================================================================
# bounce_scorer.classify_from_setup_column
#   'IntradayCapitch' in name -> 'IntradayCapitch'
#   'weakstock' in name       -> 'GapFade_weakstock'
#   else                      -> 'GapFade_strongstock'
# ===========================================================================
class TestClassifyFromSetupColumn:
    """String-membership classification of the Setup column."""

    def test_intradaycapitch_first(self):
        """IntradayCapitch wins even if 'weakstock' substring co-occurs."""
        assert classify_from_setup_column('IntradayCapitch_weakstock') == 'IntradayCapitch'

    def test_weakstock(self):
        """'weakstock' substring -> weak profile."""
        assert classify_from_setup_column('GapFade_weakstock') == 'GapFade_weakstock'

    def test_strongstock_default(self):
        """Anything else -> strong profile."""
        assert classify_from_setup_column('GapFade_strongstock') == 'GapFade_strongstock'

    def test_unknown_defaults_strong(self):
        """Unrecognized label falls through to strong."""
        assert classify_from_setup_column('SomethingElse') == 'GapFade_strongstock'


# ===========================================================================
# breakout_scorer.classify_breakout_setup
#   t == 0 -> 'D1_news_break'
#   t == 1 -> 'D2_continuation'
#   else   -> 'D2_continuation' (default)
# ===========================================================================
class TestClassifyBreakoutSetup:
    """Day-since-catalyst (t) breakout classification."""

    def test_t0_is_d1(self):
        """t==0 -> D1 news-day break."""
        setup, details = classify_breakout_setup({'t': 0})
        assert setup == 'D1_news_break'
        assert details['classification'] == 'D1_news_break'

    def test_t1_is_d2(self):
        """t==1 -> D2 continuation."""
        setup, _ = classify_breakout_setup({'t': 1})
        assert setup == 'D2_continuation'

    def test_t0_float_string_coerced(self):
        """t given as '0' string coerces via int() to 0 -> D1."""
        setup, _ = classify_breakout_setup({'t': '0'})
        assert setup == 'D1_news_break'

    def test_t2_defaults_d2(self):
        """t==2 has no clean class -> default D2."""
        setup, _ = classify_breakout_setup({'t': 2})
        assert setup == 'D2_continuation'

    def test_t_missing_defaults_d2(self):
        """Missing t -> default D2."""
        setup, _ = classify_breakout_setup({})
        assert setup == 'D2_continuation'

    def test_t_nan_defaults_d2(self):
        """NaN t -> t_int None -> default D2."""
        setup, _ = classify_breakout_setup({'t': np.nan})
        assert setup == 'D2_continuation'


# ===========================================================================
# breakout_scorer.get_intensity_threshold
#   pure dict lookup: _INTENSITY_THRESHOLDS.get(profile, {}).get(key)
# ===========================================================================
class TestGetIntensityThreshold:
    """Lookup semantics of the intensity-threshold helper (pure dict access)."""

    def test_unknown_profile_returns_none(self):
        """Profile absent from table -> None."""
        assert get_intensity_threshold('NOT_A_PROFILE', 'tradable_min') is None

    def test_unknown_key_returns_none(self):
        """Known profile but unknown key -> None (when profile present)."""
        if 'D1_news_break' in breakout_scorer._INTENSITY_THRESHOLDS:
            assert get_intensity_threshold('D1_news_break', 'no_such_key') is None
        else:
            pytest.skip('D1 reference distribution not loaded; lookup table empty')

    def test_lookup_matches_module_table(self):
        """Helper returns exactly what the module table holds for a present key."""
        table = breakout_scorer._INTENSITY_THRESHOLDS
        if 'D2_continuation' in table and 'tradable_min' in table['D2_continuation']:
            assert (get_intensity_threshold('D2_continuation', 'tradable_min')
                    == table['D2_continuation']['tradable_min'])
        else:
            pytest.skip('D2 reference distribution not loaded')

    def test_default_key_is_tradable_min(self):
        """Default key arg is 'tradable_min' -> same as explicit lookup."""
        table = breakout_scorer._INTENSITY_THRESHOLDS
        if 'D2_continuation' in table:
            assert (get_intensity_threshold('D2_continuation')
                    == table['D2_continuation'].get('tradable_min'))
        else:
            pytest.skip('D2 reference distribution not loaded')


# ===========================================================================
# reversal_scorer.check_archetype
#   Passes if ANY of:
#     breaks_fifty_two_wk truthy
#     pct_from_200mav > 0.50  (strict)
#     pct_change_30   > 0.30  (strict)
# ===========================================================================
class TestCheckArchetype:
    """Parabolic-at-highs archetype gate (OR of three sub-rules)."""

    def test_constants_are_expected(self):
        """Lock the threshold constants the rest of the math relies on."""
        assert ARCHETYPE_PCT_FROM_200MAV == 0.50
        assert ARCHETYPE_PCT_CHANGE_30 == 0.30

    def test_empty_metrics_fails(self):
        """No metrics -> no sub-rule trips -> False (safe default)."""
        passed, detail = check_archetype({})
        assert passed is False
        assert detail['passed_by_52wk'] is False
        assert detail['passed_by_200mav'] is False
        assert detail['passed_by_30d'] is False

    def test_breaks_52wk_passes(self):
        """breaks_fifty_two_wk True alone -> pass."""
        passed, detail = check_archetype({'breaks_fifty_two_wk': True})
        assert passed is True
        assert detail['passed_by_52wk'] is True

    def test_200mav_above_threshold_passes(self):
        """pct_from_200mav 0.51 > 0.50 -> pass via 200MA rule."""
        passed, detail = check_archetype({'pct_from_200mav': 0.51})
        assert passed is True
        assert detail['passed_by_200mav'] is True

    def test_200mav_at_threshold_fails(self):
        """Boundary: 0.50 is NOT > 0.50 -> 200MA rule does not trip."""
        passed, detail = check_archetype({'pct_from_200mav': 0.50})
        assert passed is False
        assert detail['passed_by_200mav'] is False

    def test_30d_above_threshold_passes(self):
        """pct_change_30 0.31 > 0.30 -> pass via 30d rule."""
        passed, detail = check_archetype({'pct_change_30': 0.31})
        assert passed is True
        assert detail['passed_by_30d'] is True

    def test_30d_at_threshold_fails(self):
        """Boundary: 0.30 is NOT > 0.30 -> 30d rule does not trip."""
        passed, detail = check_archetype({'pct_change_30': 0.30})
        assert passed is False
        assert detail['passed_by_30d'] is False

    def test_nan_values_do_not_trip(self):
        """NaN numeric metrics are treated as not-satisfying."""
        passed, _ = check_archetype({'pct_from_200mav': np.nan, 'pct_change_30': np.nan})
        assert passed is False


# ===========================================================================
# reversal_pretrade.classify_reversal_setup
#   Core gate (all required):
#     consecutive_up_days >= 2 AND gap_pct >= 0.04 AND pct_from_9ema >= 0.30
#   Supplementary rejections (only when that metric is present/valid):
#     pct_from_50mav < 0.4                                 -> None
#     atr_pct < 0.04 or > 0.20                             -> None
#     fade_day_return > 0.02 AND fade_day_close_position > 0.75 -> None
#   Otherwise -> '3DGapFade'
# ===========================================================================
class TestClassifyReversalSetup:
    """3DGapFade core gate + supplementary rejection filters."""

    # --- core gate: minimal passing input (no supplementary metrics) ---
    def test_minimal_core_pass(self):
        """All three core conditions exactly at threshold -> 3DGapFade."""
        m = {'consecutive_up_days': 2, 'gap_pct': 0.04, 'pct_from_9ema': 0.30}
        assert classify_reversal_setup(m) == '3DGapFade'

    def test_core_pass_well_inside(self):
        """Comfortably-inside core values -> 3DGapFade."""
        m = {'consecutive_up_days': 3, 'gap_pct': 0.10, 'pct_from_9ema': 0.50}
        assert classify_reversal_setup(m) == '3DGapFade'

    # --- core gate boundaries: each condition just failing ---
    def test_one_up_day_fails(self):
        """consecutive_up_days 1 < 2 -> None."""
        m = {'consecutive_up_days': 1, 'gap_pct': 0.10, 'pct_from_9ema': 0.50}
        assert classify_reversal_setup(m) is None

    def test_gap_below_threshold_fails(self):
        """gap_pct 0.039 < 0.04 -> None."""
        m = {'consecutive_up_days': 2, 'gap_pct': 0.039, 'pct_from_9ema': 0.50}
        assert classify_reversal_setup(m) is None

    def test_ema_below_threshold_fails(self):
        """pct_from_9ema 0.29 < 0.30 -> None."""
        m = {'consecutive_up_days': 2, 'gap_pct': 0.10, 'pct_from_9ema': 0.29}
        assert classify_reversal_setup(m) is None

    def test_missing_core_metric_fails(self):
        """Missing one of the three core metrics -> gate not entered -> None."""
        m = {'consecutive_up_days': 2, 'gap_pct': 0.10}  # no pct_from_9ema
        assert classify_reversal_setup(m) is None

    # --- supplementary: pct_from_50mav ---
    def test_50ma_below_floor_rejected(self):
        """Core passes but pct_from_50mav 0.39 < 0.4 -> rejected -> None."""
        m = {'consecutive_up_days': 2, 'gap_pct': 0.10, 'pct_from_9ema': 0.50,
             'pct_from_50mav': 0.39}
        assert classify_reversal_setup(m) is None

    def test_50ma_at_floor_accepted(self):
        """pct_from_50mav == 0.4 is NOT < 0.4 -> accepted -> 3DGapFade."""
        m = {'consecutive_up_days': 2, 'gap_pct': 0.10, 'pct_from_9ema': 0.50,
             'pct_from_50mav': 0.40}
        assert classify_reversal_setup(m) == '3DGapFade'

    # --- supplementary: atr_pct bounds [0.04, 0.20] ---
    def test_atr_too_low_rejected(self):
        """atr_pct 0.03 < 0.04 -> rejected -> None."""
        m = {'consecutive_up_days': 2, 'gap_pct': 0.10, 'pct_from_9ema': 0.50,
             'atr_pct': 0.03}
        assert classify_reversal_setup(m) is None

    def test_atr_too_high_rejected(self):
        """atr_pct 0.21 > 0.20 -> rejected -> None."""
        m = {'consecutive_up_days': 2, 'gap_pct': 0.10, 'pct_from_9ema': 0.50,
             'atr_pct': 0.21}
        assert classify_reversal_setup(m) is None

    def test_atr_in_bounds_accepted(self):
        """atr_pct 0.10 inside [0.04, 0.20] -> 3DGapFade."""
        m = {'consecutive_up_days': 2, 'gap_pct': 0.10, 'pct_from_9ema': 0.50,
             'atr_pct': 0.10}
        assert classify_reversal_setup(m) == '3DGapFade'

    # --- supplementary: reversal-confirmation (continuation rejection) ---
    def test_continuation_rejected(self):
        """Closed green at highs (fade_return 0.03 > 0.02 AND close_pos 0.80 > 0.75) -> None."""
        m = {'consecutive_up_days': 2, 'gap_pct': 0.10, 'pct_from_9ema': 0.50,
             'fade_day_return': 0.03, 'fade_day_close_position': 0.80}
        assert classify_reversal_setup(m) is None

    def test_green_but_not_at_highs_accepted(self):
        """Green close (0.03) but mid-range (close_pos 0.50) -> not continuation -> 3DGapFade."""
        m = {'consecutive_up_days': 2, 'gap_pct': 0.10, 'pct_from_9ema': 0.50,
             'fade_day_return': 0.03, 'fade_day_close_position': 0.50}
        assert classify_reversal_setup(m) == '3DGapFade'

    def test_at_highs_but_not_green_accepted(self):
        """At highs (close_pos 0.90) but red/flat close (fade_return 0.0) -> 3DGapFade."""
        m = {'consecutive_up_days': 2, 'gap_pct': 0.10, 'pct_from_9ema': 0.50,
             'fade_day_return': 0.0, 'fade_day_close_position': 0.90}
        assert classify_reversal_setup(m) == '3DGapFade'

    def test_all_supplementary_present_and_passing(self):
        """Full metric dict where every supplementary filter passes -> 3DGapFade."""
        m = {'consecutive_up_days': 2, 'gap_pct': 0.10, 'pct_from_9ema': 0.50,
             'pct_from_50mav': 0.60, 'atr_pct': 0.08,
             'fade_day_return': -0.05, 'fade_day_close_position': 0.20}
        assert classify_reversal_setup(m) == '3DGapFade'
