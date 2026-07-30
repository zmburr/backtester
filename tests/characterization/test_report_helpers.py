"""Characterization + unit tests for the PURE helpers in scripts/generate_report.py.

Importing the module is a one-time heavy cost (it reads data/bounce_data.csv and
prints a summary line at import). We import it as a module and ONLY exercise the
pure functions below — never generate_report() (which hits Polygon + sends email).

- route_playbook: pure routing logic -> UNIT tests asserting each branch.
- _fmt_pct / _fmt_dollars / _calc_price_from_frac: trivial pure math -> hand-computed
  unit asserts AND golden snapshots.
- format_window_label / _format_percentile_dict / format_reversal_intensity_html /
  compute_bounce_intensity: golden snapshots over frozen inputs.
"""
import dataclasses  # noqa: F401  (kept for parity with sibling test files)

import pandas as pd
import pytest

from scripts import generate_report as gr


# ---------------------------------------------------------------------------
# route_playbook — pure routing logic (UNIT tests, exact-bucket assertions)
# ---------------------------------------------------------------------------
class TestRoutePlaybook:
    """Each branch of the playbook router asserted independently."""

    def test_manual_override_wins(self):
        """Ticker in BREAKOUT_WATCHLIST forces breakout, no scoring."""
        bucket, reason, ambiguous = gr.route_playbook(
            {}, {}, ticker="AAPL", breakout_watchlist=["AAPL"]
        )
        assert bucket == "breakout"
        assert reason == "explicit BREAKOUT_WATCHLIST entry"
        assert ambiguous is False

    def test_incomplete_ma_data_defaults_to_bounce(self):
        """Missing one of the required MA keys -> default bounce."""
        mav = {"pct_from_10mav": 0.05, "pct_from_20mav": 0.04}  # no 50mav
        bucket, reason, ambiguous = gr.route_playbook({}, mav, ticker=None)
        assert bucket == "bounce"
        assert reason == "MA data incomplete (default bounce)"
        assert ambiguous is False

    def test_below_mas_routes_to_bounce(self):
        """All required MAs present but price below at least one -> bounce."""
        mav = {
            "pct_from_10mav": -0.05,  # below
            "pct_from_20mav": 0.02,
            "pct_from_50mav": 0.03,
        }
        bucket, reason, ambiguous = gr.route_playbook({}, mav, ticker=None)
        assert bucket == "bounce"
        assert reason.startswith("Not above all MAs")
        assert "10MA" in reason  # the below-MA label
        assert ambiguous is False

    def test_above_mas_and_go_routes_to_reversal(self, monkeypatch):
        """Above all MAs + reversal pretrade GO -> reversal (not near 52wk = not ambiguous)."""
        monkeypatch.setattr(
            gr, "score_pretrade_setup",
            lambda t, m, cap=None: {"score": 5, "recommendation": "GO"},
        )
        mav = {"pct_from_10mav": 0.05, "pct_from_20mav": 0.04, "pct_from_50mav": 0.03}
        pre = {"pct_to_52wk_high": -0.50}  # well below high -> not near level
        bucket, reason, ambiguous = gr.route_playbook(
            {}, mav, ticker=None, pretrade_metrics=pre
        )
        assert bucket == "reversal"
        assert "reversal 5/5 (GO)" in reason
        assert ambiguous is False

    def test_above_mas_go_near_52wk_high_is_ambiguous(self, monkeypatch):
        """Reversal GO AND within 10% of 52wk high -> ambiguous dual-thesis flag True."""
        monkeypatch.setattr(
            gr, "score_pretrade_setup",
            lambda t, m, cap=None: {"score": 4, "recommendation": "GO"},
        )
        mav = {"pct_from_10mav": 0.05, "pct_from_20mav": 0.04, "pct_from_50mav": 0.03}
        pre = {"pct_to_52wk_high": -0.05}  # within 10% -> near level
        bucket, reason, ambiguous = gr.route_playbook(
            {}, mav, ticker=None, pretrade_metrics=pre
        )
        assert bucket == "reversal"
        assert ambiguous is True
        assert "near 52wk high" in reason

    def test_above_mas_not_go_routes_to_breakout(self, monkeypatch):
        """Above all MAs but reversal NOT GO -> breakout (extension below short threshold)."""
        monkeypatch.setattr(
            gr, "score_pretrade_setup",
            lambda t, m, cap=None: {"score": 2, "recommendation": "NO-GO"},
        )
        mav = {"pct_from_10mav": 0.05, "pct_from_20mav": 0.04, "pct_from_50mav": 0.03}
        pre = {"pct_to_52wk_high": -0.05}  # near level, but only matters if GO
        bucket, reason, ambiguous = gr.route_playbook(
            {}, mav, ticker=None, pretrade_metrics=pre
        )
        assert bucket == "breakout"
        assert "extension below short threshold" in reason
        assert ambiguous is False

    def test_above_all_four_mas_adds_200ma_suffix(self, monkeypatch):
        """When 200MA present and above, reason carries the (+200MA) suffix."""
        monkeypatch.setattr(
            gr, "score_pretrade_setup",
            lambda t, m, cap=None: {"score": 5, "recommendation": "GO"},
        )
        mav = {
            "pct_from_10mav": 0.05, "pct_from_20mav": 0.04,
            "pct_from_50mav": 0.03, "pct_from_200mav": 0.10,
        }
        pre = {"pct_to_52wk_high": -0.50}
        bucket, reason, ambiguous = gr.route_playbook(
            {}, mav, ticker=None, pretrade_metrics=pre
        )
        assert bucket == "reversal"
        assert "(+200MA)" in reason


# ---------------------------------------------------------------------------
# _fmt_pct — trivial pure math (hand-computed asserts)
# ---------------------------------------------------------------------------
class TestFmtPct:
    """Fractional distance -> percentage string."""

    def test_typical_fraction(self):
        """0.042 -> '4.2%'."""
        assert gr._fmt_pct(0.042) == "4.2%"

    def test_half(self):
        """0.5 -> '50.0%'."""
        assert gr._fmt_pct(0.5) == "50.0%"

    def test_negative(self):
        """-0.1234 -> '-12.3%' (rounds to 1dp)."""
        assert gr._fmt_pct(-0.1234) == "-12.3%"

    def test_non_numeric_falls_back_to_str(self):
        """Non-convertible input falls back to str(val)."""
        assert gr._fmt_pct("abc") == "abc"
        assert gr._fmt_pct(None) == "None"


# ---------------------------------------------------------------------------
# _fmt_dollars — trivial pure math (hand-computed asserts)
# ---------------------------------------------------------------------------
class TestFmtDollars:
    """Number -> dollar string with thousands separators."""

    def test_basic(self):
        """12.34 -> '$12.34'."""
        assert gr._fmt_dollars(12.34) == "$12.34"

    def test_thousands_separator(self):
        """1234.5 -> '$1,234.50'."""
        assert gr._fmt_dollars(1234.5) == "$1,234.50"

    def test_none_is_na(self):
        """None -> 'N/A'."""
        assert gr._fmt_dollars(None) == "N/A"

    def test_nan_is_na(self):
        """NaN -> 'N/A'."""
        assert gr._fmt_dollars(float("nan")) == "N/A"


# ---------------------------------------------------------------------------
# _calc_price_from_frac — trivial pure math (hand-computed asserts)
# ---------------------------------------------------------------------------
class TestCalcPriceFromFrac:
    """price = base * (1 + frac)."""

    def test_positive_frac(self):
        """100 * (1 + 0.05) = 105.0."""
        assert gr._calc_price_from_frac(100, 0.05) == 105.0

    def test_negative_frac(self):
        """50 * (1 - 0.1) = 45.0."""
        assert gr._calc_price_from_frac(50, -0.1) == 45.0

    def test_none_inputs_return_none(self):
        """Any None input -> None."""
        assert gr._calc_price_from_frac(None, 0.05) is None
        assert gr._calc_price_from_frac(100, None) is None

    def test_zero_base_returns_none(self):
        """Zero base price -> None (avoids meaningless boundary)."""
        assert gr._calc_price_from_frac(0, 0.5) is None


# ---------------------------------------------------------------------------
# Golden snapshots for the pure formatting helpers
# ---------------------------------------------------------------------------
@pytest.mark.characterization
class TestFormattingGoldens:
    """Lock in current output of the string/HTML formatters with frozen inputs."""

    def test_format_window_label_golden(self, assert_golden):
        """Snapshot the (label, color) mapping for all three recommendations."""
        result = {
            rec: list(gr.format_window_label(rec))
            for rec in ("GO", "CAUTION", "NO-GO")
        }
        assert_golden("report_format_window_label", result)

    def test_fmt_pct_golden(self, assert_golden):
        """Snapshot _fmt_pct over a spread of frozen inputs."""
        inputs = [0.042, 0.5, -0.1234, 0.0, 1.5]
        assert_golden(
            "report_fmt_pct",
            {str(v): gr._fmt_pct(v) for v in inputs},
        )

    def test_fmt_dollars_golden(self, assert_golden):
        """Snapshot _fmt_dollars over a spread of frozen inputs."""
        inputs = [12.34, 1234.5, 0.0, 1000000.0, -5.5]
        assert_golden(
            "report_fmt_dollars",
            {str(v): gr._fmt_dollars(v) for v in inputs},
        )

    def test_calc_price_from_frac_golden(self, assert_golden):
        """Snapshot _calc_price_from_frac over frozen (base, frac) pairs."""
        pairs = [(100, 0.05), (50, -0.1), (12.5, 0.2), (1000, -0.5)]
        assert_golden(
            "report_calc_price_from_frac",
            {f"{b}|{f}": gr._calc_price_from_frac(b, f) for b, f in pairs},
        )

    def test_format_percentile_dict_golden(self, assert_golden):
        """Snapshot the ordered percentile-dict formatter with a frozen dict."""
        d = {
            "pct_change_3": 87.5,
            "pct_change_30": 42.1,
            "pct_from_50mav": 12.0,
            "pct_change_120": 99.9,
            "some_other_metric": 33.3,  # non-PCT key -> appended alphabetically
        }
        assert_golden("report_format_percentile_dict", gr._format_percentile_dict(d))

    def test_format_reversal_intensity_html_golden(self, assert_golden):
        """Snapshot reversal-intensity HTML for a frozen intensity dict."""
        intensity = {
            "composite": 72.0,
            "details": {
                "ema9_atr": {"pctile": 88.0, "weight": 0.30},
                "mom3_atr": {"pctile": 64.0, "weight": 0.25},
                "pct_from_50mav": {"pctile": 50.0, "weight": 0.20},
                "gap_atr": {"pctile": None, "weight": 0.10},  # missing -> N/A path
            },
        }
        assert_golden(
            "report_format_reversal_intensity_html",
            gr.format_reversal_intensity_html(intensity),
        )

    def test_format_reversal_intensity_html_none_golden(self, assert_golden):
        """Snapshot the missing-inputs (composite=None) branch."""
        assert_golden(
            "report_format_reversal_intensity_html_none",
            gr.format_reversal_intensity_html({"composite": None}),
        )


# ---------------------------------------------------------------------------
# compute_bounce_intensity — hermetic snapshot with a FIXED ref_df
# ---------------------------------------------------------------------------
@pytest.mark.characterization
class TestComputeBounceIntensity:
    """Percentile-rank composite scoring against a frozen reference frame."""

    @staticmethod
    def _frozen_ref_df():
        # Small fixed reference frame covering all 6 spec columns.
        # Values are representative GapFade magnitudes (all stored as fractions,
        # negative for down-moves) — hardcoded so the test is hermetic.
        return pd.DataFrame({
            "pct_change_3":     [-0.05, -0.10, -0.15, -0.20, -0.25],
            "pct_change_15":    [-0.10, -0.20, -0.30, -0.40, -0.50],
            "selloff_total_pct": [-0.20, -0.30, -0.40, -0.50, -0.60],
            "gap_pct":          [-0.02, -0.04, -0.06, -0.08, -0.10],
            "pct_off_30d_high": [-0.15, -0.25, -0.35, -0.45, -0.55],
            "pct_off_52wk_high": [-0.30, -0.40, -0.50, -0.60, -0.70],
        })

    def test_compute_bounce_intensity_golden(self, assert_golden):
        """Snapshot the composite + per-metric percentiles for a frozen candidate."""
        metrics = {
            "pct_change_3": -0.18,
            "pct_change_15": -0.35,
            "selloff_total_pct": -0.45,
            "gap_pct": -0.07,
            "pct_off_30d_high": -0.40,
            "pct_off_52wk_high": -0.55,
        }
        result = gr.compute_bounce_intensity(metrics, ref_df=self._frozen_ref_df())
        assert_golden("report_compute_bounce_intensity", result)

    def test_compute_bounce_intensity_missing_metric_golden(self, assert_golden):
        """Snapshot behavior when a candidate metric is missing (pctile None)."""
        metrics = {
            "pct_change_3": -0.12,
            # pct_change_15 missing entirely
            "selloff_total_pct": -0.30,
            "gap_pct": -0.04,
            "pct_off_30d_high": -0.25,
            "pct_off_52wk_high": -0.40,
        }
        result = gr.compute_bounce_intensity(metrics, ref_df=self._frozen_ref_df())
        assert_golden("report_compute_bounce_intensity_missing", result)
