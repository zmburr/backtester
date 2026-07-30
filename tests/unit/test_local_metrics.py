"""Unit + characterization tests for data_queries/local_metrics.py.

The private ``_compute_*`` helpers are pure metric math (no network). Each is
fed a tiny hand-built tz-aware OHLCV DataFrame and asserted against values
worked out BY HAND in the docstrings/comments. A single characterization test
locks the end-to-end ``compute_screener_metrics`` dict for a frozen 30-row
synthetic frame.
"""
import pandas as pd
import pytest

from data_queries import local_metrics as lm


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _df(dates, opens=None, highs=None, lows=None, closes=None, volumes=None):
    """Build a tz-aware (US/Eastern) OHLCV DataFrame from parallel lists."""
    idx = pd.to_datetime(list(dates)).tz_localize("US/Eastern")
    n = len(idx)

    def _col(vals, default):
        return list(vals) if vals is not None else [default] * n

    return pd.DataFrame(
        {
            "open": _col(opens, 0.0),
            "high": _col(highs, 0.0),
            "low": _col(lows, 0.0),
            "close": _col(closes, 0.0),
            "volume": _col(volumes, 0.0),
        },
        index=idx,
    )


# ======================================================================
# _compute_atr
# ======================================================================
class TestComputeAtr:
    """True Range = max(H-L, |H-prevC|, |L-prevC|); ATR = rolling mean."""

    def _frame(self):
        # Row0: H10 L8  C9   -> prevC NaN -> TR = H-L              = 2
        # Row1: H11 L9  C10  prevC=9  hl=2 hpc=|11-9|=2 lpc=|9-9|=0 -> 2
        # Row2: H12 L9  C11  prevC=10 hl=3 hpc=|12-10|=2 lpc=|9-10|=1 -> 3
        return _df(
            ["2024-01-02", "2024-01-03", "2024-01-04"],
            highs=[10, 11, 12],
            lows=[8, 9, 9],
            closes=[9, 10, 11],
        )

    def test_atr_window_capped_to_len_averages_all_tr(self):
        """window=14 on 3 rows -> ATR = mean(2,2,3) = 7/3."""
        atr = lm._compute_atr(self._frame(), window=14)
        assert atr == pytest.approx(7 / 3)

    def test_atr_window_2_uses_last_two_tr(self):
        """window=2 -> rolling(2).mean of last two TRs = mean(2,3) = 2.5."""
        atr = lm._compute_atr(self._frame(), window=2)
        assert atr == pytest.approx(2.5)

    def test_atr_shift1_uses_previous_close(self):
        """A spike in row1's range relative to PREVIOUS close drives TR."""
        # Row0: H10 L9 C9  -> TR = 1
        # Row1: H20 L9 C19 prevC=9 hl=11 hpc=|20-9|=11 lpc=|9-9|=0 -> 11
        df = _df(
            ["2024-01-02", "2024-01-03"],
            highs=[10, 20],
            lows=[9, 9],
            closes=[9, 19],
        )
        atr = lm._compute_atr(df, window=14)  # window capped to 2 -> mean(1,11)=6
        assert atr == pytest.approx(6.0)

    def test_atr_single_row_returns_none(self):
        """Fewer than 2 rows cannot form a True Range -> None."""
        df = _df(["2024-01-02"], highs=[10], lows=[8], closes=[9])
        assert lm._compute_atr(df) is None


# ======================================================================
# _close_n_trading_days_ago  (counts STRICTLY before the ref date)
# ======================================================================
class TestCloseNTradingDaysAgo:
    """Uses '<' so the reference date itself is excluded (off-by-one matters)."""

    def _frame(self):
        # closes:        100   101   102   103   104
        return _df(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"],
            closes=[100, 101, 102, 103, 104],
        )

    def test_one_day_ago_excludes_ref_date(self):
        """ref 2024-01-08 is excluded; '1 day ago' = the 01-05 close (103)."""
        assert lm._close_n_trading_days_ago(self._frame(), "2024-01-08", 1) == 103.0

    def test_three_days_ago(self):
        """From the 01-05 anchor, 3 trading days back = 01-03 close (101)."""
        assert lm._close_n_trading_days_ago(self._frame(), "2024-01-08", 3) == 101.0

    def test_four_days_ago_reaches_first_bar(self):
        """4 trading days back from the anchor lands on 01-02 close (100)."""
        assert lm._close_n_trading_days_ago(self._frame(), "2024-01-08", 4) == 100.0

    def test_too_far_back_returns_none(self):
        """5 days back would index < 0 -> None."""
        assert lm._close_n_trading_days_ago(self._frame(), "2024-01-08", 5) is None

    def test_ref_date_not_in_frame_uses_last_prior_bar(self):
        """A Saturday ref (2024-01-06) anchors on the last prior bar 01-05."""
        assert lm._close_n_trading_days_ago(self._frame(), "2024-01-06", 1) == 103.0

    def test_no_bars_before_ref_returns_none(self):
        """If every bar is on/after the ref date there is nothing prior."""
        assert lm._close_n_trading_days_ago(self._frame(), "2024-01-02", 1) is None


# ======================================================================
# _compute_mav_data
# ======================================================================
class TestComputeMavData:
    """EMA-9, SMA-10/20/50/200 and ATR-distance from the 50-SMA."""

    def test_constant_price_all_mavs_equal_price(self):
        """Close=50 const, H51/L49 -> every MA = 50, ATR = 2.

        ref=55 -> pct_from_*mav = (55-50)/50 = 0.1; 200-SMA absent (len 60<200);
        atr_distance_from_50mav = (55-50)/2 = 2.5.
        """
        df = _df(
            pd.bdate_range("2024-01-01", periods=60),
            highs=[51] * 60,
            lows=[49] * 60,
            closes=[50] * 60,
        )
        result = lm._compute_mav_data(df, reference_price=55)
        assert result["pct_from_9ema"] == pytest.approx(0.1)
        assert result["pct_from_10mav"] == pytest.approx(0.1)
        assert result["pct_from_20mav"] == pytest.approx(0.1)
        assert result["pct_from_50mav"] == pytest.approx(0.1)
        assert "pct_from_200mav" not in result  # only 60 bars
        assert result["atr_distance_from_50mav"] == pytest.approx(2.5)

    def test_ema9_only_when_under_10_bars(self):
        """9 bars -> EMA-9 computed, SMAs skipped (need >=10).

        closes [10x8, 20], span9 adjust=False (alpha=0.2):
        ema stays 10 then 0.2*20+0.8*10 = 12. ref=18 -> (18-12)/12 = 0.5.
        """
        df = _df(
            pd.bdate_range("2024-01-01", periods=9),
            highs=[100] * 9,
            lows=[1] * 9,
            closes=[10, 10, 10, 10, 10, 10, 10, 10, 20],
        )
        result = lm._compute_mav_data(df, reference_price=18)
        assert result == {"pct_from_9ema": pytest.approx(0.5)}


# ======================================================================
# _compute_range_data  (formatted "(num / den) = pct" strings)
# ======================================================================
class TestComputeRangeData:
    """Volume-vs-20DAV and TR-vs-ATR comparison strings by recency."""

    def _result(self):
        # TRs:  2     2     3      ATR rolling(3,min1): 2, 2, 7/3=2.33
        # vols: 1000  2000  3000   20DAV rolling(3,min1): 1000,1500,2000
        df = _df(
            ["2024-01-02", "2024-01-03", "2024-01-04"],
            highs=[10, 11, 12],
            lows=[8, 9, 9],
            closes=[9, 10, 11],
            volumes=[1000, 2000, 3000],
        )
        return lm._compute_range_data(df, has_today_bar=True)

    def test_volume_strings(self):
        """day_of=3000/2000, 1-before=2000/1500, 2-before=1000/1000, 3-before N/A."""
        r = self._result()
        assert r["day_of_vol_pct"] == "(3,000 / 2,000) = 1.50"
        assert r["percent_of_vol_one_day_before"] == "(2,000 / 1,500) = 1.33"
        assert r["percent_of_vol_two_day_before"] == "(1,000 / 1,000) = 1.00"
        assert r["percent_of_vol_three_day_before"] == "N/A"

    def test_range_strings(self):
        """day_of TR3/ATR2.33 -> '(3 / 2) = 1.29'; older bars 2/2 = 1.00; 3-before N/A."""
        r = self._result()
        assert r["day_of_range_pct"] == "(3 / 2) = 1.29"
        assert r["one_day_before_range_pct"] == "(2 / 2) = 1.00"
        assert r["two_day_before_range_pct"] == "(2 / 2) = 1.00"
        assert r["three_day_before_range_pct"] == "N/A"

    def test_too_few_rows_returns_empty(self):
        """A single bar cannot form ranges -> empty dict."""
        df = _df(["2024-01-02"], highs=[10], lows=[8], closes=[9], volumes=[100])
        assert lm._compute_range_data(df, has_today_bar=True) == {}


# ======================================================================
# _compute_volume_data + _add_percent_of_adv
# ======================================================================
class TestComputeVolumeData:
    """ADV (tail-mean), prior-day volume offsets, percent-of-ADV columns."""

    def test_adv_and_prior_day_offsets(self):
        """vols 1000/2000/3000/4000 -> ADV=2500; -2=3000,-3=2000,-4=1000."""
        df = _df(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            volumes=[1000, 2000, 3000, 4000],
        )
        r = lm._compute_volume_data(df, df, has_today_bar=True, intraday=None)
        assert r["avg_daily_vol"] == pytest.approx(2500.0)
        assert r["vol_one_day_before"] == 3000.0
        assert r["vol_two_day_before"] == 2000.0
        assert r["vol_three_day_before"] == 1000.0

    def test_no_intraday_yields_none_intraday_metrics(self):
        """Without minute data, all intraday + percent_of_* fields are None."""
        df = _df(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            volumes=[1000, 2000, 3000, 4000],
        )
        r = lm._compute_volume_data(df, df, has_today_bar=True, intraday=None)
        for k in [
            "vol_on_breakout_day", "premarket_vol", "vol_in_first_5_min",
            "vol_in_first_10_min", "vol_in_first_15_min", "vol_in_first_30_min",
        ]:
            assert r[k] is None
        for k in [
            "percent_of_premarket_vol", "percent_of_vol_in_first_5_min",
            "percent_of_vol_on_breakout_day",
        ]:
            assert r[k] is None

    def test_add_percent_of_adv_division(self):
        """percent_of_X = X / avg_daily_vol; None when value is None."""
        d = {
            "avg_daily_vol": 2000,
            "premarket_vol": 1000,
            "vol_in_first_5_min": 500,
            "vol_in_first_15_min": None,
            "vol_in_first_10_min": None,
            "vol_in_first_30_min": None,
            "vol_on_breakout_day": 3000,
        }
        lm._add_percent_of_adv(d)
        assert d["percent_of_premarket_vol"] == pytest.approx(0.5)
        assert d["percent_of_vol_in_first_5_min"] == pytest.approx(0.25)
        assert d["percent_of_vol_on_breakout_day"] == pytest.approx(1.5)
        assert d["percent_of_vol_in_first_15_min"] is None

    def test_add_percent_of_adv_zero_adv_yields_none(self):
        """Zero/absent ADV cannot divide -> all percent_of_* are None."""
        d = {"avg_daily_vol": 0, "premarket_vol": 1000}
        lm._add_percent_of_adv(d)
        assert d["percent_of_premarket_vol"] is None


# ======================================================================
# End-to-end characterization
# ======================================================================
@pytest.mark.characterization
def test_compute_screener_metrics_golden(assert_golden):
    """Lock the full metric dict for a frozen 30-row synthetic daily frame.

    Deterministic formula (no randomness): close=100+0.5*i, high=close+1,
    low=close-1, open=close-0.5, volume=1e6+1e4*i over 30 business days. The
    reference date is the last bar (a past date), so today is finalized and
    mav_ref falls back to current_price -> fully reproducible.
    """
    dates = pd.bdate_range("2024-05-06", periods=30)
    closes = [100 + 0.5 * i for i in range(30)]
    df = _df(
        dates,
        opens=[c - 0.5 for c in closes],
        highs=[c + 1 for c in closes],
        lows=[c - 1 for c in closes],
        closes=closes,
        volumes=[1_000_000 + 10_000 * i for i in range(30)],
    )
    ref_date = dates[-1].strftime("%Y-%m-%d")
    result = lm.compute_screener_metrics(df, ref_date)
    assert_golden("local_metrics_screener_30row", result)
