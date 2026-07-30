"""Unit tests for support/csv_utils.py.

validate_csv is pure (no I/O) — tested for raise/no-raise/empty behavior.
load_csv / save_csv_atomic touch disk — tested with pytest's tmp_path fixture.
"""
import logging

import pandas as pd
import pytest

from support.csv_utils import validate_csv, load_csv, save_csv_atomic


class TestValidateCsv:
    """validate_csv raises on missing required columns, warns otherwise."""

    def test_valid_df_no_raise(self):
        """A df with all required columns for 'breakout' does not raise."""
        df = pd.DataFrame({'date': ['01/15/2024'], 'ticker': ['NVDA']})
        # Should return None and not raise.
        assert validate_csv(df, 'breakout') is None

    def test_missing_required_raises_valueerror(self):
        """Missing a required column ('ticker') raises ValueError."""
        df = pd.DataFrame({'date': ['01/15/2024']})
        with pytest.raises(ValueError) as exc:
            validate_csv(df, 'breakout')
        assert 'ticker' in str(exc.value)

    def test_missing_all_required_raises(self):
        """A df with neither required column raises ValueError."""
        df = pd.DataFrame({'foo': [1], 'bar': [2]})
        with pytest.raises(ValueError):
            validate_csv(df, 'reversal')

    def test_missing_expected_warns_no_raise(self, caplog):
        """Missing only expected (optional) columns logs a warning, no raise."""
        df = pd.DataFrame({'date': ['01/15/2024'], 'ticker': ['COIN']})
        with caplog.at_level(logging.WARNING, logger='support.csv_utils'):
            result = validate_csv(df, 'bounce')
        assert result is None
        assert any('missing expected columns' in r.message for r in caplog.records)

    def test_empty_df_warns_and_returns(self, caplog):
        """An empty df logs a warning and returns early without raising."""
        df = pd.DataFrame(columns=['date', 'ticker'])
        with caplog.at_level(logging.WARNING, logger='support.csv_utils'):
            result = validate_csv(df, 'breakout')
        assert result is None
        assert any('Empty DataFrame' in r.message for r in caplog.records)

    def test_unknown_csv_type_no_required(self):
        """An unknown csv_type has no required columns, so never raises."""
        df = pd.DataFrame({'whatever': [1]})
        assert validate_csv(df, 'mystery') is None


class TestSaveAndLoadRoundTrip:
    """save_csv_atomic + load_csv round-trip on disk via tmp_path."""

    def test_save_then_load_equal(self, tmp_path):
        """Writing a df and reading it back yields an equal DataFrame."""
        df = pd.DataFrame({
            'date': ['01/15/2024', '01/16/2024'],
            'ticker': ['NVDA', 'COIN'],
            'trade_grade': ['A', 'B'],
        })
        path = tmp_path / 'breakout.csv'
        save_csv_atomic(df, path)

        assert path.exists(), "atomic replace did not produce the target file"

        loaded = load_csv(str(path), 'breakout')
        pd.testing.assert_frame_equal(
            loaded.reset_index(drop=True), df.reset_index(drop=True)
        )

    def test_atomic_replace_overwrites_existing(self, tmp_path):
        """A second save_csv_atomic overwrites the prior file content."""
        path = tmp_path / 'reversal.csv'
        first = pd.DataFrame({'date': ['01/01/2024'], 'ticker': ['AAA']})
        save_csv_atomic(first, path)

        second = pd.DataFrame({'date': ['02/02/2024'], 'ticker': ['BBB']})
        save_csv_atomic(second, path)

        loaded = load_csv(str(path), 'reversal')
        assert list(loaded['ticker']) == ['BBB']
        assert list(loaded['date']) == ['02/02/2024']

    def test_no_tmp_files_left_behind(self, tmp_path):
        """After a successful save, no .tmp scratch files remain in the dir."""
        df = pd.DataFrame({'date': ['01/15/2024'], 'ticker': ['NVDA']})
        path = tmp_path / 'bounce.csv'
        save_csv_atomic(df, path)
        leftover = [p.name for p in tmp_path.iterdir() if p.suffix == '.tmp']
        assert leftover == []

    def test_save_index_false_by_default(self, tmp_path):
        """save_csv_atomic defaults index=False, so no unnamed index column appears."""
        df = pd.DataFrame({'date': ['01/15/2024'], 'ticker': ['NVDA']})
        path = tmp_path / 'breakout.csv'
        save_csv_atomic(df, path)
        raw = pd.read_csv(path)
        assert list(raw.columns) == ['date', 'ticker']


class TestLoadCsvCleansRows:
    """load_csv drops rows missing ticker/date before validating."""

    def test_drops_rows_missing_ticker_or_date(self, tmp_path):
        """Rows with NaN ticker or date are dropped on load."""
        # Build a raw CSV by hand including blank ticker / date cells.
        path = tmp_path / 'breakout.csv'
        path.write_text(
            "date,ticker\n"
            "01/15/2024,NVDA\n"
            "01/16/2024,\n"        # missing ticker -> dropped
            ",COIN\n"              # missing date -> dropped
            "01/17/2024,MSFT\n"
        )
        loaded = load_csv(str(path), 'breakout')
        assert list(loaded['ticker']) == ['NVDA', 'MSFT']
