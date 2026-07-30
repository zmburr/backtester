"""Pure unit tests for support/date_utils.py.

Expected outputs are hand-computed literals (no snapshots) since these are
deterministic, pure date-string transforms.
"""
import pytest
import pandas as pd

from support.date_utils import csv_date_to_iso, parse_row_date


class TestCsvDateToIso:
    """csv_date_to_iso converts MM/DD/YYYY (or Timestamp/ISO) -> 'YYYY-MM-DD'."""

    def test_mmddyyyy_to_iso(self):
        """'01/15/2024' -> '2024-01-15'."""
        assert csv_date_to_iso('01/15/2024') == '2024-01-15'

    def test_single_digit_month_day_padded(self):
        """'3/5/2023' -> '2023-03-05' (strptime accepts unpadded, output is padded)."""
        assert csv_date_to_iso('3/5/2023') == '2023-03-05'

    def test_end_of_year(self):
        """'12/31/2024' -> '2024-12-31'."""
        assert csv_date_to_iso('12/31/2024') == '2024-12-31'

    def test_idempotent_on_iso_string(self):
        """An already-ISO string is returned unchanged."""
        assert csv_date_to_iso('2024-01-15') == '2024-01-15'

    def test_iso_string_other_date(self):
        """Another ISO string passes through unchanged."""
        assert csv_date_to_iso('1999-12-31') == '1999-12-31'

    def test_accepts_pandas_timestamp(self):
        """A pd.Timestamp is formatted via strftime to ISO."""
        ts = pd.Timestamp('2024-01-15 09:30:00')
        assert csv_date_to_iso(ts) == '2024-01-15'

    def test_timestamp_date_only(self):
        """A date-only Timestamp formats to ISO."""
        ts = pd.Timestamp(year=2022, month=7, day=4)
        assert csv_date_to_iso(ts) == '2022-07-04'

    def test_invalid_type_raises_typeerror(self):
        """A non-str / non-Timestamp input raises TypeError."""
        with pytest.raises(TypeError):
            csv_date_to_iso(12345)

    def test_malformed_string_raises_valueerror(self):
        """A string that is neither ISO nor MM/DD/YYYY raises ValueError from strptime."""
        with pytest.raises(ValueError):
            csv_date_to_iso('not-a-date')


class TestParseRowDate:
    """parse_row_date reads row['date'] then converts to ISO."""

    def test_reads_date_key_and_converts(self):
        """row['date'] in MM/DD/YYYY is converted to ISO."""
        row = {'date': '01/15/2024', 'ticker': 'NVDA'}
        assert parse_row_date(row) == '2024-01-15'

    def test_reads_iso_date_key(self):
        """row['date'] already ISO passes through."""
        row = {'date': '2023-06-09', 'ticker': 'COIN'}
        assert parse_row_date(row) == '2023-06-09'

    def test_reads_timestamp_date_key(self):
        """row['date'] as Timestamp is formatted to ISO."""
        row = {'date': pd.Timestamp('2021-11-22'), 'ticker': 'GME'}
        assert parse_row_date(row) == '2021-11-22'

    def test_missing_date_key_raises_keyerror(self):
        """A row without 'date' raises KeyError."""
        with pytest.raises(KeyError):
            parse_row_date({'ticker': 'AAPL'})
