"""Centralized date-parsing helpers for the backtester pipeline."""

from datetime import datetime
from pandas import Timestamp


def csv_date_to_iso(date_str):
    """Convert a CSV date string to ISO format ('YYYY-MM-DD').

    Handles:
      - 'MM/DD/YYYY' (primary CSV format)
      - pandas Timestamps
      - Already-ISO strings ('YYYY-MM-DD') — returned as-is
    """
    if isinstance(date_str, Timestamp):
        return date_str.strftime('%Y-%m-%d')
    if isinstance(date_str, str):
        # Already ISO?
        if len(date_str) == 10 and date_str[4] == '-':
            return date_str
        return datetime.strptime(date_str, '%m/%d/%Y').strftime('%Y-%m-%d')
    raise TypeError(f"Unsupported date type: {type(date_str)}")


def parse_row_date(row):
    """Convenience wrapper: extract and convert row['date'] to ISO format."""
    return csv_date_to_iso(row['date'])
