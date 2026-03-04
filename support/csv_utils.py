"""CSV validation and atomic write utilities for the backtester pipeline."""

import os
import tempfile
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Required columns per CSV type (missing = hard error)
REQUIRED_COLUMNS = {
    'breakout': ['date', 'ticker'],
    'reversal': ['date', 'ticker'],
    'bounce': ['date', 'ticker'],
}

# Expected columns per CSV type (missing = warning only, fill_data adds them)
EXPECTED_COLUMNS = {
    'breakout': [
        'date', 'ticker', 'trade_grade', 'atr_pct', 'avg_daily_vol',
        'breakout_open_close_pct', 'breakout_open_high_pct', 'gap_pct',
    ],
    'reversal': [
        'date', 'ticker', 'trade_grade', 'cap', 'atr_pct', 'avg_daily_vol',
        'gap_pct', 'reversal_open_close_pct',
    ],
    'bounce': [
        'date', 'ticker', 'trade_grade', 'cap', 'gap_pct',
        'bounce_open_close_pct', 'consecutive_down_days',
    ],
}


def validate_csv(df, csv_type):
    """Validate a DataFrame against the schema for *csv_type*.

    Raises ValueError if required columns are missing.
    Logs a warning for missing expected columns.
    """
    if df.empty:
        logger.warning(f"Empty DataFrame for csv_type='{csv_type}'")
        return

    required = REQUIRED_COLUMNS.get(csv_type, [])
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"CSV type '{csv_type}' missing required columns: {missing_required}"
        )

    expected = EXPECTED_COLUMNS.get(csv_type, [])
    missing_expected = [c for c in expected if c not in df.columns]
    if missing_expected:
        logger.warning(
            f"CSV type '{csv_type}' missing expected columns (will be filled): {missing_expected}"
        )


def load_csv(path, csv_type):
    """Load a CSV, drop rows missing ticker/date, and validate schema.

    Malformed rows (wrong number of fields) are skipped with a warning
    rather than crashing the pipeline.
    """
    df = pd.read_csv(path, on_bad_lines='warn')
    df = df.dropna(subset=['ticker'])
    df = df.dropna(subset=['date'])
    validate_csv(df, csv_type)
    return df


def save_csv_atomic(df, path, **kwargs):
    """Write *df* to a temp file, then atomically replace *path*.

    Uses os.replace() for an atomic swap on the same filesystem.
    Any extra **kwargs are forwarded to df.to_csv().
    """
    path = str(path)
    dir_name = os.path.dirname(path) or '.'
    kwargs.setdefault('index', False)

    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
    try:
        os.close(fd)
        df.to_csv(tmp_path, **kwargs)
        os.replace(tmp_path, path)
        logger.info(f"Saved {len(df)} rows to {path}")
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
