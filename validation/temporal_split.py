"""Temporal splitting of trade DataFrames into train / validate / test periods."""

import pandas as pd
from typing import Tuple


def temporal_split(
    df: pd.DataFrame,
    train_end: str,
    validate_end: str,
    date_col: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train, validate, and test sets by date.

    Args:
        df: DataFrame with a date column.
        train_end: Last date included in training set (inclusive), e.g. "2023-12-31".
        validate_end: Last date included in validation set (inclusive).
        date_col: Name of the date column.

    Returns:
        (train_df, validate_df, test_df)
    """
    dates = pd.to_datetime(df[date_col], format="mixed", dayfirst=False)
    train_cutoff = pd.to_datetime(train_end)
    validate_cutoff = pd.to_datetime(validate_end)

    train_mask = dates <= train_cutoff
    validate_mask = (dates > train_cutoff) & (dates <= validate_cutoff)
    test_mask = dates > validate_cutoff

    return df[train_mask].copy(), df[validate_mask].copy(), df[test_mask].copy()
