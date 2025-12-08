"""Utilities for validating credit card fraud datasets."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from pandas.api import types as ptypes

REQUIRED_COLUMNS: tuple[str, ...] = (
    "Time",
    "Amount",
    *[f"V{i}" for i in range(1, 29)],
    "Class",
)

# Expected imbalance ratio for the Kaggle credit card fraud dataset is ~0.17%.
# Allow a reasonable band to accommodate filtered or augmented samples.
EXPECTED_CLASS_RATIO_BOUNDS = (0.0005, 0.02)


class DataValidationError(ValueError):
    """Error raised when input data fails validation."""


def _validate_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataValidationError(
            "Missing required column(s): " + ", ".join(sorted(missing))
        )


def _validate_dtypes(df: pd.DataFrame, required: Iterable[str]) -> None:
    non_numeric = [col for col in required if not ptypes.is_numeric_dtype(df[col])]
    if non_numeric:
        raise DataValidationError(
            "Non-numeric dtypes detected for: " + ", ".join(sorted(non_numeric))
        )


def _validate_nan_counts(df: pd.DataFrame, required: Iterable[str]) -> None:
    nan_counts = df[required].isna().sum()
    failing = nan_counts[nan_counts > 0]
    if not failing.empty:
        formatted = ", ".join(f"{col} ({count} NaN)" for col, count in failing.items())
        raise DataValidationError("NaN values found in column(s): " + formatted)


def _validate_class_balance(df: pd.DataFrame) -> None:
    class_counts = df["Class"].value_counts(dropna=False)
    invalid_labels = [label for label in class_counts.index if label not in (0, 1)]
    if invalid_labels:
        raise DataValidationError(
            "Unexpected labels in 'Class' column: " + ", ".join(map(str, invalid_labels))
        )

    total = len(df)
    if total == 0:
        raise DataValidationError("Dataset is empty after loading")

    fraud_count = class_counts.get(1, 0)
    ratio = fraud_count / total
    lower, upper = EXPECTED_CLASS_RATIO_BOUNDS
    if not (lower <= ratio <= upper):
        raise DataValidationError(
            "Class imbalance ratio out of expected range: "
            f"observed {ratio:.4%}, expected between {lower:.2%} and {upper:.2%}"
        )


def validate_creditcard_data(df: pd.DataFrame) -> None:
    """Validate credit card fraud dataset columns, dtypes, and balance.

    Parameters
    ----------
    df:
        Input dataframe loaded from the standard Kaggle credit card fraud CSV.

    Raises
    ------
    DataValidationError
        If required columns are missing, dtypes are invalid, NaN values are present,
        labels are unexpected, or the class imbalance falls outside the allowed band.
    """

    _validate_required_columns(df, REQUIRED_COLUMNS)
    _validate_dtypes(df, REQUIRED_COLUMNS)
    _validate_nan_counts(df, REQUIRED_COLUMNS)
    _validate_class_balance(df)

