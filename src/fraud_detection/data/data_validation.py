"""Utilities for validating credit card fraud datasets."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

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
DEFAULT_CLASS_RATIO_BOUNDS = (0.0005, 0.02)


class DataValidationError(ValueError):
    """Error raised when input data fails validation."""


@dataclass(frozen=True)
class ValidationOptions:
    """Options for dataset validation."""
    required_columns: tuple[str, ...] = REQUIRED_COLUMNS
    check_balance: bool = True
    class_ratio_bounds: tuple[float, float] = DEFAULT_CLASS_RATIO_BOUNDS


def _validate_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataValidationError("Missing required column(s): " + ", ".join(sorted(missing)))


def _validate_dtypes(df: pd.DataFrame, required: list[str]) -> None:
    non_numeric = [col for col in required if not ptypes.is_numeric_dtype(df[col])]
    if non_numeric:
        raise DataValidationError("Non-numeric dtypes detected for: " + ", ".join(sorted(non_numeric)))


def _validate_nan_counts(df: pd.DataFrame, required: list[str]) -> None:
    # Pandas indexing expects list-like, not arbitrary Iterable
    nan_counts = df[required].isna().sum()
    failing = nan_counts[nan_counts > 0]
    if not failing.empty:
        formatted = ", ".join(f"{col} ({int(count)} NaN)" for col, count in failing.items())
        raise DataValidationError("NaN values found in column(s): " + formatted)


def _validate_class_balance(df: pd.DataFrame, ratio_bounds: tuple[float, float]) -> None:
    if "Class" not in df.columns:
        raise DataValidationError("Missing required column: Class")

    total = len(df)
    if total == 0:
        raise DataValidationError("Dataset is empty after loading")

    class_counts = df["Class"].value_counts(dropna=False)

    # Allow only binary labels 0/1 (also tolerates 0.0/1.0 since they compare equal).
    invalid_labels = [label for label in class_counts.index if label not in (0, 1)]
    if invalid_labels:
        raise DataValidationError(
            "Unexpected labels in 'Class' column: " + ", ".join(map(str, invalid_labels))
        )

    fraud_count = int(class_counts.get(1, 0))
    ratio = fraud_count / total
    lower, upper = ratio_bounds
    if not (lower <= ratio <= upper):
        raise DataValidationError(
            "Class imbalance ratio out of expected range: "
            f"observed {ratio:.4%}, expected between {lower:.2%} and {upper:.2%}"
        )


def validate_creditcard_data(df: pd.DataFrame, *, options: ValidationOptions | None = None) -> None:
    """Validate credit card fraud dataset columns, dtypes, NaNs, labels, and (optionally) balance.

    Parameters
    ----------
    df:
        Input dataframe loaded from the Kaggle credit card fraud dataset (or compatible schema).
    options:
        ValidationOptions controlling required columns and balance checks.

    Raises
    ------
    DataValidationError
        If required columns are missing, dtypes are invalid, NaNs are present,
        labels are unexpected, or the class imbalance is outside bounds (if enabled).
    """
    opts = options or ValidationOptions()

    required = list(opts.required_columns)  # ensure safe for df[required]
    _validate_required_columns(df, required)
    _validate_dtypes(df, required)
    _validate_nan_counts(df, required)

    if opts.check_balance:
        _validate_class_balance(df, opts.class_ratio_bounds)
