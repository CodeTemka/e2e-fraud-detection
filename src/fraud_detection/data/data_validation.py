"""Utilities for validating credit card fraud datasets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

REQUIRED_COLUMNS: tuple[str, ...] = (
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
    "Class",
)

# Expected imbalance ratio for the Kaggle credit card fraud dataset is ~0.17%.
# Allow a reasonable band to accommodate filtered or augmented samples.
DEFAULT_CLASS_RATIO_BOUNDS = (0.0005, 0.02)
DEFAULT_LABEL_COLUMN = "Class"


class DataValidationError(ValueError):
    """Error raised when input data fails validation."""


@dataclass(frozen=True)
class ValidationOptions:
    """Options for dataset validation."""
    required_columns: tuple[str, ...] = REQUIRED_COLUMNS
    check_balance: bool = True
    class_ratio_bounds: tuple[float, float] = DEFAULT_CLASS_RATIO_BOUNDS
    label_column: str = DEFAULT_LABEL_COLUMN


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


def _validate_class_balance(df: pd.DataFrame, ratio_bounds: tuple[float, float], label_column: str) -> None:
    if label_column not in df.columns:
        raise DataValidationError(f"Missing required column: {label_column}")

    total = len(df)
    if total == 0:
        raise DataValidationError("Dataset is empty after loading")

    class_counts = df[label_column].value_counts(dropna=False)

    # Allow only binary labels 0/1 (also tolerates 0.0/1.0 since they compare equal).
    invalid_labels = [label for label in class_counts.index if label not in (0, 1)]
    if invalid_labels:
        raise DataValidationError(
            f"Unexpected labels in '{label_column}' column: " + ", ".join(map(str, invalid_labels))
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
        _validate_class_balance(df, opts.class_ratio_bounds, opts.label_column)


def collect_validation_metrics(
    df: pd.DataFrame, *, options: ValidationOptions | None = None
) -> dict[str, float]:
    """Collect schema/quality metrics without raising exceptions."""
    opts = options or ValidationOptions()
    required = list(opts.required_columns)

    missing = [col for col in required if col not in df.columns]
    missing_count = float(len(missing))
    total_rows = float(len(df))

    nan_total = 0.0
    non_numeric_count = 0.0
    if missing_count == 0:
        nan_total = float(df[required].isna().sum().sum())
        non_numeric_count = float(
            sum(1 for col in required if not ptypes.is_numeric_dtype(df[col]))
        )

    class_ratio = np.nan
    label_column = opts.label_column
    if label_column in df.columns and len(df) > 0:
        class_ratio = float(df[label_column].sum() / len(df))

    return {
        "validation.total_rows": total_rows,
        "validation.missing_columns_count": missing_count,
        "validation.non_numeric_columns_count": float(non_numeric_count),
        "validation.nan_total": float(nan_total),
        "validation.class_ratio": float(class_ratio),
    }


def missing_required_columns(
    df: pd.DataFrame, *, options: ValidationOptions | None = None
) -> list[str]:
    """Return missing required columns for the schema."""
    opts = options or ValidationOptions()
    required = list(opts.required_columns)
    return [col for col in required if col not in df.columns]


def required_columns_with_label(label_column: str) -> tuple[str, ...]:
    """Return REQUIRED_COLUMNS with the label column substituted."""
    return tuple(label_column if col == DEFAULT_LABEL_COLUMN else col for col in REQUIRED_COLUMNS)


def _safe_percentiles(values: np.ndarray, bins: int) -> np.ndarray:
    quantiles = np.linspace(0, 100, bins + 1)
    breaks = np.percentile(values, quantiles)
    return np.unique(breaks)


def compute_psi(reference: pd.Series, current: pd.Series, *, bins: int = 10) -> float:
    """Compute population stability index for a single feature."""
    reference_values = reference.dropna().to_numpy()
    current_values = current.dropna().to_numpy()
    if reference_values.size == 0 or current_values.size == 0:
        return float("nan")

    breaks = _safe_percentiles(reference_values, bins)
    if breaks.size < 2:
        return float("nan")

    ref_hist, _ = np.histogram(reference_values, bins=breaks)
    cur_hist, _ = np.histogram(current_values, bins=breaks)

    ref_pct = ref_hist / max(ref_hist.sum(), 1)
    cur_pct = cur_hist / max(cur_hist.sum(), 1)

    epsilon = 1e-6
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)
    psi = np.sum((ref_pct - cur_pct) * np.log(ref_pct / cur_pct))
    return float(psi)


def compute_ks(reference: pd.Series, current: pd.Series) -> float:
    """Compute KS statistic between two samples."""
    reference_values = np.sort(reference.dropna().to_numpy())
    current_values = np.sort(current.dropna().to_numpy())
    if reference_values.size == 0 or current_values.size == 0:
        return float("nan")

    data_all = np.sort(np.concatenate([reference_values, current_values]))
    ref_cdf = np.searchsorted(reference_values, data_all, side="right") / reference_values.size
    cur_cdf = np.searchsorted(current_values, data_all, side="right") / current_values.size
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def compute_drift_metrics(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    method: str = "psi",
    bins: int = 10,
) -> dict[str, float]:
    """Compute drift metrics between reference and current datasets."""
    metrics: dict[str, float] = {}
    shared_columns = [
        col
        for col in reference_df.columns
        if col in current_df.columns and ptypes.is_numeric_dtype(reference_df[col])
    ]

    for col in shared_columns:
        if method == "psi":
            metrics[f"drift.psi.{col}"] = compute_psi(reference_df[col], current_df[col], bins=bins)
        elif method == "ks":
            metrics[f"drift.ks.{col}"] = compute_ks(reference_df[col], current_df[col])
        else:
            raise ValueError(f"Unsupported drift method '{method}'.")

    return metrics
