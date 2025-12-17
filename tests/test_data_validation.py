import pandas as pd
import pytest

from fraud_detection.data.data_validation import (
    DataValidationError,
    ValidationOptions,
    validate_creditcard_data,
)


def _base_dataframe():
    columns = ["Time", "Amount", *[f"V{i}" for i in range(1, 29)], "Class"]
    data = {col: [0.0, 1.0, 2.0] for col in columns}
    data["Class"] = [0, 0, 1]
    return pd.DataFrame(data)


def test_validate_creditcard_data_happy_path():
    df = _base_dataframe()

    # Should not raise
    validate_creditcard_data(df, options=ValidationOptions(check_balance=False))


@pytest.mark.parametrize(
    "drop_column",
    ["Time", "V1", "Class"],
)
def test_missing_required_columns_raises(drop_column):
    df = _base_dataframe().drop(columns=[drop_column])

    with pytest.raises(DataValidationError) as exc:
        validate_creditcard_data(df)

    assert drop_column in str(exc.value)


def test_non_numeric_column_raises():
    df = _base_dataframe()
    df["V3"] = ["a", "b", "c"]

    with pytest.raises(DataValidationError) as exc:
        validate_creditcard_data(df)

    assert "V3" in str(exc.value)


def test_nan_values_raises():
    df = _base_dataframe()
    df.loc[1, "Amount"] = None

    with pytest.raises(DataValidationError) as exc:
        validate_creditcard_data(df)

    assert "Amount" in str(exc.value)


def test_invalid_labels_raises():
    df = _base_dataframe()
    df.loc[0, "Class"] = 2

    with pytest.raises(DataValidationError) as exc:
        validate_creditcard_data(df)

    assert "Unexpected labels" in str(exc.value)


def test_class_balance_out_of_bounds():
    df = _base_dataframe()
    df["Class"] = [0, 0, 0]

    with pytest.raises(DataValidationError) as exc:
        validate_creditcard_data(df, options=ValidationOptions(class_ratio_bounds=(0.5, 1)))

    assert "out of expected range" in str(exc.value)
