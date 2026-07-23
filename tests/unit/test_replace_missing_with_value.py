import numpy as np
import pandas as pd
import pytest

from synthpop.data_processing.missing_value_handling import ReplaceMissingWithValue
from synthpop.utils import str_dtype


# ----- test data -----


def get_test_data():
    missing_indicators = ["missing", "N.a.N."]

    test_data_np_arrays = [
        # X_in, y_in, y_exp, missing_marker
        *[
            (
                {"a": np.array(["a", "b", "a"], str_dtype)},
                np.array(["x", "y", np.nan], dtype=str_dtype),
                np.array(["x", "y", missing_indicator], dtype=str_dtype),
                missing_indicator,
            )
            for missing_indicator in missing_indicators
        ],
        *[
            (
                {"b": np.array(["a", "b", np.nan], dtype=str_dtype)},
                np.array(["x", "y", "y"], dtype=str_dtype),
                np.array(["x", "y", "y"], dtype=str_dtype),
                "N.a.N.",
            )
        ],
        *[
            (
                {"a": np.array(["a", "b", "a"], dtype=str_dtype)},
                np.array(["x", "y", "y"], dtype=str_dtype),
                np.array(["x", "y", "y"], dtype=str_dtype),
                "N.a.N.",
            )
        ],
        *[
            (
                {"a": np.array(["a", "b", "a"], dtype=str_dtype)},
                np.array([np.nan, np.nan, np.nan], dtype=str_dtype),
                np.array([missing_indicator, missing_indicator,
                          missing_indicator], dtype=str_dtype),
                missing_indicator,
            )
            for missing_indicator in missing_indicators],
        (
            {"a": np.array(["a", "b", "a"], dtype=str_dtype)},
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            -8,
        ),
        (
            {"a": np.array(["a", "b", "a"], dtype=str_dtype)},
            np.array([1, np.nan, 3]),
            np.array([1, -8, 3], dtype=np.float64),
            -8,
        ),
    ]

    return test_data_np_arrays


def get_post_synth_test_data():
    x_values = {"a": np.array(["a", "b", "c", "d"], dtype=str_dtype)}
    missing_markers = ["N.a.N.", "missing marker"]

    # y_in, y_exp, missing_marker
    y_values_np_array = [
        (
            np.array(["a", missing, "b", missing], dtype=str_dtype),
            np.array(["a", np.nan, "b", np.nan], dtype=str_dtype),
            missing,
        )
        for missing in missing_markers] + \
        [
            (
                np.array(["a", "c", "b", missing], dtype=str_dtype),
                np.array(["a", "c", "b", np.nan], dtype=str_dtype),
                missing,
            )
            for missing in missing_markers] + \
        [
            (
                np.array(["a", "c", "b", not_missing], dtype=str_dtype),
                np.array(["a", "c", "b", not_missing], dtype=str_dtype),
                None,
            )
            for not_missing in ["not missing", "missing", "Nan", "None"]]

    return [(x_values, *y_val) for y_val in y_values_np_array]


# ----- prepare_data_for_fit tests -----


@pytest.mark.parametrize(
        "X_in, y_in, y_exp, missing_indicator",
        get_test_data()
)
def test_prepare_data_for_fit_numeric_correctness(X_in, y_in, y_exp, missing_indicator):
    replace_nan = ReplaceMissingWithValue(missing_marker=missing_indicator)

    X_res, y_res = replace_nan.prepare_data_for_fit(X_in, y_in)
    assert X_res is X_in
    assert np.array_equal(y_exp, y_res, equal_nan=True)
    assert y_exp.dtype == y_res.dtype


def test_prepare_data_for_fit_does_not_change_arguments():
    x_orig = {"a": np.array(["a", "b"], dtype=str_dtype)}
    y = np.array(["x", np.nan], dtype=str_dtype)
    replace_nan = ReplaceMissingWithValue()

    x_res, y_res = replace_nan.prepare_data_for_fit(x_orig, y)

    assert pd.isna(y[1])
    assert y_res[1] == "N.a.N."


def test_prepare_data_for_fit_error_when_nan_is_a_value():
    X = {"a": np.array(["a", "b"], dtype=str_dtype)}
    y = np.array(["x", "N.a.N.", np.nan], dtype=str_dtype)

    replace_nan = ReplaceMissingWithValue(missing_marker="N.a.N.")

    with pytest.raises(ValueError):
        replace_nan.prepare_data_for_fit(X, y)


def test_prepare_data_for_fit_empty():
    replace_nan = ReplaceMissingWithValue()
    with pytest.raises(ValueError):
        replace_nan.prepare_data_for_fit(
            {"a": np.array(["s"], dtype=str_dtype)},
            np.array([])
        )
    # no longer gives an empty array back, X and y must have the same number of rows


# ----- post_synth_transform tests -----


@pytest.mark.parametrize("x, y_in ,y_exp, missing_marker", get_post_synth_test_data())
def test_post_synth_transform_correct_on_data(x, y_in, y_exp, missing_marker):
    if missing_marker is None:
        transform = ReplaceMissingWithValue()
    else:
        transform = ReplaceMissingWithValue(missing_marker=missing_marker)

    y_res = transform.post_synth_transform(x, y_in)
    assert np.array_equal(y_res, y_exp, equal_nan=True)
    assert y_res.dtype == y_exp.dtype


def test_post_synth_transform_replaces_nan():
    X = np.array([1, 2, 3])  # X is used for the validation of y
    y = np.array(["a", "b", "N.a.N."], dtype=str_dtype)

    replace_nan = ReplaceMissingWithValue()

    result = replace_nan.post_synth_transform(X, y)
    assert result[0] == "a"
    assert result[1] == "b"
    assert pd.isna(result[2])


def test_post_synth_transform_does_nothing_when_no_nan():
    X = np.array([1, 2])  # X is used for the validation of y
    y = np.array(["a", "b"], dtype=str_dtype)

    replace_nan = ReplaceMissingWithValue()

    result = replace_nan.post_synth_transform(X, y)
    assert (result == y).all()


# ----- clonability tests -----


def test_clone_works_and_fitted_does_not_preserve_state():
    replace_missing = ReplaceMissingWithValue(missing_marker="N.a.N.")
    replace_missing.prepare_data_for_fit(
        X={"a": np.array(["a", "b", "c", "c"], dtype=str_dtype)},
        y=np.array(["x", "y", np.nan, "z"], dtype=str_dtype),
    )

    cloned = replace_missing.clone()

    # Does not have learned attributes
    assert hasattr(cloned, "missing_marker")
    assert hasattr(replace_missing, "missing_marker")
