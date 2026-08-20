import re
import warnings

import pytest
import numpy as np
import pandas as pd

from synthpop.utils import (
    _warn_on_rare_category,
    str_dtype,
    _validate_stringdtype_array,
    _validate_1d_target,
    _validate_2d_dict,
    _standardise_array_dtypes,
    _to_standardised_array_dict,
)


# ----- _validate_stringdtype_array -----


def test_validate_stringdtype_array_accepts_valid_array():
    arr = np.array(["a", "b", np.nan], dtype=str_dtype)
    _validate_stringdtype_array(arr)


@pytest.mark.parametrize(
    "arr",
    [
        np.array(["a", "b"]),
        np.array(["a", "b", np.nan], dtype=object),
        np.array([1, 2]),
        np.array(["a", "b", None], dtype=np.dtypes.StringDType(na_object=None)),
    ],
)
def test_validate_stringdtype_array_rejects_invalid_dtypes(arr):
    with pytest.raises(TypeError):
        _validate_stringdtype_array(arr)


# ----- _validate_2d_dict -----


def test_validate_2d_dict_accepts_numeric_and_stringdtypes():
    X = {
        "num": np.array([[1], [2]]),
        "cat": np.array([["a"], ["b"]], dtype=str_dtype),
    }
    result, _ = _validate_2d_dict(X)

    assert result["num"].shape == (2, 1)
    assert result["cat"].shape == (2, 1)


def test_validate_2d_dict_reshapes_1d_arrays():
    X = {"num": np.array([1, 2])}
    result, _ = _validate_2d_dict(X)

    assert result["num"].shape == (2, 1)


def test_validate_2d_dict_rejects_non_dict():
    with pytest.raises(TypeError):
        _validate_2d_dict(np.array([1, 2]))


def test_validate_2d_dict_rejects_empty_dict():
    with pytest.raises(ValueError):
        _validate_2d_dict({})


def test_validate_2d_dict_rejects_multiple_columns():
    with pytest.raises(ValueError):
        _validate_2d_dict({"num": np.array([[1, 2], [3, 4]])})


def test_validate_2d_dict_rejects_inconsistent_row_counts():
    X = {"a": np.array([1, 2]), "b": np.array([1])}

    with pytest.raises(ValueError):
        _validate_2d_dict(X)


@pytest.mark.parametrize(
    "x_dict",
    [
        {"cat": np.array(["a", "b"])},
        {"cat": np.array(["a", "b", np.nan], dtype=object)},
        {"cat": np.array(
            ["a", None], dtype=np.dtypes.StringDType(na_object=None))},
    ],
)
def test_validate_2d_dict_rejects_wrong_non_numeric_dtype(x_dict):
    with pytest.raises(TypeError):
        _validate_2d_dict(x_dict)


# ----- _validate_1d_target -----


def test_validate_1d_target_accepts_numeric_1d():
    y = np.array([1, 2])
    result = _validate_1d_target(y, None)

    assert result.shape == (2,)


def test_validate_1d_target_accepts_stringdtype():
    y = np.array(["a", np.nan], dtype=str_dtype)
    result = _validate_1d_target(y, None)

    assert result.dtype == str_dtype


def test_validate_1d_target_reshapes_single_column_2d():
    y = np.array([[1], [2]])
    result = _validate_1d_target(y, 2)

    assert result.shape == (2,)


def test_validate_1d_target_rejects_wrong_sample_count():
    with pytest.raises(ValueError):
        _validate_1d_target(np.array([1, 2,]), n_samples=3)


@pytest.mark.parametrize(
    "y",
    [
        np.array(["a", "b"]),
        np.array(["a", "b", np.nan], dtype=object),
        np.array(["a", None], dtype=np.dtypes.StringDType(na_object=None)),
    ],
)
def test_validate_1d_target_rejects_wrong_non_numeric_dtype(y):
    with pytest.raises(TypeError):
        _validate_1d_target(y, None)


# ----- _standardise_array_dtypes -----


def get_one_var_data_str_standardise_array_dtypes():
    return [
        pd.Series(["a", "b", "c"], dtype="string"),
        pd.Series(["1", "2", "3"], dtype="category"),
        ["a", "b", None],
        pd.Series(["a", pd.NA, "c"]),
        np.array(["a", np.nan, "b"], dtype=str_dtype),
        pd.Series([1, pd.NA, 2], dtype=object)
    ]


def get_one_var_data_num_standardise_array_dtypes():
    return [
        pd.Series([1, 2, 3]),
        [1.1, 2.2, np.nan],
        pd.Series([1.1, pd.NA, 2.2], dtype="Float64"),
        pd.Series([1, None, 3], dtype="Int64"),
        np.array([1.2, np.nan, 3.4]),
    ]


# NON-NUMERIC CASES
@pytest.mark.parametrize("y", get_one_var_data_str_standardise_array_dtypes())
def test_standardise_array_dtypes_string_output_is_ndarray(y):
    result = _standardise_array_dtypes(y)

    assert isinstance(result, np.ndarray)


@pytest.mark.parametrize("y", get_one_var_data_str_standardise_array_dtypes())
def test_standardise_array_dtypes_string_output_has_stringdtype(y):
    result = _standardise_array_dtypes(y)

    assert result.dtype == str_dtype


@pytest.mark.parametrize("y", get_one_var_data_str_standardise_array_dtypes())
def test_standardise_array_dtypes_string_missing_values_are_normalised(y):
    result = _standardise_array_dtypes(y)

    assert np.array_equal(
        pd.isna(y),
        pd.isna(result),
    ), "Missing values are not handled correctly"


def test_standardise_array_dtypes_preserves_2d_string_shape():
    arr = np.array([["a", None], ["c", "d"]])
    result = _standardise_array_dtypes(arr)

    assert result.shape == (2, 2)
    assert result.dtype == str_dtype
    assert pd.isna(result[0, 1])


def test_standardise_array_dtypes_mixed_object_array_becomes_stringdtype():
    arr = np.array([1, "a", None], dtype=object)

    result = _standardise_array_dtypes(arr)

    assert result.dtype == str_dtype
    assert result[0] == "1"
    assert result[1] == "a"
    assert pd.isna(result[2])


# NUMERIC CASES
@pytest.mark.parametrize("y", get_one_var_data_num_standardise_array_dtypes())
def test_standardise_array_dtypes_numeric_output_is_ndarray(y):
    result = _standardise_array_dtypes(y)

    assert isinstance(result, np.ndarray)


@pytest.mark.parametrize("y", get_one_var_data_num_standardise_array_dtypes())
def test_standardise_array_dtypes_numeric_output_has_float32_dtype(y):
    result = _standardise_array_dtypes(y)
    assert result.dtype == np.float32


@pytest.mark.parametrize("y", get_one_var_data_num_standardise_array_dtypes())
def test_standardise_array_dtypes_numeric_missing_values_are_normalised(y):
    result = _standardise_array_dtypes(y)

    assert np.array_equal(
        pd.isna(y),
        pd.isna(result),
    ), "Missing values are not handled correctly"


def test_standardise_array_dtypes_preserves_2d_numeric_shape():
    arr = np.array([[1, np.nan], [3, 4]])
    result = _standardise_array_dtypes(arr)

    assert result.shape == (2, 2)
    assert result.dtype == np.float32
    assert np.isnan(result[0, 1])


@pytest.mark.parametrize(
    "bad_input",
    [
        {"a": object()},
        object(),
        lambda x: x,
    ],
)
def test_standardise_array_dtypes_unsupported_inputs_raise(bad_input):
    with pytest.raises((TypeError, ValueError)):
        _standardise_array_dtypes(bad_input)


@pytest.mark.parametrize(
    "series",
    [
        pd.Series([1, None, 3], dtype="Int64"),
        pd.Series([1, None, 3], dtype="Float64"),
    ],
)
def test_standardise_array_dtypes_nullable_dtype(series):
    result = _standardise_array_dtypes(series)

    assert result.dtype == np.float32

    expected = np.array([1, np.nan, 3], dtype=np.float32)

    assert np.array_equal(result, expected, equal_nan=True)


# ----- _to_standardised_array_dict -----


def get_x_input_data_to_standardised_array_dict():
    x_dict = {
        "a": [1, 2, np.nan],
        "b": ["x", "y", np.nan]
    }

    x_dataframe = pd.DataFrame(x_dict)

    return [x_dict, x_dataframe]


@pytest.mark.parametrize("X", get_x_input_data_to_standardised_array_dict())
def test_to_standardised_array_dict_returns_dict(X):
    result = _to_standardised_array_dict(X)

    assert isinstance(result, dict)

    for key, value in result.items():
        assert isinstance(key, str)
        assert isinstance(value, np.ndarray)


@pytest.mark.parametrize("X", get_x_input_data_to_standardised_array_dict())
def test_to_standardised_array_dict_standardises_numeric_columns(X):
    result = _to_standardised_array_dict(X)

    assert result["a"].dtype == np.float32

    expected = np.array([1, 2, np.nan], dtype=np.float32)

    assert np.array_equal(result["a"], expected, equal_nan=True)


@pytest.mark.parametrize("X", get_x_input_data_to_standardised_array_dict())
def test_to_standardised_array_dict_standardises_string_columns(X):
    result = _to_standardised_array_dict(X)

    assert result["b"].dtype == str_dtype

    expected = np.array(["x", "y", np.nan], dtype=str_dtype)

    assert np.array_equal(result["b"], expected, equal_nan=True)


def test_to_standardised_array_dict_preserves_2d_column_shapes():
    X = {
        "a": np.array([[1], [2], [3]]),
        "b": np.array([["x"], ["y"], ["z"]]),
    }

    result = _to_standardised_array_dict(X)

    assert result["a"].shape == (3, 1)
    assert result["b"].shape == (3, 1)

    assert result["a"].dtype == np.float32
    assert result["b"].dtype == str_dtype


def test_to_standardised_array_dict_preserves_column_names():
    X = pd.DataFrame({
        "col1": [1, 2],
        "col2": ["a", "b"],
    })

    result = _to_standardised_array_dict(X)

    assert list(result.keys()) == ["col1", "col2"]


def test_to_standardised_array_dict_with_numpy_inputs():
    X = {
        "num": np.array([1, 2, 3], dtype=np.int64),
        "cat": np.array(["a", "b", "c"], dtype=object),
    }

    result = _to_standardised_array_dict(X)

    assert result["num"].dtype == np.float32
    assert result["cat"].dtype == str_dtype

# ----- _warn_on_rare_category -----


def unique_cats(count):
    rng = np.random.default_rng(seed=123)

    return [str(val) for val in rng.random(size=count)]


RARE_CATEGORIES_WARN_CASES = [
    # one row, so unique value , and more than quarter.
    (np.array(["a"]), 5),
    (np.array(["a"] * 4), 5),  # rowcount below threshold.
    (np.array(["a", "b", "c", "d", "e"]), 5),  # all values unique
    (np.array(unique_cats(100) * 5), 6),  # all categories rare

    # 6*5 = 30 rows with values that occur 5 times
    # So 30% of the rows have a rare category
    (np.array(unique_cats(6) * 5 + ["b"] * 70), 6),
    (np.array(unique_cats(6) * 5 + [np.nan] * 70), 6),  # same, but with nan.

    # 9*5 = 45 rows with values that occur 5 times
    # So 45% of the rows have a rare category
    (np.array(unique_cats(9) * 5 + ["b"] * 55), 6),
    (np.array(unique_cats(9) * 5 + [np.nan] * 55), 6),  # same, but with nan.

    # nan is the rare category.
    (np.array([np.nan]*26 + ["b"] * 74), 52),
    (np.array([True] * 26 + [False] * 74), 52),  # boolean

]

@pytest.mark.parametrize("x, threshold", RARE_CATEGORIES_WARN_CASES)
def test_warn_on_rare_category_proportion_more_than_quarter_warns(x, threshold):
    txt_threshold = re.escape(f"fewer than {threshold} times")
    txt_column_name = re.escape(f"column_with_threshold{threshold}")

    message_regex = f".* {txt_column_name}.*{txt_threshold}.*"

    with pytest.warns(UserWarning, match=message_regex):
        _warn_on_rare_category(
            x, threshold, name=f"column_with_threshold{threshold}")


def test_warn_on_rare_category_no_name():
    x_in = np.array(["a"], dtype=str_dtype)

    txt_threshold = re.escape(f"fewer than 2 times")
    txt_column_name = re.escape(f"unnamed predictor")

    message_regex = f".* {txt_column_name}.*{txt_threshold}.*"
    with pytest.warns(UserWarning, match=message_regex) as record:
        _warn_on_rare_category(
            x_in, 2, name=None)


@pytest.mark.parametrize("x, threshold", [
    (["a"] * 5, 4),  # just above threshold
    # number of occurrences exactly equal to threshold
    (["a"] * 5 + ["b"] * 6, 5),
    # number of occurrences strictly higher than threshold
    (["a"] * 7 + ["b"] * 10, 5),
    ([np.nan] * 5, 4),  # just above threshold, with nan
    # number of occurrences strictly higher than threshold, with nan
    ([np.nan] * 7 + ["b"] * 10, 5),
    # number of occurrences exactly equal to threshold, with missing values.
    ([np.nan] * 5 + ["b"] * 6, 5),
    # do not raise on nan only array containing a number of values equalling the threshold.
    (np.array([np.nan] * 7), 7),

    # number of 'a' is below threshold (rare category), but is less than 50% of the rows
    (["a"] * 4 + ["b"] * 13, 5),
    ([np.nan] * 4 + ["b"] * 13, 5),  # same, but with nan.
    ([True] * 4 + [False] * 13, 5),  # same, but with boolean.


    # 4*5 = 20 rows with values that occur 5 times
    # So 20% of the rows have a rare category
    (np.array(unique_cats(4) * 5 + ["b"] * 80), 6),
    (np.array(unique_cats(4) * 5 + [np.nan] * 80), 6),  # same, but with nan.


])
def test_warn_on_rare_category_does_not_warn_below_quarter(x, threshold):
    x_in = np.array(x)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _warn_on_rare_category(
            x_in, threshold, name="some_name") is None


@pytest.mark.parametrize("x, threshold", RARE_CATEGORIES_WARN_CASES)
def test_warn_on_rare_category_check_disabled(x, threshold):

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _warn_on_rare_category(x, 0, name="some_name") is None
