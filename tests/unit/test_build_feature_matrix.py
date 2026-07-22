import numpy as np
import pytest

from synthpop.methods.tree_utils import build_feature_matrix


# ----- test build_feature_matrix -----


def test_build_feature_matrix_empty_input_empty_output():
    X = {}
    result = build_feature_matrix(X, [])
    assert result.shape == (0, 0)


def test_build_feature_matrix_single_column():
    X = {"a": np.array([1, 2])}
    result = build_feature_matrix(X, ["a"])

    assert np.array_equal(result, np.array([[1], [2]]))
    assert result.dtype == np.float32


def test_build_feature_matrix_two_columns():
    X = {
        "a": np.array([1, np.nan]),
        "b": np.array([3, 4]),
    }
    result = build_feature_matrix(X, ["a", "b"])

    assert np.array_equal(
        result,
        np.array([[1, 3], [np.nan, 4]]),
        equal_nan=True,
    )
    assert result.dtype == np.dtype(np.float32)


def test_build_feature_matrix_2d_features():
    X = {
        "a": np.array([[1, 3], [2, 4]]),
        "b": np.array([5, 6]),
    }
    result = build_feature_matrix(X, ["a", "b"])

    assert np.array_equal(
        result,
        np.array([[1, 3, 5], [2, 4, 6]]),
    )
    assert result.dtype == np.dtype(np.float32)


def test_build_feature_matrix_respects_order():
    X = {
        "a": np.array([[1, 3], [2, 4]]),
        "b": np.array([5, 6]),
    }
    result = build_feature_matrix(X, ["b", "a"])

    assert np.array_equal(
        result,
        np.array([[5, 1, 3], [6, 2, 4]]),
    )
    assert result.dtype == np.dtype(np.float32)


def test_build_feature_matrix_raises_on_columns_mismatch():
    X = {
        "a": np.array([1, np.nan]),
        "b": np.array([3, 4]),
    }
    with pytest.raises(
        ValueError,
        match="cannot build feature matrix: received more columns than expected",
    ):
        build_feature_matrix(X, ["a"])

    with pytest.raises(
        ValueError,
        match="cannot build feature matrix: received less columns than expected",
    ):
        build_feature_matrix(X, ["a", "b", "c"])


def test_build_feature_matrix_raises_on_wrong_column_names():
    X = {
        "a": np.array([1, 2]),
        "c": np.array([3, 4]),
    }

    with pytest.raises(ValueError):
        build_feature_matrix(X, ["a", "b"])


def test_build_feature_matrix_raises_on_row_mismatch():
    X = {
        "a": np.array([1, 2]),
        "b": np.array([3, 4, 5]),
    }

    with pytest.raises(ValueError):
        build_feature_matrix(X, ["a", "b"])
