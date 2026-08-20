import re

import numpy as np
import pandas as pd
import pytest

from synthpop.data_processing.missing_value_handling import (
    MissingValuePredictor,
    ReplaceMissingWithValue
)
from synthpop.methods.cart_synth import (
    CartMethod,
    TreeClassifierMethod,
    TreeRegressorMethod,
    tune_cart,
)
from synthpop.reproducibility import RandomStateManager
from synthpop.utils import str_dtype

# This imports an auto-use fixture to set the seed,
# in order to make the test reproducible.
from tests.integration.make_int_test_reproducible import control_random_state_manager


def test_numeric_target_uses_regressor():
    X = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "income": [1000.0, 2000.0, 3000.0, 4000.0],
            "blood type": ["A", "O", "AB", "O"],
        }
    )

    y = pd.Series([1.1, 2.2, 3.3, 4.4], name="target")

    cart = tune_cart(rare_categories_threshold=0)()
    cart.fit(X, y)

    assert isinstance(cart.method_, TreeRegressorMethod)

    out = cart.transform(X)

    assert isinstance(out, pd.Series)
    assert len(out) == len(X)
    assert out.name == "target"


def test_string_target_uses_classifier():
    X = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "letter": ["A", "B", "C", "A"],
        }
    )

    y = pd.Series(["A", "B", "A", "B"], name="group")

    cart = tune_cart(rare_categories_threshold=0)()
    cart.fit(X, y)

    assert isinstance(cart.method_, TreeClassifierMethod)

    out = cart.transform(X)

    assert isinstance(out, pd.Series)
    assert len(out) == len(X)
    assert out.name == "group"


def test_dirty_dataframe_roundtrip():
    X = pd.DataFrame(
        {
            "float_col": pd.Series(
                [1.1, np.nan, 3.3, 4.4],
                dtype="Float64",
            ),
            "int_col": pd.Series(
                [1, pd.NA, 3, 4],
                dtype="Int64",
            ),
            "bool_col": pd.Series(
                [True, False, pd.NA, True],
                dtype="boolean",
            ),
            "str_col": pd.Series(
                ["a", None, "c", "d"],
                dtype="string",
            ),
            "cat_col": pd.Series(
                pd.Categorical(
                    ["x", "y", "x", "z"],
                ),
            ),
        }
    )

    y = pd.Series(["yes", "no", "yes", "no"], dtype="string", name="target")

    cart = tune_cart(rare_categories_threshold=0)()
    cart.fit(X, y)

    out = cart.transform(X)

    assert isinstance(out, pd.Series)
    assert len(out) == len(X)
    assert out.name == "target"


def test_transform_accepts_reordered_columns():
    X_fit = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [5, 6, 7, 8],
            "c": ["9", "10", "11", "12"],
        }
    )

    y = pd.Series([1.0, 2.0, 3.0, 4.0], name="target")

    cart = tune_cart(rare_categories_threshold=0)()
    cart.fit(X_fit, y)

    X_transform = X_fit[["c", "a", "b"]]

    out = cart.transform(X_transform)

    assert len(out) == len(X_transform)
    assert out.name == "target"


def test_transform_accepts_extra_columns():
    X_fit = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": ["a", "b", "c", "d"],
        }
    )

    y = pd.Series(
        [10.0, 20.0, 30.0, 40.0],
        name="target"
    )

    cart = tune_cart(rare_categories_threshold=0)()
    cart.fit(X_fit, y)

    X_transform = X_fit.assign(
        extra_numeric=[100, 200, 300, 400],
        extra_string=["x", "y", "z", "w"],
    )

    out = cart.transform(X_transform)

    assert len(out) == len(X_transform)


def test_transform_preserves_index_and_name():
    X = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        },
        index=["row1", "row2", "row3"],
    )

    y = pd.Series(
        [10.0, 20.0, 30.0],
        index=["row1", "row2", "row3"],
        name="salary",
    )

    cart = tune_cart(rare_categories_threshold=0)()
    cart.fit(X, y)

    out = cart.transform(X)

    pd.testing.assert_index_equal(
        out.index,
        X.index,
    )

    assert out.name == "salary"


def test_fit_sets_all_fitted_attributes():
    X = pd.DataFrame(
        {
            "a": [1, 2],
            "b": ["3", "4"],
        }
    )

    y = pd.Series([10.0, 20.0], name="target")

    cart = tune_cart(rare_categories_threshold=0)()
    cart.fit(X, y)

    assert cart.feature_names_in_ == ["a", "b"]
    assert cart.target_name_ == "target"
    assert isinstance(cart.method_, TreeRegressorMethod)


@pytest.mark.parametrize(
    "y",
    [
        pd.Series([1, 2, 3, 4], dtype=np.int64),
        pd.Series([1, 2, 3, 4], dtype=np.float64),
        pd.Series([1, 2, 3, 4], dtype="Int64"),
        pd.Series([1.1, 2.2, 3.3, 4.4], dtype="Float64"),
        pd.Series([True, False, True, False], dtype="boolean"),
    ],
)
def test_numeric_target_dtypes_dispatch_to_regressor(y):
    X = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4],
            "x2": ["y", "n", "n", "y"],
        }
    )

    cart = tune_cart(rare_categories_threshold=0)()
    cart.fit(X, y)

    assert isinstance(cart.method_, TreeRegressorMethod)


@pytest.mark.parametrize(
    "y",
    [
        pd.Series(["a", "b", "a", "c"], dtype="string"),
        pd.Series(pd.Categorical(["a", "b", "a", "c"])),
        pd.Series([1, 2, 3, 4], dtype="object"),
    ],
)
def test_non_numeric_target_dtypes_dispatch_to_classifier(y):
    X = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4],
            "x2": ["hello", "world", "hello", "world"],
        }
    )

    cart = tune_cart(rare_categories_threshold=0)()
    cart.fit(X, y)

    assert isinstance(cart.method_, TreeClassifierMethod)


def test_fit_transform_with_missing_values_in_predictors():
    X = pd.DataFrame(
        {
            "float_col": pd.Series(
                [1.1, np.nan, 3.3, 4.4],
                dtype="Float64",
            ),
            "int_col": pd.Series(
                [1, pd.NA, 3, 4],
                dtype="Int64",
            ),
            "string_col": pd.Series(
                ["a", None, "c", "d"],
                dtype="string",
            ),
            "category_col": pd.Series(
                pd.Categorical(
                    ["x", np.nan, "x", "y"],
                ),
            ),
        }
    )

    y = pd.Series([10.0, 20.0, 30.0, 40.0], name="target")

    cart = tune_cart(rare_categories_threshold=0)()
    cart.fit(X, y)

    out = cart.transform(X)

    assert len(out) == len(X)


@pytest.mark.parametrize(
    "y",
    [
        pd.Series(["A", None, "A", "B"]*100, dtype="string", name="target"),
        pd.Series([1, 2, np.nan, 4]*100, name="target"),
        pd.Series([True, False, True, pd.NA]*100, name="target"),
        pd.Series([1, None, 2, np.nan]*100, name="target"),
    ],
)
def test_fit_transform_with_missing_values_in_target(y):
    X = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4] * 100,
            "x2": ["a", "b", "y", "z"] * 100,
        }
    )

    cart = CartMethod()
    cart.fit(X, y)

    out = cart.transform(X)

    assert len(out) == len(y)
    assert out.isna().sum() > 0


@pytest.mark.parametrize(
    "y",
    [
        pd.Series([1, 2, 3, 4], name='target', dtype=np.int64),
        pd.Series([1, 2, 3, 4], name='target', dtype=np.float64),
        pd.Series([1, 2, 3, 4], name='target', dtype=np.float32),
        pd.Series([1, 2, 3, 4], name='target', dtype="Int64"),
        pd.Series([1.1, 2.2, 3.3, 4.4], name='target', dtype="Float32"),

        pd.Series(['x', 'y', 'x', 'y'], name='target', dtype="str"),
        pd.Series(['x', 'y', 'x', 'y'], name='target', dtype="string"),
        pd.Series(['x', 'y', 'x', 'y'], name='target', dtype=str_dtype),

        pd.Series(['x', 'y', 'x', 'y'], name='target', dtype="object"),
        pd.Series(['x', 'y', 'x', 'y'], name='target', dtype="category"),

        pd.Series([True, False, True, False], name='target', dtype="boolean"),
        pd.Series([True, False, True, False], name='target', dtype=np.bool_),
    ],
)
def test_cart_preserves_target_dtype_end_to_end(y):
    X = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "group": ["a", "a", "b", "b"],
        }
    )

    cart = tune_cart(rare_categories_threshold=0)()
    cart.fit(X, y)
    result = cart.transform(X)

    assert result.dtype == y.dtype


@pytest.mark.parametrize(
    "y",
    [
        pd.Series([None, None], name='c'),
        pd.Series([np.nan, np.nan], name='c'),
        pd.Series([pd.NA, pd.NA], name='c'),
    ],
)
def test_fitting_handles_all_missing_target(y):
    cart = CartMethod()

    X = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
        },
    )

    res = cart.fit(X, y)
    assert res.method_._all_missing
    assert res.target_name_ == 'c'


@pytest.mark.parametrize(
    "y, dtype",
    [
        (pd.Series([None, None], name='c', dtype='object'), 'object'),
        (pd.Series([np.nan, np.nan], name='c', dtype=np.float64), np.float64),
        (pd.Series([pd.NA, pd.NA], name='c',  dtype='string'), 'string'),
    ],
)
def test_transform_handles_entire_nan_array(y, dtype):
    cart = CartMethod()

    X = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
        },
    )

    res = cart.fit(X, y)

    out = res.transform(X)

    assert len(out) == len(y)
    assert pd.isna(out).all()
    assert out.name == "c"
    assert out.dtype == dtype


@pytest.mark.parametrize(
    "y",
    [
        pd.Series([1.1, 2.2, np.nan, 4.4] * 5,
                  dtype=np.float64, name="target"),
        pd.Series([1, 2, np.nan, 4] * 5,
                  dtype="Float64", name="target"),
    ]
)
def test_regressor_method_and_replace_missing_with_value(y):
    X = pd.DataFrame(
        {
            "age": [20, 30, 40, 50] * 5,
            "income": [1000.0, 2000.0, 3000.0, 4000.0] * 5,
            "blood type": ["A", "O", "AB", "O"] * 5,
        }
    )

    cart = CartMethod(
        regressor=TreeRegressorMethod(
            tree=None, missing_handler=ReplaceMissingWithValue(missing_marker=-8)
        )
    )

    cart.fit(X, y)

    assert isinstance(cart.method_.missing_handler_, ReplaceMissingWithValue)
    assert cart.method_.missing_handler_.missing_marker == -8

    out = cart.transform(X)

    assert isinstance(out, pd.Series)
    assert len(out) == len(X)
    assert out.name == "target"


@pytest.mark.parametrize(
    "y, expected_none_missing ",
    [
        (pd.Series(['a', 'b', np.nan, 'c'] * 10,
                   dtype=str, name="target"), False),
        (pd.Series(['a', 'b', np.nan, 'c'] * 10,
                   dtype=str_dtype, name="target"), False),
        (pd.Series(['a', 'b', np.nan, 'c'] * 10,
                   dtype="category", name="target"), False),
        (pd.Series(['a', 'b', np.nan, 'c'] * 10,
                   dtype=object, name="target"), False),

        (pd.Series(['a', 'b', 'N.a.N', 'c'] * 10,
                   dtype=str, name="target"), True),
    ]
)
def test_classifier_method_and_missing_value_predictor(y, expected_none_missing):
    X = pd.DataFrame(
        {
            "age": [20, 30, 40, 50] * 10,
            "income": [1000.0, 2000.0, 3000.0, 4000.0] * 10,
            "blood type": ["A", "O", "AB", "O"] * 10,
        }
    )

    cart = CartMethod(
        classifier=TreeClassifierMethod(
            tree=None, missing_handler=MissingValuePredictor()
        ),
    )

    cart.fit(X, y)

    assert isinstance(cart.method_.missing_handler_, MissingValuePredictor)
    assert not cart.method_.missing_handler_._all_missing
    assert cart.method_.missing_handler_._none_missing == expected_none_missing

    out = cart.transform(X)

    assert isinstance(out, pd.Series)
    assert len(out) == len(X)
    assert out.name == "target"
    if expected_none_missing:
        assert not out.isna().any()
    else:
        assert out.isna().any()


@pytest.mark.parametrize(
    "y",
    [
        pd.Series([1, 2, 3, 4] * 5, name='target', dtype=np.int64),
        pd.Series([1, 2, 3, 4] * 5, name='target', dtype=np.float64),
        pd.Series([1, 2, 3, 4] * 5, name='target', dtype=np.float32),
        pd.Series([1, 2, 3, 4] * 5, name='target', dtype="Int64"),
        pd.Series([1.1, 2.2, 3.3, 4.4] * 5, name='target', dtype="Float32"),

        pd.Series(['x', 'y', 'x', 'y'] * 5, name='target', dtype="str"),
        pd.Series(['x', 'y', 'x', 'y'] * 5, name='target', dtype="string"),
        pd.Series(['x', 'y', 'x', 'y'] * 5, name='target', dtype=str_dtype),

        pd.Series(['x', 'y', 'x', 'y'] * 5, name='target', dtype="object"),
        pd.Series(['x', 'y', 'x', 'y'] * 5, name='target', dtype="category"),

        pd.Series([True, False, True, False] * 5,
                  name='target', dtype="boolean"),
        pd.Series([True, False, True, False] * 5,
                  name='target', dtype=np.bool_),
    ],
)
def test_missing_handler_does_not_mutate_output_no_missing(y):
    X = pd.DataFrame(
        {
            "age": [20, 30, 40, 50] * 5,
            "income": [1000.0, 2000.0, 3000.0, 4000.0] * 5,
            "blood type": ["A", "O", "AB", "O"] * 5,
        }
    )

    with RandomStateManager(seed=0):
        cart_standard = CartMethod()

        cart_standard.fit(X, y)
        out_standard = cart_standard.transform(X)

    with RandomStateManager(seed=0):
        cart_different_missing_handling = CartMethod(
            regressor=TreeRegressorMethod(
                tree=None, missing_handler=ReplaceMissingWithValue(missing_marker=-8)
            ),

            classifier=TreeClassifierMethod(
                tree=None, missing_handler=MissingValuePredictor()
            )
        )

        cart_different_missing_handling.fit(X, y)
        out_different_missing_handling = cart_different_missing_handling.transform(
            X)

    pd.testing.assert_series_equal(
        out_standard, out_different_missing_handling)


@pytest.mark.parametrize(
    "y",
    [
        np.array(["a", "b", "c", "d", "e"] * 6, dtype=str_dtype),
        np.array([1, 2, 3, 4, 5] * 6),
    ],
)
def test_cart_method_raises_on_rare_category(y):
    """
    Test for an exception when there is a value of a categorical variable that occurs once.
    Since the decision trees use randomness, this error does not happen for all seeds.

    If the root seed is 0, the overfitting happens.
    The control_random_state_manager fixture in make_int_test_reproducible.py sets the root seed to 0.

    """
    # This test is affected by #210
    rng = np.random.default_rng(seed=123)

    feature = [str(val) for val in rng.random(size=30)]

    X = {
        "column": np.array(feature, dtype=str_dtype)
    }

    method = tune_cart(n_leaves=2)()

    with pytest.warns(UserWarning, match=".* contains categories occurring fewer than 2 times.*"):
        result = method.fit_transform(pd.DataFrame(X), pd.Series(y))


def test_tune_cart_disable_rare_categories_check():

    feature = ["x", "y", "z"] * 10
    y = [1, 2, 3] * 10
    feature[3] = "unique value"
    X = {
        "column": np.array(feature, dtype=str_dtype)
    }

    method = tune_cart(n_leaves=2, rare_categories_threshold=0)()

    result = method.fit_transform(pd.DataFrame(X), pd.Series(y))

    assert result[3] == y[3], "attribute disclosure for sample 3"
