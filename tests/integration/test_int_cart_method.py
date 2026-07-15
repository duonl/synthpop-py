import pytest
import pandas as pd
import numpy as np

from synthpop.methods.cart_synth import CartMethod, TreeRegressorMethod, TreeClassifierMethod

# This imports an auto-use fixture to set the seed, in order to make the test reproducible
from tests.integration.make_int_test_reproducible import control_random_state_manager

from synthpop.utils import str_dtype

def test_numeric_target_uses_regressor():
    X = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "income": [1000.0, 2000.0, 3000.0, 4000.0],
            "blood type": ["A", "O", "AB", "O"],
        }
    )

    y = pd.Series([1.1, 2.2, 3.3, 4.4], name="target")

    cart = CartMethod()
    cart.fit(X, y)

    assert isinstance(cart.method_, TreeRegressorMethod)

    out = cart.transform(X)

    assert isinstance(out, pd.Series)
    assert len(out) == len(X)
    assert out.name == "target"

def test_string_target_uses_classifier():
    X = pd.DataFrame(
        {"age": [20, 30, 40, 50],
         "letter": ["A", "B", "C", "A"]}
    )

    y = pd.Series(["A", "B", "A", "B"], name="group")

    cart = CartMethod()
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
                    ["x", "y", "x", "z"]
                )
            ),
        }
    )

    y = pd.Series(["yes", "no", "yes", "no"], dtype="string", name="target")

    cart = CartMethod()
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

    cart = CartMethod()
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

    y = pd.Series([10.0, 20.0, 30.0, 40.0], name="target")

    cart = CartMethod()
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

    y = pd.Series([10.0, 20.0, 30.0], index=["row1", "row2", "row3"], name="salary")

    cart = CartMethod()
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

    cart = CartMethod()

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
        {"x1": [1, 2, 3, 4],
         "x2": ["y", "n", "n", "y"]}
    )

    cart = CartMethod()
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
        {"x1": [1, 2, 3, 4],
         "x2": ["hello", "world", "hello", "world"]}
    )

    cart = CartMethod()
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
                    ["x", np.nan, "x", "y"]
                )
            ),
        }
    )

    y = pd.Series([10.0, 20.0, 30.0, 40.0], name="target",)

    cart = CartMethod()
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
        {"x1": [1, 2, 3, 4]*100,
         "x2": ["a", "b", "y", "z"]*100}
    )

    cart = CartMethod()
    cart.fit(X, y)

    out = cart.transform(X)

    assert len(out) == len(y)
    assert out.isna().sum() > 0

    
@pytest.mark.parametrize(
    "y",
    [
        (pd.Series([None, None], name='c')),
        (pd.Series([np.nan, np.nan], name='c')),
        (pd.Series([pd.NA, pd.NA], name='c'))
    ]
)
def test_fitting_handles_all_missing_target(y):
    cart = CartMethod()

    X = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
    })

    res = cart.fit(X, y)
    assert res.method_._all_missing is True
    assert res.target_name_ == 'c'

@pytest.mark.parametrize(
    "y, dtype",
    [
        (pd.Series([None, None], name='c'), 'object'), # This should be changed after pull request 171
        (pd.Series([np.nan, np.nan], name='c'), np.float32),
        (pd.Series([pd.NA, pd.NA], name='c'), 'object')
    ]
)
def test_transform_handles_entire_nan_array(y, dtype):
    cart = CartMethod()

    X = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
    })

    res = cart.fit(X, y)
    out = res.transform(X)
    
    assert len(out) == len(y)
    assert pd.isna(out).all()
    assert out.name == "c"
    assert out.dtype == dtype