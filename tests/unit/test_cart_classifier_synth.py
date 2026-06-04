import pytest
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import NotFittedError

from synthpop.methods.cart_synth import CartMethod
from synthpop.utils import str_dtype


# ----- stubs and fixtures -----
class StubRegressor(TransformerMixin, BaseEstimator):
    def __init__(self, transform_result=None):
        self.transform_result = transform_result

    def fit(self, X, y):
        self.fit_x = X
        self.fit_y = y
        return self

    def transform(self, X):
        self.transform_x = X
        return self.transform_result
    
    def get_feature_names_out(self, input_features=None):
        return ["fake_output"]
    
class StubClassifier(TransformerMixin, BaseEstimator):
    def __init__(self, transform_result=None):
        self.transform_result = transform_result

    def fit(self, X, y):
        self.fit_x = X
        self.fit_y = y
        return self

    def transform(self, X):
        self.transform_x = X
        return self.transform_result
    
    def get_feature_names_out(self, input_features=None):
        return ["fake_output"]
    
# ----- fit tests -----
@pytest.mark.parametrize(
    ("X", "y", "expected", "message"),
    [
        ({}, pd.Series([1]), TypeError, "X must be a pandas DataFrame"),
        (pd.DataFrame({"a": [1]}), [1], TypeError, "y must be a pandas Series"),
        (pd.DataFrame({"a": [1, 2]}), pd.Series([1]), ValueError, "X and y must contain the same number of samples"),
    ],
)
def test_fit_validates_inputs(X, y, expected, message):
    cart = CartMethod()

    with pytest.raises(expected, match=message):
        cart.fit(X, y)

@pytest.mark.parametrize(
    ("y_array", "expected_type"),
    [
        (np.array([1, 2, 3], dtype=np.float32), StubRegressor),
        (np.array(["a", "b", "c"], dtype=str_dtype), StubClassifier),
    ],
)
def test_fit_selects_correct_method(mocker, y_array, expected_type):
    regressor = StubRegressor()
    classifier = StubClassifier()

    cart = CartMethod(regressor=regressor, classifier=classifier)
    X_df = pd.DataFrame({"x": [1, 2, 3]})
    y = pd.Series([1, 2, 3])

    mocker.patch(
        "synthpop.methods.cart_synth.utils.to_standardised_array_dict",
        return_value={"x": np.array([1, 2, 3], dtype=np.float32)},
    )

    mocker.patch(
        "synthpop.methods.cart_synth.utils.standardise_array_dtypes",
        return_value=y_array,
    )

    cart.fit(X_df, y)

    if pd.api.types.is_numeric_dtype(y_array.dtype):
        assert isinstance(cart.method_, expected_type)
        assert np.array_equal(cart.method_.fit_y, y_array)
    else:
        assert isinstance(cart.method_, expected_type)
        assert np.array_equal(cart.method_.fit_y, y_array)

def test_fit_passes_standardised_data_to_tree(mocker):
    clean_X = {"a": np.array([1, 2, 3], dtype=np.float32)}
    clean_y = np.array([4, 5, 6], dtype=np.float32)

    cart = CartMethod(regressor = StubRegressor(), classifier = StubClassifier())

    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([4, 5, 6])

    mocked_X = mocker.patch(
        "synthpop.methods.cart_synth.utils.to_standardised_array_dict",
        return_value=clean_X,
    )

    mocked_y = mocker.patch(
        "synthpop.methods.cart_synth.utils.standardise_array_dtypes",
        return_value=clean_y,
    )

    cart.fit(X, y)

    mocked_X.assert_called_once_with(X)
    mocked_y.assert_called_once_with(y)

    assert cart.method_.fit_x is clean_X
    assert cart.method_.fit_y is clean_y

@pytest.mark.parametrize(
        ("y", "expected_type"),
        [
            (pd.Series([1.0, 2.0]), StubRegressor),
            (pd.Series(["a", "b"]), StubClassifier)
        ]
)
def test_fit_clones_methods(y, expected_type):
    regressor = StubRegressor()
    classifier = StubClassifier()
    cart = CartMethod(regressor = regressor, classifier = classifier)

    X = pd.DataFrame({"a": [1, 2]})

    cart.fit(X, y)

    if pd.api.types.is_numeric_dtype(y.dtype):
        assert isinstance(cart.method_, expected_type)
        assert cart.method_ is not regressor
    else:
        assert isinstance(cart.method_, expected_type)
        assert cart.method_ is not classifier

@pytest.mark.parametrize(
        "y",
        [
            pd.Series([1.0, 2.0], name = "target"),
            pd.Series(["a", "b"], name = "target"),
        ]
)
def test_fit_sets_fitted_attributes(mocker, y):
    clean_X = {
        "a": np.array([1, 2], dtype=np.float32),
        "b": np.array([3, 4], dtype=np.float32),
    }

    clean_y = np.array([10, 20], dtype=np.float32)

    mocker.patch(
        "synthpop.methods.cart_synth.utils.to_standardised_array_dict",
        return_value=clean_X,
    )

    mocker.patch(
        "synthpop.methods.cart_synth.utils.standardise_array_dtypes",
        return_value=clean_y,
    )

    cart = CartMethod(regressor = StubRegressor(), classifier = StubClassifier())

    X = pd.DataFrame({"a": [1, 2], "b": [3, 4],})

    cart.fit(X, y)

    assert cart.feature_names_in_ == ["a", "b"]

    assert cart.target_name_ == "target"

    assert hasattr(cart, "method_")

    assert cart.method_.fit_x is clean_X
    assert cart.method_.fit_y is clean_y

# ----- transform tests -----
@pytest.mark.parametrize(
        ("result", "method"),
        [
            (pd.Series([10, 20]), StubRegressor(transform_result = np.array([10, 20]))),
            (pd.Series(["a", "b"]), StubClassifier(transform_result = np.array(["a", "b"])))
        ]
)
def test_transform_returns_series(mocker, result, method):
    cart = CartMethod(regressor = StubRegressor(result), classifier = StubClassifier(result))

    cart.feature_names_in_ = ["a"]
    cart.target_name_ = "target"
    cart.method_ = method

    clean_X = {"a": np.array([1, 2])}

    mocked_X = mocker.patch(
        "synthpop.methods.cart_synth.utils.to_standardised_array_dict",
        return_value=clean_X,
    )

    X = pd.DataFrame(
        {"a": [1, 2]},
        index=["r1", "r2"],
    )

    out = cart.transform(X)

    pd.testing.assert_frame_equal(mocked_X.call_args.args[0], X)

    assert isinstance(out, pd.Series)
    assert out.name == "target"

    pd.testing.assert_index_equal(out.index, X.index)

    np.testing.assert_array_equal(out.to_numpy(), result)

@pytest.mark.parametrize(
        ("result", "method"),
        [
            (pd.Series([10, 20]), StubRegressor(transform_result = np.array([10, 20]))),
            (pd.Series(["a", "b"]), StubClassifier(transform_result = np.array(["a", "b"])))
        ]
)
def test_transform_preserves_metadata_and_feature_order(mocker, result, method):
    cart = CartMethod(regressor = StubRegressor(result), classifier = StubClassifier(result))

    cart.method_ = method
    cart.feature_names_in_ = ["b", "a"]
    cart.target_name_ = "synthetic_target"

    mocked_standardise = mocker.patch(
        "synthpop.methods.cart_synth.utils.to_standardised_array_dict",
        return_value={"dummy": np.array([1, 2])},
    )

    X = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
        },
        index=["row1", "row2"],
    )

    out = cart.transform(X)

    pd.testing.assert_frame_equal(
        mocked_standardise.call_args.args[0],
        X[["b", "a"]],
    )

    assert out.name == "synthetic_target"

    pd.testing.assert_index_equal(
        out.index,
        X.index,
    )

    np.testing.assert_array_equal(
        out.to_numpy(),
        result,
    )

def test_transform_rejects_missing_columns():
    cart = CartMethod(regressor = StubRegressor(), classifier = StubClassifier())

    cart.method_ = StubRegressor()
    cart.feature_names_in_ = ["a", "b"]
    cart.target_name_ = "y"

    X = pd.DataFrame({"a": [1]})

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        cart.transform(X)

@pytest.mark.parametrize(
        ("result", "method"),
        [
            (pd.Series([10, 20]), StubRegressor(transform_result = np.array([10, 20]))),
            (pd.Series(["a", "b"]), StubClassifier(transform_result = np.array(["a", "b"])))
        ]
)
def test_transform_ignores_extra_columns(mocker, result, method):
    cart = CartMethod(regressor = StubRegressor(result), classifier = StubClassifier(result))

    cart.method_ = method
    cart.feature_names_in_ = ["b", "a"]
    cart.target_name_ = "target"

    mocked_standardise = mocker.patch(
        "synthpop.methods.cart_synth.utils.to_standardised_array_dict",
        return_value={}
    )

    X = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6],  # extra
            "d": [7, 8],  # extra
        }
    )

    cart.transform(X)

    mocked_standardise.assert_called_once()

    pd.testing.assert_frame_equal(
        mocked_standardise.call_args.args[0],
        X[["b", "a"]],
    )

def test_transform_requires_fit():
    cart = CartMethod()
    with pytest.raises(NotFittedError):
        cart.transform(pd.DataFrame({"a": [1]}))

# ----- get_feature_names_out test -----
def test_get_feature_names_out_delegates():
    cart = CartMethod()
    cart.method_ = StubRegressor()

    assert cart.get_feature_names_out() == ["fake_output"]

def test_get_feature_names_out_raises_unfitted():
    cart = CartMethod()

    with pytest.raises(NotFittedError):
        cart.get_feature_names_out()

# ----- clonability test -----
def test_clone_works_and_fitted_cart_does_not_preserve_state():
    X = pd.DataFrame({"a": [1]})
    y = pd.Series([1])
    cart = CartMethod(regressor = StubRegressor(), classifier = StubClassifier())
    cart.fit(X, y)

    cloned = clone(cart)

    # Fitted attributes should NOT be copied, original remains intact
    for attr in ["method_", "feature_names_in_", "target_name_"]:
        assert not hasattr(cloned, attr)
        assert hasattr(cart, attr)
    assert hasattr(cloned, "regressor")
    assert hasattr(cloned, "classifier")
    assert hasattr(cart, "regressor")
    assert hasattr(cart, "classifier")