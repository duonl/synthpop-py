import numpy as np
import pandas as pd
import pytest 
from sklearn.tree import BaseDecisionTree
from sklearn.base import BaseEstimator

from synthpop.data_processing.missing_value_handling import MissingValuePredictor
# ----- fixtures -----
class DummyEncoder(BaseEstimator):
    def __init__(self):
        self.fit_called = False
        self.transform_called = False
        pass

    def fit(self, X, y):
        self.fitted = True
        self.mapping_ = True
        self.fit_called = True
        return self
    
    def transform(self, X):
        self.transform_called = True
        return np.array([1.0 if x is not None else 0.0 for x in X])

class DummySampler:
    def __init__(self):
        self.fit_called = False
        self.sample_called = False
        pass

    def fit_sampler(self, leaf_ids, z):
        self._leaf_map = True
        self.random_state_ = True
        self.fit_called = True
        self.last_z = z
        self.last_leaf_ids = leaf_ids
    
    def sample_from_leaves(self, leaf_ids):
        self.sample_called = True
        return np.arange(len(leaf_ids)) % 2
    
    def clone(self):
        return self
    
class DummyTree(BaseDecisionTree):
    def __init__(self):
        self.fit_called = False
        pass

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.fit_called = True
        return self

    def apply(self, X):
        return np.arange(len(X))

def make_predictor():
    return MissingValuePredictor(
        encoding=DummyEncoder(),
        tree = DummyTree(),
        tree_sampler= DummySampler()
    )

# ----- validate input tests -----
@pytest.mark.parametrize(
    "X, y",
    [
        ({"a": [1, 2], "b": [3, 4]}, [1, 2]),
        ({"a": np.array([1, 2]), "b": pd.Series([3, 4])}, np.array(["a", "b"])),
        ({"a": pd.Series(["a", "b"]), "b": np.array([3, 4])}, pd.Series([1, 2])),
        ({1: [1, 2], 2: [3, 4]}, [None, pd.NA]),
        ({"a": [1, "1"], "b": [1, "a"]}, np.array([1, "1"], dtype=object))
    ]
)
def test_validate_accepts_valid_X_y(X, y):
    out_X, out_y = MissingValuePredictor._validate_X_y_dict(X, y)
    assert isinstance(out_X, dict)
    assert isinstance(out_y, np.ndarray)
    assert len(out_X) == 2
    assert len(out_y) == 2

@pytest.mark.parametrize(
    "X",[None, [1, 2, 3], "invalid", 123, [[1, 2], [3, 4], [5, 6] ]])
def test_validate_raises_invalid_X(X):
    with pytest.raises(TypeError):
        MissingValuePredictor._validate_X_y_dict(X, [0, 1, 2])

@pytest.mark.parametrize(
        "y", [{"a": [1, 2]}, [[1], [2]], None, "invalid", 123, []])
def test_validate_raises_invalid_y(y):
    with pytest.raises(ValueError):
        MissingValuePredictor._validate_X_y_dict({1: [1, 2]}, y)

@pytest.mark.parametrize(
    "X",
    [
        {"a": [[1, 2]], "b": [1, 2]},   # 2-dimensional
        {"a": [], "b": [1, 2]},         # empty column
        {"a": [1, 2], "b": [1]},        # length mismatch
        {"a": []}                       # empty key
    ],
)
def test_validate_raises_bad_shapes(X):
    y = [0,1]
    with pytest.raises(ValueError):
        MissingValuePredictor._validate_X_y_dict(X, y)

# ----- prepare data for fit tests -----
def test_prepare_data_for_fit_mixed_types():
    model = make_predictor()
    X = {"cat": ["a", "b", "c", "d", "e", "f"], "num": [1, 2, 3, 4, 5, 6]}
    y = [0, np.nan, 1, None, 2, pd.NA]

    X_out, y_out = model.prepare_data_for_fit(X, y)
    assert np.array_equal(X_out["cat"], ["a", "c", "e"])
    assert np.array_equal(X_out["num"], [1, 3, 5])
    assert np.array_equal(y_out, [0, 1, 2])

def test_prepare_data_all_missing():
    model = make_predictor()
    X = {"a": [1, 2, 3]}
    y = [np.nan, None, pd.NA]
    X_out, y_out = model.prepare_data_for_fit(X, y)
    assert model.tree_ is None
    assert model.tree_sampler_ is None
    assert len(X_out["a"]) == 0
    assert len(y_out) == 0

def test_prepare_data_no_missing():
    model = make_predictor()
    X = {"a": [1, 2, 3], "b": [1, 2, 3]}
    y = [0, 1, 0]
    X_out, y_out = model.prepare_data_for_fit(X, y)
    assert model.tree_ is None
    for k in X:
        assert np.array_equal(X_out[k], X[k])
    assert np.array_equal(y_out, y)

def test_prepare_data_encoding_called_only_to_categorical():
    model = make_predictor()
    X = {"cat": ["a", "b", "c"], "num": [1, 2, 3]}
    y = [0, 1, 0]
    model.prepare_data_for_fit(X, y)
    encoder = model.encoders_["cat"]
    assert "cat" in model.encoders_
    assert encoder.fit_called
    assert encoder.transform_called
    assert model.encoders_["num"] is None

def test_prepare_data_tree_and_sampler_called():
    model = make_predictor()
    X = {"cat": ["a", "b", "c", "d"]}
    y = [0, np.nan, 1, 0]
    model.prepare_data_for_fit(X, y)
    assert model.tree_.fit_called
    assert model.tree_sampler_.fit_called

def test_missingness_indicator_correctness():
    model = make_predictor()
    X = {"a": [1, 2, 3, 4, 5, 6]}
    y = [0, np.nan, 1, None, 2, pd.NA]
    model.prepare_data_for_fit(X, y)
    expected_z = np.array([0, 1, 0, 1, 0, 1])
    assert np.array_equal(model.tree_sampler_.last_z, expected_z)

def test_prepare_data_does_not_mutate_inputs():
    model = make_predictor()
    X = {"a": [1, 2, 3]}
    y = [0, None, 1]
    X_copy = {k: v.copy() for k, v in X.items()}
    y_copy = list(y)
    model.prepare_data_for_fit(X, y)
    for k in X:
        assert np.array_equal(X_copy[k], X[k])
    assert y == y_copy
    assert X == X_copy

    # ----- post synth transform tests -----
def test_post_synth_transform_basic():
    model = make_predictor()

    model.tree_ = DummyTree()
    model.tree_sampler_ = DummySampler()
    model.encoders_ = {"a": None, "b": None}

    model._all_missing = False
    model._no_missing = False

    X = {"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]}
    y = np.array([100, 200, 300, 400])
    out = model.post_synth_transform(X, y)

    assert len(out) == 4
    assert np.isnan(out[1])
    assert not np.isnan(out[0])

def test_post_synth_all_missing():
    model = make_predictor()
    model._all_missing = True
    model._no_missing = False

    model.tree_ = DummyTree()
    model.tree_sampler_ = DummySampler()
    model.encoders_ = {}

    X = {"a": [1, 2, 3]}
    y = np.array([1, 2, 3])
    out = model.post_synth_transform(X, y)

    assert np.all(np.isnan(out))

def test_post_synth_no_missing():
    model = make_predictor()
    model._all_missing = False
    model._no_missing = True

    model.tree_ = DummyTree()
    model.tree_sampler_ = DummySampler()
    model.encoders_ = {}

    X = {"a": [1, 2, 3]}
    y = np.array([1, 2, 3])

    out = model.post_synth_transform(X, y)

    assert np.array_equal(out, y)

def test_post_synth_with_encoder():
    model = make_predictor()

    model.tree_ = DummyTree()
    model.tree_sampler_ = DummySampler()
    model.encoders_ = {"cat": DummyEncoder()}

    model._all_missing = False
    model._no_missing = False

    X = {"cat": ["a", None, "c"]}
    y = np.array([1, 2, 3])

    model.post_synth_transform(X, y)
    assert model.encoders_["cat"].transform_called

def post_synth_transform_raises_unfitted():
    model = MissingValuePredictor()
    with pytest.raises(AttributeError):
        model.post_synth_transform({"a": [1]}, [1])

# ----- clonability tests -----
def test_clone_works_and_fitted_does_not_preserve_state():
    mvp = MissingValuePredictor()
    mvp.prepare_data_for_fit(X={"a": [1, 2, 3]}, y=[1, 2, 3])

    cloned = mvp.clone()

    # Fitted attributes should NOT be copied, original remains intact
    for attr in ["encoders_", "tree_", "tree_sampler_"]:
        assert not hasattr(cloned, attr)
        assert hasattr(mvp, attr)
    for attr in ["encoding", "tree", "tree_sampler"]:
        assert hasattr(cloned, attr)
        assert hasattr(mvp, attr)