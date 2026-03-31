import numpy as np
import pandas as pd
import pytest 
from sklearn.tree import BaseDecisionTree
from sklearn.base import BaseEstimator

from synthpop.data_processing.missing_value_handling import MissingValuePredictor
# ----- fixtures -----    
@pytest.fixture
def dummy_encoder():
    class _DummyEncoder(BaseEstimator):
        def __init__(self):
            self.fit_called = False
            self.transform_called = False
            self.fitted = None
            self.mapping_ = None
            pass

        def fit(self, X, y):
            self.fitted = True
            self.mapping_ = True
            self.fit_called = True
            return self
    
        def transform(self, X):
            self.transform_called = True
            return np.array([1.0 if x is not None else 0.0 for x in X])
    
    return _DummyEncoder()

@pytest.fixture
def dummy_sampler():
    class _DummySampler:
        def __init__(self):
            self.fit_called = False
            self.sample_called = False
            self.last_leaf_ids = None
            self.last_z = None
            pass

        def fit_sampler(self, leaf_ids, z):
            self._leaf_map = True
            self.random_state_ = True
            self.last_z = z
            self.last_leaf_ids = leaf_ids
    
        def sample_from_leaves(self, leaf_ids):
            self.sample_called = True
            self.last_leaf_ids = leaf_ids
            return (np.arange(len(leaf_ids)) % 2).astype(bool)
    
        def clone(self):
            return self
    
    return _DummySampler()

@pytest.fixture
def dummy_tree():
    class _DummyTree(BaseDecisionTree):
        def __init__(self):
            self.fit_called = False
            self.X = None
            self.y = None
            pass

        def fit(self, X, y):
            self.X = X
            self.y = y
            self.fit_called = True
            return self

        def apply(self, X):
            return np.arange(len(X))
    
    return _DummyTree()

@pytest.fixture
def predictor(dummy_encoder, dummy_tree, dummy_sampler):
    return MissingValuePredictor(
        encoding=dummy_encoder,
        tree = dummy_tree,
        tree_sampler= dummy_sampler
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

# ----- build X matrix tests -----
def test_build_X_matrix_respects_feature_order(predictor, dummy_encoder):
    predictor.encoders_ = {
        "b": dummy_encoder,
        "a": None
    }

    X = {
        "a": [1, 2, 3],
        "b": ["x", None, "y"]
    }

    # explicitly override order
    predictor.feature_order_ = ["b", "a"]

    X_matrix = predictor._build_X_matrix(X)

    # b uses encoder: non-None -> 1.0, None -> 0.0
    b_encoded = np.array([1.0, 0.0, 1.0], dtype=np.float32).reshape(-1, 1)

    # a is raw float cast
    a_encoded = np.array([1, 2, 3], dtype=np.float32).reshape(-1, 1)

    expected = np.hstack([b_encoded, a_encoded])

    assert np.array_equal(X_matrix, expected)

    # ensure order is preserved (b first, then a)
    assert predictor.feature_order_ == ["b", "a"]

# ----- prepare data for fit tests -----
def test_prepare_data_missing_data_flow_correct(predictor):
    X = {"cat": ["a", "b", "a", "b"]}
    y = [0, None, 0, 1]

    predictor.prepare_data_for_fit(X, y)

    # Check encoding output used as tree input
    tree_X = predictor.tree_.X
    assert tree_X.shape == (4, 1)

    # Since DummyEncoder maps non-missing → 1.0
    expected_encoded = np.array([[1.0], [1.0], [1.0], [1.0]])
    assert np.array_equal(tree_X, expected_encoded)

    # Check sampler received correct leaf_ids and z
    sampler = predictor.tree_sampler_
    assert np.array_equal(sampler.last_leaf_ids, np.arange(4))
    expected_z = np.array([False, True, False, False])
    assert np.array_equal(sampler.last_z, expected_z)

def test_prepare_data_no_missing_data_flow(predictor):
    X = {"cat": ["a", "b", "c", "d"], "num": [1, 2, 3, 4]}
    y = [10, 20, 30, 40] 

    X_out, y_out = predictor.prepare_data_for_fit(X, y)

    # core branch behaviour
    assert predictor._no_missing
    assert not predictor._all_missing

    assert predictor.tree_ is None
    assert predictor.tree_sampler_ is None

    # encoding still happens
    assert "cat" in predictor.encoders_
    assert predictor.encoders_["cat"].fit_called
    assert predictor.encoders_["cat"].transform_called

    assert predictor.encoders_["num"] is None

    for col in X:
        assert np.array_equal(X_out[col], np.array(X[col]))

    assert np.array_equal(y_out, np.array(y))

def test_prepare_data_for_fit_mixed_types(predictor):
    X = {"cat": ["a", "b", "c", "d", "e", "f"], "num": [1, 2, 3, 4, 5, 6]}
    y = [0, np.nan, 1, None, 2, pd.NA]

    X_out, y_out = predictor.prepare_data_for_fit(X, y)
    assert np.array_equal(X_out["cat"], ["a", "c", "e"])
    assert np.array_equal(X_out["num"], [1, 3, 5])
    assert np.array_equal(y_out, [0, 1, 2])

def test_prepare_data_all_missing(predictor):
    X = {"a": [1, 2, 3]}
    y = [np.nan, None, pd.NA]
    X_out, y_out = predictor.prepare_data_for_fit(X, y)
    assert predictor.tree_ is None
    assert predictor.tree_sampler_ is None
    assert len(X_out["a"]) == 0
    assert len(y_out) == 0

def test_prepare_data_no_missing(predictor):
    X = {"a": [1, 2, 3], "b": [1, 2, 3]}
    y = [0, 1, 0]
    X_out, y_out = predictor.prepare_data_for_fit(X, y)
    assert predictor.tree_ is None
    for k in X:
        assert np.array_equal(X_out[k], X[k])
    assert np.array_equal(y_out, y)

def test_prepare_data_encoding_called_only_to_categorical(predictor):
    X = {"cat": ["a", "b", "c"], "num": [1, 2, 3]}
    y = [0, 1, 0]
    predictor.prepare_data_for_fit(X, y)
    encoder = predictor.encoders_["cat"]
    assert "cat" in predictor.encoders_
    assert encoder.fit_called
    assert encoder.transform_called
    assert predictor.encoders_["num"] is None

def test_prepare_data_tree_and_sampler_called(predictor):
    X = {"cat": ["a", "b", "c", "d"]}
    y = [0, np.nan, 1, 0]
    predictor.prepare_data_for_fit(X, y)
    assert predictor.tree_.fit_called
    assert hasattr(predictor.tree_sampler_, "last_z")
    assert hasattr(predictor.tree_sampler_, "last_leaf_ids")

def test_missingness_indicator_correctness(predictor):
    X = {"a": [1, 2, 3, 4, 5, 6]}
    y = [0, np.nan, 1, None, 2, pd.NA]
    predictor.prepare_data_for_fit(X, y)
    expected_z = np.array([0, 1, 0, 1, 0, 1])
    assert np.array_equal(predictor.tree_sampler_.last_z, expected_z)

def test_prepare_data_does_not_mutate_inputs(predictor):
    X = {"a": [1, 2, 3]}
    y = [0, None, 1]
    X_copy = {k: v.copy() for k, v in X.items()}
    y_copy = list(y)
    predictor.prepare_data_for_fit(X, y)
    for k in X:
        assert np.array_equal(X_copy[k], X[k])
    assert y == y_copy
    assert X == X_copy

    # ----- post synth transform tests -----
def test_post_synth_transform_basic(predictor, dummy_tree, dummy_sampler):
    predictor.tree_ = dummy_tree
    predictor.tree_sampler_ = dummy_sampler
    predictor.encoders_ = {"a": None, "b": None}

    predictor._all_missing = False
    predictor._no_missing = False

    X = {"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]}
    y = np.array([100, 200, 300, 400])
    predictor.feature_order_ = list(X.keys())
    out = predictor.post_synth_transform(X, y)

    expected_mask = np.array([False, True, False, True]) #from DummySampler
    assert out.shape == y.shape
    assert np.array_equal(np.isnan(out), expected_mask)


def test_post_synth_all_missing(predictor, dummy_tree, dummy_sampler):
    predictor._all_missing = True
    predictor._no_missing = False

    predictor.tree_ = dummy_tree
    predictor.tree_sampler_ = dummy_sampler
    predictor.encoders_ = {}

    X = {"a": [1, 2, 3]}
    y = np.array([1, 2, 3])
    predictor.feature_order_ = list(X.keys())
    out = predictor.post_synth_transform(X, y)

    assert np.all(np.isnan(out))

def test_post_synth_no_missing(predictor, dummy_tree, dummy_sampler):
    predictor._all_missing = False
    predictor._no_missing = True

    predictor.tree_ = dummy_tree
    predictor.tree_sampler_ = dummy_sampler
    predictor.encoders_ = {}

    X = {"a": [1, 2, 3]}
    y = np.array([1, 2, 3])
    predictor.feature_order_ = list(X.keys())
    out = predictor.post_synth_transform(X, y)

    assert np.array_equal(out, y)

def test_post_synth_transform_dataflow(predictor, dummy_tree, dummy_sampler):
    predictor.tree_ = dummy_tree
    predictor.tree_sampler_ = dummy_sampler
    predictor.encoders_ = {"a": None, "b": None}
    predictor._all_missing = False
    predictor._no_missing = False

    X = {"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]}
    y = np.array([100, 200, 300, 400])

    predictor.feature_order_ = list(X.keys())
    out = predictor.post_synth_transform(X, y)

    assert len(out) == len(y)

    expected_X_encoded = np.array([
        [1., 10.],
        [2., 20.],
        [3., 30.],
        [4., 40.]
    ])

    # reconstruct what SHOULD have been fed into tree
    X_encoded = np.hstack([
        np.array(X["a"]).reshape(-1, 1).astype(np.float32),
        np.array(X["b"]).reshape(-1, 1).astype(np.float32),
    ])

    assert np.array_equal(X_encoded, expected_X_encoded)

    leaf_ids = predictor.tree_.apply(X_encoded)

    assert np.array_equal(leaf_ids, np.arange(4))

    expected_z = predictor.tree_sampler_.sample_from_leaves(leaf_ids)

    # sampler output matches internal state
    assert np.array_equal(predictor.tree_sampler_.last_leaf_ids, leaf_ids)

    expected_out = y.astype(float).copy()
    expected_out[expected_z] = np.nan

    assert np.array_equal(np.isnan(out), expected_z)
    assert np.array_equal(out[~np.isnan(out)], expected_out[~np.isnan(out)])

def test_post_synth_with_encoder(predictor, dummy_tree, dummy_sampler, dummy_encoder):
    predictor.tree_ = dummy_tree
    predictor.tree_sampler_ = dummy_sampler
    predictor.encoders_ = {"cat": dummy_encoder}

    predictor._all_missing = False
    predictor._no_missing = False

    X = {"cat": ["a", None, "c"]}
    y = np.array([1, 2, 3])
    predictor.feature_order_ = list(X.keys())

    predictor.post_synth_transform(X, y)
    assert predictor.encoders_["cat"].transform_called

def post_synth_transform_raises_unfitted():
    model = MissingValuePredictor()
    with pytest.raises(AttributeError):
        model.post_synth_transform({"a": [1]}, [1])
    
def test_post_synth_consistent_with_build_X_matrix(predictor, dummy_tree, dummy_sampler):
    predictor.tree_ = dummy_tree
    predictor.tree_sampler_ = dummy_sampler
    predictor.encoders_ = {"a": None, "b": None}
    predictor._all_missing = False
    predictor._no_missing = False

    X = {
        "a": [1, 2, 3, 4],
        "b": [10, 20, 30, 40]
    }
    y = np.array([100, 200, 300, 400])

    predictor.feature_order_ = ["a", "b"]

    out = predictor.post_synth_transform(X, y)

    X_expected = np.hstack([
        np.array(X["a"], dtype=np.float32).reshape(-1, 1),
        np.array(X["b"], dtype=np.float32).reshape(-1, 1),
    ])

    leaf_ids = predictor.tree_.apply(X_expected)
    assert np.array_equal(leaf_ids, np.arange(4))

    z = predictor.tree_sampler_.last_leaf_ids is not None
    z = (np.arange(len(y)) % 2).astype(bool)  # matches DummySampler definition

    expected = y.astype(float).copy()
    expected[z] = np.nan

    np.testing.assert_allclose(out, expected, equal_nan=True)

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