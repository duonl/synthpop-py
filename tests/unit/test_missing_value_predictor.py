import numpy as np
import pandas as pd
import pytest 
from sklearn.tree import BaseDecisionTree
from sklearn.base import BaseEstimator

from synthpop.data_processing.missing_value_handling import MissingValuePredictor
# ----- fixtures -----    
@pytest.fixture
def stub_encoder():
    class EncoderStub(BaseEstimator):
        def __init__(self, transform_return=None):
            self.transform_return = transform_return
            self.fit_inputs = None
            self.transform_inputs = None
            pass

        def fit(self, X, y = None):
            self.fit_inputs = (X, y)
            return self
    
        def transform(self, X):
            self.transform_inputs = X
            return self.transform_return
    
    return EncoderStub()

@pytest.fixture
def stub_sampler():
    class SamplerStub:
        def __init__(self, sample_return=None):
            self.sample_return = sample_return
            self.fit_inputs = None
            self.sample_inputs = None
            pass

        def fit_sampler(self, leaf_ids, z):
            self.fit_inputs = (leaf_ids, z)
    
        def sample_from_leaves(self, leaf_ids):
            self.sample_inputs = leaf_ids
            return self.sample_return
    
        def clone(self):
            return self
    
    return SamplerStub()

@pytest.fixture
def stub_tree():
    class TreeStub(BaseDecisionTree):
        def __init__(self, apply_return=None):
            self.apply_return = apply_return
            self.fit_inputs = None
            self.apply_inputs = None
            pass

        def fit(self, X, y):
            self.fit_inputs = (X, y)
            return self

        def apply(self, X):
            self.apply_inputs = X
            return self.apply_return
    
    return TreeStub()

@pytest.fixture
def predictor(stub_encoder, stub_tree, stub_sampler):
    return MissingValuePredictor(
        encoding=stub_encoder,
        tree = stub_tree,
        tree_sampler= stub_sampler
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
def test_validate_fit_accepts_valid_X_y(predictor, X, y):
    predictor.encoding.transform_return = np.array([1, 2])
    out_X, out_y = predictor.prepare_data_for_fit(X, y)
    assert isinstance(out_X, dict)
    assert isinstance(out_y, np.ndarray)

@pytest.mark.parametrize(
    "X",[None, [1, 2, 3], "invalid", 123, [[1, 2], [3, 4], [5, 6] ]])
def test_validate_fit_raises_invalid_X(predictor, X):
    with pytest.raises(TypeError):
        predictor.prepare_data_for_fit(X, [0, 1, 2])

@pytest.mark.parametrize(
        "y", [{"a": [1, 2]}, [[1], [2]], None, "invalid", 123, []])
def test_validate_fit_raises_invalid_y(predictor, y):
    with pytest.raises(ValueError):
        predictor.prepare_data_for_fit({1: [1, 2]}, y)

@pytest.mark.parametrize(
    "X",
    [
        {"a": [[1, 2]], "b": [1, 2]},   # 2-dimensional
        {"a": [], "b": [1, 2]},         # empty column
        {"a": [1, 2], "b": [1]},        # length mismatch
        {"a": []}                       # empty key
    ],
)
def test_validate_fit_raises_bad_shapes(predictor, X):
    y = [0,1]
    with pytest.raises(ValueError):
        predictor.prepare_data_for_fit(X, y)

# ----- prepare data for fit tests -----
def test_prepare_data_respects_feature_order_through_flow(predictor):
    predictor.encoding.transform_return = np.array([1, 1, 1])
    X = {
        "a": [1, 2, 3],
        "b": ["x", None, "y"]
    }
    y = np.array([0, 1, 0])

    X_out, y_out = predictor.prepare_data_for_fit(X, y)
    
    assert predictor.feature_order_ == ["a", "b"], "Feature order contract should be handled correctly"
    assert np.array_equal(predictor.feature_order_, list(X_out.keys())), "output feature order should be the same as input"

def test_prepare_data_missing_data_flow_correct(predictor):
    predictor.encoding.transform_return = np.array([1, 1, 1, 1])
    predictor.tree.apply_return = np.array([0, 1, 2, 3])

    X = {"cat": ["a", "b", "a", "b"]}
    y = [0, None, 0, 1]

    X_out, y_out = predictor.prepare_data_for_fit(X, y)

    assert np.array_equal(predictor.encoders_["cat"].fit_inputs, (X["cat"], pd.isna(y))), "encoding input should be original X and and missingness mask"

    tree = predictor.tree_
    tree_X, z = tree.fit_inputs
    
    assert tree_X.shape == (4, 1) #Tree expects matrix
    assert np.array_equal(tree_X, predictor.encoding.transform_return.reshape(-1, 1)), "X input of the tree should be the output of the encoding but reshaped to 2D"

    sampler = predictor.tree_sampler_
    leaf_ids, z_sampler = sampler.fit_inputs

    assert np.array_equal(leaf_ids, predictor.tree.apply_return), "Sampler must receive tree.apply output as leaf IDs"

    expected_z = np.array([False, True, False, False]) #from y
    assert np.array_equal(z_sampler, expected_z), "missingness mask is not retrieved correctly"

    assert np.array_equal(X_out["cat"], ["a", "a", "b"])
    assert np.array_equal(y_out, [0, 0, 1])

def test_prepare_data_no_missing_data_flow(predictor):
    predictor.encoding.transform_return = np.array([1, 1, 1, 1])
    X = {"cat": ["a", "b", "c", "d"], "num": [1, 2, 3, 4]}
    y = [10, 20, 30, 40] 

    X_out, y_out = predictor.prepare_data_for_fit(X, y)

    assert predictor.tree_ is None, "no tree should be build when there are no missing values"
    assert predictor.tree_sampler_ is None, "no sampler should be used when there are no missing values"

    enc_num = predictor.encoders_["num"]
    assert enc_num is None

    enc_cat = predictor.encoders_["cat"]
    assert enc_cat is not None
    fit_X, fit_y = enc_cat.fit_inputs
    assert np.array_equal(fit_X, X["cat"]), "encoding input should be original X"
    assert np.array_equal(fit_y, pd.isna(y)), "encoding input should be original missingness mask"

    for col in X:
        assert np.array_equal(X_out[col], np.array(X[col]))

    assert np.array_equal(y_out, np.array(y)), "nothing should change to output data if no missing"
    assert np.array_equal(list(X_out.keys()), predictor.feature_order_)

def test_prepare_data_for_fit_mixed_types(predictor):
    predictor.encoding.transform_return = pd.array([1, 1, 1, 1, 1, 1]) #without the code fails
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

def test_prepare_data_does_not_mutate_inputs(predictor):
    predictor.encoding.transform_return = np.array([1, 1, 1])
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
def test_post_synth_transform_basic(predictor, stub_tree, stub_sampler):
    predictor.tree_ = stub_tree
    predictor.tree_.apply_return = np.array([0, 1, 2, 3])
    predictor.tree_sampler_ = stub_sampler
    predictor.tree_sampler_.sample_return = np.array([False, True, False, True])
    predictor.encoders_ = {"a": None, "b": None}
    predictor._all_missing = False
    predictor._no_missing = False

    X = {"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]}
    y = np.array([100, 200, 300, 400])
    predictor.feature_order_ = list(X.keys())

    out = predictor.post_synth_transform(X, y)

    assert predictor.tree_sampler_.sample_inputs is not None, "Sampler should be called"
    assert out.shape == y.shape
    assert np.array_equal(np.isnan(out), predictor.tree_sampler_.sample_return)


def test_post_synth_all_missing(predictor, stub_tree, stub_sampler):
    predictor._all_missing = True
    predictor._no_missing = False

    predictor.tree_ = stub_tree
    predictor.tree_sampler_ = stub_sampler
    predictor.encoders_ = {}

    X = {"a": [1, 2, 3]}
    y = np.array([1, 2, 3])
    predictor.feature_order_ = list(X.keys())
    out = predictor.post_synth_transform(X, y)

    assert np.all(np.isnan(out))

def test_post_synth_no_missing(predictor, stub_tree, stub_sampler):
    predictor._all_missing = False
    predictor._no_missing = True

    predictor.tree_ = stub_tree
    predictor.tree_sampler_ = stub_sampler
    predictor.encoders_ = {}

    X = {"a": [1, 2, 3]}
    y = np.array([1, 2, 3])
    predictor.feature_order_ = list(X.keys())
    out = predictor.post_synth_transform(X, y)

    assert np.array_equal(out, y)

def test_post_synth_transform_dataflow(predictor, stub_tree, stub_sampler):
    predictor.tree_ = stub_tree
    predictor.tree_.apply_return = np.array([0, 1, 2, 3])
    predictor.tree_sampler_ = stub_sampler
    predictor.tree_sampler_.sample_return = np.array([False, True, False, False])
    predictor._all_missing = False
    predictor._no_missing = False
    predictor.encoders_ = {
        "a": type("E", (), {"transform_return": np.array([1, 2, 3, 4]),
                           "transform": lambda self, x: self.transform_return})(),
        "b": type("E", (), {"transform_return": np.array([10, 20, 30, 40]),
                           "transform": lambda self, x: self.transform_return})()
    }
    predictor.feature_order_ = ["a", "b"]

    X = {"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]}
    y = np.array([100, 200, 300, 400])

    out = predictor.post_synth_transform(X, y)

    tree = predictor.tree_
    tree_X = tree.apply_inputs
    expected_X = np.column_stack([predictor.encoders_["a"].transform_return.reshape(-1, 1), predictor.encoders_["b"].transform_return.reshape(-1, 1)])

    assert np.array_equal(tree_X, expected_X), "tree input must be encoded X matrix"

    sampler = predictor.tree_sampler_
    leaf_ids = sampler.sample_inputs
    
    assert np.array_equal(leaf_ids, predictor.tree_.apply_return), "Sampler must receive tree.apply output as leaf IDs."
    assert np.array_equal(np.isnan(out), predictor.tree_sampler_.sample_return)
    

def post_synth_transform_raises_unfitted():
    model = MissingValuePredictor()
    with pytest.raises(AttributeError):
        model.post_synth_transform({"a": [1]}, [1])
    
def test_post_synth_uses_feature_order(predictor, stub_tree, stub_sampler):
    predictor.tree_ = stub_tree
    predictor.tree_.apply_return = np.array([0, 1, 2, 3])
    predictor.tree_sampler_ = stub_sampler
    predictor.tree_sampler_.sample_return = np.array([False, True, False, False])
    predictor._all_missing = False
    predictor._no_missing = False
    predictor.encoders_ = {"a": None, "b": None}
    predictor.feature_order_ = ["a", "b"]

    X = {
        "b": [10, 20, 30, 40],
        "a": [1, 2, 3, 4]
    }
    y = np.array([100, 200, 300, 400])

    out = predictor.post_synth_transform(X, y)

    tree = predictor.tree_
    tree_X = tree.apply_inputs
    expected_X = np.column_stack((X["a"], X["b"]))

    assert np.array_equal(tree_X, expected_X), "feature_order_ must override input dict ordering"

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