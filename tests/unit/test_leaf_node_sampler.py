from sklearn.utils.validation import NotFittedError
from sklearn.base import BaseEstimator, clone
import numpy as np
import pandas as pd
import pytest 
import copy

from synthpop.methods.tree_utils import LeafNodeSampler

class DummyTree(BaseEstimator):
    def __init__(self, leaf_ids):
        self._leaf_ids = np.asarray(leaf_ids)

    def fit(self, X, y):
        self.tree_ = True
        return self

    def apply(self, X):
        X = np.asarray(X)
        if len(X) != len(self._leaf_ids):
            return np.array([self._leaf_ids[0]] * len(X))
        return self._leaf_ids

# ----- fit sampler test cases -----
@pytest.mark.parametrize(
    "X, y, leaf_ids, expected_map",
    [
        # --- numerical numpy arrays ---
        (np.array([0, 1, 2, 3]), np.array([0, 0, 1, 1]), np.array([10, 10, 20, 20]),
            {10: {0: 2}, 20: {1: 2}}),

        # --- lists ---
        ([[0], [1], [2]], [0, 1, 1], [10, 10, 20],
            {10: {0: 1, 1: 1}, 20: {1: 1}}),

        # --- pandas Series ---
        (pd.DataFrame([0, 1, 2]).values, pd.Series([0, 1, 1]), [10, 10, 20],
            {10: {0: 1, 1: 1}, 20: {1: 1}}),

        # --- strings ---
        (np.array(["a", "b", "c"]), np.array(["x", "z", "y"]), np.array([1, 1, 2]),
            {1: {"x": 1, "z": 1}, 2: {"y": 1}}),

        (np.array([1, 2, 3]), np.array(["a", "a", "b"]), np.array([1, 1, 2]),
            {1: {"a": 2}, 2: {"b": 1}}),

        # --- mixed: None, np.nan, pd.NA ---
        (np.array([0, 1, 2, 3]), np.array([None, np.nan, pd.NA, 1], dtype=object), np.array([10, 10, 20, 20]),
            {10: {None: 1, np.nan: 1}, 20: {pd.NA: 1, 1: 1,}}),
        
        # --- mixed string and integer ---
        (np.array([1, "1", 1], dtype=object), np.array([1, "1", 1.0], dtype=object), [1, 2, 1],
            {1: {1: 2}, 2: {"1": 1}}),

        # Single input
        (np.array([1]), np.array([5]), np.array([10]),
            {10: {5: 1}}),

    ],
)

def test_fit_sampler_parametrized_inputs(X, y, leaf_ids, expected_map):
    """
    From various inputs the correct count mapping should be created.
    """
    tree = DummyTree(leaf_ids)
    tree.tree_ = True

    sampler = LeafNodeSampler()
    sampler.fit_sampler(tree, X, y)

    assert hasattr(sampler, "_leaf_map")
    assert hasattr(sampler, "random_state_")
    assert hasattr(sampler, "tree_")
    assert sampler._leaf_map.keys() == expected_map.keys()

    for leaf_id in expected_map:
        for key, count in expected_map[leaf_id].items():
            assert sampler._leaf_map[leaf_id][key] == count

def test_fit_sampler_empty_input():
    tree = DummyTree([])
    tree.tree_ = True
    sampler = LeafNodeSampler()
    X = np.empty((0, 1))
    y = np.array([])

    sampler.fit_sampler(tree, X, y)
    assert hasattr(sampler, "_leaf_map")
    assert sampler._leaf_map == {}

def test_fit_sampler_throws_shape_mismatch():
    X = np.array([0, 1, 2])
    y = np.array([0, 1])

    leaf_ids = np.array([10, 10, 20])
    tree = DummyTree(leaf_ids)
    tree.tree_ = True
    sampler = LeafNodeSampler()

    with pytest.raises(ValueError, match="X and y must have the same number of samples"):
        sampler.fit_sampler(tree, X, y)

def test_fit_sampler_throws_unfitted():
    X = np.array([0, 1])
    y = np.array([0, 1])
    tree = DummyTree([10, 20])
    sampler = LeafNodeSampler()

    with pytest.raises(NotFittedError):
        sampler.fit_sampler(tree, X, y)

# ----- sample from leaves test cases -----
def helper_make_sampler(leaf_map, leaf_ids, random_state=42):
    """
    Helper to construct a minimally fitted sampler
    """
    sampler = LeafNodeSampler(random_state=random_state)
    sampler._leaf_map = leaf_map
    sampler.tree_ = DummyTree(leaf_ids)
    sampler.tree_.tree_ = True
    sampler.random_state_ = np.random.RandomState(random_state)

    return sampler

@pytest.mark.parametrize(
    "X_syn",
    [
        np.array([0, 1, 2]),                         # numpy numeric
        [0, 1, 2],                                   # python list numeric
        pd.Series([0, 1, 2]),                        # pandas Series numeric
        np.array(["a", "b", "c"]),                   # numpy strings
        ["a", "b", "c"],                             # python list strings
        pd.Series(["a", "b", "c"]),                  # pandas Series strings
        np.array([None, np.nan, pd.NA], dtype=object),  # mixed missing
        [None, np.nan, pd.NA],                       # python list mixed missing
        np.array(["1", 1, "2"], dtype=object),       # mixed dtypes
        np.array([[0], [1], [2]]),                   # 2d input
        np.array([[[0]], [[1]], [[2]]])              # 3d input
    ],
)

def test_sample_from_leaves_various_input_types(X_syn):
    """
    Ensure sample_from_leaves works with multiple input container types and dtypes.
    """

    leaf_map = {10: {0: 3, 1: 1}}
    leaf_ids = [10] * 3

    sampler = helper_make_sampler(leaf_map, leaf_ids, random_state=42)
    
    y_syn = sampler.sample_from_leaves(X_syn)

    expected_values = set(leaf_map[10].keys())

    for val in y_syn:
        if any(pd.isna(k) for k in expected_values):
            assert pd.isna(val) or val in expected_values
        else:
            assert val in expected_values

    assert len(y_syn) == len(X_syn)

def test_sample_from_leaves_respects_probabilities():
    """
    Sampling should follow empirical probabilities.
    """
    sampler = helper_make_sampler(
        leaf_map={10: {0: 3, 1: 1}},
        leaf_ids=[10] * 1000,
        random_state=42
    )

    X_syn = np.arange(1000).reshape(-1, 1)
    y_syn = sampler.sample_from_leaves(X_syn)

    proportion_ones = np.mean(y_syn == 1)

    assert 0.15 < proportion_ones < 0.35

def test_sample_large_imbalanced_distribution():
    sampler = helper_make_sampler(
        leaf_map={10: {0: 10**6, 1: 1}},
        leaf_ids=[10] * 100
    )

    X = np.arange(100).reshape(-1, 1)
    y = sampler.sample_from_leaves(X)

    # Should overwhelmingly favour 0 but still include 1 occasionally
    proportion_ones = np.mean(y == 1)

    assert proportion_ones < 0.01

def test_sample_from_leaves_handles_nan():
    """
    NaN values should be sampled correctly.
    """
    sampler = helper_make_sampler(
        leaf_map={10: {np.nan: 2, pd.NA: 1, None: 1, 1: 1}},
        leaf_ids=[10, 10, 10, 10, 10]
    )

    X_syn = np.array([0, 1, 2, 3, 4])
    y_syn = sampler.sample_from_leaves(X_syn)

    assert len(y_syn) == 5
    assert any(pd.isna(v) for v in y_syn)

def test_sample_determinism_with_same_seed():
    leaf_map = {10: {0: 3, 1: 1}}
    leaf_ids = [10] * 5

    sampler1 = helper_make_sampler(leaf_map, leaf_ids, random_state=42)
    sampler2 = helper_make_sampler(leaf_map, leaf_ids, random_state=42)

    X = np.arange(5).reshape(-1, 1)

    y1 = sampler1.sample_from_leaves(X)
    y2 = sampler2.sample_from_leaves(X)

    assert np.array_equal(y1, y2)

def test_sample_empty_histogram_allows_nan():
    sampler = helper_make_sampler(
        leaf_map={10: {0: 0, 1: 0}},
        leaf_ids=[10]
    )

    y = sampler.sample_from_leaves(np.array([[1]]))

    assert len(y) == 1
    assert pd.isna(y[0])

def test_sample_all_values_same_leaf():
    sampler = helper_make_sampler(
        leaf_map={10: {5: 3}},
        leaf_ids=[10] * 5
    )

    X = np.arange(5).reshape(-1, 1)
    y = sampler.sample_from_leaves(X)

    assert np.all(y == 5)

def test_sample_from_leaves_raises_unseen():
    """
    Unseen leaf id should raise ValueError.
    """
    sampler = helper_make_sampler(
        leaf_map={10: {0: 1}},
        leaf_ids=[999]  # not in leaf_map
    )

    with pytest.raises(ValueError, match="Leaf id .* not seen during fitting"):
        sampler.sample_from_leaves(np.array([0]))


def test_sample_from_leaves_empty_histogram_returns_nan():
    """
    Empty histogram should return NaN.
    """
    sampler = helper_make_sampler(
        leaf_map={10: {}},
        leaf_ids=[10]
    )

    y_syn = sampler.sample_from_leaves(np.array([0]))

    assert len(y_syn) == 1
    assert pd.isna(y_syn[0])

def test_sample_from_leaves_raises_unfitted():
    """
    Missing required attributes should raise AttributeError.
    """
    sampler = LeafNodeSampler()
    with pytest.raises(AttributeError):
        sampler.sample_from_leaves(np.array([0]))
    sampler.tree_ = True
    with pytest.raises(AttributeError):
        sampler.sample_from_leaves(np.array([0]))
    sampler._leaf_map = {}
    with pytest.raises(AttributeError):
        sampler.sample_from_leaves(np.array([0]))

def test_sample_unhashable_tree_output_raises():
    class BadTree(BaseEstimator):
        def fit(self, X, y):
            self.tree_ = True
            return self

        def apply(self, X):
            return [[1]] * len(X)  # unhashable leaf ids

    tree = BadTree()
    tree.fit(None, None)

    sampler = LeafNodeSampler()
    sampler._leaf_map = {1: {0: 1}}
    sampler.tree_ = tree
    sampler.random_state_ = np.random.RandomState(42)

    with pytest.raises(TypeError):
        sampler.sample_from_leaves(np.array([[1]]))

# ----- clonability tests -----
def test_clone_works_and_fitted_sampler_does_not_preserve_state():
    tree = DummyTree([10, 10, 20])
    tree.tree_ = True

    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 1])

    sampler = LeafNodeSampler(random_state=42)
    sampler.fit_sampler(tree, X, y)

    #cloned = clone(sampler)
    cloned = sampler.clone()

    # Fitted attributes should NOT be copied
    assert not hasattr(cloned, "_leaf_map")
    assert not hasattr(cloned, "tree_")
    assert not hasattr(cloned, "random_state_")

    # Original remains intact
    assert hasattr(sampler, "_leaf_map")
    assert hasattr(sampler, "tree_")
    assert hasattr(sampler, "random_state_")

