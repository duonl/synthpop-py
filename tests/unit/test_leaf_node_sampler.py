from sklearn.utils.validation import NotFittedError
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
import pytest 

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
            {10: {np.nan: 2}, 20: {np.nan: 1, 1: 1,}}),
    ],
)

def test_fit_sampler_parametrized_inputs(X, y, leaf_ids, expected_map):
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
            if isinstance(key, float) and np.isnan(key):
                assert any(
                    np.isnan(k) and v == count
                    for k, v in sampler._leaf_map[leaf_id].items()
                )
            else:
                assert sampler._leaf_map[leaf_id][key] == count

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
        [None, np.nan, pd.NA],                        # python list mixed missing
        np.array(["1", 1, "2"], dtype=object)
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


def test_sample_from_leaves_unseen_leaf_raises():
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

def test_sample_from_leaves_not_fitted():
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

