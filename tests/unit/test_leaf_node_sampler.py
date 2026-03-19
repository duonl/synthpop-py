from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import NotFittedError
import numpy as np
import pandas as pd
import pytest 

from synthpop.methods.tree_utils import LeafNodeSampler

from sklearn.base import BaseEstimator

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