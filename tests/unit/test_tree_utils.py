import re

import numpy as np
import pandas as pd
import pytest 
from sklearn.base import clone

from synthpop.methods.tree_utils import _sample_array, _fit_decision_tree_with_reachable_leaves, _check_all_leaf_nodes_are_reached

# ----- stubs -----
class StubRNG:
    def __init__(self, output_array):
        self.output_array = np.array(output_array)
        self.called_with = []

    def integers(self, low, high, size=None):
        self.called_with.append((low, high, size))

        if size is None:
            return self.output_array[0]
        
        return self.output_array[:size]

class MockTree:
    def __init__(self, tree=None, random_state=None):
        self.tree = tree
        self.random_state = random_state

    def fit(self, X, y):
        self.fit_X = X
        self.fit_y = y
        self.tree_ = self.tree
        self.random_state_ = self.random_state
        return self

    def __sklearn_clone__(self):
        result = MockTree(tree=None, random_state=self.random_state)
        if self.tree is None:
            return result
        else:
            tree = clone(self.tree)
            result.tree = tree
            result.tree.parent = result
        return result

class BaseTree:
    # Used for tests about fitting a reachable tree.
    # A fitted tree is reachable when:
    # all leaf nodes are returned when apply is called on the same data used for fitting
    # all leaf nodes can be found by checking tree.children_left == tree.children_right
    def __init__(self, leaf_node_ids=None, n_nodes=1, apply_return_val=None, parent=None):
        self.n_nodes = n_nodes
        self.apply_return_val = apply_return_val
        # to help make assertions about the correct tree being used.
        self.parent = parent
        self.leaf_node_ids = leaf_node_ids

        self.children_left = np.arange(0, n_nodes, dtype=np.int_)
        self.children_right = self.children_left + 10
        self.children_right[leaf_node_ids] = self.children_left[leaf_node_ids]

    def apply(self, X):
        self.apply_X = X
        return self.apply_return_val

    def __sklearn_clone__(self):

        return BaseTree(leaf_node_ids=self.leaf_node_ids,
                         n_nodes=self.n_nodes,
                         apply_return_val=self.apply_return_val,)

# ----- _fit_decision_tree_with_reachable_leaves tests -----

def test_check_all_leaf_nodes_are_reached_returns_true_on_reachable_tree():

    tree = BaseTree(n_nodes=10, leaf_node_ids=[
                     2, 3, 4], apply_return_val=[2, 4, 4, 3, 3, 3])
    X_train = np.array([1, 2, 3])

    result = _check_all_leaf_nodes_are_reached(tree=tree, X_train=X_train)

    assert result
    assert tree.apply_X is X_train


def test_check_all_leaf_nodes_are_reached_returns_false_on_unreachable_tree():

    # leaf 5 not reached by apply
    tree = BaseTree(n_nodes=10, leaf_node_ids=[
                     2, 3, 4, 5], apply_return_val=[2, 4, 4, 3, 3, 3])
    X_train = np.array([1, 2, 3])

    result = _check_all_leaf_nodes_are_reached(tree=tree, X_train=X_train)

    assert not result
    assert tree.apply_X is X_train


def test_fit_decision_tree_with_reachable_leaves_first_time_reachable(mocker):

    mock_check_all_leaf_nodes_are_reached = mocker.patch(
        "synthpop.methods.tree_utils._check_all_leaf_nodes_are_reached", return_value=True)
    decision_tree = MockTree(tree=BaseTree())

    X = np.array([1, 2, 3])
    y = np.array([4, 5, 6])

    result = _fit_decision_tree_with_reachable_leaves(
        decision_tree=decision_tree, X=X, y=y)

    assert result is decision_tree
    assert np.array_equal(result.fit_X, X)
    assert np.array_equal(result.fit_y, y)

    mock_check_all_leaf_nodes_are_reached.assert_called_once_with(
        tree=decision_tree.tree_, X_train=X)


def test_fit_decision_tree_with_reachable_leaves_retry_when_unreachable(mocker):

    mock_check_all_leaf_nodes_are_reached = mocker.patch(
        "synthpop.methods.tree_utils._check_all_leaf_nodes_are_reached", side_effect=[False, False, True])
    new_random_states = [222, 333]
    mock_create_instance_seed = mocker.patch(
        "synthpop.reproducibility.RandomStateManager.create_instance_seed", side_effect=new_random_states)
    expected_random_states = [3]+new_random_states
    decision_tree = MockTree(random_state=3)
    decision_tree.tree = BaseTree(parent=decision_tree)

    X = np.array([1, 2, 3])
    y = np.array([4, 5, 6])

    result = _fit_decision_tree_with_reachable_leaves(
        decision_tree=decision_tree, X=X, y=y)

    # Assert that _check_all_leaf_nodes_are_reached has been called with the correct arguments.
    # In this case, that is a tree with the random_state parameter set to the return values of create_instance_seed

    for i in range(len(expected_random_states)):
        kwargs = mock_check_all_leaf_nodes_are_reached.mock_calls[i][2]
        tree = kwargs["tree"].parent

        # The following assertion asserts that:
        # - _check_all_leaf_nodes_are_reached has been called the expected number of times
        # - for each call, the random state has been changed before the tree is fit again.
        # - The first attempt is made without altering the seed.
        # - new instances are used (the call to clone)
        assert tree.random_state_ == expected_random_states[i], "The random_state has not been set"

        # The following asserts that:
        # - The tree passed to _check_all_leaf_nodes_are_reached has been fitted before the call.
        # - The call to fit happened with the correct parameters.
        assert np.array_equal(
            tree.fit_X, X), "tree has not been fitted again with correct X before passing it to _check_all_leaf_nodes_are_reached"
        assert np.array_equal(
            tree.fit_y, y), "tree has not been fitted again with correct y  before passing it to _check_all_leaf_nodes_are_reached"

    assert result.random_state_ == expected_random_states[
        -1], "_fit_decision_tree_with_reachable_leaves returned the wrong results"


def test_fit_decision_tree_with_reachable_leaves_raises_after_100_tries(mocker):
    mock_check_all_leaf_nodes_are_reached = mocker.patch(
        "synthpop.methods.tree_utils._check_all_leaf_nodes_are_reached", return_value=False)
    new_random_states = range(99)
    mock_create_instance_seed = mocker.patch(
        "synthpop.reproducibility.RandomStateManager.create_instance_seed", side_effect=new_random_states)
    decision_tree = MockTree(random_state=3)
    decision_tree.tree = BaseTree(parent=decision_tree)

    X = np.array([1, 2, 3])
    y = np.array([4, 5, 6])

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Failed to fit a decision tree with reachable leaves after 100 attempts")
    ):
        _fit_decision_tree_with_reachable_leaves(
            decision_tree=decision_tree, X=X, y=y)

# ----- _sample_array tests -----

def test_sample_array_maps_rng_output_to_sampled_values():
    rng = StubRNG([0, 3, 1, 2])

    counts = np.array([3, 1])   # total = 4
    values = np.array([0, 1])

    out = _sample_array(rng, counts, values, n_samples=4)

    # cumulative bins: [3, 4]
    # r=0 → idx=0 → 0
    # r=3 → idx=1 → 1
    # r=1 → idx=0 → 0
    # r=2 → idx=0 → 0
    expected = np.array([0, 1, 0, 0])

    assert np.array_equal(out, expected)

    assert rng.called_with == [(0, 4, 4)] #low is 0, high is 4, n_samples is 4

@pytest.mark.parametrize(
    "counts, values",
    [
        ([1, 1, 1], [10, 20, np.nan]),        # balanced
        ([9, 1], [10, np.nan]),               # rare missing
        ([1, 9], [10, np.nan]),               # dominant missing
    ],
)
def test_sample_array_samples_missing(counts, values):
    rng = np.random.default_rng(42)

    counts = np.array(counts)
    values = np.array(values, dtype=object)

    out = _sample_array(rng, counts, values, n_samples=1000)

    if any(pd.isna(v) for v in values):
        assert any(pd.isna(out))

def test_sample_array_shape_and_dtype():
    rng = np.random.default_rng(42)

    counts = np.array([2, 3, 5])
    values = np.array([10, 20, 30], dtype=np.int32)

    out = _sample_array(rng, counts, values, n_samples=7)

    assert out.shape == (7,)
    assert out.dtype == values.dtype

def test_sample_array_values_in_support():
    rng = np.random.default_rng(0)

    counts = np.array([1, 1, 1])
    values = np.array(["a", "b", "c"], dtype=object)

    out = _sample_array(rng, counts, values, n_samples=50)

    assert set(out).issubset(set(values))

def test_sample_array_zero_count_never_sampled():
    rng = np.random.default_rng(0)

    counts = np.array([5, 0, 5])
    values = np.array([1, 2, 3])

    out = _sample_array(rng, counts, values, n_samples=100)

    assert 2 not in out

@pytest.mark.parametrize(
    "values, counts",
    [
        ([1, 2, 3], [1, 1, 1]),     # Uniform distribution
        ([1, 2, 3], [5, 3, 2]),     # Moderate skew
        ([1, 2, 3], [98, 1, 1]),    # Extreme skew
        ([1], [10]),                # Degenerate (single outcome)
        (["a", "b"], [999, 1]),         # Binary extreme
    ],
)
def test_sample_array_distribution(values, counts):
    rng = np.random.default_rng(123)

    counts = np.array(counts)
    values = np.array(values)

    n = 10000
    out = _sample_array(rng, counts, values, n_samples=n)

    total = counts.sum()
    expected = {v: c / total for v, c in zip(values, counts)}

    for v in values:
        observed = np.mean(out == v)

        if len(values) == 1:
            # Degenerate case should be exact
            assert observed == 1.0
        else:
            assert np.isclose(observed, expected[v], atol=0.02)

def test_sample_array_zero_samples():
    rng = np.random.default_rng(0)

    counts = np.array([1, 2])
    values = np.array([10, 20])

    out = _sample_array(rng, counts, values, n_samples=0)

    assert out.shape == (0,)

def test_sample_array_all_zero_counts():
    rng = np.random.default_rng(0)

    counts = np.array([0, 0])
    values = np.array([1, 2])

    with pytest.raises(ValueError):
        _sample_array(rng, counts, values, n_samples=3)


