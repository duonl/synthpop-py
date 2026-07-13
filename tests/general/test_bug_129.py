"""
These are regression tests for Bug #129.

Bug #129 occurred when `DecisionTree.apply()` returned leaf IDs during prediction
that were never reached by the training data. This caused downstream sampling to fail.

After making the synthesis process reproducible, several combinations of tree seeds
and generated datasets were found to reproduce the problem deterministically.
These tests verify that those combinations no longer produce empty leaves
and that prediction succeeds.
"""
import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from synthpop.data_processing.missing_value_handling import MissingValuePredictor
from synthpop.methods.cart_synth import TreeClassifierMethod, TreeRegressorMethod
from synthpop.methods.tree_utils import LeafNodeSampler

from tests.integration.data_generated_for_tests import get_test_data_classifier, get_test_data_regressor


class SpyDecisionTreeClassifier(DecisionTreeClassifier):
    def fit(self, X, y):
        self.fit_X = X
        self.fit_y = y
        return super().fit(X, y)

    def apply(self, X):
        self.apply_X = X
        return super().apply(X)


class SpyDecisionTreeRegressor(DecisionTreeRegressor):
    def fit(self, X, y):
        self.fit_X = X
        self.fit_y = y
        return super().fit(X, y)

    def apply(self, X):
        self.apply_X = X
        return super().apply(X)
    

def assert_all_leaf_nodes_reached(tree):
    reached = tree.apply(tree.fit_X)
    leaves = np.where(
        tree.tree_.children_left == tree.tree_.children_right
    )[0]

    assert set(leaves) == set(reached)


@pytest.mark.noautofixt
# These are four seeds that reproduced bug #129 before the fix.
@pytest.mark.parametrize("tree_seed,data_seed", [(i, j) for i in [89, 88, 52, 91]for j in [59, 14, 51, 80]])
def test_regression_bug_129_regressor_no_empty_leaf_failure(tree_seed, data_seed):

    seeds = np.random.SeedSequence(0).generate_state(6)

    # the seed for the tree and the seed for the data make bug 129 fully deterministic,
    # including the exact leaf id.
    # This means that the randomness from the missing value predictor or the leaf node sampler does not affect this bug.

    # A tree regressor method is constructed in this test to make this test stable w.r.t. changes in this package.
    method = TreeRegressorMethod(
        tree=SpyDecisionTreeRegressor(
            min_samples_leaf=5,    # equivalent to minbucket in synthpop-r
            min_impurity_decrease=1e-08,   # equivalent to cp in synthpop-r
            random_state=tree_seed  # None#bad_tree_seed#seeds[0]
        ),
        missing_handler=MissingValuePredictor(
            tree=DecisionTreeClassifier(
                min_samples_leaf=5, random_state=seeds[5])
        ),
        tree_sampler=LeafNodeSampler(random_state=seeds[2])
    )

    X, y = get_test_data_regressor(
        n_samples=50, seed=data_seed, with_cats=True, with_missing_features=True, with_missing_target=True)

    method.fit(X, y)

    assert_all_leaf_nodes_reached(method.tree_)

    rng = np.random.default_rng(seeds[4])

    X_pred = {}
    for col in X.keys():
        X_pred[col] = rng.choice(X[col], size=len(X[col])*100, replace=True)

    result = method.transform(X_pred)

    assert len(result) == len(y)*100


@pytest.mark.parametrize("tree_seed,data_seed", [(i, j) for i in [89, 88, 52, 91]for j in [59, 14, 51, 80]])
def test_regression_bug_129_classifier_no_empty_leaf_failure(tree_seed, data_seed):
    method = TreeClassifierMethod(
        tree=SpyDecisionTreeClassifier(
            min_samples_leaf=5,    # equivalent to minbucket in synthpop-r
            min_impurity_decrease=1e-08,   # equivalent to cp in synthpop-r
            random_state=tree_seed  # None#bad_tree_seed#seeds[0]
        ),
        tree_sampler=LeafNodeSampler(random_state=0)
    )

    X, y = get_test_data_classifier(
        n_samples=50, seed=data_seed, with_cats=True, with_missing_features=True, with_missing_target=True)

    method.fit(X, y)

    assert_all_leaf_nodes_reached(method.tree_)

    rng = np.random.default_rng([tree_seed, data_seed])

    X_pred = {}
    for col in X.keys():
        X_pred[col] = rng.choice(X[col], size=len(X[col])*100, replace=True)
        make_missing = rng.choice(
            [True, False], size=len(X[col])*100, replace=True)
        X_pred[col][make_missing] = np.nan

    result = method.transform(X_pred)
    assert len(result) == len(y)*100
