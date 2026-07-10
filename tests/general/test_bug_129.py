
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
import pytest 

from synthpop.data_processing.encoders import PCAEncoder
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


@pytest.mark.parametrize("seed",[i for i in range(50)])
def test_regression_bug_129_classifier_no_empty_leaf_failure(seed):

    seeds = np.random.SeedSequence(seed).generate_state(5)

    method = TreeClassifierMethod(
            tree=DecisionTreeClassifier(
                min_samples_leaf=5,    # equivalent to minbucket in synthpop-r
                min_impurity_decrease=1e-08,   # equivalent to cp in synthpop-r
                random_state=seeds[0]
            ),
            encoder=PCAEncoder(
                pca_transform=PCA(random_state=seeds[1])
            ),
            tree_sampler=LeafNodeSampler(random_state=seeds[2])
        )
    
    X,y = get_test_data_classifier(seed=seeds[3],with_cats=True,with_missing_features= True,with_missing_target=True)

    method.fit(X,y)

    
    rng = np.random.default_rng(seeds[4])

    for attempts in range(100):
        X_pred = {}
        for col in X.keys():
            X_pred[col] = rng.choice(X[col],size = len(X[col]),replace=True)
            make_missing = rng.choice([True,False],size = len(X[col]),replace=True)
            X_pred[col][make_missing] = np.nan
            

        result = method.transform(X_pred)


@pytest.mark.noautofixt
@pytest.mark.parametrize("tree_seed,data_seed",[(i,j) for i in [89,88,52,91]for j in [59,14,51,80]])
def test_regression_bug_129_regressor_no_empty_leaf_failure(tree_seed,data_seed):


    seeds = np.random.SeedSequence(0).generate_state(6)

    # the seed for the tree and the seed for the data make bug 129 fully deterministic, 
    # including the exact leaf id.
    # This means that the randomness from the missing value predictor or the leaf node sampler does not affect this bug.

    # A tree regressor method is constructed in this test to make this test stable w.r.t. changes in this package.
    method = TreeRegressorMethod(
            tree=DecisionTreeRegressor(
                min_samples_leaf=5,    # equivalent to minbucket in synthpop-r
                min_impurity_decrease=1e-08,   # equivalent to cp in synthpop-r
                random_state=tree_seed#None#bad_tree_seed#seeds[0]
            ),
            missing_handler=MissingValuePredictor(
                tree=DecisionTreeClassifier(min_samples_leaf=5,random_state=seeds[5])
            )
            ,
            tree_sampler=LeafNodeSampler(random_state=seeds[2])
        )

    X,y = get_test_data_regressor(n_samples=50,seed=data_seed,with_cats=True,with_missing_features= True,with_missing_target=True)

    method.fit(X,y)

    
    rng = np.random.default_rng(seeds[4])

    for attempts in range(100):
        X_pred = {}
        for col in X.keys():
            X_pred[col] = rng.choice(X[col],size = len(X[col]),replace=True)
            

        result = method.transform(X_pred)

@pytest.mark.parametrize("tree_seed,data_seed",[(i,j) for i in [89,88,52,91]for j in [59,14,51,80]])
def test_regressor_trainings_data_reaches_all_nodes(tree_seed,data_seed):
    method = TreeRegressorMethod(
            tree=SpyDecisionTreeRegressor(
                min_samples_leaf=5,    # equivalent to minbucket in synthpop-r
                min_impurity_decrease=1e-08,   # equivalent to cp in synthpop-r
                random_state=tree_seed#None#bad_tree_seed#seeds[0]
            ),
            missing_handler=MissingValuePredictor(
                tree=DecisionTreeClassifier(min_samples_leaf=5,random_state=1)
            )
            ,
            tree_sampler=LeafNodeSampler(random_state=0)
        )
    
    X,y = get_test_data_regressor(n_samples=50,seed=data_seed,with_cats=True,with_missing_features= True,with_missing_target=True)

    method.fit(X,y)

    X_train = method.tree_.fit_X
    reached_nodes = method.tree_.apply(X_train)

    all_leaf_nodes = np.where(method.tree_.tree_.children_left==method.tree_.tree_.children_right)

    assert set(all_leaf_nodes[0]) == set(reached_nodes)

@pytest.mark.parametrize("tree_seed,data_seed",[(i,j) for i in [89,88,52,91]for j in [59,14,51,80]])
def test_classifier_trainings_data_reaches_all_nodes(tree_seed,data_seed):
    method = TreeClassifierMethod(
            tree=SpyDecisionTreeClassifier(
                min_samples_leaf=5,    # equivalent to minbucket in synthpop-r
                min_impurity_decrease=1e-08,   # equivalent to cp in synthpop-r
                random_state=tree_seed#None#bad_tree_seed#seeds[0]
            ),
            missing_handler=MissingValuePredictor(
                tree=DecisionTreeClassifier(min_samples_leaf=5,random_state=1)
            )
            ,
            tree_sampler=LeafNodeSampler(random_state=0)
        )
    
    X,y = get_test_data_classifier(n_samples=50,seed=data_seed,with_cats=True,with_missing_features= True,with_missing_target=True)

    method.fit(X,y)

    X_train = method.tree_.fit_X
    reached_nodes = method.tree_.apply(X_train)

    all_leaf_nodes = np.where(method.tree_.tree_.children_left==method.tree_.tree_.children_right)

    assert set(all_leaf_nodes[0]) == set(reached_nodes)