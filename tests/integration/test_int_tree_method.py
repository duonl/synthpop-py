import copy
import pytest
import pandas as pd
import numpy as np
import string

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from synthpop.data_processing.encoders import PCAEncoder
from synthpop.data_processing.missing_value_handling import MissingValuePredictor
from synthpop.methods.cart_synth import TreeClassifierMethod, TreeRegressorMethod
from synthpop.methods.tree_utils import LeafNodeSampler
from synthpop.utils import str_dtype
from sklearn.datasets import make_classification, make_regression

# This imports an auto-use fixture to set the seed, in order to make the test reproducible
from tests.integration.make_int_test_reproducible import control_random_state_manager
from tests.integration.data_generated_for_tests import get_test_data_classifier,get_test_data_regressor

def test_treemethod_classifier_fit_and_transform():
    tree_method = TreeClassifierMethod()

    X = {
        "column1":np.array([1.1,2.2]),
        "column2":np.array([1.4,1.2]),
        "column3":np.array(["a","b"],dtype = str_dtype)
        }
    y = np.array(["x","y"],dtype = str_dtype)

    tree_method.fit(X,y)
    assert tree_method.n_features_in_ >= 3
    assert y.dtype == str_dtype

    result = tree_method.transform(X)

    assert result.shape[0] ==2


def test_treemethod_regressor_fit_and_transform():
    tree_method = TreeRegressorMethod()

    X = {
        "column1":np.array([1.1,2.2]),
        "column2":np.array([1.4,1.2]),
        "column3":np.array(["a","b"],dtype = str_dtype)
        }
    y = np.array([1,2])

    tree_method.fit(X,y)
    assert tree_method.n_features_in_ >= 3
    
    result = tree_method.transform(X)

    assert result.shape[0] ==2


def make_data_missing(X):

    #We need a pattern of missingness that is different for each column
    # The missingness pattern should not be too predictable.

    for ik, k in enumerate(X.keys()):

        # The missingness is periodic. ever p-th element is missing.
        # The value of p decreases for each column.
        p = (len(X.keys())-ik) 

        values = [v if i % p !=1 else np.nan for i,v in enumerate(X[k])]
        if pd.api.types.is_numeric_dtype(X[k].dtype):
            X[k]=np.array(values)
        else:
            X[k] = np.array(values,dtype=str_dtype)

    return X


# def get_test_data_classifier(with_cats = False,with_missing_features=False,with_missing_target=False):
#     X,y = make_classification(n_classes=10,n_informative=11)
    
#     X = {i:X[:,i] for i in range(X.shape[1])}


#     idx_cats = [3,4,6]
#     if with_cats:
#         for idx in idx_cats:
#             x = (X[idx]*10).astype(int)
#             x_i = [f %5 for f in x]
#             X[idx] = np.array([string.ascii_lowercase[i%26] for i in x_i],dtype=str_dtype)

#     if with_missing_features:
#         X = make_data_missing(X)

#     if with_missing_target:
#         y = np.array([string.ascii_lowercase[i%26] if i%5 !=0 else np.nan for i in y],dtype=str_dtype)
#     else:
#         y = np.array([string.ascii_lowercase[i%26] for i in y],dtype=str_dtype)
#     return (X,y)


# def get_test_data_regressor(with_cats=False,with_missing_features=False,with_missing_target=False):
#     X,y = make_regression()
#     X = {i:X[:,i] for i in range(X.shape[1])}

#     idx_cats = [3,4,6]
#     if with_cats:
#         for idx in idx_cats:
#             x = (X[idx]*10).astype(int)
#             x_i = [f %26 for f in x]
#             X[idx] = np.array([string.ascii_lowercase[i] for i in x_i],dtype = str_dtype)

#     if with_missing_features:
#         X = make_data_missing(X)

#     if with_missing_target:
#         y = np.array([v if i%5 !=0 else np.nan for i,v in enumerate(y)])

#     return (X,y)

    
class SpyDecisionTreeClassifier(DecisionTreeClassifier):
        def fit(self,X,y):
            self.fit_X = X
            self.fit_y = y
            return super().fit(X,y)
        
        def apply(self,X):
            self.apply_X = X
            return super().apply(X)
        
class SpyDecisionTreeRegressor(DecisionTreeRegressor):
        def fit(self,X,y):
            self.fit_X = X
            self.fit_y = y
            return super().fit(X,y)
        
        def apply(self,X):
            self.apply_X = X
            return super().apply(X)
        

def rigged_tree_classifier_method(pca_components = 1):
    tree = SpyDecisionTreeClassifier(min_samples_leaf=5)
    return TreeClassifierMethod(tree=tree,encoder=PCAEncoder(pca_transform= PCA(n_components=pca_components)))

def rigged_tree_regressor_method():
    tree = SpyDecisionTreeRegressor(min_samples_leaf=5)
    return TreeRegressorMethod(tree=tree)

CLASSIFIER_CASES = [
    (rigged_tree_classifier_method(),*get_test_data_classifier()),
    (rigged_tree_classifier_method(),*get_test_data_classifier(with_cats=True)),
    (rigged_tree_classifier_method(),*get_test_data_classifier(with_missing_features= True)),
    (rigged_tree_classifier_method(),*get_test_data_classifier(with_cats=True,with_missing_features= True)),
    (rigged_tree_classifier_method(),*get_test_data_classifier(with_cats=True,with_missing_features= True,with_missing_target=True)),

]

REGRESSOR_CASES = [
    (rigged_tree_regressor_method(),*get_test_data_regressor()),
    (rigged_tree_regressor_method(),*get_test_data_regressor(with_cats=True)),
    (rigged_tree_regressor_method(),*get_test_data_regressor(with_missing_features= True)),
    (rigged_tree_regressor_method(),*get_test_data_regressor(with_cats=True,with_missing_features= True)),
    (rigged_tree_regressor_method(),*get_test_data_regressor(with_cats=True,with_missing_features= True,with_missing_target=True)),
]

NO_MISSING_TARGET = [
    (rigged_tree_classifier_method(),*get_test_data_classifier()),
    (rigged_tree_classifier_method(),*get_test_data_classifier(with_cats=True)),
    (rigged_tree_classifier_method(),*get_test_data_classifier(with_missing_features= True)),
    (rigged_tree_classifier_method(),*get_test_data_classifier(with_cats=True,with_missing_features= True)),
    (rigged_tree_regressor_method(),*get_test_data_regressor()),
    (rigged_tree_regressor_method(),*get_test_data_regressor(with_cats=True)),
    (rigged_tree_regressor_method(),*get_test_data_regressor(with_missing_features= True)),
    (rigged_tree_regressor_method(),*get_test_data_regressor(with_cats=True,with_missing_features= True)),

]
@pytest.mark.parametrize("method,X,y",[*CLASSIFIER_CASES,*REGRESSOR_CASES ])
def test_tree_received_float32_feature_matrix(method,X,y):
    

    method.fit(X,y)
    assert isinstance(method.tree_.fit_X,np.ndarray)
    assert method.tree_.fit_X.dtype == np.dtype(np.float32)

    assert isinstance(method.tree_.apply_X,np.ndarray)
    assert method.tree_.apply_X.dtype == np.dtype(np.float32)

@pytest.mark.parametrize("method,X,y",REGRESSOR_CASES)
def test_regressor_y_is_array_of_float32(method,X,y):
    

    result = method.fit_transform(X,y)
    assert isinstance(method.tree_.fit_y,np.ndarray)
    assert method.tree_.fit_y.dtype == np.dtype(np.float32)

    assert result.dtype ==np.float32

@pytest.mark.parametrize("method,X,y",CLASSIFIER_CASES)
def test_classifier_result_is_array_of_str_dtype(method,X,y):
    
    result = method.fit_transform(X,y)
    assert result.dtype ==str_dtype

@pytest.mark.parametrize("method,X,y",CLASSIFIER_CASES)
def test_output_is_not_a_copy_classifier(method,X,y):

    method = TreeClassifierMethod()
    result = method.fit_transform(X,y)

    assert not np.array_equal(y,result,equal_nan= True)

@pytest.mark.parametrize("method,X,y",[*REGRESSOR_CASES,])
def test_output_is_not_a_copy_regressor(method,X,y):

    method = TreeRegressorMethod()
    result = method.fit_transform(X,y)

    assert not np.array_equal(y,result,equal_nan= True)

def test_output_is_not_a_copy_unique_data():
    X,y = get_test_data_classifier()

    for k in X.keys():
        X[k] = X[k].astype(str_dtype)

    method = TreeClassifierMethod()

    result = method.fit_transform(X,y)

    assert not np.array_equal(y,result)
@pytest.mark.parametrize("method,X,y",NO_MISSING_TARGET)
def test_order_of_input_dict_does_not_change_output(method,X,y):

    method.fit(X,y)

    feature_matrix_fit = copy.copy(method.tree_.fit_X)

    method.transform(X)

    feature_matrix_apply = copy.copy(method.tree_.apply_X)

    assert np.array_equal(feature_matrix_fit,feature_matrix_apply,equal_nan=True)

    X_different_order = {k:X[k] for k in sorted(X.keys(),reverse=True)}

    assert list(X_different_order.keys()) != list(X.keys()), "test invalid, order of features has not been changed"

    method.transform(X_different_order)

    feature_matrix_different_order = copy.copy(method.tree_.apply_X)

    assert np.array_equal(feature_matrix_fit,feature_matrix_different_order,equal_nan=True)


def multiset_frequency_structure_matches(a,b):

    a_not_nan = ~pd.isna(a)
    b_not_nan = ~pd.isna(b)

    hist_a =np.sort(np.unique(a[a_not_nan],return_counts=True,equal_nan=True)[1])
    hist_b =np.sort(np.unique(b[b_not_nan],return_counts=True,equal_nan=True)[1])

    return np.array_equal(hist_a,hist_b) and (pd.isna(a).sum()==pd.isna(b).sum())


@pytest.mark.parametrize("method,X,y",NO_MISSING_TARGET )
def test_fit_preserves_feature_value_frequencies(method,X,y):
    """
    test bijection of X and tree input.
    """
    method.fit(X,y)

    for (i,k) in enumerate(X.keys()):
        assert multiset_frequency_structure_matches(X[k],method.tree_.fit_X[:,i]), f"histogram mismatch on key {k}"

    assert method.tree_.fit_X.shape[0] == list(X.values())[0].shape[0]
    assert method.tree_.fit_X.shape[1] == len(X.values())


@pytest.mark.parametrize("method,X,y",NO_MISSING_TARGET )
def test_no_information_lost_when_apply_tree(method,X,y):
    """
    test bijection of X and tree input.
    """

    method.fit(X,y)

    for (i,k) in enumerate(X.keys()):
        assert multiset_frequency_structure_matches(X[k],method.tree_.apply_X[:,i])

    assert method.tree_.apply_X.shape[0] == list(X.values())[0].shape[0]
    assert method.tree_.apply_X.shape[1] == len(X.values())

@pytest.mark.parametrize("estimator,y", [(TreeRegressorMethod(),np.array([0, 1])),(TreeClassifierMethod(),np.array(["aaa","bbb"],dtype=str_dtype))])
def test_transform_raises_when_feature_names_differ(estimator,y):
    X_fit = {
        "a": np.array([1, 2]),
        "b": np.array([3, 4]),
    }

    estimator.fit(X_fit, y)

    X_bad = {
        "a": np.array([1, 2]),
        "c": np.array([3, 4]),
    }

    with pytest.raises(ValueError):
        estimator.transform(X_bad)


@pytest.mark.parametrize("method,X,y",[
    (TreeClassifierMethod(),*get_test_data_classifier(with_cats=True)),
    (TreeClassifierMethod(),*get_test_data_classifier(with_cats=True,with_missing_features=True)),
                                       ])
def test_classifier_missing_target(method,X,y):
    y = np.array([v if i%3 == 0 else np.nan for (i,v) in enumerate(y)], dtype = str_dtype)

    result = method.fit_transform(X,y)

    assert result.dtype == str_dtype
    assert len(y) == len(result)
    n_missing = pd.isna(result).sum()

    frac_missing_result = n_missing/len(result)
    frac_missing_observed = pd.isna(y).sum()/len(y)

    assert frac_missing_result >(frac_missing_observed-0.1)
    assert frac_missing_result <(frac_missing_observed+0.1)


@pytest.mark.parametrize("method,X,y",[
    (TreeRegressorMethod(),*get_test_data_regressor(with_cats=True)),
    (TreeRegressorMethod(),*get_test_data_regressor(with_cats=True,with_missing_features=True)),
                                       ])
def test_regressor_missing_target(method,X,y):
    y = np.array([v if i%3 == 0 else np.nan for (i,v) in enumerate(y)])

    result = method.fit_transform(X,y)

    assert result.dtype == np.float32
    assert len(y) == len(result)
    n_missing = pd.isna(result).sum()

    frac_missing_result = n_missing/len(result)
    frac_missing_observed = pd.isna(y).sum()/len(y)

    assert frac_missing_result >(frac_missing_observed-0.1)
    assert frac_missing_result <(frac_missing_observed+0.1)


def test_regressor_nondefault_missing_value_predictor():
    method = TreeRegressorMethod(
        missing_handler=MissingValuePredictor(tree = DecisionTreeClassifier(min_samples_leaf=10))
    )
    X = {"a":np.array([1,2])}
    y = np.array([4,3])

    result = method.fit_transform(X,y)

    assert result.shape[0] == y.shape[0]

@pytest.mark.parametrize("seed",[i for i in range(50)])
def test_bug_129_regression_classifier(seed):

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
def test_bug_129_regression_regressor(tree_seed,data_seed):


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
