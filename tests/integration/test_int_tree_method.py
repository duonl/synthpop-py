import copy
import pytest
import pandas as pd
import numpy as np
import string

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from synthpop.data_processing.encoders import PCAEncoder
from synthpop.methods.cart_synth import TreeClassifierMethod, TreeRegressorMethod
from synthpop.utils import str_dtype
from sklearn.datasets import make_classification, make_regression

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

    for ik, k in enumerate(X.keys()):
        if pd.api.types.is_numeric_dtype(X[k].dtype):
            X[k]=np.array([v if (i )% (len(X.keys())-ik) !=1 else np.nan for i,v in enumerate(X[k])])
        else:
            X[k] = np.array([v if (i )% (len(X.keys())-ik) !=1 else np.nan for i,v in enumerate(X[k])],dtype=np.dtypes.StringDType(na_object=np.nan))

    return X


def get_test_data_classifier(with_cats = False,with_missing_features=False,with_missing_target=False):
    X,y = make_classification(n_classes=10,n_informative=11)
    
    X = {i:X[:,i] for i in range(X.shape[1])}


    idx_cats = [3,4,6]
    if with_cats:
        for idx in idx_cats:
            x = (X[idx]*10).astype(int)
            x_i = [f %5 for f in x]
            X[idx] = np.array([string.ascii_lowercase[i] for i in x_i],dtype=str_dtype)

    if with_missing_features:
        X = make_data_missing(X)

    if with_missing_target:
        y = np.array([string.ascii_lowercase[i] if i%5 !=0 else np.nan for i in y],dtype=str_dtype)
    else:
        y = np.array([string.ascii_lowercase[i] for i in y],dtype=str_dtype)
    return (X,y)


def get_test_data_regressor(with_cats=False,with_missing_features=False,with_missing_target=False):
    X,y = make_regression()
    X = {i:X[:,i] for i in range(X.shape[1])}

    idx_cats = [3,4,6]
    if with_cats:
        for idx in idx_cats:
            x = (X[idx]*10).astype(int)
            x_i = [f %26 for f in x]
            X[idx] = np.array([string.ascii_lowercase[i] for i in x_i],dtype = np.dtypes.StringDType(na_object=np.nan))

    if with_missing_features:
        X = make_data_missing(X)

    if with_missing_target:
        y = np.array([v if i%5 !=0 else np.nan for i,v in enumerate(y)])

    return (X,y)

    
    
def spy_tree_wrapper(obj):
    class spy_wrapper(obj.__class__):
        def fit(self,X,y):
            self.fit_X = X
            self.fit_y = y
            return super().fit(X,y)
        
        def apply(self,X):
            self.apply_X = X
            return super().apply(X)
        
    obj.__class__ = spy_wrapper

    return obj
        

def rigged_tree_classifier_method(pca_components = 1):
    tree = spy_tree_wrapper(DecisionTreeClassifier())
    return TreeClassifierMethod(tree=tree,encoder=PCAEncoder(pca_transform= PCA(n_components=pca_components)))

def rigged_tree_regressor_method():
    tree = spy_tree_wrapper(DecisionTreeRegressor())
    return TreeRegressorMethod(tree=tree)

    
@pytest.mark.parametrize("method,X,y",[
    (rigged_tree_classifier_method(),*get_test_data_classifier()),
    (rigged_tree_classifier_method(),*get_test_data_classifier(with_cats=True)),
    (rigged_tree_regressor_method(),*get_test_data_regressor()),
    (rigged_tree_regressor_method(),*get_test_data_regressor(with_cats=True)),
    (rigged_tree_classifier_method(),*get_test_data_classifier(with_missing_features= True)),
    (rigged_tree_classifier_method(),*get_test_data_classifier(with_cats=True,with_missing_features= True)),
    (rigged_tree_regressor_method(),*get_test_data_regressor(with_missing_features= True)),
    (rigged_tree_regressor_method(),*get_test_data_regressor(with_cats=True,with_missing_features= True)),
    (rigged_tree_regressor_method(),*get_test_data_regressor(with_cats=True,with_missing_features= True,with_missing_target=True)),
    (rigged_tree_classifier_method(),*get_test_data_classifier(with_cats=True,with_missing_features= True,with_missing_target=True)),
                                       ])
def test_input_to_tree_is_array_of_float32(method,X,y):

    method.fit(X,y)
    assert isinstance(method.tree_.fit_X,np.ndarray)
    assert method.tree_.fit_X.dtype == np.dtype(np.float32)

    assert isinstance(method.tree_.apply_X,np.ndarray)
    assert method.tree_.apply_X.dtype == np.dtype(np.float32)


@pytest.mark.parametrize("method,X,y",[
    (TreeClassifierMethod(),*get_test_data_classifier()),
    (TreeClassifierMethod(),*get_test_data_classifier(with_cats=True)),
    (TreeRegressorMethod(),*get_test_data_regressor()),
    (TreeRegressorMethod(),*get_test_data_regressor(with_cats=True)),
    (TreeClassifierMethod(),*get_test_data_classifier(with_missing_features= True)),
    (TreeClassifierMethod(),*get_test_data_classifier(with_cats=True,with_missing_features= True)),
    (TreeRegressorMethod(),*get_test_data_regressor(with_missing_features= True)),
    (TreeRegressorMethod(),*get_test_data_regressor(with_cats=True,with_missing_features= True)),
    (TreeRegressorMethod(),*get_test_data_regressor(with_cats=True,with_missing_features= True,with_missing_target=True)),
    (TreeClassifierMethod(),*get_test_data_classifier(with_cats=True,with_missing_features= True,with_missing_target=True)),
                                       ])
def test_output_is_not_a_copy(method,X,y):

    result = method.fit_transform(X,y)

    assert not np.array_equal(y,result,equal_nan= True)

def test_output_is_not_a_copy_unique_data():
    X,y = get_test_data_classifier()

    for k in X.keys():
        X[k] = X[k].astype(str_dtype)

    method = TreeClassifierMethod()

    result = method.fit_transform(X,y)

    assert not np.array_equal(y,result)
@pytest.mark.parametrize("method,X,y",[
    (rigged_tree_classifier_method(),*get_test_data_classifier()),
    (rigged_tree_classifier_method(),*get_test_data_classifier(with_cats=True)),
    (rigged_tree_regressor_method(),*get_test_data_regressor()),
    (rigged_tree_regressor_method(),*get_test_data_regressor(with_cats=True)),
    (rigged_tree_classifier_method(),*get_test_data_classifier(with_missing_features= True)),
    (rigged_tree_classifier_method(),*get_test_data_classifier(with_cats=True,with_missing_features= True)),
    (rigged_tree_regressor_method(),*get_test_data_regressor(with_missing_features= True)),
    (rigged_tree_regressor_method(),*get_test_data_regressor(with_cats=True,with_missing_features= True)),
                                       ])
def test_order_of_input_dict_does_not_change_output(method,X,y):

    method.fit(X,y)

    feature_matrix_fit = method.tree_.fit_X

    method.transform(X)

    feature_matrix_apply = method.tree_.apply_X

    assert np.array_equal(feature_matrix_fit,feature_matrix_apply,equal_nan=True)

    X_different_order = {k:X[k] for k in sorted(X.keys(),reverse=True)}

    assert list(X_different_order.keys()) != list(X.keys()), "test invalid, order of features has not been changed"

    method.transform(X_different_order)

    feature_matrix_different_order = method.tree_.apply_X

    assert np.array_equal(feature_matrix_fit,feature_matrix_different_order,equal_nan=True)


def histogram_matches(a,b):
    hist_a =np.sort(np.unique(a,return_counts=True)[1])
    hist_b =np.sort(np.unique(b,return_counts=True)[1])

    return np.array_equal(hist_a,hist_b)


@pytest.mark.parametrize("method,X,y",[
    (rigged_tree_classifier_method(),*get_test_data_classifier()),
    (rigged_tree_classifier_method(),*get_test_data_classifier(with_cats=True)),
    (rigged_tree_regressor_method(),*get_test_data_regressor()),
    (rigged_tree_regressor_method(),*get_test_data_regressor(with_cats=True)),
                                       ])
def test_no_information_lost_when_fitting_tree(method,X,y):
    """
    test bijection of X and tree input.
    """
    method.fit(X,y)

    for (i,k) in enumerate(X.keys()):
        assert histogram_matches(X[k],method.tree_.fit_X[:,i]), f"histogram mismatch on key {k}"

    assert method.tree_.fit_X.shape[0] == list(X.values())[0].shape[0]
    assert method.tree_.fit_X.shape[1] == len(X.values())


@pytest.mark.parametrize("method,X,y",[
    (rigged_tree_classifier_method(),*get_test_data_classifier()),
    (rigged_tree_classifier_method(),*get_test_data_classifier(with_cats=True)),
    (rigged_tree_regressor_method(),*get_test_data_regressor()),
    (rigged_tree_regressor_method(),*get_test_data_regressor(with_cats=True)),
                                       ])
def test_no_information_lost_when_apply_tree(method,X,y):
    """
    test bijection of X and tree input.
    """

    method.fit(X,y)

    for (i,k) in enumerate(X.keys()):
        assert histogram_matches(X[k],method.tree_.apply_X[:,i])

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

