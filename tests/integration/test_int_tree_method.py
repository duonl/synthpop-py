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

    result = tree_method.transform(X)

    assert result.shape[0] ==2


def test_treemethod_regressor_fit_and_transform():
    tree_method = TreeRegressorMethod()

    X = {
        "column1":np.array([1.1,2.2]),
        "column2":np.array([1.4,1.2]),
        "column3":np.array(["a","b"])
        }
    y = np.array([1,2])

    tree_method.fit(X,y)
    assert tree_method.n_features_in_ >= 3

    result = tree_method.transform(X)

    assert result.shape[0] ==2


def get_basic_numeric_data():
    return  np.array([1,1,1,3,2,4,5],dtype=np.float32)
def get_basic_string_data():
    return np.array(["a","a","b","a","c","c","b"],dtype = str_dtype)

def set_value_at_index(a,idx,val):
    result = copy.copy(a)
    result[idx] = val
    return result
def get_x_test_data():

    num_d = get_basic_numeric_data()
    str_d = get_basic_string_data()
    x1 = {"first":num_d,"second":str_d}
    x2 = {"first":num_d,"second":str_d, "third": num_d*1.2,"fourth":np.array([s*3 for s in str_d])}
    x3 = {"first":set_value_at_index(num_d,3,np.nan),"second":set_value_at_index(str_d,4,None), "third":np.array(num_d)*1.2,"fourth":np.array([s*3 for s in str_d])}
    return [x1,x2,x3]

def get_test_data():

    x_data = get_x_test_data()

    str_target = get_basic_string_data()
    int_target = get_basic_numeric_data()

    str_data = [(TreeClassifierMethod(),x,str_target) for x in x_data]
    num_data = [(TreeRegressorMethod(),x,int_target) for x in x_data]
    return str_data +num_data

@pytest.mark.parametrize("method,X,y",get_test_data())
def test_general_usage(method,X,y):

    result = method.fit_transform(X,y)

    #assert result.dtype == y.dtype
    assert X["first"].shape[0] == len(y)
    assert not np.array_equal(y,result)

    y_pd = pd.Series(y,name="target_variable")

    method = method.set_output(transform="pandas")

    result2 = method.fit_transform(X,y_pd)
    assert isinstance(result2,pd.DataFrame)
    assert "target_variable" in result2


def make_data_missing(X):

    for ik, k in enumerate(X.keys()):


        X[k] = np.array([v if (i )% (len(X.keys())-ik) !=1 else np.nan for i,v in enumerate(X[k])],dtype=np.dtypes.StringDType(na_object=np.nan))

    return X


def get_test_data_classifier(with_cats = False,with_missing_features=False):
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


    y = np.array([string.ascii_lowercase[i] for i in y])
    return (X,y)


def get_test_data_regressor(with_cats=False,with_missing_features=False):
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
                                       ])
def test_input_to_tree_is_array_of_float32(method,X,y):

    method.fit(X,y)
    assert isinstance(method.tree_.fit_X,np.ndarray)
    assert method.tree_.fit_X.dtype == np.dtype(np.float32)

    assert isinstance(method.tree_.apply_X,np.ndarray)
    assert method.tree_.apply_X.dtype == np.dtype(np.float32)

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
        assert histogram_matches(X[k],method.tree_.fit_X[:,i])

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

@pytest.mark.parametrize("estimator", [TreeRegressorMethod(),TreeClassifierMethod()])
def test_transform_raises_when_feature_names_differ(estimator):
    X_fit = {
        "a": np.array([1, 2]),
        "b": np.array([3, 4]),
    }

    y = np.array([0, 1])

    estimator.fit(X_fit, y)

    X_bad = {
        "a": np.array([1, 2]),
        "c": np.array([3, 4]),
    }

    with pytest.raises(ValueError):
        estimator.transform(X_bad)
