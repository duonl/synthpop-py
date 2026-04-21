import copy
import pytest
import pandas as pd
import numpy as np
import string

from sklearn.tree import DecisionTreeClassifier

from synthpop.methods.cart_synth import TreeClassifierMethod, TreeRegressorMethod
from sklearn.datasets import make_classification

def test_treemethod_classifier_fit_and_transform():
    tree_method = TreeClassifierMethod()

    X = {
        "column1":np.array([1.1,2.2]),
        "column2":np.array([1.4,1.2]),
        "column3":np.array(["a","b"])
        }
    y = np.array(["x","y"])

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
    return np.array(["a","a","b","a","c","c","b"])

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

    assert result.dtype == y.dtype
    assert X["first"].shape[0] == len(y)
    assert not np.array_equal(y,result)

    y_pd = pd.Series(y,name="target_variable")

    method = method.set_output(transform="pandas")

    result2 = method.fit_transform(X,y_pd)
    assert isinstance(result2,pd.DataFrame)
    assert "target_variable" in result2


def get_test_data_classifier():
    X,y = make_classification(n_classes=10,n_informative=11)
    y = np.array([string.ascii_lowercase[i] for i in y])
    X = {i:X[:,i] for i in range(X.shape[1])}
    return (X,y)
    
def spy_fit_transfrom(obj):
    class spy_fit_wrapper(obj.__class__):
        def fit(self,X,y):
            self.fit_X = X
            self.fit_y = y
            return super().fit(X,y)
        
        def transform(self,X):
            self.transform_X = X
            return super().transform(X)
        
    obj.__class__ = spy_fit_wrapper

    return obj
        
def spy_decisionTreeClassifier():
    tree = DecisionTreeClassifier()
    return spy_fit_transfrom(tree)

def rigged_tree_classifier_method():
    tree = spy_fit_transfrom(DecisionTreeClassifier())
    return TreeClassifierMethod(tree=tree)

    
@pytest.mark.parametrize("method,X,y",[(rigged_tree_classifier_method(),*get_test_data_classifier())])
def test_input_to_tree_is_array_of_float32(method,X,y):

    method.fit(X,y)
    assert isinstance(method.tree_.fit_X,np.ndarray)
    assert method.tree_.fit_X.dtype == np.dtype(np.float32)

def test_no_information_lost():
    """
    test bijection of X and tree input.
    """
    assert False,"test not made yet"

def test_TreeRegressorMethod_shape():
    """
    test if the input to the decision trees has the right shape
    """
    assert False,"test not made yet"

def test_TreeClassifierMethod_shape():
    """
    test if the input to the decision trees has the right shape
    """
    assert False,"test not made yet"

def test_synthetic_nowhere_missing_when_observed_nowhere_missing():
    """
    Test if the synthetic column is nowhere missing when the original is nowhere missing.
    """
    assert False,"test not made yet"

def test_something_no_missing():
    #test should contain:
    # int features
    # float64 features
    # float32 features
    # string features

    #aspects of tests:
    # data (correlations, shape,count )
    # types (numpy, pd.categorical, float32/64, int, string, object, pd.DataFrame)
    # missing
    # consistency (order of dict/columns, reproducible)
    # sklearn compatibility

    # invariants:
    # the y of the decision tree can never be missing.
    # the input to the decision tree when transforming should be invariant under the order of the features.
    # equivalent datatypes: {string,pd.Categorical}, {int, float32,float64}
    # for any valid X and y, the input to decision tree should be a np.array of dtype float32. (fit and transform)
    # for any valid X with 1-column encoding and y nowhere missing, the shape of X and the input of the decision tree should be the same.

    # co variants:
    # for nowhere missing numeric targets: the shape of the features should be the shape of the array given to decision tree.
    # non-degenerate encoding: bijection X <-> input decision tree

    # for all tests: the outcome should be invariant under equivalent datatypes

    pass