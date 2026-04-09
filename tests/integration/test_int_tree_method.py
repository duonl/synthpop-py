import copy
import pytest
import pandas as pd
import numpy as np

from synthpop.methods.cart_synth import TreeClassifierMethod, TreeRegressorMethod

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
