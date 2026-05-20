"""
Requirements:

- input can be "dirty" pandas dataframes/series
- input numpy arrays supported?
- implements set_output api
- passes only (dicts of) numpy arrays with correct dtypes to its dependencies
- selects correct tree method

- tune_cart method


Approach:
- one function that "cleans" data
- assert that CartMethod uses that function and delegates correctly.
"""
import pandas as pd
import numpy as np
from synthpop.methods.cart_synth import CartMethod
import pytest
str_dtype = np.dtypes.StringDType(na_object=np.nan)

class StubTreeMethod():
    def __init__(self,transform_result):
        self.transform_result = transform_result

    def fit(self,X,y):
        self.fit_x = X
        self.fit_y = y
        return self

    def transform(self,X):
        self.transform_x = X
        return self.transform_result
    
@pytest.mark.parametrize("y_clean,cat_target",[(np.array([1,2,3],dtype=np.float32),False),
                                               (np.array(["a","2","3"],dtype=str_dtype),True)
                                               ])
def test_cartmethod_fit_dataflow_classifier(y_clean,cat_target,mocker):

    exp_result = np.array([1,2,3])
    tree_method = StubTreeMethod(transform_result=exp_result)

    clean_X = {}
    clean_y = []

    mocked_conventionalise_x = mocker.patch('synthpop.methods.cart_synth.conventionalize_x_data',return_value=clean_X)
    mocked_conventionalise_y = mocker.patch('synthpop.methods.cart_synth.conventionalize_1d_array_like',return_value=y_clean)

    X = {
        "a":[1,2,3,4]
    }
    y = [1,1,2,2]

    if cat_target:
        cart = CartMethod(classifier=tree_method)
    else:
        cart = CartMethod(regressor=tree_method)

    fit_res = cart.fit(X,y)

    assert fit_res is cart

    mocked_conventionalise_x.assert_called_with(X)
    mocked_conventionalise_y.assert_called_with(y)

    assert tree_method.fit_x == clean_X
    assert np.array_equal(tree_method.fit_y, y_clean)


#TODO: test get_feature_names_out.
#TODO: test cloning of tree methods
#TODO: implement cloning of CartMethod 
