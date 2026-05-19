
import pandas as pd
import numpy as np
from synthpop.methods.cart_synth import CartMethod,conventionalize_x_data,conventionalize_1d_array_like
import pytest

def get_x_input_data():
    #TODO: more input types
    x_dict = {
        "a": [1,2,3],
        "b": ["x","y","z"]
    }

    x_dataframe = pd.DataFrame(x_dict)

    return [x_dict,x_dataframe]

def get_one_var_data():
    #TODO: more input types
    y1 = pd.Series(["a","b","c"])
    return [y1]


@pytest.mark.parametrize("X",get_x_input_data())
def test_conventionalize_output_is_dict(X):

    result =conventionalize_x_data(X)

    assert isinstance(result,dict)

    for (k,v) in result.items():
        assert isinstance(k,str)
        assert isinstance(v,np.ndarray)

@pytest.mark.parametrize("y",get_one_var_data())
def test_conventionalize_output_is_dict(y):

    result =conventionalize_1d_array_like(y)

    assert isinstance(result,np.ndarray)

#TODO: test normalisation of different missings
#TODO: assert shapes


