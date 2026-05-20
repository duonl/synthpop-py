
import pandas as pd
import numpy as np
from synthpop.methods.cart_synth import CartMethod,conventionalize_x_data,conventionalize_1d_array_like
import pytest
str_dtype = np.dtypes.StringDType(na_object=np.nan)

def get_x_input_data():
    #TODO: more input types
    x_dict = {
        "a": [1,2,3],
        "b": ["x","y","z"]
    }

    x_dataframe = pd.DataFrame(x_dict)

    return [x_dict,x_dataframe]

def get_one_var_data_str():
    #TODO: more input types
    
    return [
        pd.Series(["a","b","c"]),
        ["a","b"],
        pd.Series(["a",pd.NA,"c"]),
        np.array(["a",np.nan,"b"],dtype=str_dtype),
    ]

def get_one_var_data_num():
    #TODO: more input types
    
    return [
        pd.Series([1,2,3]),
        [1.1,2.2],
        pd.Series([1,np.nan,2]),
        np.array([1.2,np.nan,3.4],dtype=np.float32),
    ]

@pytest.mark.parametrize("X",get_x_input_data())
def test_conventionalize_output_is_dict(X,mocker):

    clean_column = np.array([1,2,3])

    mocked_conventionalise_y = mocker.patch('synthpop.methods.cart_synth.conventionalize_1d_array_like',return_value=clean_column)

    result =conventionalize_x_data(X)

    assert isinstance(result,dict)

    for (k,v) in result.items():
        assert isinstance(k,str)
        assert np.array_equal(v,clean_column)

    mocked_conventionalise_y.assert_called()

@pytest.mark.parametrize("y",get_one_var_data_str())
def test_conventionalize_y_output_is_np_array(y):

    result =conventionalize_1d_array_like(y)

    assert isinstance(result,np.ndarray)

@pytest.mark.parametrize("y",get_one_var_data_str())
def test_conventionalize_cat_y_output_is_np_array_string_dytpe(y):

    result =conventionalize_1d_array_like(y)

    assert result.dtype == str_dtype

    #str_dtype has exactly one way of representing missing values: np.nan.
    #So if the dtype of result is str_dtype and the missingness pattern aligns, then the conversion was succesfull.
    assert np.array_equal(pd.isna(y),pd.isna(result)), "missings are not handled correctly"

@pytest.mark.parametrize("y",get_one_var_data_num())
def test_conventionalize_cat_y_output_is_np_array_float32(y):

    result =conventionalize_1d_array_like(y)

    assert result.dtype == np.float32

    
#TODO: test normalisation of different missings
#TODO: test numeric dtypes
#TODO: test shape (2D).
#TODO: assert shapes


