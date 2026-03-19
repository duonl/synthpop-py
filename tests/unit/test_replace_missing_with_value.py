from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import NotFittedError
import numpy as np
import pandas as pd
import pytest 
from synthpop.data_processing.missing_value_handling import ReplaceNoneWithValue


def get_test_data():
    missing_indicators = ["missing","N.a.N."]
    return [
#               X_in                y_in                          X_exp               y_exp
    (np.array(["a","b"]),      np.array(["x","y"]),np.array(["a","b"]),np.array(["x","y"]),"N.a.N."),
    *[(np.array(["a","b","a"]),np.array(["x","y",missing_target],dtype=np.object_),np.array(["a","b","a"]),np.array(["x","y",missing_indicator]),missing_indicator) for missing_target in [None,pd.NA,np.nan] for missing_indicator in missing_indicators],
    *[(np.array(["a","b",None],dtype=np.object_),np.array(["x","y","y"]),np.array(["a","b",None],dtype=np.object_),np.array(["x","y","y"]),"N.a.N.") ]
    
]
@pytest.mark.parametrize("X_in,y_in,X_exp,y_exp,missing_indicator", get_test_data())
def test_prepare_data_for_fit_numeric_correctness(X_in,y_in,X_exp,y_exp,missing_indicator):
    replace_nan = ReplaceNoneWithValue(missing_marker = missing_indicator)

    X_res,y_res = replace_nan.prepare_data_for_fit(X_in,y_in)
    assert np.array_equal(X_exp,X_res)
    assert np.array_equal(y_exp,y_res)

@pytest.mark.parametrize("X_in,y_in,X_exp,y_exp,missing_indicator", get_test_data())
def test_prepare_data_for_fit_numeric_correctness_pandas(X_in,y_in,X_exp,y_exp,missing_indicator):
    replace_nan = ReplaceNoneWithValue(missing_marker = missing_indicator)

    X_res,y_res = replace_nan.prepare_data_for_fit(pd.Series(X_in,name="someName_X"),pd.Series(y_in,name="someName_y"))
    assert np.array_equal(X_exp,X_res)
    assert np.array_equal(y_exp,y_res)

    assert isinstance(y_res,pd.Series)
    assert isinstance(X_res,pd.Series)

    assert y_res.name == "someName_y"
    assert X_res.name == "someName_X"

def test_prepare_data_for_fit_error_when_nan_is_a_value():
    X = np.array(["a","b"])
    y = np.array(["x","N.a.N."])

    replace_nan = ReplaceNoneWithValue()

    with pytest.raises(ValueError):
        replace_nan.prepare_data_for_fit(X,y)

def test_post_synth_transform_replaces_nan():
    X = None #should not use X
    y = np.array(["a","b","N.a.N."])

    replace_nan = ReplaceNoneWithValue()

    result = replace_nan.post_synth_transform(X,y)
    assert result[0] == "a"
    assert result[1] == "b"
    assert result[2] is None

def test_post_synth_transform_does_nothing_when_no_nan():
    X = None #should not use X
    y = np.array(["a","b"])

    replace_nan = ReplaceNoneWithValue()

    result = replace_nan.post_synth_transform(X,y)
    assert (result == y).all()