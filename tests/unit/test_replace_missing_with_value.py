from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import NotFittedError
import numpy as np
import pandas as pd
import pytest 
from synthpop.data_processing.missing_value_handling import ReplaceNoneWithValue



@pytest.mark.parametrize("X_in,y_in,X_exp,y_exp", [
#               X_in                y_in                          X_exp               y_exp
    (np.array(["a","b"]),      np.array(["x","y"]),np.array(["a","b"]),np.array(["x","y"])),
    *[(np.array(["a","b","a"]),np.array(["x","y",missing_target],dtype=np.object_),np.array(["a","b","a"]),np.array(["x","y","N.a.N."])) for missing_target in [None,pd.NA,np.nan]],
    *[(np.array(["a","b",None],dtype=np.object_),np.array(["x","y","y"]),np.array(["a","b",None],dtype=np.object_),np.array(["x","y","y"]))],
])
def test_prepare_data_for_fit_numeric_correctness(X_in,y_in,X_exp,y_exp):
    replace_nan = ReplaceNoneWithValue()

    X_res,y_res = replace_nan.prepare_data_for_fit(X_in,y_in)
    assert np.array_equal(X_exp,X_res)
    assert np.array_equal(y_exp,y_res)

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