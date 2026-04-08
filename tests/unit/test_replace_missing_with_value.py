import numpy as np
import pandas as pd
import pytest
from synthpop.data_processing.missing_value_handling import ReplaceNoneWithValue


def get_test_data():
    missing_indicators = ["missing","N.a.N."]

    test_data_np_arrays = [
#               X_in                y_in                          X_exp               y_exp
    (np.array(["a","b"]),      np.array(["x","y"]),np.array(["x","y"]),"N.a.N."),
    *[(np.array(["a","b","a"]),np.array(["x","y",missing_target],dtype=np.object_),np.array(["x","y",missing_indicator],dtype=np.str_),missing_indicator) for missing_target in [None,pd.NA,np.nan] for missing_indicator in missing_indicators],
    *[(np.array(["a","b",None],dtype=np.object_),np.array(["x","y","y"]),np.array(["x","y","y"]),"N.a.N.") ],
    *[(np.array(["a","b","a"]),np.array(["x","y",missing_indicator],dtype=np.str_),np.array(["x","y",missing_indicator],dtype=np.str_),missing_indicator) for missing_indicator in missing_indicators],
    *[(np.array(["a","b","a"]),np.array([missing_target,missing_target,missing_target],dtype=np.object_),np.array([missing_indicator,missing_indicator,missing_indicator],dtype=np.str_),missing_indicator) for missing_target in [None,pd.NA,np.nan] for missing_indicator in missing_indicators]
    ]

    test_data_lists = [(X_in.tolist(),y_in.tolist(),y_exp,missing_indicator) for X_in,y_in,y_exp,missing_indicator in test_data_np_arrays]
    return test_data_np_arrays + test_data_lists
    

@pytest.mark.parametrize("X_in,y_in,y_exp,missing_indicator", get_test_data())
def test_prepare_data_for_fit_numeric_correctness(X_in,y_in,y_exp,missing_indicator):
    replace_nan = ReplaceNoneWithValue(missing_marker = missing_indicator)

    X_res,y_res = replace_nan.prepare_data_for_fit(X_in,y_in)
    assert X_res is X_in
    #assert np.array_equal(X_exp,X_res)
    assert np.array_equal(y_exp,y_res)
    assert y_exp.dtype == y_res.dtype

def test_prepare_data_for_fit_does_not_change_arguments():
    x_orig = np.array(["a","b"])
    y = np.array(["x",None])
    replace_nan = ReplaceNoneWithValue()

    x_res,y_res = replace_nan.prepare_data_for_fit(x_orig,y)

    assert y[1] is None
    assert y_res[1] == "N.a.N."

@pytest.mark.parametrize("X_in,y_in,y_exp,missing_indicator", get_test_data())
def test_prepare_data_for_fit_numeric_correctness_pandas(X_in,y_in,y_exp,missing_indicator):
    replace_nan = ReplaceNoneWithValue(missing_marker = missing_indicator)

    x_pd = pd.Series(X_in,name="someName_X")
    y_pd = pd.Series(y_in,name="someName_y")
    X_res,y_res = replace_nan.prepare_data_for_fit(x_pd,y_pd)
    assert X_res is x_pd
    assert np.array_equal(y_exp,y_res)

    assert isinstance(y_res,pd.Series)
    assert isinstance(X_res,pd.Series)

    assert y_res.name == "someName_y"
    assert X_res.name == "someName_X"

    assert (y_res.index == y_pd.index).all()

@pytest.mark.parametrize("missing",[None,pd.NA,np.nan])
def test_prepare_data_for_fit_error_when_nan_is_a_value(missing):
    X = np.array(["a","b"])
    y = np.array(["x","N.a.N.",missing],dtype=np.object_)

    replace_nan = ReplaceNoneWithValue(missing_marker="N.a.N.")

    with pytest.raises(ValueError):
        replace_nan.prepare_data_for_fit(X,y)

    with pytest.raises(ValueError):
        replace_nan.prepare_data_for_fit(X,pd.Series(y))

    with pytest.raises(ValueError):
        replace_nan.prepare_data_for_fit(X,y.tolist())

@pytest.mark.parametrize("empty_data",
                         [(np.array([])),
                          (pd.Series([])),
                          ([])
                          ])
def test_prepare_data_for_fit_empty(empty_data):
    replace_nan = ReplaceNoneWithValue()
    x_res,y_res = replace_nan.prepare_data_for_fit(np.array(["s"]),empty_data)
    assert len(y_res) == 0



def get_post_synth_test_data():
    x_values = [np.array([]),np.array([1,2,3]),pd.Series(["a","b"])]
    missing_markers = ["N.a.N.","missing marker"]
    #               y_in                            y_exp                              missing_marker
    y_values_np_array = [(np.array(["a",missing,"b"]),np.array(["a",None,"b"],dtype=np.object_),missing) for missing in missing_markers] + \
    [(np.array(["a","c","b",missing]),np.array(["a","c","b",None],dtype=np.object_),missing) for missing in missing_markers] + \
    [(np.array(["a","c","b",not_missing]),np.array(["a","c","b",not_missing],dtype=np.str_),None) for not_missing in ["not missing","missing","Nan","None"]]

    y_values_list = [(y_in.tolist(),y_exp,missmarker) for y_in,y_exp,missmarker in y_values_np_array]
    return [(x_val,*y_val) for x_val in x_values for y_val in y_values_np_array +y_values_list ]

@pytest.mark.parametrize("x,y_in,y_exp,missing_marker", get_post_synth_test_data())
def test_post_synth_transform_correct_on_data(x,y_in,y_exp,missing_marker):
    if missing_marker is None:
        transform = ReplaceNoneWithValue()
    else:
        transform = ReplaceNoneWithValue(missing_marker=missing_marker)

    y_res = transform.post_synth_transform(x,y_in)
    assert np.array_equal(y_res,y_exp)
    assert y_res.dtype == y_exp.dtype



@pytest.mark.parametrize("x,y_in,y_exp,missing_marker", get_post_synth_test_data())
def test_post_synth_transform_correct_on_data_pandas(x,y_in,y_exp,missing_marker):
    if missing_marker is None:
        transform = ReplaceNoneWithValue()
    else:
        transform = ReplaceNoneWithValue(missing_marker=missing_marker)

    y_res = transform.post_synth_transform(x,pd.Series(y_in,name="name_y"))
    assert isinstance(y_res,pd.Series)
    assert np.array_equal(y_res.to_numpy(),y_exp)
    assert y_res.name == "name_y"

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

@pytest.mark.parametrize("empty_data",
                         [(np.array([])),
                          (pd.Series([])),
                          ([])
                          ])
def test_post_synth_transform_empty(empty_data):
    replace_nan = ReplaceNoneWithValue()
    y_res = replace_nan.post_synth_transform(np.array(["s"]),empty_data)
    assert len(y_res) == 0

# ----- clonability tests -----
def test_clone_works_and_fitted_does_not_preserve_state():
    rpnwv = ReplaceNoneWithValue(missing_marker="N.a.N.")
    rpnwv.prepare_data_for_fit(X=np.array(["a","b","c","c"]), y=np.array(["x","y",None,"z"]))

    cloned = rpnwv.clone()

    # Does not have learned attributes
    assert hasattr(cloned, "missing_replacement")
    assert hasattr(rpnwv, "missing_replacement")