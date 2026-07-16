import pytest
import pandas as pd
import numpy as np

from synthpop.data_processing.missing_value_handling import ReplaceMissingWithValue

str_dtype = np.dtypes.StringDType(na_object=np.nan)

def test_round_trip():
    replace_missing = ReplaceMissingWithValue()
    X = {"a":np.array(["a","b","c","c"], dtype=str_dtype)}
    y = np.array(["x","y",np.nan,"z"], dtype=str_dtype)

    y_res = replace_missing.post_synth_transform(*replace_missing.prepare_data_for_fit(X,y))

    assert (y_res[0:2] == np.array(["x","y"],dtype= np.object_)).all()
    assert pd.isna(y_res[2])
    assert y_res[3] =="z"
