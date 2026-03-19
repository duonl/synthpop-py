import pytest
import pandas as pd
import numpy as np
from synthpop.data_processing.missing_value_handling import ReplaceNoneWithValue

def test_round_trip():
    replace_missing = ReplaceNoneWithValue()
    X = np.array(["a","b","c","c"])
    y = np.array(["x","y",None,"z"])

    y_res = replace_missing.post_synth_transform(*replace_missing.prepare_data_for_fit(X,y))

    assert (y_res[0:2] == np.array(["x","y"],dtype= np.object_)).all()
    assert y_res[2] is None
    assert y_res[3] =="z"