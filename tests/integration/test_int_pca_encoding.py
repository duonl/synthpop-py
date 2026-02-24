import pytest
import pandas as pd
from synthpop.data_processing.encoders import PCAEncoder

def test_fit_full_data():
    X = pd.Series(["a", "a","b","b","c"],name="input_feature")
    y = pd.Series(["x", "x","y","z","w"])

    encoder = PCAEncoder()

    encoder.fit(X=X,y=y)
    assert len(encoder.mapping_) == 3
    assert len(encoder.mapping_["b"]) == 3