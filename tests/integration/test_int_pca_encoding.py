import pytest
import pandas as pd
from synthpop.data_processing.encoders import PCAEncoder

def test_pca_encoding_fit_full_data():
    X = pd.Series(["a", "a","b","b","c"],name="input_feature")
    y = pd.Series(["x", "x","y","z","w"])

    encoder = PCAEncoder()

    encoder.fit(X=X,y=y)
    assert len(encoder.mapping_) == 3
    assert len(encoder.mapping_["b"]) == 3

def test_pca_encoding_fit_constant_target():
    X = pd.Series(["a", "a","b","b","c"],name="input_feature")
    y = pd.Series(["x", "x","x","x","x"])

    encoder = PCAEncoder()

    encoder.fit(X=X,y=y)

    assert len(encoder.mapping_) == 3
    assert len(encoder.mapping_["b"]) == 1
    assert encoder.mapping_["a"][0] == pytest.approx(encoder.mapping_["b"][0])

    # The behaviour is different as described in the functional descriptions.
    # Instead of a pure count encoding, it is count encoding + scaling + centering. 

def test_pca_encoding_fit_constant_feature():
    X = pd.Series(["a", "a","a","a","a"],name="input_feature")
    y = pd.Series(["x", "y","y","w","q"])

    encoder = PCAEncoder()

    encoder.fit(X=X,y=y)

    assert len(encoder.mapping_) == 1
    assert len(encoder.mapping_["a"]) == 1
    # The behaviour is different as described in the functional descriptions.
    # Instead of the number of rows, it is encoded by the value 0.0