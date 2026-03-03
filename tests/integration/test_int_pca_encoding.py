import pytest
import pandas as pd
import numpy as np
from synthpop.data_processing.encoders import PCAEncoder
from sklearn.utils.estimator_checks import parametrize_with_checks,check_estimator


def test_pca_encoding_fit_full_data():
    X = np.array(["a", "a","b","b","c"])
    y = np.array(["x", "x","y","z","w"])

    encoder = PCAEncoder()

    encoder.fit(X=X,y=y)
    assert len(encoder.mapping_) == 3
    assert len(encoder.mapping_["b"]) == 3

def test_pca_encoding_fit_constant_target():
    X = np.array(["a", "a","b","b","c"])
    y = np.array(["x", "x","x","x","x"])

    encoder = PCAEncoder()

    encoder.fit(X=X,y=y)

    assert len(encoder.mapping_) == 3
    assert len(encoder.mapping_["b"]) == 1
    assert encoder.mapping_["a"][0] == pytest.approx(encoder.mapping_["b"][0])

    # The behaviour is different as described in the functional descriptions.
    # Instead of a pure count encoding, it is count encoding + scaling + centring. 

def test_pca_encoding_fit_constant_feature():
    X = pd.Series(["a", "a","a","a","a"],name="input_feature")
    y = pd.Series(["x", "y","y","w","q"])

    encoder = PCAEncoder()

    encoder.fit(X=X,y=y)

    assert len(encoder.mapping_) == 1
    assert len(encoder.mapping_["a"]) == 1
    # The behaviour is different as described in the functional descriptions.
    # Instead of the number of rows, it is encoded by the value 0.0

def test_pca_encoding_fit_transform_regular_feature_output_api():
    X = pd.Series(["a", "a","b","b","c"],name="input_feature")
    y = pd.Series(["x", "x","y","z","w"])

    encoder = PCAEncoder().set_output(transform="pandas")

    result = encoder.fit_transform(X=X,y=y)
    assert result.shape[0]== 5
    assert isinstance(result,pd.DataFrame)
    assert result.columns.equals(pd.Index(["input_feature_pca0","input_feature_pca1","input_feature_pca2"]))
