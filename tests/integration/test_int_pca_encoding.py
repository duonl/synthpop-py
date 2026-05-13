import pytest
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from synthpop.data_processing.encoders import PCAEncoder
from sklearn import config_context

str_dtype = np.dtypes.StringDType(na_object=np.nan)

def test_pca_encoding_fit_full_data():
    X = np.array(["a", "a","b","b","c","d"])
    y = np.array(["x", "x","y","z","w",np.nan],dtype=str_dtype)

    encoder = PCAEncoder()

    encoder.fit(X=X,y=y)
    assert len(encoder.mapping_) == 4
    assert len(encoder.mapping_["b"]) == 3

def test_pca_encoding_fit_constant_target():
    X = np.array(["a", "a","b","b","c"])
    y = np.array(["x", "x","x","x","x"])

    encoder = PCAEncoder()

    encoder.fit(X=X,y=y)

    # Contingency table:
    #   |x|
    #   |-|
    # a |2|
    # b |2|
    # c |1|

    # centring (mean = 5/3 = 1 2/3):
    #   |x|
    #   |-|
    # a |1/3|
    # b |1/3|
    # c |-2/3|

    # The variance is ((1/3)^2 + (1/3)^2 + (-2/3)^2)/3 =
    #                 ( 1/9 + 1/9 + 4/9)/3 =
    #                 (6/9)/3 = 2/9
    # sigma = (1/3)sqrt(2)

    # scaling:
    #   |x|
    #   |-|
    # a |1/sqrt(2)|   (1/2)sqrt(2)
    # b |1/sqrt(2)|   (1/2)sqrt(2)
    # c |-sqrt(2)|    -sqrt(2)

    sqrt2 = np.sqrt(2)
    assert len(encoder.mapping_) == 3
    assert encoder.mapping_["a"] == pytest.approx([1/sqrt2])#floating point errors occur.
    assert encoder.mapping_["b"] == pytest.approx( [1/sqrt2])
    assert encoder.mapping_["c"] == pytest.approx([-1*sqrt2])


def test_pca_encoding_fit_constant_feature():
    X = np.array(["a", "a","a","a","a"])
    y = np.array(["x", "y","y","w","q"])


    encoder = PCAEncoder()

    encoder.fit(X=X,y=y)

    assert len(encoder.mapping_) == 1
    assert len(encoder.mapping_["a"]) == 1
    assert np.array_equal(encoder.mapping_["a"],np.array([0]))
    # The behaviour is different as described in the functional descriptions.
    # Instead of the number of rows, it is encoded by the value 0.0


def test_pca_encoding_changed_number_of_components():
    X = np.array(["a", "a","b","b","c"])
    y = np.array(["x", "x","y","z","w"],dtype=str_dtype)

    encoder = PCAEncoder(pca_transform= PCA(n_components=2))

    result = encoder.fit_transform(X=X,y=y)
    assert result.shape[0]== 5


def test_pca_encoding_not_broken_by_setoutput_api():

    X = np.array(["a", "a","b","b","c"],dtype=str_dtype)
    y = np.array(["x", "x","y","z","w"],dtype=str_dtype)

    encoder = PCAEncoder(pca_transform= PCA(n_components=2))

    before = encoder.fit_transform(X,y)
    
    #Using the setoutput api here via a context manager could have effect on sklearn.PCA
    with config_context(transform_output="pandas"):
        after = encoder.fit_transform(X,y)

    assert np.array_equal(before,after), "Scikit-learn's set_output API breaks PCA encoding"
    
