import pandas as pd
import numpy as np
import pytest
from sklearn.base import BaseEstimator,TransformerMixin

from synthpop.data_processing.encoders import PCAEncoder

class TransformStub(TransformerMixin, BaseEstimator):

    def __init__(self, fit_return_value=None,transform_return_value=None):
        self.transform_value = transform_return_value
        self.fit_return_value = fit_return_value

    def fit(self,X,y):
        self.fit_X_ = X
        self.fit_y_ = y
        return self.fit_return_value
    
    def transform(self,X):
        self.transform_X_ = X
        return self.transform_value
    
    def fit_transform(self, X, y = None, **fit_params):
        self.fit_X_ = X
        self.transform_X_ = X
        self.fit_y_ = y


        return self.transform_value

def test_pca_fit_when_given_full_features_and_targets():
    #Given features and targets that are nowhere missing and an unfitted pcaEncoder
    X = pd.Series(["a", "a","b","b"])
    y = pd.Series(["x", "x","y","z"])


    pca_return_value = pd.DataFrame(
        [   #pc1, pc2, pc3
            [1.2, 3.4, 5],#a
            [11.22,33.44,6]#b
        ],columns=["pc1","pc2","pc3"]
        )
    stub_pca_transform = TransformStub(transform_return_value=pca_return_value)
    encoder = PCAEncoder(pca_transform = stub_pca_transform )

    result = encoder.fit(X=X,y=y)

    # Then the return value is self
    assert result is encoder
    # The contingency table has been made and passed to the PCA transform

    expected_contingency_table = pd.DataFrame(
        [#   x  y  z 
            [2, 0, 0],# a
            [0, 1, 1] #b
        ]
        ,columns= ["x","y","z"]
        ,index=["a","b"]
        )
    
    
    assert np.array_equal(stub_pca_transform.transform_X_.columns,expected_contingency_table.columns), f"columns do not match. Expected:{str(expected_contingency_table.columns)}, actual: {str(stub_pca_transform.transform_X_.columns)}"
    assert stub_pca_transform.transform_X_.equals(expected_contingency_table), "contingency table not calculated correctly when no missing values in target or feature."
    assert result.mapping_["a"]== pd.DataFrame
    

