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

#TODO: assert data types
#TODO: test empty mapping
#TODO: test constant feature and constant target
#TODO: validation in fit and transform. (check feature names, check is fitted)
#TODO: test with different index

def test_pca_fit_when_given_full_features_and_targets():
    #Given features and targets that are nowhere missing and an unfitted pcaEncoder
    X = pd.Series(["a", "a","b","b"],name="input_feature")
    y = pd.Series(["x", "x","y","z"])


    pca_return_value = pd.DataFrame(
        [   #pc1, pc2, pc3
            [1.2, 3.4, 5],#a
            [11.22,33.44,6]#b
        ],columns=["pca0","pca1","pca2"],index=["a","b"]
        )
    stub_pca_transform = TransformStub(transform_return_value=pca_return_value)
    encoder = PCAEncoder(_pca_transform = stub_pca_transform )

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
    assert np.array_equal(result.mapping_["a"],[1.2, 3.4, 5]), "first row of map incorrect"
    assert np.array_equal(result.mapping_["b"],[11.22,33.44,6]), "second row of map incorrect"

    assert result.n_features_in_ == 1
    assert result.feature_names_in_ == ["input_feature"]
    assert result.n_features_out_ == 3
    
def test_pca_fit_when_target_is_missing():

    X = pd.Series(["a", "a","b","b","c","c"],name="input_feature")
    y = pd.Series(["x", None,"y","z",None,None])#Target is always missing for X=c, but not always missing for X=a

    expected_contingency_table = pd.DataFrame(
        [#   x  y  z Note that the expected behaviour is that C is not in the contingency table. 
            [1, 0, 0],# a
            [0, 1, 1] #b
        ]
        ,columns= ["x","y","z"]
        ,index=["a","b"]
        )
    
    pca_return_value = pd.DataFrame(
        [   #pc1, pc2, pc3
            [1.2, 3.4, 5],#a
            [11.22,33.44,6]#b
        ],columns=["pca0","pca1","pca2"],index=["a","b"]
        )
    
    stub_pca_transform = TransformStub(transform_return_value=pca_return_value)
    encoder = PCAEncoder(_pca_transform = stub_pca_transform )

    result = encoder.fit(X=X,y=y)

    assert np.array_equal(stub_pca_transform.transform_X_.columns,expected_contingency_table.columns), f"columns do not match. Expected:{str(expected_contingency_table.columns)}, actual: {str(stub_pca_transform.transform_X_.columns)}"
    assert len(result.mapping_["b"])==3
    assert len(result.mapping_["c"]) == 3
    assert result.mapping_["c"].count(None) == 3
    assert result.n_features_out_ == 3

def test_pca_fit_when_feature_contains_missing():
    X = pd.Series(["a", None,"b","b"],name="input_feature")
    y = pd.Series(["x", "x","y","z"])

    stub_pca_transform = TransformStub(transform_return_value=pd.DataFrame())
    encoder = PCAEncoder(_pca_transform = stub_pca_transform )

    encoder.fit(X=X,y=y)

    expected_contingency_table = pd.DataFrame(
        [#   x  y  z 
            [1, 0, 0],# a
            [0, 1, 1] #b
        ]
        ,columns= ["x","y","z"]
        ,index=["a","b"]
        )
    
    assert np.array_equal(stub_pca_transform.transform_X_.columns,expected_contingency_table.columns), f"columns do not match. Expected:{str(expected_contingency_table.columns)}, actual: {str(stub_pca_transform.transform_X_.columns)}"

def test_pca_transform_given_fitted_estimator_when_transforming_non_missing_values():
    X = pd.Series(["a", "a","b","b","c","c"],name="input_feature")

    encoder = PCAEncoder(_pca_transform=None)
    encoder.mapping_ = {
        "a":[1.2,3.4],
        "b":[None,None],
        "c":[5.6,7.8]
        }
    encoder.n_features_out_ = 2
    encoder.feature_names_in_ = ["input_feature"]
    
    result = encoder.transform(X)

    expected_result = pd.DataFrame(
        [
            [1.2,3.4],#a
            [1.2,3.4],#a
            [None,None],#b
            [None,None],#b
            [5.6,7.8],#c
            [5.6,7.8]#c
        ]
        ,columns=["input_feature_pca0","input_feature_pca1"]
    )

    assert expected_result.equals(result)


def test_pca_transform_given_fitted_estimator_when_transforming_missing_values():
    X = pd.Series(["a", None],name="input_feature")

    encoder = PCAEncoder(_pca_transform=None)
    encoder.mapping_ = {
        "a":[1.2,3.4],
        }
    encoder.n_features_out_ = 2
    encoder.feature_names_in_ = ["input_feature"]
     
    result = encoder.transform(X)

    expected_result = pd.DataFrame(
        [
            [1.2,3.4],#a
            [None,None],#None

        ]
        ,columns=["input_feature_pca0","input_feature_pca1"]
    )

    assert expected_result.equals(result)

def test_pca_encoder_get_freature_names_out_no_input():

    encoder = PCAEncoder(_pca_transform=None)
    encoder.feature_names_in_ = ["test_feature_name"]
    encoder.n_features_out_ = 3

    result = encoder.get_feature_names_out()
    assert result == ["test_feature_name_pca0","test_feature_name_pca1","test_feature_name_pca2"]

def test_pca_encoder_get_freature_names_out_correct_input():

    encoder = PCAEncoder(_pca_transform=None)
    encoder.feature_names_in_ = ["test_feature_name"]
    encoder.n_features_out_ = 2

    result = encoder.get_feature_names_out(["test_feature_name"])
    assert result == ["test_feature_name_pca0","test_feature_name_pca1"]

def test_pca_encoder_get_freature_names_out_incorrect_input():

    encoder = PCAEncoder(_pca_transform=None)
    encoder.feature_names_in_ = ["test_feature_name"]

    with pytest.raises(ValueError):
        result = encoder.get_feature_names_out(["wrong_feature_name"])



