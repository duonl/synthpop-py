import pandas as pd
import numpy as np
import pytest
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn import set_config

from synthpop.data_processing.encoders import PCAEncoder
from sklearn.utils.estimator_checks import parametrize_with_checks

class TransformStub(TransformerMixin, BaseEstimator):

    def __init__(self, fit_return_value=None,transform_return_value=None):
        self.transform_return_value = transform_return_value
        self.fit_return_value = fit_return_value

    def fit(self,X,y):
        self.fit_X_ = X
        self.fit_y_ = y
        return self.fit_return_value
    
    def transform(self,X):
        self.transform_X_ = X
        return self.transform_return_value
    
    def fit_transform(self, X, y = None, **fit_params):
        self.fit_X_ = X
        self.transform_X_ = X
        self.fit_y_ = y


        return self.transform_return_value


#TODO: test empty mapping/empty input

#TODO: validation in fit and transform. (check feature names, check is fitted)
#TODO: parameters
#TODO: fitting idempotence

def get_pca_return_and_dict():
    pca_return_value = np.array(
        [   #pc1, pc2, pc3
            [1.2, 3.4, 5],#a
            [11.22,33.44,6]#b
        ]
        )
    expected_dict = {
        "a": np.array([1.2, 3.4, 5]),
        "b": np.array([11.22,33.44,6])
    }
    return [(pca_return_value,expected_dict),(pd.DataFrame(pca_return_value,columns=["pca0","pca1","pca2"],index=["a","b"]),expected_dict)]
     

def get_test_data_full():
     # full data 

    X = np.array(["a", "a","b","b"])
    y = np.array(["x", "x","y","z"])
    
    ## contingency table
    # x   y  z
    # [2, 0, 0],# a
    # [0, 1, 1] #b

    ## centred table
    #   x   y    z 
    # [1, -0.5, -0.5],# a
    # [-1, 0.5, 0.5] #b
    #  1   1/2,   1/2 = sigma

    ## scaled
    expected_input_pca = np.array([
        #x y z
        [1,-1,-1],#a 
        [-1,1,1]#b
    ])

   
    return [(X,y,expected_input_pca,pca_result,expected_dict) for pca_result,expected_dict in get_pca_return_and_dict()]

def get_test_data_feature_constants():
#   constant feature
    X = np.array(["a", "a","a","a"])
    y = np.array(["x", "x","y","z"])
    
    ## contingency table
    # x   y  z
    # [2, 1, 1],# a


    ## centred table
    #   x y  z 
    # [0, 0, 0],# a

    ## scaled
    expected_input_pca = np.array([
        #x y z
        [0,0,0],#a 
    ])
    return [(X,y,expected_input_pca,expected_input_pca,{"a":[0,0,0]})]

def get_test_data_target_constants():
#   constant feature
    X = np.array(["a", "a","a","b"])
    y = np.array(["x", "x","x","x"])
    
    ## contingency table
    # x  
    # [3,], a
    # [1] b


    ## centred table
    # x  
    # [1,], a
    # [-1] b
    #  1 = sigma

    ## scaled
    expected_input_pca = np.array([
        #x y z
        [1],#a 
        [-1],#a 
    ])
    return [(X,y,expected_input_pca,expected_input_pca,{"a":[1],"b":[-1]})]



def get_test_data_missing_target():
    missing_types = [None,np.nan,pd.NA]
    X = np.array(["a", "a","b","b","c","c"])
    y = np.array(["x", None,"y","z",None,None])#Target is always missing for X=c, but not always missing for X=a

     ## contingency table
    # x   y  z
    # [1, 0, 0],# a
    # [0, 1, 1] #b
    #           # no row for c

    ## centred table
    #   x   y    z 
    # [0.5, -0.5, -0.5],# a
    # [-0.5, 0.5, 0.5] #b
    #  1/2   1/2,   1/2 = sigma

    v = np.sqrt(0.5)

    ## scaled
    expected_input_pca = np.array([
        #x y z
        [1,-1,-1],#a 
        [-1,1,1]#b
    ])


    return [(X,np.array(["x", missing,"y","z",missing,missing],dtype=np.object_)
             ,expected_input_pca
             ,pca_result
             ,expected_dict | {"c":[None]*len(expected_dict["a"])}) 
             for pca_result,expected_dict in get_pca_return_and_dict()
             for missing in missing_types]

def get_test_data_feature_missing():
    X = np.array(["a", None,"b","b"])
    y = np.array(["x", "x","y","z"])
    
    ## contingency table
    # x   y  z
    # [1, 0, 0],# a
    # [0, 1, 1] #b

    ## centred table
    #   x   y    z 
    # [0.5, -0.5, -0.5],# a
    # [-0.5, 0.5, 0.5] #b
    #  1/2   1/2,   1/2 = sigma

    ## scaled
    expected_input_pca = np.array([
        #x y z
        [1,-1,-1],#a 
        [-1,1,1]#b
    ])
    missing_types = [None,np.nan,pd.NA]
    return [(np.array(["a", missing,"b","b"], dtype=np.object_),
             y,
             expected_input_pca,
             pca_result,
             expected_dict) for pca_result,expected_dict in get_pca_return_and_dict() for missing in missing_types]


def get_test_data():
    return [*get_test_data_full(),
            *get_test_data_missing_target(),
            *get_test_data_feature_missing(),
            *get_test_data_feature_constants(),
            *get_test_data_target_constants()]

   
def assert_dict(expected, actual):
    for (k,v) in expected.items():
        assert np.array_equal(expected[k],actual[k]), "values do not match"

    assert len(expected.keys()) == len(actual.keys()), "keys don't match"
def validate_mapping(result_mapping,pca_result,expected_keys):
    for (i,key) in enumerate(expected_keys):
        assert np.array_equal(result_mapping[key],pca_result[i]), f"invalid mapping for {key}. Expected {pca_result[i]}, actual: {result_mapping[key]}"

def validate_set_inout_count(result_encoder,expected_n_feat):
    assert result_encoder.n_features_in_ == 1
    assert result_encoder.n_features_out_ == expected_n_feat

    if hasattr(result_encoder,"pca_transform_"):
        assert result_encoder.pca_transform is not result_encoder.pca_transform_ #to be compatible with sklearn.

@pytest.mark.parametrize("X,y,expected_input_for_PCA,pca_result,expected_dict",get_test_data())
def test_pca_fit_numeric_correctness(X,y,expected_input_for_PCA,pca_result,expected_dict):
    """
    test that the correct numeric output is produced for each numeric input.

    """
    #Given features and targets that are nowhere missing and an unfitted pcaEncoder
    # X = np.array(["a", "a","b","b"])
    # y = np.array(["x", "x","y","z"])


    stub_pca_transform = TransformStub(transform_return_value=pca_result)
    encoder = PCAEncoder(pca_transform = stub_pca_transform )

    result = encoder.fit(X=X,y=y)

    # Then the return value is self
    assert result is encoder
    assert np.array_equal(result.pca_transform_.transform_X_,expected_input_for_PCA), "input for PCA not calculated correctly"
    assert_dict(expected_dict,result.mapping_)

    validate_set_inout_count(result,expected_n_feat=len(expected_dict["a"]))

@pytest.mark.parametrize("X,y,expected_input_for_PCA,pca_result,expected_dict",get_test_data())
def test_pca_output_api(X,y,expected_input_for_PCA,pca_result,expected_dict):
    """
    Given that
        The features are a pandas series with name
        and that PCA returns 3 columns

    When the encoder is being fit

    then self.feature_names_in_ should be set to a one item array containing the name of X
        the result should be equal as if the features are a numpy array.
    """
    #Given features and targets that are nowhere missing and an unfitted pcaEncoder
    X = pd.Series(X,name="input_feature")


    stub_pca_transform = TransformStub(transform_return_value=pca_result)
    encoder = PCAEncoder(pca_transform = stub_pca_transform )

    result = encoder.fit(X=X,y=y)

    # Then the return value is self
    assert result is encoder
    # The contingency table has been made and passed to the PCA transform
    # expected_contingency_table = np.array(
    #     [#   x  y  z 
    #         [2, 0, 0],# a
    #         [0, 1, 1] #b
    #     ]
    #     )
    
    assert np.array_equal(result.pca_transform_.transform_X_,expected_input_for_PCA), "input for PCA not calculated correctly"
    assert_dict(expected_dict,result.mapping_)
    validate_set_inout_count(result,expected_n_feat=len(expected_dict["a"]))

    assert result.feature_names_in_ == ["input_feature"]


def test_pca_transform_given_fitted_estimator_when_transforming_non_missing_values():
    "When a value is mapped to an array of Nones, the mapping should still apply when transforming."
    X = np.array(["a", "a","b","b","c","c"])

    encoder = PCAEncoder(pca_transform=None)
    encoder.mapping_ = {
        "a":[1.2,3.4],
        "b":[None,None],
        "c":[5.6,7.8]
        }
    encoder.n_features_out_ = 2
    
    result = encoder.transform(X)

    expected_result = np.array(
        [
            [1.2,3.4],#a
            [1.2,3.4],#a
            [None,None],#b, note that numpy turns these Nones to nans when casting to float32
            [None,None],#b
            [5.6,7.8],#c
            [5.6,7.8]#c
        ],
        dtype=np.float32
    )

    assert np.array_equal(expected_result, result,equal_nan=True)

def test_pca_fit_empty_input():
    """
    fitting on empty data should result in an empty mapping.
    """
    encoder = PCAEncoder(pca_transform=None)

    encoder.fit(X=np.array([]),y=np.array([]))

    assert encoder.mapping_ == {}
    validate_set_inout_count(encoder,expected_n_feat=0)

def test_pca_transform_empty_input():
    """
    transforming an empty X should always result in an empty array, even if the mapping is not empty.
    """
    encoder = PCAEncoder(pca_transform=None)
    encoder.mapping_={"a":[1.1,2.2]}
    encoder.n_features_out_=2

    result = encoder.transform(np.array([]))

    assert len(result) == 0
    
def test_pca_transform_when_mapping_is_empty_transform_empty_to_empty():
    """
    transforming an empty X should always result in an empty array, even if the mapping is empty.
    """
    encoder = PCAEncoder(pca_transform=None)
    encoder.mapping_={}
    encoder.n_features_out_=0

    result = encoder.transform(X=np.array([]))
    assert len(result) == 0 

def test_pca_transform_exception_on_new_value():
    """
    An informative exception should be raised when the user attempts to encode a value that was not in the data during fitting.
    """
    encoder = PCAEncoder(pca_transform=None)
    encoder.mapping_={"a":[1.1,2.2]}
    encoder.n_features_out_=2

    with pytest.raises(ValueError,match="values not seen during fitting"):
        result = encoder.transform(np.array(["b"]))


def test_pca_fit_exception_on_not_1d_datatype():
    """
    The PCA encoder expects 1D inputs for both X and y.
    """

    encoder = PCAEncoder(pca_transform= None)

    with pytest.raises(ValueError):
        encoder.fit(pd.DataFrame([["a", None,"b","b"]]),np.array(["a", None,"b","b"]))

    with pytest.raises(ValueError):
        encoder.fit(np.array(["a", None,"b","b"]),pd.DataFrame([["a", None,"b","b"]]))

    with pytest.raises(ValueError):
        encoder.fit(np.array([["a", None,"b","b"]]),np.array(["a", None,"b","b"]))

    with pytest.raises(ValueError):
        encoder.fit(np.array(["a", None,"b","b"]),np.array([["a", None,"b","b"]]))

def test_pca_transform_given_fitted_estimator_when_transforming_missing_values():
    """
    When transforming a X that contains Nones, it should always be mapped to an array of Nones.
    """
    X = np.array(["a", None])

    encoder = PCAEncoder(pca_transform=None)
    encoder.mapping_ = {
        "a":[1.2,3.4],
        }
    encoder.n_features_out_ = 2
    encoder.feature_names_in_ = ["input_feature"]
     
    result = encoder.transform(X)

    expected_result = np.array(
        [
            [1.2,3.4],#a
            [None,None],#None

        ],
        dtype=np.float32
    )

    assert np.array_equal(expected_result, result,equal_nan=True)

def test_pca_encoder_get_feature_names_out_no_input():

    encoder = PCAEncoder(pca_transform=None)
    encoder.feature_names_in_ = ["test_feature_name"]
    encoder.n_features_out_ = 3

    result = encoder.get_feature_names_out()
    assert result == ["test_feature_name_pca0","test_feature_name_pca1","test_feature_name_pca2"]

def test_pca_encoder_get_feature_names_out_correct_input():

    encoder = PCAEncoder(pca_transform=None)
    encoder.feature_names_in_ = ["test_feature_name"]
    encoder.n_features_out_ = 2

    result = encoder.get_feature_names_out(["test_feature_name"])
    assert result == ["test_feature_name_pca0","test_feature_name_pca1"]

def test_pca_encoder_get_feature_names_out_incorrect_input():

    encoder = PCAEncoder(pca_transform=None)
    encoder.feature_names_in_ = ["test_feature_name"]

    with pytest.raises(ValueError):
        result = encoder.get_feature_names_out(["wrong_feature_name"])


@parametrize_with_checks([PCAEncoder()],legacy=False,expected_failed_checks= lambda x: {
    "check_dont_overwrite_parameters":"tests with multiple features",
    "check_n_features_in_after_fitting":"tests with multiple features"
})
def test_pca_encoder_is_sklearn_compatible(estimator,check):
    check(estimator)
