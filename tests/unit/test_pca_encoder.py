import pandas as pd
import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from sklearn.utils.estimator_checks import parametrize_with_checks
from synthpop.data_processing.encoders import PCAEncoder

str_dtype = np.dtypes.StringDType(na_object=np.nan)

class TransformStub(TransformerMixin, BaseEstimator):

    def __init__(self, fit_return_value=None, transform_return_value=None):
        self.transform_return_value = transform_return_value
        self.fit_return_value = fit_return_value

    def fit(self, X, y):
        self.fit_X_ = X
        self.fit_y_ = y
        return self.fit_return_value

    def transform(self, X):
        self.transform_X_ = X
        return self.transform_return_value

    def fit_transform(self, X, y=None, **fit_params):
        self.fit_X_ = X
        self.transform_X_ = X
        self.fit_y_ = y

        return self.transform_return_value


def get_pca_return_and_dict():
    pca_return_value = np.array(
        [  # pc1, pc2, pc3
            [1.2, 3.4, 5],  # a
            [11.22, 33.44, 6]  # b
        ]
    )
    expected_dict = {
        "a": np.array([1.2, 3.4, 5]),
        "b": np.array([11.22, 33.44, 6])
    }
    return [(pca_return_value, expected_dict), (pd.DataFrame(pca_return_value, columns=["pca0", "pca1", "pca2"], index=["a", "b"]), expected_dict)]


def get_test_data_full():
    # full data

    X = np.array(["a", "a", "b", "b"],dtype = str_dtype)
    y = np.array(["x", "x", "y", "z"],dtype = str_dtype)

    expected_input_pca = np.array([  # centred table        contingency table
        # x y z                      #   x   y    z              x   y  z
        [1, -1, -1],  # a                 a [1, -0.5, -0.5],      a [2, 0, 0]
        [-1, 1, 1]  # b                   b [-1, 0.5, 0.5]        b [0, 1, 1]
    ])  # sigma=   1   1/2,   1/2

    input_with_np_arrays = [(X, y, expected_input_pca, pca_result, expected_dict) for pca_result, expected_dict in get_pca_return_and_dict()]
    input_2D = [(X.reshape((-1,1)), y, expected_input_pca, pca_result, expected_dict) for pca_result, expected_dict in get_pca_return_and_dict()]
    input_with_lists = [(X.tolist(), y.tolist(), expected_input_pca, pca_result, expected_dict) for pca_result, expected_dict in get_pca_return_and_dict()]
    return [*input_with_np_arrays,*input_with_lists,*input_2D]


def get_test_data_feature_constants():
    #   constant feature
    X = np.array(["a", "a", "a", "a"],dtype = str_dtype)
    y = np.array(["x", "x", "y", "z"],dtype = str_dtype)

    expected_input_pca = np.array([  # centred table       contingency table
        # x y z                             x  y  z              x  y  z
        [0, 0, 0],  # a                      a [0, 0, 0]            [2, 1, 1]
    ])
    return [(X, y, None, None, {"a": [0, 0, 0]})]


def get_test_data_target_constants():
    #   constant feature
    X = np.array(["a", "a", "a", "b"],dtype = str_dtype)
    y = np.array(["x", "x", "x", "x"],dtype = str_dtype)

    expected_input_pca = np.array([  # centred table      contingency table
        # x y z                            x                   x
        [1],  # a                       a  [ 1]              a   [3]
        [-1],  # b                      b  [-1]              b   [1]
    ])  # 1 = sigma
    return [(X, y, expected_input_pca, expected_input_pca, {"a": [1], "b": [-1]})]

def get_test_data_target_constant_missing():
    return [(np.array(["a", "a","b","b","c"],dtype = str_dtype), np.array([np.nan]*5,dtype = str_dtype), None, None, {"a": [np.nan], "b": [np.nan], "c":[np.nan]})]


def get_test_data_missing_target():
    X = np.array(["a", "a", "b", "b", "c", "c"],dtype = str_dtype)
    # y = np.array(["x", None,"y","z",None,None])#Target is always missing for X=c, but not always missing for X=a

    expected_input_pca = np.array([  # centred table        contingency table
        # x  y  z  'None'                   x      y        z    'None'                  x  y  z  'None'
        # a                 a [0.5, -0.5, -0.5,     0.5 ],               a [1, 0, 0,    1   ]
        [1, -1, -1, 1],
        # b                   b [-0.5, 0.5, 0.5,     -0.5 ]                b [0, 1, 1,    0   ]
        [-1, 1, 1, -1]
    ])  # sigma=   1/2   1/2,   1/2      1/2

    regular_target = [(X, np.array(["x", np.nan, "y", "z", np.nan, np.nan],dtype = str_dtype), expected_input_pca, pca_result, expected_dict | {"c": [np.nan]*len(expected_dict["a"])})
            for pca_result, expected_dict in get_pca_return_and_dict()]
    
    small_target = [(np.array(["a","a","b"],dtype = str_dtype), np.array(["x", "y", np.nan],dtype = str_dtype),None, None, {"a":[0],"b": [np.nan]})]

    return [*regular_target,*small_target]


def get_test_data_feature_missing():
    # X = np.array(["a", None,"b","b"])
    y = np.array(["x", "x", "y", "z"])

    expected_input_pca = np.array([  # centred table        contingency table
        # x y z                      #   x   y    z              x   y  z
        [1, -1, -1],  # a                 a [0.5, -0.5, -0.5],      a [1, 0, 0]
        [-1, 1, 1]  # b                   b [-0.5, 0.5, 0.5]        b [0, 1, 1]
    ])
    return [(np.array(["a", np.nan, "b", "b"],dtype = str_dtype),
             y,
             expected_input_pca,
             pca_result,
             expected_dict) for pca_result, expected_dict in get_pca_return_and_dict()]


def get_test_fit_data():
    return [*get_test_data_full(),
            *get_test_data_missing_target(),
            *get_test_data_feature_missing(),
            *get_test_data_target_constants(),
            *get_test_data_target_constant_missing()]


def assert_dict(expected, actual):
    for k in expected.keys():
        assert np.allclose(expected[k], actual[k],equal_nan=True), "values do not match"

    assert len(expected.keys()) == len(actual.keys()), "keys don't match"


def validate_mapping(result_mapping, pca_result, expected_keys):
    for (i, key) in enumerate(expected_keys):
        assert np.allclose(
            result_mapping[key], pca_result[i]), f"invalid mapping for {key}. Expected {pca_result[i]}, actual: {result_mapping[key]}"


def validate_set_inout_count(result_encoder, expected_n_feat):
    assert result_encoder.n_features_in_ == 1
    assert result_encoder.n_features_out_ == expected_n_feat

    if hasattr(result_encoder, "pca_transform_"):
        # to be compatible with sklearn.
        assert result_encoder.pca_transform is not result_encoder.pca_transform_

# test fitting pca encoder ------------------------------------------------------------------------


@pytest.mark.parametrize("X,y,expected_input_for_PCA,pca_result,expected_dict", get_test_fit_data())
def test_pca_fit_numeric_correctness(X, y, expected_input_for_PCA, pca_result, expected_dict):
    """
    test that the correct numeric output is produced for each numeric input.

    """
    # Given features and targets that are nowhere missing and an unfitted pcaEncoder
    # X = np.array(["a", "a","b","b"])
    # y = np.array(["x", "x","y","z"])

    stub_pca_transform = TransformStub(transform_return_value=pca_result)
    encoder = PCAEncoder(pca_transform=stub_pca_transform)

    result = encoder.fit(X=X, y=y)

    # Then the return value is self
    assert result is encoder
    if expected_input_for_PCA is not None:
        assert np.allclose(result.pca_transform_.transform_X_,
                          expected_input_for_PCA), "input for PCA not calculated correctly"
    assert_dict(expected_dict, result.mapping_)

    validate_set_inout_count(result, expected_n_feat=len(expected_dict["a"]))

def test_pca_fit_constant_feature():
    X = np.array(["a", "a", "a", "a"])
    y = np.array(["x", "x", "y", "z"])

    stub_pca_transform = TransformStub()

    encoder = PCAEncoder(pca_transform=stub_pca_transform)

    result = encoder.fit(X,y)

    assert result is encoder
    assert result.n_features_in_ == 1
    assert result.n_features_out_ == 1
    assert result.pca_transform is stub_pca_transform
    assert not hasattr(result.pca_transform,"transform_X_")
    assert np.array_equal(result.mapping_["a"], np.array([0],dtype=np.float32))
    


def test_pca_fit_empty_input():
    """
    For now, we do not have a clear usage scenario in which empty inputs should be supported.
    Although outputting an empty `mapping_` for an empty array makes sense, it is more likely to hide a serious bug than to help the user for now.
    This test ensures that poviding empty arrays raises a ValueError, since the expected behaviour for empty datasets is currently undefined in the functional description.
    """
    encoder = PCAEncoder(pca_transform=None)
    
    with pytest.raises(ValueError):
        encoder.fit(X=np.array([]), y=np.array([]))


def test_pca_fit_exception_on_not_1d_datatype():
    """
    The PCA encoder expects 1D inputs for both X and y.
    """

    encoder = PCAEncoder(pca_transform=None)

    #assert that a ValueError is raised when X has multiple columns.
    with pytest.raises(ValueError):
        encoder.fit(np.array([["a", np.nan],["b","c"]],dtype=str_dtype),
                    np.array(["a", np.nan, "b", "b"]))
        
    with pytest.raises(ValueError):
        encoder.fit(np.array([["a", None, "b", "b"]]),
                    np.array([["a", np.nan],["b","c"]],dtype=str_dtype))


# testing transform--------------------------------------------------------------------------------


def get_test_data_transform():

    shapes = [-1,(-1,1)]
    data = [

        (
            np.array(["a", "a", "b", "b", "c",np.nan, "c"],dtype = str_dtype).reshape(s), #X
            {  # mapping_
                "a": [1.2, 3.4],
                "b": [np.nan,np.nan],
                "c": [5.6, 7.8]
            },
            np.array(#Expected output
                [
                    [1.2, 3.4],  # a
                    [1.2, 3.4],  # a
                    # b, note that numpy turns these Nones to nans when casting to float32
                    [np.nan, np.nan],
                    [np.nan, np.nan],  # b
                    [5.6, 7.8],  # c
                    [np.nan,np.nan],
                    [5.6, 7.8]  # c
                ],
                dtype=np.float32
            )
        ) for s in shapes]

    return data


@pytest.mark.parametrize("X,mapping,expected_output", [*get_test_data_transform()])
def test_pca_transform_numeric_correctness(X,mapping,expected_output):
    #X = np.array(["a", "a", "b", "b", "c", "c"])

    encoder = PCAEncoder(pca_transform=None)
    encoder.mapping_ = mapping
    encoder.n_features_out_ = 2

    result = encoder.transform(X)

    assert np.array_equal(expected_output, result, equal_nan=True)


def test_pca_transform_empty_input():
    """
    transforming an empty X should always result in an empty array, even if the mapping is not empty.
    """
    encoder = PCAEncoder(pca_transform=None)
    encoder.mapping_ = {"a": [1.1, 2.2]}
    encoder.n_features_out_ = 2

    result = encoder.transform(np.array([]))

    assert len(result) == 0
    assert result.dtype == np.float32
    assert result.ndim == 2


@pytest.mark.parametrize("missing", [np.nan])
def test_pca_transform_when_mapping_is_empty_transform_missing_to_nan(missing):
    """
    transforming an always missing X should result in an always nan array, even if the mapping is empty.
    """
    encoder = PCAEncoder(pca_transform=None)

    encoder.n_features_out_ = 2
    encoder.mapping_ = {}

    result = encoder.transform(X=np.array([missing,missing]))
    assert len(result) == 2
    assert np.isnan(result).all()


def test_pca_transform_exception_on_new_value():
    """
    An informative exception should be raised when the user attempts to encode a value that was not in the data during fitting.
    """
    encoder = PCAEncoder(pca_transform=None)
    encoder.mapping_ = {"a": [1.1, 2.2], "c":[3.3,4.4]}
    encoder.n_features_out_ = 2

    with pytest.raises(ValueError, match="transform received unseen categories. Unseen values:"):
         encoder.transform(np.array(["b","c"]))


def test_pca_transform_given_fitted_estimator_when_transforming_missing_values():
    """
    When transforming a X that contains Nones, it should always be mapped to an array of Nones.
    """
    X = np.array(["a", np.nan],dtype = str_dtype)

    encoder = PCAEncoder(pca_transform=None)
    encoder.mapping_ = {
        "a": [1.2, 3.4],
    }
    encoder.n_features_out_ = 2
    encoder.feature_names_in_ = ["input_feature"]

    result = encoder.transform(X)

    expected_result = np.array(
        [
            [1.2, 3.4],  # a
            [np.nan, np.nan],  # None

        ],
        dtype=np.float32
    )

    assert np.array_equal(expected_result, result, equal_nan=True)


# test scikit-learn compatibility------------------------------------------------------------------


@parametrize_with_checks([PCAEncoder()], legacy=False, expected_failed_checks=lambda x: {
    "check_dont_overwrite_parameters": "tests with multiple features",
    "check_n_features_in_after_fitting": "tests with multiple features",
    "check_fit_score_takes_y":"tests with a score component"
})
def test_pca_encoder_is_sklearn_compatible(estimator, check):
    check(estimator)


def test_pca_encoding_transform_error_when_transforming_not_fitted():

    encoder = PCAEncoder()
    with pytest.raises(NotFittedError):
        encoder.transform(X=np.array(["a", "b"]))
