import pytest
import numpy as np
from synthpop.methods.tree_utils import build_feature_matrix

def test_build_feature_matrix_empty_input_empty_output():
    X = {}
    result = build_feature_matrix(X,[])
    assert result.shape == (0,0)

def test_build_feature_matrix_single_column():
    X = {"a":np.array([1,2])}
    result = build_feature_matrix(X,["a"])

    assert np.array_equal(result,np.array([[1],[2]]))
    assert result.dtype == np.dtype(np.float32)

def test_build_feature_matrix_two_columns():
    X = {
        "a":np.array([1,np.nan]),
        "b":np.array([3,4]),
         }
    result = build_feature_matrix(X,["a","b"])

    assert np.array_equal(result,np.array([[1,3],
                                           [np.nan,4]]),equal_nan=True)
    assert result.dtype == np.dtype(np.float32)
    
def test_build_feature_matrix_2D_features():
    X = {
        "a":np.array([[1,3],
                      [2,4]]),

        "b":np.array([5,6]),
         }
    result = build_feature_matrix(X,["a","b"])

    assert np.array_equal(result,np.array([[1,3,5],
                                           [2,4,6]]))
    assert result.dtype == np.dtype(np.float32)
    
def test_build_feature_matrix_respects_order():
    X = {
        "a":np.array([[1,3],
                      [2,4]]),

        "b":np.array([5,6]),
         }
    result = build_feature_matrix(X,["b","a"])

    assert np.array_equal(result,np.array([[5,1,3],
                                           [6,2,4]]))
    assert result.dtype == np.dtype(np.float32)

