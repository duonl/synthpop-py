import pandas as pd
import numpy as np
import pytest 

from synthpop.data_processing.encoders import MeanEncoder

# ----- test data fixtures -----
@pytest.fixture
def simple_data():
    X = np.array(["a", "a", "b", "b", "c"])
    y = np.array([1, 0, 2, 0, 3])
    return X, y

@pytest.fixture
def fractional_data():
    X = np.array(["a", "b", "c"])
    y = np.array([3/2, 5/2, 7/2])

# ----- fit test cases -----
def test_fit_raises_for_non_numeric_target():
    X = np.array(["a", "b", "c"])
    y = np.array(["x", "y", 0])
    enc = MeanEncoder()
    with pytest.raises(TypeError):
        enc.fit(X, y)

def test_fit_calculates_means(simple_data):
    X, y = simple_data
    enc = MeanEncoder()
    result = enc.fit(X, y)
    expected_mapping = {"a": 0.5, "b": 1, "c": 3}
    assert result is enc, "Fit should return self"
    assert enc.n_features_in_ == 1, "self.n_features_in_ must equal 1 after fitting"
    assert result.mapping_ == expected_mapping

def test_fit_calculates_means_fractional(fractional_data):
    X, y = fractional_data
    enc = MeanEncoder()
    result = enc.fit(X, y)
    expected_mapping = {"a": 1.5, "b": 2.5, "c": 3.5}
    assert result is enc, "Fit should return self"
    assert enc.n_features_in_ == 1, "self.n_features_in_ must equal 1 after fitting"
    assert result.mapping_ == expected_mapping, "Encoder calculates incorrect mean values with fractional values"

def test_fit_empty_target_category():
    X = np.array(["a", "b", "c", "d"])
    y = np.array([1, 2, np.nan, None])
    enc = MeanEncoder()
    enc.fit(X, y)
    assert enc.mapping_["a"] == 1
    assert enc.mapping_["b"] == 2
    assert np.isnan(enc.mapping_["c"]), "When y is always empty for a specific X category, encoding map must be empty for that category only."
    assert np.isnan(enc.mapping_["d"]), "When y is always empty for a specific X category, encoding map must be empty for that category only."

def test_fit_ignores_some_missing_targets():
    X = np.array(["a", "a", "b", "b"])
    y = np.array([1, np.nan, 2, np.nan])
    enc = MeanEncoder().fit(X, y)
    assert enc.mapping_["a"] == 1, "When some values are missing for a specific X category, they should be ignored."
    assert enc.mapping_["b"] == 2, "When some values are missing for a specific X category, they should be ignored."

def test_fit_ignores_missing_features():
    X = np.array(["a", None, "b", None])
    y = np.array([1, 2, 3, 4])
    enc = MeanEncoder().fit(X, y)
    assert enc.mapping_["a"] ==  1
    assert enc.mapping_["b"] == 3.0
    assert None not in enc.mapping_, "Encoder should ignore missing values in the feature column"

def test_fit_all_missing_feature():
    X = np.array([None, None, None])
    y = np.array([1, 2, 3])
    enc = MeanEncoder().fit(X, y)
    assert enc.mapping_ == {}, "When X has only missing values, the mapping dictionary should be empty"

# ----- transform test cases -----
def test_transforms_uses_mapping():
    #Given a fitted estimator
    encoder = MeanEncoder()
    encoder.mapping_ = {"red":1,"blue":2,"green":None}
    encoder.feature_names_in_ = np.array(["colour"])
    encoder.n_features_in_ = 1
    X = np.array(["red", "blue","red", "green"])
    result = encoder.transform(X)
    expected_result = np.array([1,2,1,None], dtype=np.float32)
    assert np.equal(expected_result,result), "Transform does not apply mapping correctly"

def test_transform_with_empty_mapping_returns_nans():
    X = np.array([np.nan, None])
    enc = MeanEncoder()
    enc.mapping_ = {}
    X_transformed = enc.transform(X)
    assert X_transformed.shape == (2,1), "With an empty mapping, transform should preserve the number of rows when all values are missing"
    assert np.all(np.isnan(X_transformed)), "With an empty mapping, transform should give only NaNs"

def test_constant_feature_gives_constant_encoding():
    X = np.array(["a", "a", "a"])
    y = np.array([1, 2, 3])
    enc = MeanEncoder().fit(X, y)
    X_transformed = enc.transform(X)
    assert np.all(X_transformed == np.mean(y)), "Constant feature should lead to constant encoding"

def test_constant_target_gives_constant_encoding():
    X = np.array(["a", "b", "c"])
    y = np.array([5, 5, 5])
    enc = MeanEncoder().fit(X, y)
    X_transformed = enc.transform(X)
    assert np.all(X_transformed == 5), "Constant target should lead to constant encoding"

def test_transform_raises_for_unseen_categories(simple_data):
    X, y = simple_data
    enc = MeanEncoder().fit(X, y)
    X_new = np.array(["a", "b", "z"])  # 'z' is unseen
    with pytest.raises(ValueError):
        enc.transform(X_new)

    