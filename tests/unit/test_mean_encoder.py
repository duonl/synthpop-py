from sklearn.utils.estimator_checks import check_estimator
import numpy as np
import pandas as pd
import pytest 

from synthpop.data_processing.encoders import MeanEncoder

# ----- fit test cases -----
def test_fit_raises_for_non_numeric_target():
    X = np.array(["a", "b", "c"])
    y = np.array(["x", "y", 0])
    enc = MeanEncoder()
    with pytest.raises(ValueError): #validate_data gives a ValueError instead of a TypeError
        enc.fit(X, y)

@pytest.mark.parametrize(
    "X, y, expected_mapping",
    [
        (np.array(["a", "a", "b", "b", "c"]), np.array([1, 0, 2, 0, 3]), {"a": 0.5, "b": 1, "c": 3}),
        (np.array(["a", "b", "c"]), np.array([3/2, 5/2, 7/2]), {"a": 1.5, "b": 2.5, "c": 3.5}),
        (pd.Series(["a", "a", "b", "b", "c"]), pd.Series([1, 0, 2, 0, 3]), {"a": 0.5, "b": 1, "c": 3}),
        (pd.Series(["a", "a", "b", "b"]), pd.Series([1, -1, 6, -4]), {"a": 0, "b": 1}),
    ]
)
def test_fit_calculates_means(X, y, expected_mapping):
    enc = MeanEncoder()
    result = enc.fit(X, y)
    assert result is enc, "Fit should return self"
    assert enc.n_features_in_ == 1, "self.n_features_in_ must equal 1 after fitting"
    assert result.mapping_ == expected_mapping, "Encoder calculates incorrect mean values"

def test_fit_empty_target_category():
    X = np.array(["a", "b", "c", "d", "e", "f", "f", "f"])
    y = np.array([1, 2, np.nan, None, pd.NA, np.nan, None, pd.NA])
    enc = MeanEncoder()
    enc.fit(X, y)
    assert enc.mapping_["a"] == 1
    assert enc.mapping_["b"] == 2
    assert np.isnan(enc.mapping_["c"]), "When y is always None for a specific X category, encoding map must be empty for that category only."
    assert np.isnan(enc.mapping_["d"]), "When y is always np.nan for a specific X category, encoding map must be empty for that category only."
    assert np.isnan(enc.mapping_["e"]), "When y is always pd.NA for a specific X category, encoding map must be empty for that category only"
    assert np.isnan(enc.mapping_["f"]), "When y is always empty for a specific X category, encoding map must be empty for that category only"

def test_fit_ignores_some_missing_targets():
    X = np.array(["a", "a", "a", "a", "b", "b"])
    y = np.array([1, np.nan, None, pd.NA, 2, np.nan])
    enc = MeanEncoder().fit(X, y)
    assert enc.mapping_["a"] == 1, "When some values are missing for a specific X category, they should be ignored."
    assert enc.mapping_["b"] == 2, "When some values are missing for a specific X category, they should be ignored."

def test_fit_ignores_missing_features():
    X = np.array(["a", None, "b", np.nan, pd.NA])
    y = np.array([1, 2, 3, 4, 5])
    enc = MeanEncoder().fit(X, y)
    expected_mapping = {"a": 1, "b": 3}
    assert enc.mapping_ == expected_mapping
    for missing in [None, np.nan, pd.NA]:
        assert missing not in enc.mapping_, f"Encoder should ignore missing values {missing} in the feature column"

def test_fit_all_missing_feature():
    X = np.array([None, None, None])
    y = np.array([1, 2, 3])
    enc = MeanEncoder().fit(X, y)
    assert enc.mapping_ == {}, "When X has only missing values, the mapping dictionary should be empty"

def test_fit_multiple_times():
    X1 = np.array(["a", "b"])
    y1 = np.array([1, 2])
    X2 = np.array(["a", "b"])
    y2 = np.array([10, 20])
    
    enc = MeanEncoder().fit(X1, y1)
    enc.fit(X2, y2)
    
    assert enc.mapping_["a"] == 10, "Refitting does not work properly"
    assert enc.mapping_["b"] == 20, "Refitting does not work properly"

def test_fit_with_empty_arrays():
    enc = MeanEncoder()
    with pytest.raises(ValueError):
        enc.fit(np.array([]), np.array([]))

def test_fit_and_transform_with_lists():
    X = ["a", "b", "a"]
    y = [1, 2, 3]
    enc = MeanEncoder().fit(X, y)
    X_trans = enc.transform(X)
    assert X_trans.shape[0] == len(X), "mean encoder does not work with lists as input"

def test_fit_with_mixed_types():
    X = np.array(["a", 1, "b"], dtype=object)
    y = np.array([1, 2, 3])
    enc = MeanEncoder().fit(X, y)
    assert "a" in enc.mapping_, "fit does not handle mixed types in the feature"
    assert "1" in enc.mapping_, "fit does not handle mixed types in the feature" #makes all string

# ----- transform test cases -----
def test_transforms_uses_mapping():
    #Given a fitted estimator
    encoder = MeanEncoder()
    encoder.mapping_ = {"red": 1, "blue": 2, "green": None}
    encoder.feature_names_in_ = np.array(["colour"])
    encoder.n_features_in_ = 1
    X = np.array(["red", "blue","red", "green"])
    result = encoder.transform(X)
    expected_result = np.array([1, 2, 1, np.nan], dtype=np.float32)
    assert np.allclose(expected_result, result, equal_nan=True), "Transform does not apply mapping correctly"

def test_transform_with_empty_mapping_returns_nans():
    X = np.array([np.nan, None])
    enc = MeanEncoder()
    enc.mapping_ = {}
    enc.feature_names_in_ = np.array(["Missing"])
    enc.n_features_in_ = 1
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

def test_transform_raises_for_unseen_categories():
    X = np.array(["a", "a", "b", "b", "c"])
    y = np.array([1, 0, 2, 0, 3])
    enc = MeanEncoder().fit(X, y)
    X_new = np.array(["a", "b", "z"])  # 'z' is unseen
    with pytest.raises(ValueError):
        enc.transform(X_new)

def test_transform_wrong_shape():
    X = np.array(["a", "a", "b", "b", "c"])
    y = np.array([1, 0, 2, 0, 3])
    enc = MeanEncoder().fit(X, y)
    with pytest.raises(ValueError):
        enc.transform(np.array([["a"], ["b"]]))  # 2D instead of 1D

# ----- get_feature_names_out test cases -----
def test_get_feature_names_out():
    X = np.array(["a", "a", "b", "b", "c"])
    y = np.array([1, 0, 2, 0, 3])
    enc = MeanEncoder().fit(X, y)
    names = enc.get_feature_names_out()
    assert names[0].endswith("_mean")

# ----- sklearn test suite -----

def test_sklearn_compatibility():
    check_estimator(MeanEncoder())
    