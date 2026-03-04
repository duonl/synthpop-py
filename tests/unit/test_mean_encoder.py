import pandas as pd
import numpy as np
import pytest 

from synthpop.data_processing.encoders import MeanEncoder

def test_fitted_encoder_transforms_using_mapping():
    #Given a fitted estimator
    encoder = MeanEncoder()
    encoder.mapping_ = {"red":1,"blue":2,"green":None}
    encoder.feature_names_in_ = np.array(["color"])
    encoder.n_features_in_ = 1

    #When transform is called on a series without a None
    X = np.array(["red", "blue","red", "green"])
    result = encoder.transform(X)

    #Then the result is a numeric array with the values replaced numeric values as in mapping_, and None are still Nones.
    expected_result = np.array([1,2,1,None], dtype=np.float32)
    assert np.equal(expected_result,result), "incorrect transformed output"


def test_fit_error_when_y_not_numeric():

    #Given an unfitted mean encoder
    encoder = MeanEncoder()

    #raises an TypeError exception when called with non-numeric data as target.
    X = np.array(["red", "blue"])
    y = np.array(["value", 0])
    with pytest.raises(TypeError):
        encoder.fit(X, y)

def test_fit_calculate_means():
    #Given an uninitialised mean encoder
    encoder = MeanEncoder()

    #When being fitted with non-missing features and numeric target
    X = np.array(["red", "blue", "red", "blue", "red"])
    y = np.array([1, 0, 2, 0, 3])
    returned =encoder.fit(X, y)

    # should return self
    assert returned is encoder, "Fit should return self"

    # and set attributes for the feature name and number
    assert encoder.n_features_in_==1, "self.n_featured_in_ must equal 1 after fitting"
    #assert encoder.feature_names_in_ == ["color"], "self.feature_names_in_ has incorrect name"

    # and should calculate the conditional mean and store it in mapping_
    expected_mapping = {
        "red":2,
        "blue":0
    }

    assert encoder.mapping_ == expected_mapping


def test_fit_calculate_means_with_fractional_values():
    X = np.array(["red", "blue", "red"])
    y = np.array([1/3, 2/3, 1/2])

    encoder = MeanEncoder()
    encoder.fit(X, y)

    assert encoder.mapping_['red'] == pytest.approx(5/12), "Encoder calculates incorrect mean values with fractional values"
    assert encoder.mapping_['blue'] == pytest.approx(2/3), "Encoder calculates incorrect mean values with fractional values"

def test_fit_empty_target_category_gives_NaN_only_for_that_category(): 
    X = np.array(["red", "blue", "red"])
    y = np.array([np.nan, 1, None])

    encoder = MeanEncoder()
    encoder.fit(X, y)

    assert np.isnan(encoder.mapping_["red"]), "If y is always empty for a specific X category, encoding map must be empty for that category only."
    assert encoder.mapping_["blue"] == 1, "If y is always empty for a specific X category, encoding map must be empty for that category only."


def test_fit_some_missing_target_values_are_ignored():  
    X = np.array(["red", "blue", "red", "blue", "red"])
    y = np.array([1, 0, 2, np.nan, None])

    encoder = MeanEncoder()
    encoder.fit(X, y)

    assert encoder.mapping_ == {'red': 1.5, 'blue': 0}, "If some values are missing for a specific X category, they should be ignored."

def test_fit_partially_empty_feature_ignores_missing_values():
    X = np.array(['red', np.nan, None])
    y = np.array([1, 0, 2])

    encoder = MeanEncoder()
    encoder.fit(X, y)

    assert encoder.mapping_ == {'red': 1}, "Encoder should ignore missing values in the feature column"

def test_fit_empty_feature_gives_empty_encoding_map():
    X = np.array([np.nan, np.nan, None])
    y = np.array([1, 0, 2])

    encoder = MeanEncoder()
    encoder.fit(X, y)

    assert encoder.mapping_ == {}, "If X has only missing values, the mapping dictionary should be empty"


def test_empty_mapping_and_empty_input_transforms_into_only_NaNs():
    encoder = MeanEncoder()
    encoder.mapping_ = {}

    Z = np.array([np.nan, None])
    Z_transformed = encoder.transform(Z)
    
    assert Z_transformed.shape == (2,1), 'With empty mapping, transform should preserve rows if always nan'
    assert np.isnan(Z_transformed).all(),'With empty mapping, transform should give only NaNs.' 


def test_constant_encoding_by_constant_feature():
    X = np.array(["red", "red", "red"])
    y = np.array([1, 2, 3])

    encoder = MeanEncoder()
    encoder.fit(X, y)

    assert encoder.mapping_ == {'red': 2.0}, "Constant feature should lead to constant encoding"

def test_constant_encoding_with_constant_target():  
    X = np.array(["red", "blue", "red"])
    y = np.array([2, 2, 2])

    encoder = MeanEncoder()
    encoder.fit(X, y)

    assert encoder.mapping_ == {'red': 2.0, 'blue': 2.0}, "Constant target should lead to constant encoding"

def test_error_if_X_has_unseen_values():
    encoder = MeanEncoder()
    encoder.mapping_ = {"red": 1.5, "blue": 0}

    Z = np.array(["red", "blue", "green"])
    with pytest.raises(ValueError):
        encoder.transform(Z)    

    