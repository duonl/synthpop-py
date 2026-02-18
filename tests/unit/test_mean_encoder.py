import pandas as pd
import numpy as np
import pytest 

from synthpop.data_processing.Encoders import MeanEncoder

def test_error_when_y_not_numeric():
    X = pd.Series(["red", "blue", "red"], name='color')
    y = pd.Series(["value", 0, 2,], name='score')

    encoder = MeanEncoder()

    with pytest.raises(TypeError):
        encoder.fit(X, y)

def test_calculate_means():
    X = pd.Series(["red", "blue", "red", "blue", "red"], name='color')
    y = pd.Series([1, 0, 2, 0, 3], name='score')

    encoder = MeanEncoder()
    encoder.fit(X, y)

    assert any(encoder.mapping_), "Mapping_ parameter should be filled during fitting"
    assert encoder.mapping_['red'] == 2, "Encoder calculates incorrect mean values"
    assert encoder.mapping_['blue'] == 0, "Encoder calculates incorrect mean values"

def test_calculate_means_with_fractional_values():
    X = pd.Series(["red", "blue", "red"], name='color')
    y = pd.Series([1/3, 2/3, 1/2], name='score')

    encoder = MeanEncoder()
    encoder.fit(X, y)

    assert encoder.mapping_['red'] == pytest.approx(5/12), "Encoder calculates incorrect mean values with fractional values"
    assert encoder.mapping_['blue'] == pytest.approx(2/3), "Encoder calculates incorrect mean values with fractional values"

def test_fit_returns_self():
    X = pd.Series(["red", "blue", "red"], name="color")
    y = pd.Series([1, 0, 2], name="score")

    encoder = MeanEncoder()

    returned = encoder.fit(X, y)

    assert returned is encoder, "Fit should return self"

def test_is_transformed_into_numeric():
    X = pd.Series(["red", "blue", "red"], name='color')
    y = pd.Series([1, 0, 2], name='score')

    encoder = MeanEncoder()
    encoder.fit(X, y)
    X_transformed = encoder.transform(X)

    assert pd.api.types.is_numeric_dtype(X_transformed.iloc[:,0]), "Transformed column should be numeric"

def test_transform_returns_series_of_same_length():
    X = pd.Series(["red", "blue", "red"], name='color')
    y = pd.Series([1, 0, 2], name='score')

    encoder = MeanEncoder()
    encoder.fit(X, y)

    X_transformed = encoder.transform(X)

    assert isinstance(X_transformed, pd.DataFrame), "Output should be a pandas DataFrame"
    assert len(X_transformed) == len(X), "Output should have same length as input"

def test_empty_target_gives_NaN():
    X = pd.Series(["red", "blue", "red"], name='color')
    y = pd.Series([np.nan, np.nan, None], name='score')

    encoder = MeanEncoder()
    encoder.fit(X, y)

    X_transformed = encoder.transform(X)
    assert X_transformed.equals(pd.DataFrame({'color':[np.nan, np.nan, np.nan]})), "y has only missing values. Output should have only missing values."

def test_empty_target_category_gives_NaN():
    X = pd.Series(["red", "blue", "red"], name='color')
    y = pd.Series([np.nan, 1, None], name='score')

    encoder = MeanEncoder()
    encoder.fit(X, y)

    X_transformed = encoder.transform(X)
    assert X_transformed.equals(pd.DataFrame({'color':[np.nan, 1, np.nan]})), "If y is always empty for a specific X category, encoding must be empty."

def test_some_missing_target_values_are_ignored():
    X = pd.Series(["red", "blue", "red", "blue", "red"], name='color')
    y = pd.Series([1, 0, 2, np.nan, None], name='score')

    encoder = MeanEncoder()
    encoder.fit(X, y)

    X_transformed = encoder.transform(X)
    assert X_transformed.equals(pd.DataFrame({'color':[1.5, 0, 1.5, 0, 1.5]})), "If some values are missing for a specific X category, they are ignored."

def test_constant_encoding_by_constant_feature():
    X = pd.Series(["red", "red", "red"], name='color')
    y = pd.Series([1, 2, 3], name='score')

    encoder = MeanEncoder()
    encoder.fit(X, y)

    X_transformed = encoder.transform(X)
    assert X_transformed.equals(pd.DataFrame({'color':[2.0, 2.0, 2.0]})), "Constant feature should lead to constant encoding"

def test_constant_encoding_with_constant_target():
    X = pd.Series(["red", "blue", "red"], name='color')
    y = pd.Series([2, 2, 2], name='score')

    encoder = MeanEncoder()
    encoder.fit(X, y)

    X_transformed = encoder.transform(X)
    assert X_transformed.equals(pd.DataFrame({'color':[2.0, 2.0, 2.0]})), "Constant target should lead to constant encoding"

def test_error_if_X_has_unseen_values():
    X = pd.Series(["red", "blue", "red"], name='color')
    y = pd.Series([1, 0, 2,], name='score')
    
    encoder = MeanEncoder()
    encoder.fit(X, y)    

    Z = pd.Series(["red", "blue", "green"], name='color')
    with pytest.raises(ValueError):
        encoder.transform(Z)    

def test_get_feature_names_out():
    X = pd.Series(["red", "blue"], name="color")
    y = pd.Series([1, 0], name="score")

    encoder = MeanEncoder()
    names = encoder.fit(X, y).get_feature_names_out()

    assert isinstance(names, np.ndarray), "Output must be an numpy array"
    assert names.tolist() == ["color"], "Output has incorrect name"