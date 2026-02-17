import pandas as pd
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

    assert encoder.mapping_['red'] == 2
    assert encoder.mapping_['blue'] == 0

def test_is_fitted():
    X = pd.Series(["red", "blue", "red", "blue", "red"], name='color')
    y = pd.Series([1, 0, 2, 0, 3], name='score')

    encoder = MeanEncoder()
    encoder.fit(X, y)

    assert any(encoder.mapping_)

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
