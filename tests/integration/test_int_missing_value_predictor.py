import pytest
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import NotFittedError

from synthpop.data_processing.missing_value_handling import MissingValuePredictor
from synthpop.data_processing.encoders import MeanEncoder
from synthpop.methods.tree_utils import LeafNodeSampler, build_feature_matrix

str_dtype = np.dtypes.StringDType(na_object=np.nan)

@pytest.fixture
def predictor():
    return MissingValuePredictor(
    encoder=MeanEncoder(),
    tree=DecisionTreeClassifier(min_samples_leaf=5, random_state=0),
    tree_sampler=LeafNodeSampler()
    )

def test_missing_value_predictor_happy_path(predictor):
    X = {"a": np.array([1, 2, 3, 4]), "b": np.array([10, 20, 30, 40])}
    y_train = np.array([100, np.nan, 300, np.nan])

    X_filtered, y_filtered = predictor.prepare_data_for_fit(X, y_train)

    # sanity: rows removed correctly
    assert len(y_filtered) == 2
    assert pd.isna(y_filtered).sum() == 0

    y_input = np.array([100, 200, 300, 400])

    out = predictor.post_synth_transform(X, y_input)

    assert out.shape == y_input.shape

    # must preserve non-missing positions OR introduce new NaNs
    assert pd.isna(out).any()
    assert np.all(out[~pd.isna(out)] == y_input[~pd.isna(out)])

def test_categorical_encoder_integration(predictor):
    X = {"cat": np.array(["A", "B", "A", "B"], dtype=str_dtype), "num": np.array([1, 2, 3, 4])}
    y = np.array([1, np.nan, 3, np.nan])

    predictor.prepare_data_for_fit(X, y)
    
    assert "cat" in predictor.encoders_

def test_tree_sampler_integration(predictor):
    X = {"a": np.array([1, 2, 3, 4]), "b": np.array([10, 20, 30, 40])}
    y = np.array([100, np.nan, 300, np.nan])

    predictor.prepare_data_for_fit(X, y)

    # with the removal of the _build_X_matrix we have to reconstruct
    X_encoded = {
        col: (
            predictor.encoders_[col].transform(X[col])
            if col in predictor.encoders_
            else X[col]
        )
        for col in predictor.feature_order_
    }

    X_matrix = build_feature_matrix(X_encoded, predictor.feature_order_,)

    leaf_ids = predictor.tree_.apply(X_matrix)

    # sampler should be fitted on same leaf structure
    assert len(np.unique(leaf_ids)) > 0

    mask = predictor.tree_sampler_.sample_from_leaves(leaf_ids)

    assert len(mask) == len(y)
    assert mask.dtype == bool

@pytest.mark.xfail(reason="known issue, see #139")
def test_missingness_determinism(predictor):
    X = {"a": np.array([1, 2, 3, 4]), "b": np.array([10, 20, 30, 40])}

    y = np.array([100, np.nan, 300, np.nan])

    predictor.prepare_data_for_fit(X, y)

    y_input = np.array([100, 200, 300, 400])

    out1 = predictor.post_synth_transform(X, y_input)
    out2 = predictor.post_synth_transform(X, y_input)

    assert np.array_equal(out1, out2, equal_nan=True)

def test_full_pipeline_stability(predictor):
    X = {"a": np.array([1, 2, 3, 4]), "b": np.array([10, 20, 30, 40])}

    y = np.array([100, np.nan, 300, np.nan])

    predictor.prepare_data_for_fit(X, y)

    y_input = np.array([100, 200, 300, 400])

    out = predictor.post_synth_transform(X, y_input)

    assert out.shape == y_input.shape
    assert np.all(np.isfinite(out[~np.isnan(out)]))

def test_encoded_values_are_numeric(predictor):
    X = {"cat": np.array(["A", "B", "A", "B"], dtype=str_dtype), "num": np.array([1, 2, 3, 4])}
    y = np.array([1.0, np.nan, 3.0, np.nan])

    predictor.prepare_data_for_fit(X, y)

    X_encoded = {
        col: (
            predictor.encoders_[col].transform(X[col])
            if col in predictor.encoders_
            else X[col]
        )
        for col in predictor.feature_order_
    }

    X_matrix = build_feature_matrix(X_encoded, predictor.feature_order_,)

    assert np.issubdtype(X_matrix.dtype, np.floating)

def test_feature_order_controls_matrix_construction(predictor):
    X = {"b": np.array([10, 20, 30, 40]), "a": np.array([1, 2, 3, 4]),}
    y = np.array([100, np.nan, 300, np.nan])

    predictor.prepare_data_for_fit(X, y)

    X_new = {
        "a": np.array([1, 2, 3, 4]),
        "b": np.array([10, 20, 30, 40]),
    }

    X_encoded = {
        col: (
            predictor.encoders_[col].transform(X_new[col])
            if col in predictor.encoders_
            else X_new[col]
        )
        for col in predictor.feature_order_
    }

    X_matrix = build_feature_matrix(X_encoded, predictor.feature_order_,)

    assert np.array_equal(X_matrix[:, 0], np.array([10, 20, 30, 40]),)
    assert np.array_equal(X_matrix[:, 1], np.array([1, 2, 3, 4]),)

def test_feature_matrix_dtype_is_float32(predictor):
    X = {"a": np.array([1, 2, 3, 4]), "b": np.array([10, 20, 30, 40])}
    y = np.array([1, np.nan, 3, np.nan])

    predictor.prepare_data_for_fit(X, y)

    X_encoded = {
        col: (
            predictor.encoders_[col].transform(X[col])
            if col in predictor.encoders_
            else X[col]
        )
        for col in predictor.feature_order_
    }

    X_matrix = build_feature_matrix(X_encoded, predictor.feature_order_,)

    assert X_matrix.dtype == np.float32