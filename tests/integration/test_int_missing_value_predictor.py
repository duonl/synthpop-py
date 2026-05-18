import pytest
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from synthpop.data_processing.missing_value_handling import MissingValuePredictor
from synthpop.data_processing.encoders import MeanEncoder
from synthpop.methods.tree_utils import LeafNodeSampler

str_dtype = np.dtypes.StringDType(na_object=np.nan)

def test_missing_value_predictor_happy_path():
    predictor = MissingValuePredictor(
        encoder=MeanEncoder(),
        tree=DecisionTreeClassifier(min_samples_leaf=5, random_state=0),
        tree_sampler=LeafNodeSampler()
    )

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

def test_feature_order_and_encoder_integration():
    predictor = MissingValuePredictor(
        encoder=MeanEncoder(),
        tree=DecisionTreeClassifier(min_samples_leaf=5, random_state=0),
        tree_sampler=LeafNodeSampler()
    )

    X = {"b": np.array([10, 20, 30, 40]), "a": np.array([1, 2, 3, 4])}
    y = np.array([100, np.nan, 300, np.nan])

    predictor.prepare_data_for_fit(X, y)
    assert predictor.feature_order_ == ["b", "a"]

    X_matrix = predictor._build_X_matrix(X)

    assert X_matrix.shape == (4, 2)

def test_tree_sampler_integration():
    predictor = MissingValuePredictor(
        encoder=MeanEncoder(),
        tree=DecisionTreeClassifier(min_samples_leaf=5, random_state=0),
        tree_sampler=LeafNodeSampler()
    )

    X = {"a": np.array([1, 2, 3, 4]), "b": np.array([10, 20, 30, 40])}
    y = np.array([100, np.nan, 300, np.nan])

    predictor.prepare_data_for_fit(X, y)

    X_matrix = predictor._build_X_matrix(X)

    leaf_ids = predictor.tree_.apply(X_matrix)

    # sampler should be fitted on same leaf structure
    assert len(np.unique(leaf_ids)) > 0

    mask = predictor.tree_sampler_.sample_from_leaves(leaf_ids)

    assert len(mask) == len(y)
    assert mask.dtype == bool

def test_missingness_determinism():
    predictor = MissingValuePredictor(
        encoder=MeanEncoder(),
        tree=DecisionTreeClassifier(min_samples_leaf=5, random_state=42),
        tree_sampler=LeafNodeSampler()
    )

    X = {"a": np.array([1, 2, 3, 4]), "b": np.array([10, 20, 30, 40])}

    y = np.array([100, np.nan, 300, np.nan])

    predictor.prepare_data_for_fit(X, y)

    y_input = np.array([100, 200, 300, 400])

    out1 = predictor.post_synth_transform(X, y_input)
    out2 = predictor.post_synth_transform(X, y_input)

    assert np.array_equal(out1, out2, equal_nan=True)