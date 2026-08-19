import copy

import numpy as np
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.tree import BaseDecisionTree

from synthpop.data_processing.missing_value_handling import MissingValuePredictor
from synthpop.utils import str_dtype


# ----- fixtures -----


@pytest.fixture
def stub_encoder():
    class EncoderStub(BaseEstimator):
        def __init__(self, transform_return=None):
            self.transform_return = transform_return
            self.fit_inputs = None
            self.transform_inputs = None

        def fit(self, X, y=None):
            self.fit_inputs = (X, y)
            return self

        def transform(self, X):
            self.transform_inputs = X
            if self.transform_return is None:
                self.transform_return = np.ones(len(X))
            return self.transform_return

        def fit_transform(self, X, y=None):
            self.fit_transform_inputs = (X, y)
            self.fit(X, y)
            self.fit_transform_return = np.ones(len(X))
            return self.fit_transform_return

    return EncoderStub()


@pytest.fixture
def stub_sampler():
    class SamplerStub:
        def __init__(self, sample_return=None):
            self.sample_return = sample_return
            self.fit_inputs = None
            self.sample_inputs = None

        def fit_sampler(self, leaf_ids, z):
            self.fit_inputs = (leaf_ids, z)

        def sample_from_leaves(self, leaf_ids):
            self.sample_inputs = leaf_ids
            return self.sample_return

        def clone(self):
            return self

    return SamplerStub()


@pytest.fixture
def stub_tree():
    class TreeStub(BaseDecisionTree):
        def __init__(self, apply_return=None):
            self.apply_return = apply_return
            self.fit_inputs = None
            self.apply_inputs = None

        def fit(self, X, y):
            self.fit_inputs = (X, y)
            return self

        def apply(self, X):
            self.apply_inputs = X
            return self.apply_return

    return TreeStub()


@pytest.fixture
def predictor(stub_encoder, stub_tree, stub_sampler):
    return MissingValuePredictor(
        encoder=stub_encoder,
        tree=stub_tree,
        tree_sampler=stub_sampler
    )


@pytest.fixture(autouse=True)
def mock_fit_decision_tree(request, mocker, stub_tree):
    if 'noautofixt' in request.keywords:
        return
    mocker.patch(
        "synthpop.methods.tree_utils._fit_decision_tree_with_reachable_leaves",
        return_value=stub_tree
    )


# ----- prepare data for fit tests -----


def test_prepare_data_respects_feature_order_through_flow(predictor):
    X = {
        "a": np.array([1, 2, 3]),
        "b": np.array(["x", np.nan, "y"], dtype=str_dtype),
    }
    y = np.array([0, 1, 0])

    X_out, y_out = predictor.prepare_data_for_fit(X, y)

    assert predictor.feature_order_ == ["a", "b"], (
        "Feature order contract should be handled correctly"
    )
    assert np.array_equal(predictor.feature_order_, list(X_out.keys())), (
        "output feature order should be the same as input"
    )


def test_prepare_data_for_fit_accepts_1d_inputs(predictor):
    y = np.array([0, np.nan, 0, 1])
    X = {
        "cat": np.array(["a", "b", "a", "b"], dtype=str_dtype),
        "num": np.array([1, 2, 3, 4]),
    }

    predictor.prepare_data_for_fit(X, y)

    fit_X, _ = predictor.encoders_["cat"].fit_transform_inputs

    assert fit_X.shape == (4, 1)


@pytest.mark.noautofixt
def test_prepare_data_missing_data_flow_correct(predictor, mocker):
    predictor.encoder.transform_return = np.array([2, 2, 2, 2])
    predictor.tree.apply_return = np.array([0, 1, 2, 3])

    expected_tree = clone(predictor.tree)

    mock_fit_decision_tree_with_reachable_leaves = mocker.patch(
        "synthpop.methods.tree_utils._fit_decision_tree_with_reachable_leaves",
        return_value=expected_tree
    )

    y = np.array([0, np.nan, 0, 1])

    X = {
        "cat1": np.array([["a"], ["b"], ["a"], ["b"]], dtype=str_dtype),
        "cat2": np.array([["x"], ["y"], ["x"], ["y"]], dtype=str_dtype),
        "num1": np.array([[1], [2], [3], [4]]),
        "num2": np.array([[10.0], [20.0], [30.0], [40.0]]),
    }

    X_out, y_out = predictor.prepare_data_for_fit(X, y)

    expected_mask = np.array([False, True, False, False])

    for col in ["cat1", "cat2"]:
        enc = predictor.encoders_[col]
        fit_X, fit_y = enc.fit_transform_inputs

        assert np.array_equal(fit_X, X[col])
        assert np.array_equal(fit_y, expected_mask)

    kwargs = mock_fit_decision_tree_with_reachable_leaves.call_args.kwargs

    tree_X = kwargs["X"]  # tree.fit_inputs

    # --- check full matrix composition ---
    encoded_cat1 = predictor.encoders_[
        "cat1"].fit_transform_return.reshape(-1, 1)
    encoded_cat2 = predictor.encoders_[
        "cat2"].fit_transform_return.reshape(-1, 1)

    expected_matrix = np.column_stack(
        [
            encoded_cat1,
            encoded_cat2,
            np.array(X["num1"]).reshape(-1, 1),
            np.array(X["num2"]).reshape(-1, 1),
        ],
    )

    assert np.array_equal(tree_X, expected_matrix), (
        "Tree input must combine encoded categorical and raw numeric features in correct order"
    )
    assert np.array_equal(expected_mask, kwargs["y"]), (
        "incorrect argument for _fit_decision_with_reachable_leaves"
    )

    sampler = predictor.tree_sampler_
    given_sampler_input_leaf_ids, given_sampler_input_y_values = sampler.fit_inputs

    assert np.array_equal(
        given_sampler_input_leaf_ids,
        predictor.tree.apply_return,
    ), "Sampler must receive tree.apply output as leaf IDs"

    expected_z = np.array([False, True, False, False])  # from y
    assert np.array_equal(
        given_sampler_input_y_values,
        expected_z,
    ), "missingness mask is not retrieved correctly"

    keep_idx = ~expected_mask

    for col in X:
        assert np.array_equal(X_out[col], np.array(
            X[col])[keep_idx]), f"{col} not correctly filtered"
    assert np.array_equal(y_out, [0, 0, 1])


def test_prepare_data_no_missing_data_flow(predictor):
    X = {
        "cat": np.array([["a"], ["b"], ["c"], ["d"]], dtype=str_dtype),
        "num": np.array([[1], [2], [3], [4]]),
    }
    y = np.array([10, 20, 30, 40])

    X_out, y_out = predictor.prepare_data_for_fit(X, y)

    assert predictor.tree_.fit_inputs is None, (
        "no tree should be built when there are no missing values"
    )
    assert predictor.tree_sampler_.fit_inputs is None, (
        "no sampler should be used when there are no missing values"
    )

    assert len(predictor.encoders_) == 1

    enc_cat = predictor.encoders_["cat"]
    fit_X, fit_y = enc_cat.fit_transform_inputs
    assert np.array_equal(
        fit_X, X["cat"]), "encoder input should be original X"
    assert np.array_equal(fit_y, [False] * len(y))

    for col in X:
        assert np.array_equal(X_out[col], np.array(X[col]))

    assert np.array_equal(y_out, np.array(y)), (
        "nothing should change to output data if no missing"
    )
    assert np.array_equal(list(X_out.keys()), predictor.feature_order_)


def test_prepare_data_all_missing(predictor):
    X = {"a": np.array([1, 2, 3])}
    y = np.array([np.nan, np.nan, np.nan])
    X_out, y_out = predictor.prepare_data_for_fit(X, y)
    assert predictor.tree_.fit_inputs is None
    assert predictor.tree_sampler_.fit_inputs is None
    assert len(X_out["a"]) == 0
    assert len(y_out) == 0


def test_prepare_data_no_missing(predictor):
    X = {"a": np.array([1, 2, 3]), "b": np.array([1, 2, 3])}
    y = np.array([0, 1, 0])
    X_out, y_out = predictor.prepare_data_for_fit(X, y)
    expected = {k: v.reshape(-1, 1)
                for k, v in X.items()}  # reshape for internal 2D
    assert predictor.tree_.fit_inputs is None
    for k in X:
        assert np.array_equal(X_out[k], expected[k])
    assert np.array_equal(y_out, y)


def test_prepare_data_does_not_mutate_inputs(predictor):
    X = {"a": np.array([1, 2, 3])}
    y = np.array([0, np.nan, 1])
    X_copy = {k: v.copy() for k, v in X.items()}
    y_copy = y.copy()
    predictor.prepare_data_for_fit(X, y)
    for k in X:
        assert np.array_equal(X_copy[k], X[k])
    assert np.array_equal(y, y_copy, equal_nan=True)


# ----- post synth transform tests -----

@pytest.mark.parametrize(
    "y",
    [
        np.array([1, 2, 3], dtype=np.float32),
        np.array(['1', '2', '3'], dtype=str_dtype),
    ],
)
def test_post_synth_all_missing(y, predictor):
    predictor._all_missing = True
    predictor._none_missing = False
    # all set to none as they are irrelevant to this test
    # but need to be set to avoid NotFittedError
    predictor.tree_ = None
    predictor.tree_sampler_ = None
    predictor.encoders_ = None
    predictor.feature_order_ = None
    X = {"a": np.array([1, 2, 3])}

    out = predictor.post_synth_transform(X, y)

    assert np.all(np.isnan(out))

@pytest.mark.parametrize(
    "y",
    [
        np.array([1, 2, 3], dtype=np.float32),
        np.array(['1', '2', '3'], dtype=str_dtype),
    ],
)
def test_post_synth_no_missing(y, predictor):
    predictor._all_missing = False
    predictor._none_missing = True
    # all set to none as they are irrelevant to this test
    # but need to be set to avoid NotFittedError
    predictor.tree_ = None
    predictor.tree_sampler_ = None
    predictor.encoders_ = None
    predictor.feature_order_ = None
    X = {"a": np.array([1, 2, 3])}

    out = predictor.post_synth_transform(X, y)

    assert np.array_equal(out, y)

@pytest.mark.parametrize(
    "y",
    [
        np.array(np.array([100, 200, 300, 400]), dtype=np.float32),
        np.array(np.array(['100', '200', '300', '400']), dtype=str_dtype),
    ],
)
def test_post_synth_transform_dataflow(y, predictor, stub_tree, stub_sampler, stub_encoder):
    predictor.tree_ = stub_tree
    predictor.tree_.apply_return = np.array([0, 1, 2, 3])
    predictor.tree_sampler_ = stub_sampler
    predictor.tree_sampler_.sample_return = np.array(
        [False, True, False, False])
    predictor._all_missing = False
    predictor._none_missing = False
    encoder_a = stub_encoder
    encoder_a.transform_return = np.array([1, 2, 3, 4])
    encoder_b = copy.copy(stub_encoder)
    encoder_b.transform_return = np.array([10, 20, 30, 40])
    predictor.encoders_ = {"a": encoder_a, "b": encoder_b}
    predictor.feature_order_ = ["a", "b"]

    X = {"a": np.array([1, 2, 3, 4]), "b": np.array([10, 20, 30, 40])}

    out = predictor.post_synth_transform(X, y)
    assert out.dtype == y.dtype

    tree = predictor.tree_
    tree_X = tree.apply_inputs
    expected_X = np.column_stack(
        [
            predictor.encoders_["a"].transform_return.reshape(-1, 1),
            predictor.encoders_["b"].transform_return.reshape(-1, 1),
        ],
    )

    assert np.array_equal(
        tree_X, expected_X), "tree input must be encoded X matrix"

    sampler = predictor.tree_sampler_
    given_sampler_input_leaf_ids = sampler.sample_inputs

    assert np.array_equal(
        given_sampler_input_leaf_ids,
        predictor.tree_.apply_return,
    ), "Sampler must receive tree.apply output as leaf IDs."
    assert np.array_equal(np.isnan(out), predictor.tree_sampler_.sample_return)


def test_post_synth_transform_raises_unfitted():
    model = MissingValuePredictor()
    with pytest.raises(NotFittedError):
        model.post_synth_transform({"a": np.array([1])}, np.array([1]))

@pytest.mark.parametrize(
    "y",
    [
        np.array(np.array([100, 200, 300, 400]), dtype=np.float32),
        np.array(np.array(['100', '200', '300', '400']), dtype=str_dtype),
    ],
)
def test_post_synth_uses_feature_order(y, predictor, stub_tree, stub_sampler, stub_encoder):
    predictor.tree_ = stub_tree
    predictor.tree_.apply_return = np.array([0, 1, 2, 3])
    predictor.tree_sampler_ = stub_sampler
    predictor.tree_sampler_.sample_return = np.array(
        [False, True, False, False])
    predictor._all_missing = False
    predictor._none_missing = False

    encoder_b = stub_encoder
    encoder_b.transform_return = np.array(
        [[1], [2], [1], [2]], dtype=np.float32)
    encoder_c = stub_encoder
    encoder_c.transform_return = np.array(
        [[5], [6], [5], [6]], dtype=np.float32)
    predictor.encoders_ = {"b": encoder_b, "c": encoder_c}

    predictor.feature_order_ = ["a", "b", "c", "d"]

    X = {
        "b": np.array([["x"], ["y"], ["x"], ["y"]], dtype=str_dtype),
        "a": np.array([[1], [2], [3], [4]]),
        "c": np.array([["u"], ["v"], ["u"], ["v"]], dtype=str_dtype),
        "d": np.array([[10], [20], [30], [40]])
    }

    predictor.post_synth_transform(X, y)

    tree = predictor.tree_
    tree_X = tree.apply_inputs
    expected_X = np.column_stack(
        (
            X["a"],
            encoder_b.transform_return,
            encoder_c.transform_return,
            X["d"],
        ),
    )

    assert np.array_equal(
        tree_X, expected_X), "feature_order_ must override input dict ordering"


# ----- clonability tests -----


def test_clone_works_and_fitted_does_not_preserve_state():
    mvp = MissingValuePredictor()
    mvp.prepare_data_for_fit(
        X={"a": np.array([1, 2, 3])}, y=np.array([1, 2, 3]))

    cloned = mvp.clone()

    # Fitted attributes should NOT be copied, original remains intact
    for attr in ["encoders_", "tree_", "tree_sampler_"]:
        assert not hasattr(cloned, attr)
        assert hasattr(mvp, attr)
    for attr in ["encoder", "tree", "tree_sampler"]:
        assert hasattr(cloned, attr)
        assert hasattr(mvp, attr)
