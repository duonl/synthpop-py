"""
These tests aim to show that the synthesis process is
reproducible and handles randomness correctly.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from synthpop.methods.cart_synth import tune_cart
from synthpop.methods.sample_synth import SampleMethod
from synthpop.methods.tree_utils import LeafNodeSampler
from synthpop.reproducibility import RandomStateManager
from synthpop.synthesiser import Synthesiser

from tests.integration.data_generated_for_tests import (
    get_test_data_classifier,
    get_test_data_regressor,
    simulate_realistic_dataset_correlations,
)


@pytest.fixture(autouse=True)
def control_random_state_manager():
    """
    Resets RandomStateManager to uninitialised.
    This is needed to test behaviour when the user does not provide a seed.
    That is why make_int_test_reproducible is not used here.
    """
    RandomStateManager._root_seed = None
    RandomStateManager._seed_sequence = None

    yield

    RandomStateManager._root_seed = None
    RandomStateManager._seed_sequence = None


class StandardTransformer(TransformerMixin, BaseEstimator):
    """
    This class implements randomness exactly as described in the developer docs.
    It simulates how components in this package should use random numbers.
    The aim is to test that the goals set out in the functional descriptions
    regarding randomness are met.
    """

    def __init__(self, random_state: int | None = None):
        self.random_state = random_state

    def fit(self, X, y):
        if self.random_state is None:
            self.random_state_ = RandomStateManager.create_instance_seed()
        else:
            self.random_state_ = self.random_state

        return self

    def transform(self, X):
        rng = RandomStateManager.create_rng(self.random_state_)
        return rng.integers(low=0, high=1000, size=100)


def test_standard_transformer_has_reproducible_transform_by_default():
    transformer = StandardTransformer()
    transformer.fit(X=0, y=0)

    result1 = transformer.transform(X=0)
    result2 = transformer.transform(X=0)

    assert np.array_equal(result1, result2), (
        "StandardTransformer.transform is not reproducible by default"
    )

    with RandomStateManager(seed=1000):
        result3 = transformer.transform(X=0)

    result4 = transformer.transform(X=0)

    assert not np.array_equal(result1, result3), (
        "StandardTransformer.transform should have a different result if the seed has been changed."
    )
    assert np.array_equal(result1, result4), (
        "StandardTransformer.transform should have the same result after exiting the context block"
    )


def test_standard_transformer_independent_instances():
    transformer1 = StandardTransformer()
    transformer2 = StandardTransformer()

    transformer1.fit(X=0, y=0)
    transformer2.fit(X=0, y=0)

    result1 = transformer1.transform(X=0)
    result2 = transformer2.transform(X=0)

    assert not np.array_equal(result1, result2)

    assert np.array_equal(result1, transformer1.transform(X=0))
    assert np.array_equal(result2, transformer2.transform(X=0))


@pytest.mark.parametrize(
    "seed",
    [
        [1, 2, 3],
        (1, 2, 3),
        np.array([1, 2, 3]),
    ],
)  # Test different arraylike objects
def test_standard_transformer_reproduces_with_arraylike_root_seed(seed):
    RandomStateManager.set_root_seed(seed)

    transformer1 = StandardTransformer()
    transformer1.fit(X=0, y=0)
    result1 = transformer1.transform(X=2)

    RandomStateManager.set_root_seed(seed)
    transformer2 = StandardTransformer()
    transformer2.fit(X=0, y=0)
    result2 = transformer2.transform(X=2)

    assert np.array_equal(result1, result2)


def combined_regressor_and_classifier_test_data(seed=10):
    X_reg, y_reg = get_test_data_regressor(
        seed=seed,
        with_cats=True,
        with_missing_features=True,
        with_missing_target=True,
    )
    X_clas, y_clas = get_test_data_classifier(
        seed=seed,
        with_cats=True,
        with_missing_features=True,
        with_missing_target=True,
    )

    d_data = {}

    available_columns = sorted(set(X_reg) & set(X_clas))

    for i, k in enumerate(available_columns):
        if i % 2 == 0:
            d_data[k] = X_reg[k]
        else:
            d_data[k] = X_clas[k]

    d_data['y1'] = y_reg
    d_data['y2'] = y_clas

    return pd.DataFrame(d_data)


def test_reproducibility_synthesis():
    obs = combined_regressor_and_classifier_test_data()

    synth = Synthesiser(
        random_seed=1,
        default_syn_method=tune_cart(rare_categories_threshold=0),
    )
    synth.fit(obs)

    syn1 = synth.generate(2000)
    syn2 = synth.generate(2000)

    pd.testing.assert_frame_equal(
        syn1,
        syn2,
        obj="generating 2 consecutive times did not produce the same synthetic dataset",
    )
    synth2 = Synthesiser(
        random_seed=1,
        default_syn_method=tune_cart(rare_categories_threshold=0),
    )
    synth2.fit(obs)

    syn3 = synth2.generate(2000)

    for col in syn3.columns:
        syn3_is_nan_mask = pd.isna(syn3[col])
        syn2_is_nan_mask = pd.isna(syn2[col])
        pd.testing.assert_series_equal(
            syn2_is_nan_mask,
            syn3_is_nan_mask,
            obj=f"missingness not reproduced for column {col}",
        )
        assert (
            syn3[col][~syn3_is_nan_mask]
            == syn2[col][~syn2_is_nan_mask]
        ).all(), f"column {col} not reproduced"


def test_generate_independent_syn_datasets():
    obs = simulate_realistic_dataset_correlations(n_samples=1010)[0]

    synth = Synthesiser(random_seed=0)
    synth.fit(obs)

    syn1 = synth.generate(n=100)
    syn2 = synth.generate(n=100, random_seed=1234)

    assert not syn1.equals(syn2)

    synth2 = Synthesiser(random_seed=1234)
    syn3 = synth2.fit(obs).generate(n=100)
    assert not syn1.equals(syn3)


def test_generate_override_seed_is_replayable():
    obs = simulate_realistic_dataset_correlations(n_samples=1010)[0]

    synth = Synthesiser(random_seed=0)
    synth.fit(obs)

    syn1 = synth.generate()
    syn2 = synth.generate(random_seed=100)
    syn3 = synth.generate(random_seed=100)

    pd.testing.assert_frame_equal(syn2, syn3)
    assert not syn2.equals(syn1)


def test_sample_method_reproducible():
    y = pd.Series(
        ["a"] * 3 + ["b"] * 5,
        name="test_target",
    )
    method = SampleMethod()

    method.fit(None, y)

    result1 = method.transform(None)
    result2 = method.transform(None)

    pd.testing.assert_series_equal(result1, result2)


def test_sample_method_explicit_seed_reproducible():
    y = pd.Series(
        ["a"] * 3 + ["b"] * 5,
        name="test_target",
    )

    m1 = SampleMethod(random_state=123).fit(None, y)
    m2 = SampleMethod(random_state=123).fit(None, y)

    pd.testing.assert_series_equal(
        m1.transform(None),
        m2.transform(None),
    )


def test_leafnode_sampler_sample_determinism_with_same_seed():
    y = np.array([0, 0, 1, 1])
    leaf_ids = np.array([10, 10, 20, 20])

    sampler1 = LeafNodeSampler(
        random_state=41,
    ).fit_sampler(
        leaf_ids=leaf_ids,
        y=y,
    )
    sampler2 = LeafNodeSampler(
        random_state=41,
    ).fit_sampler(
        leaf_ids=leaf_ids,
        y=y,
    )

    y1 = sampler1.sample_from_leaves(leaf_ids)
    y2 = sampler2.sample_from_leaves(leaf_ids)

    assert np.array_equal(y1, y2)

    # repeated calls not advance the random state, unless a generator is input
    y3 = sampler1.sample_from_leaves(leaf_ids)
    assert np.array_equal(y1, y3)

def test_reproducibility_tune_cart_regression_220():
    obs = combined_regressor_and_classifier_test_data()

    synth = Synthesiser(random_seed=1,default_syn_method=tune_cart(rare_categories_threshold=0))
    synth.fit(obs)
    syn1 = synth.generate(2000)

    synth2 = Synthesiser(random_seed=1,default_syn_method=tune_cart(rare_categories_threshold=0))
    synth2.fit(obs)
    syn2 = synth2.generate(2000)
    syn3 = synth2.generate(2000)

    pd.testing.assert_frame_equal(syn1,syn2)
    pd.testing.assert_frame_equal(syn3,syn2)

