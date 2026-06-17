"""
These test aim to show that the synthesis process is reproducible and deals correctly with randomness
"""
import pytest

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from synthpop.reproducibility import RandomStateManager


class StandardTransformer(TransformerMixin, BaseEstimator):
    """
    This class implements randomness exactly as described in the developer docs.
    It simulates how components in this package should use random numbers.
    The aim is to test that the goals set out in the functional descriptions regarding randomness are met.
    """

    def __init__(self, random_state: int | None = None):
        self.random_state = random_state

    def fit(self, X, y):
        self.random_state_ = RandomStateManager.create_new_seed(
        ) if self.random_state is None else self.random_state

    def transform(self, X):
        rng = RandomStateManager.create_rng(self.random_state_)
        return rng.integers(low=0, high=1000, size=100)


def test_standard_transformer_has_reproducible_transform_by_default():

    transformer = StandardTransformer()
    transformer.fit(X=0, y=0)

    result1 = transformer.transform(X=0)
    result2 = transformer.transform(X=0)

    assert np.array_equal(
        result1, result2), "StandardTransformer.transform is not reproducible by default"

    with RandomStateManager(seed=1000):
        result3 = transformer.transform(X=0)

    result4 = transformer.transform(X=0)

    assert not np.array_equal(
        result1, result3), "StandardTransformer.transform should have a different result if the seed has been changed."
    assert np.array_equal(
        result1, result4), "StandardTransformer.transform should have the same result after exiting the context block"


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
