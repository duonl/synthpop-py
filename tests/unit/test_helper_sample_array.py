import numpy as np
import pandas as pd
import pytest 

from synthpop.methods.tree_utils import sample_array

# ----- stubs -----
class StubRNG:
    def __init__(self, output_array):
        self.output_array = np.array(output_array)
        self.called_with = []

    def integers(self, low, high, size=None):
        self.called_with.append((low, high, size))

        if size is None:
            return self.output_array[0]
        
        return self.output_array[:size]

def test_sample_array_maps_rng_output_to_sampled_values():
    rng = StubRNG([0, 3, 1, 2])

    counts = np.array([3, 1])   # total = 4
    values = np.array([0, 1])

    out = sample_array(rng, counts, values, n_samples=4)

    # cumulative bins: [3, 4]
    # r=0 → idx=0 → 0
    # r=3 → idx=1 → 1
    # r=1 → idx=0 → 0
    # r=2 → idx=0 → 0
    expected = np.array([0, 1, 0, 0])

    assert np.array_equal(out, expected)

    assert rng.called_with == [(0, 4, 4)] #low is 0, high is 4, n_samples is 4

@pytest.mark.parametrize(
    "counts, values",
    [
        ([1, 1, 1], [10, 20, np.nan]),        # balanced
        ([9, 1], [10, np.nan]),               # rare missing
        ([1, 9], [10, np.nan]),               # dominant missing
    ],
)
def test_sample_array_samples_missing(counts, values):
    rng = np.random.default_rng(42)

    counts = np.array(counts)
    values = np.array(values, dtype=object)

    out = sample_array(rng, counts, values, n_samples=1000)

    if any(pd.isna(v) for v in values):
        assert any(pd.isna(out))

def test_sample_array_shape_and_dtype():
    rng = np.random.default_rng(42)

    counts = np.array([2, 3, 5])
    values = np.array([10, 20, 30], dtype=np.int32)

    out = sample_array(rng, counts, values, n_samples=7)

    assert out.shape == (7,)
    assert out.dtype == values.dtype

def test_sample_array_values_in_support():
    rng = np.random.default_rng(0)

    counts = np.array([1, 1, 1])
    values = np.array(["a", "b", "c"], dtype=object)

    out = sample_array(rng, counts, values, n_samples=50)

    assert set(out).issubset(set(values))

def test_sample_array_zero_count_never_sampled():
    rng = np.random.default_rng(0)

    counts = np.array([5, 0, 5])
    values = np.array([1, 2, 3])

    out = sample_array(rng, counts, values, n_samples=100)

    assert 2 not in out

@pytest.mark.parametrize(
    "values, counts",
    [
        ([1, 2, 3], [1, 1, 1]),     # Uniform distribution
        ([1, 2, 3], [5, 3, 2]),     # Moderate skew
        ([1, 2, 3], [98, 1, 1]),    # Extreme skew
        ([1], [10]),                # Degenerate (single outcome)
        (["a", "b"], [999, 1]),         # Binary extreme
    ],
)
def test_sample_array_distribution(values, counts):
    rng = np.random.default_rng(123)

    counts = np.array(counts)
    values = np.array(values)

    n = 10000
    out = sample_array(rng, counts, values, n_samples=n)

    total = counts.sum()
    expected = {v: c / total for v, c in zip(values, counts)}

    for v in values:
        observed = np.mean(out == v)

        if len(values) == 1:
            # Degenerate case should be exact
            assert observed == 1.0
        else:
            assert np.isclose(observed, expected[v], atol=0.02)

def test_sample_array_zero_samples():
    rng = np.random.default_rng(0)

    counts = np.array([1, 2])
    values = np.array([10, 20])

    out = sample_array(rng, counts, values, n_samples=0)

    assert out.shape == (0,)

def test_sample_array_all_zero_counts():
    rng = np.random.default_rng(0)

    counts = np.array([0, 0])
    values = np.array([1, 2])

    with pytest.raises(ValueError):
        sample_array(rng, counts, values, n_samples=3)


