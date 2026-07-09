import string

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from synthpop.synthesiser import Synthesiser

from synthpop.methods.sample_synth import SampleMethod
from synthpop.methods.copy_synth import CopyMethod
from synthpop.methods.cart_synth import CartMethod


def test_synthesiser_correct_default_methods():
    synth = Synthesiser(random_seed=2)
    test_data = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
        "c": [5, 6]
    })

    synth.fit(test_data)

    assert isinstance(synth.models_["a"], CartMethod)
    assert isinstance(synth.models_["b"], CartMethod)
    assert isinstance(synth.models_["c"], CartMethod)


def test_synthesiser_first_column_is_sampled_categorical():
    expected_proportions = {
        "a": 1/2,
        "b": 1/3,
        "missing": 1/6
    }

    n_samples = 300

    a_list = (["a"]*int(n_samples*expected_proportions["a"]))
    b_list = (["b"]*int(n_samples*expected_proportions["b"]))
    missing_list = ([np.nan]*int(n_samples*expected_proportions["missing"]))
    test_data = pd.DataFrame({
        "first_column": a_list + b_list+missing_list
    })

    synth = Synthesiser(random_seed=2)

    result = synth.fit(test_data).generate()

    result_proportions = result["first_column"].value_counts(
        dropna=False, normalize=True)

    assert np.abs(expected_proportions["a"] - result_proportions["a"]) < 0.05
    assert np.abs(expected_proportions["b"] - result_proportions["b"]) < 0.05
    assert np.abs(
        expected_proportions["missing"] - result_proportions[np.nan]) < 0.05


# @pytest.mark.xfail(reason="known issue, see #130")
def test_synthesiser_first_column_is_sampled_numeric():
    expected_proportions = {
        '1.1': 1/2,
        '2': 1/3,
        "missing": 1/6
    }

    n_samples = 3000

    one_list = ([1.1]*int(n_samples*expected_proportions["1.1"]))
    two_list = ([2]*int(n_samples*expected_proportions["2"]))
    missing_list = ([np.nan]*int(n_samples*expected_proportions["missing"]))
    test_data = pd.DataFrame({
        "first_column": one_list + two_list+missing_list
    })

    synth = Synthesiser(random_seed=2)

    result = synth.fit(test_data).generate()

    result_proportions = result["first_column"].value_counts(
        dropna=False, normalize=True)

    assert np.abs(expected_proportions["1.1"] - result_proportions[1.1]) < 0.05
    assert np.abs(expected_proportions["2"] - result_proportions[2]) < 0.05
    assert np.abs(
        expected_proportions["missing"] - result_proportions[np.nan]) < 0.05


def simulate_realistic_dataset_correlations(n_samples=100):
    rng = np.random.default_rng(seed=852456)

    # first column is uniform random between 0 and 1.
    first_column = rng.random((n_samples,))
    # Second column is linearly related to the first
    second_column = first_column*3 + 5.5 + rng.random((n_samples,))*0.1
    # third column is independent categorical
    third_column = rng.choice(["a", "b", "c"], size=n_samples, replace=True)

    # fourth column is correlated with both numeric and categoric variables.
    fourth_column = [first_column[i] if third_column[i] in [
        "a", "b"] else second_column[i] for i in range(n_samples)] + rng.random((n_samples,))*0.1

    # fifth column is categorial with many levels and correlated with both numeric and categorical columns
    # This is done by calculating a numeric value roughly between 0 and 26 and map that value to the alphabet.

    # The thrid column decides if the fifth is near the begin or the end of the alphabet
    distribution_general_means = [9 if third_column[i] in [
        "b", "c"] else 18 for i in range(n_samples)]

    # The first column causes variance in the fifth column
    distribution_means = distribution_general_means + (first_column - 0.5)*6

    alphabet_index = [int(rng.normal(distribution_means[i], 6)) %
                      26 for i in range(n_samples)]

    fifth_column = [string.ascii_lowercase[alphabet_index[i]]
                    for i in range(n_samples)]

    dataset = pd.DataFrame({
        "first": first_column,
        "second": second_column,
        "third": third_column,
        "fourth": fourth_column,
        "fifth": fifth_column
    })

    return (dataset, ["first", "second", "fourth"], ["third", "fifth"])


def test_synthesiser_preserves_1D_statistics():
    """
    The aim of this test is to see if the means and univariate distributions are not wildly different.
    The goal is not to test that the synthetic data has a certain utility.
    The goal is to test that the synthetic data is reasonable. Benchmarking for utility should happen in other tests.
    """

    n_samples_orig = 5000
    n_samples_synthetic = 6000
    original_data, index_num, index_cat = simulate_realistic_dataset_correlations(
        n_samples=n_samples_orig)
    synthesiser = Synthesiser(random_seed=74124)

    syn_df = synthesiser.fit(original_data).generate(n=n_samples_synthetic)

    assert syn_df.shape[0] == n_samples_synthetic

    for num_col in index_num:
        original_mean = original_data[num_col].mean()
        synthetic_mean = syn_df[num_col].mean()

        assert np.abs(
            original_mean-synthetic_mean) > 1e-3, "original and synthetic are too close"
        assert np.abs(original_mean-synthetic_mean) / \
            original_mean < 0.05, "original and synthetic are too different"

    for cat_col in index_cat:
        original_dist = original_data[cat_col].value_counts(
            dropna=False, normalize=True)
        synthetic_dist = syn_df[cat_col].value_counts(
            dropna=False, normalize=True)
        max_diff = np.max(np.abs(original_dist - synthetic_dist))
        assert max_diff < 0.02


def test_synthesiser_preserves_num_num_relation():
    n_samples_orig = 3000
    original_data, _, _ = simulate_realistic_dataset_correlations(
        n_samples=n_samples_orig)
    synthesiser = Synthesiser(random_seed=74124)

    syn_df = synthesiser.fit(original_data).generate()

    assert syn_df.shape[0] == original_data.shape[0]

    obs_corr = original_data[["first", "second"]].corr()["second"]["first"]
    syn_corr = syn_df[["first", "second"]].corr()["second"]["first"]
    assert np.abs(obs_corr - syn_corr) < 0.01


def test_synthesiser_preserves_cat_num_relation():
    n_samples_orig = 3000
    original_data, _, _ = simulate_realistic_dataset_correlations(
        n_samples=n_samples_orig)
    synthesiser = Synthesiser(random_seed=74124)

    syn_df = synthesiser.fit(original_data).generate()

    syn_means = syn_df.groupby("third")["fourth"].mean()
    obs_means = original_data.groupby("third")["fourth"].mean()
    assert np.max(np.abs(syn_means - obs_means))/max(obs_means) < 0.05


def test_synthesiser_preserves_cat_cat_relation():
    n_samples_orig = 10000
    original_data, _, _ = simulate_realistic_dataset_correlations(
        n_samples=n_samples_orig)
    synthesiser = Synthesiser(random_seed=74124)

    syn_df = synthesiser.fit(original_data).generate()

    syn_ct = pd.crosstab(syn_df["third"], syn_df["fifth"], normalize='columns')
    obs_ct = pd.crosstab(
        original_data["third"], original_data["fifth"], normalize='columns')

    for col in obs_ct.columns:
        value = np.max(np.abs(obs_ct[col] - syn_ct[col]))
        assert value < 0.1


@pytest.mark.parametrize(
    "method",
    [
        CopyMethod(),
        SampleMethod(),
        CartMethod()
    ]
)
def test_synthesizer_preserves_datatypes(method):
    """
    Reproduces bug 162, where synthesizer class returns object dtype in the synthetic data
    while the original data is string datatype.

    All columns should return their original datatype.
    """

    n_samples_orig = 1000
    original_data, _, _ = simulate_realistic_dataset_correlations(
        n_samples=n_samples_orig)
    synthesiser = Synthesiser(default_syn_method=method, random_seed=74124)

    syn_df = synthesiser.fit(original_data).generate()

    assert all(syn_df.dtypes == original_data.dtypes)
