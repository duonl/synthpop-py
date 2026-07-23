import numpy as np
import pandas as pd
import pytest

from synthpop.methods.cart_synth import CartMethod
from synthpop.methods.copy_synth import CopyMethod
from synthpop.methods.sample_synth import SampleMethod
from synthpop.synthesiser import Synthesiser

from tests.integration.data_generated_for_tests import (
    simulate_realistic_dataset_correlations,
    make_data_missing
)


def test_synthesiser_correct_default_methods():
    synth = Synthesiser(random_seed=2)
    test_data = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6],
        },
    )

    synth.fit(test_data)

    assert isinstance(synth.models_["a"], CartMethod)
    assert isinstance(synth.models_["b"], CartMethod)
    assert isinstance(synth.models_["c"], CartMethod)


def test_synthesiser_first_column_is_sampled_categorical():
    expected_proportions = {
        "a": 1 / 2,
        "b": 1 / 3,
        "missing": 1 / 6,
    }

    n_samples = 300

    a_list = (["a"] * int(n_samples * expected_proportions["a"]))
    b_list = (["b"] * int(n_samples * expected_proportions["b"]))
    missing_list = [
        np.nan
    ] * int(n_samples * expected_proportions["missing"])
    test_data = pd.DataFrame(
        {
            "first_column": a_list + b_list + missing_list,
        },
    )

    synth = Synthesiser(random_seed=2)

    result = synth.fit(test_data).generate()

    result_proportions = result["first_column"].value_counts(
        dropna=False, normalize=True)

    assert np.abs(expected_proportions["a"] - result_proportions["a"]) < 0.05
    assert np.abs(expected_proportions["b"] - result_proportions["b"]) < 0.05
    assert np.abs(
        expected_proportions["missing"] - result_proportions[np.nan]) < 0.05


def test_synthesiser_first_column_is_sampled_numeric():
    expected_proportions = {
        '1.1': 1 / 2,
        '2': 1 / 3,
        "missing": 1 / 6,
    }

    n_samples = 3000

    one_list = ([1.1] * int(n_samples * expected_proportions["1.1"]))
    two_list = ([2] * int(n_samples * expected_proportions["2"]))
    missing_list = ([np.nan] * int(n_samples *
                    expected_proportions["missing"]))
    test_data = pd.DataFrame(
        {
            "first_column": one_list + two_list + missing_list,
        },
    )

    synth = Synthesiser(random_seed=2)

    result = synth.fit(test_data).generate()

    result_proportions = result["first_column"].value_counts(
        dropna=False, normalize=True)

    assert np.abs(expected_proportions["1.1"] -
                  result_proportions.iloc[0]) < 0.05
    assert np.abs(expected_proportions["2"] -
                  result_proportions.iloc[1]) < 0.05
    assert np.abs(
        expected_proportions["missing"] - result_proportions.iloc[2]) < 0.05


def test_synthesiser_preserves_1d_statistics():
    """
    The aim of this test is to see if the means and univariate distributions
    are not wildly different.
    The goal is not to test that the synthetic data has a certain utility.
    The goal is to test that the synthetic data is reasonable.
    Benchmarking for utility should happen in other tests.
    """

    n_samples_orig = 5000
    n_samples_synthetic = 6000
    original_data, index_num, index_cat = (
        simulate_realistic_dataset_correlations(
        n_samples=n_samples_orig,
        )
    )
    synthesiser = Synthesiser(random_seed=74125)

    syn_df = synthesiser.fit(original_data).generate(n=n_samples_synthetic)

    assert syn_df.shape[0] == n_samples_synthetic

    for num_col in index_num:
        original_mean = original_data[num_col].mean()
        synthetic_mean = syn_df[num_col].mean()

        assert np.abs(
            original_mean - synthetic_mean) > 1e-3, "original and synthetic are too close"
        assert (
            np.abs(original_mean - synthetic_mean) / original_mean < 0.05
         ), "original and synthetic are too different"

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
    assert np.max(np.abs(syn_means - obs_means)) / max(obs_means) < 0.05


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
    "missing_value",
    [np.nan, pd.NA, None]
)
def test_missingness_predicts_value(missing_value):
    """A missing should always imply B == 3."""

    test_data = pd.DataFrame({
        "a": [missing_value, 1, missing_value, 2, 3, missing_value] * 20,
        "b": [3, 0, 3, 1, 2, 3] * 20,
    })

    synth = Synthesiser(random_seed=2)
    generated = synth.fit(test_data).generate(n=200)

    rows = generated["a"].isna()

    assert (generated.loc[rows, "b"] == 3).all()


@pytest.mark.parametrize(
    "missing_value",
    [np.nan, pd.NA, None]
)
def test_value_predicts_missingness(missing_value):
    """a == 'x' should always imply b is missing."""

    test_data = pd.DataFrame({
        "a": ["x", "y", "z", "x", "y", "x"] * 20,
        "b": [missing_value, 1, 2, missing_value, 3, missing_value] * 20,
    })

    synth = Synthesiser(random_seed=2)
    generated = synth.fit(test_data).generate(n=200)

    assert generated.loc[generated["a"] == "x", "b"].isna().all()


@pytest.mark.parametrize(
    "missing_value",
    [np.nan, pd.NA, None]
)
def test_joint_missingness_pattern(missing_value):
    """Missing values should occur together."""

    test_data = pd.DataFrame({
        "a": [missing_value, 1, missing_value, 2] * 30,
        "b": [missing_value, 10, missing_value, 20] * 30,
        "c": [5, 6, 7, 8] * 30,
    })

    synth = Synthesiser(random_seed=2)
    generated = synth.fit(test_data).generate(n=200)

    assert (generated["a"].isna() == generated["b"].isna()).all()
@pytest.mark.parametrize(
    "method",
    [
        CopyMethod,
        SampleMethod,
        CartMethod,
    ]
)
def test_synthesiser_preserves_datatypes(method):
    """
    Reproduces bug 162, where synthesiser class returns 
    object dtype in the synthetic data
    while the original data is string datatype.

    All columns should return their original datatype.
    """

    n_samples_orig = 1000
    original_data, _, _ = simulate_realistic_dataset_correlations(
        n_samples=n_samples_orig)

    original_data["sixth"] = pd.Series(np.zeros(n_samples_orig), dtype='Int64')
    original_data["seventh"] = pd.Series(
        pd.Categorical(
            ["medium"] * n_samples_orig,
            categories=["low", "medium", "high"],
            ordered=True,
        )
    )

    original_data = original_data.astype({
        "second": "float32",
        "third": "string",
        "fourth": "Float64",
        "fifth": "object",
    })

    synthesiser = Synthesiser(default_syn_method=method(), random_seed=74124)

    syn_df = synthesiser.fit(original_data).generate()

    assert all(syn_df.dtypes == original_data.dtypes)
    assert list(syn_df["seventh"].cat.categories) == [
        "low",
        "medium",
        "high",
    ]
    assert syn_df["seventh"].cat.ordered is True


@pytest.mark.parametrize(
    "method",
    [
        CopyMethod,
        SampleMethod,
        CartMethod,
    ]
)
def test_synthesiser_preserves_datatypes_with_missing(method):
    """
    Reproduces bug 162, where synthesiser class returns object dtype in the synthetic data
    while the original data is string datatype.

    All columns should return their original datatype.
    """

    n_samples_orig = 1000
    original_data, _, _ = simulate_realistic_dataset_correlations(
        n_samples=n_samples_orig)

    original_data["sixth"] = pd.Series(np.zeros(n_samples_orig), dtype='Int64')

    original_data["seventh"] = pd.Series(
        pd.Categorical(
            ["medium"] * n_samples_orig,
            categories=["low", "medium", "high"],
            ordered=True,
        )
    )

    original_data = original_data.astype({
        "second": "float32",
        "third": "string",
        "fourth": "Float64",
        "fifth": "object",
    })

    original_data = make_data_missing(original_data, as_series=True)

    synthesiser = Synthesiser(default_syn_method=method(), random_seed=74124)

    syn_df = synthesiser.fit(original_data).generate()

    assert all(syn_df.dtypes == original_data.dtypes)
    assert list(syn_df["seventh"].cat.categories) == [
        "low",
        "medium",
        "high",
    ]
    assert syn_df["seventh"].cat.ordered is True


@pytest.mark.parametrize(
    "missing_value",
    [np.nan, pd.NA, None]
)
def test_conditional_missingness_multiple_columns(missing_value):
    """
    c is missing only when b == 'x' and a == 1.
    """

    test_data = pd.DataFrame({
        "a": ["x", "x", "y", "y"] * 30,
        "b": [1, 0, 1, 0] * 30,
        "c": [missing_value, 2, 3, 4] * 30,
    })

    synth = Synthesiser(random_seed=2)
    generated = synth.fit(test_data).generate(n=200)

    mask = (generated["a"] == "x") & (generated["b"] == 1)

    assert generated.loc[mask, "c"].isna().all()

def test_mixed_missing_representations():
    """Different missing-value should be handled similarly"""

    test_data = pd.DataFrame({
        "a": [np.nan, pd.NA, None, 1, 2, 3,] * 20,
        "b": ["x","x", "x", "one", "two", "three",] * 20,
    })

    synth = Synthesiser(random_seed=2)
    generated = synth.fit(test_data).generate(n=200)

    missing = generated["a"].isna()

    assert (generated.loc[missing, "b"] == "x").all()

def test_object_dtype_numeric_strings():
    """Object columns with mixed numeric/string values. Should transform to string dtype"""

    test_data = pd.DataFrame({
        "a": pd.Series(
            [1, "1", 2, "2", np.nan, None] * 20,
            dtype="object",
        ),
        "b": [0, 0, 1, 1, 2, 2] * 20,
    })

    synth = Synthesiser(random_seed=2)

    generated = synth.fit(test_data).generate()

    rows_na = generated["a"].isna()

    assert (generated.loc[rows_na, "b"] == 2).all()

    assert len(generated) == len(test_data)
@pytest.mark.parametrize(
    [np.nan, None, pd.NA],
)
def test_synthesiser_handles_cart_with_all_missing_target(missing_value):
    """
    Regression test for bug 152. The CartMethod failed for a
    numerical array with only missing values, 
    Missing values are masked. As such, fitting on an entire 
    np.nan array is the same as fitting on an empty array.
    This threw an error resulting in bug issue 152. A fix was implemented
    """
    df = pd.DataFrame(
        {
            "a": [1, None],
            "b": [0, 0],
            "c": [missing_value, missing_value],
        }
    )

    special_syn_method = {
        "a": SampleMethod(),
        "b": CopyMethod(),
        "c": CartMethod(),
    }

    synth = Synthesiser(
        random_seed=2,
        special_syn_method=special_syn_method
    )
    fit = synth.fit(df)

    assert isinstance(fit.models_['a'], SampleMethod)
    assert isinstance(fit.models_['b'], CopyMethod)
    assert isinstance(fit.models_['c'], CartMethod)

    generated = fit.generate()

    assert len(generated) == len(df)

    pd.testing.assert_series_equal(
        df["b"],
        generated["b"],
    )

    assert generated['c'].isna().all()
