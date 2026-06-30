import pandas as pd
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from synthpop.methods.sample_synth import SampleMethod

# ----- helpers -----
def make_fitted_model(values: list, counts: list, target_name="target", n_samples=3, seed=42):
    model = SampleMethod()

    model.values_ = np.asarray(values)
    model.counts_ = np.asarray(counts)
    model.target_name_ = target_name
    model.n_samples_ = n_samples
    model.random_state_ = seed

    return model

# ----- fit tests -----
@pytest.mark.parametrize(
    "y, expected_values, expected_counts",
    [
        (pd.Series([1, pd.NA, 3], name="my_target"), [1, pd.NA, 3], [1, 1, 1],),
        (pd.Series([1.1, 2.2, np.nan], name="my_target"), [1.1, 2.2, np.nan], [1, 1, 1],),
        (pd.Series(["a", "b", "c"], name="my_target", dtype="category"), ["a", "b", "c"], [1, 1, 1],),
        (pd.Series([None, False, True], name="my_target"), [None, False, True], [1, 1, 1],),
        (pd.Series([0], name="my_target"), [0], [1],),
        (pd.Series([1, 1, 1, -2, 3], name="my_target"), [1, -2, 3], [3, 1, 1],),
        (pd.Series(["a", "a", "a", "a", "a", "a", "b"], name="my_target"), ["a", "b"], [6, 1],),
        (pd.Series([np.nan, np.nan, np.nan, np.nan, 0], name="my_target"), [np.nan, 0], [4, 1],),
    ],
)
def test_fit_stores_distribution_and_metadata(y, expected_values, expected_counts,mocker):
    mock_create_instance_seed = mocker.patch("synthpop.reproducibility.RandomStateManager.create_instance_seed",return_value= 333)
    model = SampleMethod().fit(None, y)

    assert model.target_name_ == "my_target"
    assert model.n_samples_ == len(y)
    assert model.counts_.sum() == len(y), "Counts must sum to total number of samples"

    assert list(model.counts_) == expected_counts
    assert model.random_state_ == 333

    for model_value, expected_value in zip(model.values_, expected_values):
        assert (pd.isna(model_value) and pd.isna(expected_value)) or model_value == expected_value, f"Mismatch: {model_value} != {expected_value}"


def test_fit_stores_provided_seed():
    y = pd.Series([1, 2, 3])
    model = SampleMethod(random_state=1001).fit(None, y)

    assert model.random_state_ == 1001


def test_fit_sets_name_when_none():
    y = pd.Series([1, 2, 3])
    model = SampleMethod().fit(None, y)

    assert model.target_name_ is None

# ----- transform tests -----

def test_transform_output_shape_matches_X():
    model = make_fitted_model(
        values=[1, 2, 3],
        counts=[2, 3, 5]
    )

    X = pd.DataFrame({"X": range(10)})
    result = model.transform(X)

    assert len(result) == len(X)
    assert result.name == "target"


def test_transform_without_X_uses_training_size():
    model = make_fitted_model(
        values=[1, 2, 3],
        counts=[2, 3, 5],
        n_samples=10
    )

    result = model.transform(None)

    assert len(result) == 10


def test_transform_values_within_observed_support():
    values = [1, 2, 3, np.nan]
    model = make_fitted_model(values = values, counts=[1, 1, 1, 1], n_samples=4)
    result = model.transform(pd.DataFrame(index=range(100)))

    generated = set(result.unique())

    observed_no_nan = {v for v in values if not pd.isna(v)}
    generated_no_nan = {v for v in generated if not pd.isna(v)}

    assert generated_no_nan.issubset(observed_no_nan)

    if any(pd.isna(v) for v in values):
        assert result.isna().any()

def test_transform_uses_random_state_manager(mocker):

    expected_rng = np.random.default_rng()
    expected_result = [6,7,67,76]
    mock_create_rng= mocker.patch("synthpop.reproducibility.RandomStateManager.create_rng",return_value= expected_rng)
    mock_sample_array = mocker.patch("synthpop.methods.tree_utils.sample_array",return_value=expected_result )
    values = [1, 2, 3, np.nan]
    model = make_fitted_model(values = values, counts=[1, 1, 1, 1], n_samples=4)

    model.random_state_= 123456

    result = model.transform(pd.DataFrame(index=range(100)))

    mock_create_rng.assert_called_with(seed=123456)
    mock_sample_array.assert_called_with(expected_rng,model.counts_,model.values_,100)
    assert (result==expected_result).all()



def test_transform_reproducibility_with_fixed_seed():
    values = [1, 2, 3]
    counts = [3, 3, 4]

    model1 = make_fitted_model(values, counts, n_samples=10, seed=123)
    model2 = make_fitted_model(values, counts, n_samples=10, seed=123)

    X = pd.DataFrame(index=range(50))

    result1 = model1.transform(X)
    result2 = model2.transform(X)

    pd.testing.assert_series_equal(result1, result2)


def test_transform_different_seeds_produce_different_results():
    values = [1, 2, 3]
    counts = [3, 3, 4]

    model1 = make_fitted_model(values, counts, n_samples=10, seed=1)
    model2 = make_fitted_model(values, counts, n_samples=10, seed=2)

    X = pd.DataFrame(index=range(50))

    result1 = model1.transform(X)
    result2 = model2.transform(X)

    # Not guaranteed, but overwhelmingly likely
    assert not result1.equals(result2)


@pytest.mark.parametrize(
    "values, counts",
    [
        ([1, 2, 3], [1, 1, 1]),     # Uniform distribution
        ([1, 2, 3], [5, 3, 2]),     # Moderate skew
        ([1, 2, 3], [98, 1, 1]),    # Extreme skew
        ([1], [10]),                # Degenerate (single outcome)
        (["a", "b"], [999, 1]),     # Binary extreme
    ],
)
def test_transform_approximate_distribution(values, counts):
    model = make_fitted_model(values, counts, n_samples=sum(counts))
    X = pd.DataFrame(index=range(10000))

    result = model.transform(X).value_counts(normalize=True)

    total = sum(counts)
    expected = {v: c / total for v, c in zip(values, counts)}

    for v in values:
        # Degenerate case: expect near-perfect match
        if len(values) == 1:
            assert np.isclose(result[v], 1.0)
        else:
            assert np.isclose(result[v], expected[v], atol=0.02)


def test_transform_raises_unfitted():
    model = SampleMethod()

    with pytest.raises(NotFittedError):
        model.transform(None)


# ----- missing values tests -----

def test_missing_values_all_types_are_sampled():
    values = [1, np.nan, None, 2]
    counts = [1, 1, 1, 1]

    model = make_fitted_model(values, counts, n_samples=4)
    result = model.transform(pd.DataFrame(index=range(500)))

    assert result.isna().any(), "transform must preserve missingness"

    non_missing_original = [v for v in values if not pd.isna(v)]
    non_missing_generated = result.dropna().unique()

    assert set(non_missing_generated).issubset(set(non_missing_original))

# ----- feature names out tests -----

@pytest.mark.parametrize(
    "target_name, input_features, expected",
    [
        ("synthetic", None, "synthetic"),
        ("synthetic", ["a", "b"], "synthetic"),
        (None, ["a", "b"], ["a", "b"]),
        (None, None, []),
    ],
)
def test_get_feature_names_out_manual_state(target_name, input_features, expected):
    model = SampleMethod()
    model.target_name_ = target_name

    result = model.get_feature_names_out(input_features)

    assert result == [expected]