import pandas as pd
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from synthpop.methods.copy_synth import CopyMethod

# ----- fit tests -----


@pytest.mark.parametrize(
    "y", [
        pd.Series([1, pd.NA, 3], name="my_target"),  # integer
        pd.Series([1.1, 2.2, np.nan], name="my_target"),  # floats
        pd.Series([1.1, 2.2, np.nan], name="my_target", dtype='Float64'),  # floats
        pd.Series(["a", "b", "c"], name="my_target", dtype="category"),  # categorical
        pd.Series(["a", "b", "c"], name="my_target", dtype="object"), # object
        pd.Series(["a", "b", "c"], name="my_target", dtype="string"),  # string
        pd.Series([None, False, True], name="my_target"),  # boolean
        pd.Series([0], name="my_target")
    ],
)
def test_fit_stores_target_and_metadata(y):
    model = CopyMethod().fit(None, y)

    assert model.target_name_ == "my_target"
    assert model.n_samples_ == len(y)
    assert model.target_dtype_ == y.dtype
    pd.testing.assert_series_equal(model.y_, y)


def test_fit_sets_name_when_none():
    y = pd.Series([1, 2, 3])
    model = CopyMethod().fit(None, y)

    assert model.target_name_ is None

# ----- transform tests -----


@pytest.mark.parametrize(
    "y, target_name, n_samples", [
        (pd.Series([1, pd.NA, 3]), "integer", 3),
        (pd.Series([1.1, 2.2, 3.3, np.nan]), "floats", 4),
        (pd.Series([1.1, 2.2, 3.3, np.nan], dtype='Float64'), "floats", 3),  # floats
        (pd.Series(["a", "b", "c"], dtype="category"), "categorical", 3),
        (pd.Series(["a", "b", "c"], dtype="object"), "object", 3),
        (pd.Series(["a", "b", "c"], dtype="string"), "string", 3),
        (pd.Series([None, False, True]), "boolean", 3),
        (pd.Series([0]), "zero", 1)
    ],
)
def test_transform_various_dtypes(y, target_name, n_samples):
    model = CopyMethod()
    model.y_ = y
    model.target_name_ = target_name
    model.n_samples_ = n_samples
    model.target_dtype_ = y.dtype

    result = model.transform(None)
    expected = pd.Series(y.values, name=target_name, dtype=y.dtype)
    pd.testing.assert_series_equal(result, expected)
    if isinstance(y.dtype, pd.CategoricalDtype):
        assert result.cat.categories.equals(
        expected.cat.categories
    )
        assert result.cat.ordered == expected.cat.ordered

def test_transform_accepts_X():
    model = CopyMethod()
    model.y_ = pd.Series([1, 2, 3])
    model.target_name_ = "target"
    model.n_samples_ = 3
    model.target_dtype_ = np.int64

    X = pd.DataFrame({"X": [1, 2, 3]})

    result = model.transform(X)

    assert len(result) == 3


@pytest.mark.parametrize(
    "missing_attr",
    [
        "y_",
        "n_samples_",
        "target_name_",
        "target_dtype_",
    ],
)
def test_transform_raises_unfitted(missing_attr):
    model = CopyMethod()

    # Set all required fitted attributes
    model.y_ = pd.Series([1, 2, 3])
    model.n_samples_ = 3
    model.target_name_ = "target"
    model.target_dtype_ = "int64"

    # Remove one attribute
    delattr(model, missing_attr)

    with pytest.raises(NotFittedError):
        model.transform(None)

def test_transform_raises_row_mismatch():
    model = CopyMethod()
    model.y_ = pd.Series([1, 2, 3])
    model.target_name_ = "target"
    model.n_samples_ = 3
    model.target_dtype_ = np.int64

    X = pd.DataFrame({"X": [1, 2,]})

    with pytest.raises(ValueError, match="Row mismatch"):
        model.transform(X)

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
    model = CopyMethod()
    model.target_name_ = target_name

    result = model.get_feature_names_out(input_features)

    assert result == [expected]
