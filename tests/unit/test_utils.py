import pytest
import numpy as np

from synthpop.utils import validate_stringdtype_array, validate_2d_dict, validate_1d_target

str_dtype = np.dtypes.StringDType(na_object=np.nan) 

# ----- validate_stringdtype_array -----
def test_validate_stringdtype_array_accepts_valid_array():
    arr = np.array(["a", "b", np.nan], dtype=str_dtype)
    validate_stringdtype_array(arr)

@pytest.mark.parametrize(
    "arr",
    [
        np.array(["a", "b"]),
        np.array(["a", "b", np.nan], dtype=object),
        np.array([1, 2]),
        np.array(["a", "b", None], dtype=np.dtypes.StringDType(na_object=None))
    ]
)
def test_validate_stringdtype_array_rejects_invalid_dtypes(arr):
    with pytest.raises(TypeError):
        validate_stringdtype_array(arr)


# ----- validate_2d_dict -----
def test_validate_2d_dict_accepts_numeric_and_stringdtypes():
    X = {"num": np.array([[1], [2]]), "cat": np.array([["a"], ["b"]], dtype=str_dtype)}
    result, _ = validate_2d_dict(X)

    assert result["num"].shape == (2, 1)
    assert result["cat"].shape == (2, 1)

def test_validate_2d_dict_reshapes_1d_arrays():
    X = {"num": np.array([1, 2])}
    result, _ = validate_2d_dict(X)

    assert result["num"].shape == (2, 1)

def test_validate_2d_dict_rejects_non_dict():
    with pytest.raises(TypeError):
        validate_2d_dict(np.array([1, 2]))

def test_validate_2d_dict_rejects_empty_dict():
    with pytest.raises(ValueError):
        validate_2d_dict({})

def test_validate_2d_dict_rejects_multiple_columns():
    with pytest.raises(ValueError):
        validate_2d_dict({"num": np.array([[1, 2], [3, 4]])})

def test_validate_2d_dict_rejects_inconsistent_row_counts():
    X = {"a": np.array([1, 2]), "b": np.array([1])}

    with pytest.raises(ValueError):
        validate_2d_dict(X)

@pytest.mark.parametrize(
        "dict",
        [
            {"cat": np.array(["a", "b"])},
            {"cat": np.array(["a", "b", np.nan], dtype=object)},
            {"cat": np.array(["a", None], dtype=np.dtypes.StringDType(na_object=None))}
        ]
)
def test_validate_2d_dict_rejects_wrong_non_numeric_dtype(dict):
    with pytest.raises(TypeError):
        validate_2d_dict(dict)
 

# ----- validate_1d_target -----
def test_validate_1d_target_accepts_numeric_1d():
    y = np.array([1, 2])
    result = validate_1d_target(y, None)

    assert result.shape == (2,) 

def test_validate_1d_target_accepts_stringdtype():
    y = np.array(["a", np.nan], dtype=str_dtype)
    result = validate_1d_target(y, None)

    assert result.dtype == str_dtype

def test_validate_1d_target_reshapes_single_column_2d():
    y = np.array([[1], [2]])
    result = validate_1d_target(y, 2)

    assert result.shape == (2,)

def test_validate_1d_target_rejects_wrong_sample_count():
    with pytest.raises(ValueError):
        validate_1d_target(np.array([1, 2,]), n_samples=3)

@pytest.mark.parametrize(
        "y",
        [
            np.array(["a", "b"]),
            np.array(["a", "b", np.nan], dtype=object),
            np.array(["a", None], dtype=np.dtypes.StringDType(na_object=None))
        ]
)
def test_validate_1D_target_rejects_wrong_non_numeric_dtype(y):
    with pytest.raises(TypeError):
        validate_1d_target(y, None)