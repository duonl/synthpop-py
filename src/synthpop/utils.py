import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Dict

str_dtype = np.dtypes.StringDType(na_object=np.nan)


def _to_stringdtype_array(x):
    if isinstance(x, list):
        return np.array(x, dtype=str_dtype)
    return x.astype(str_dtype, copy=False)


def _validate_stringdtype_array(x):
    if not isinstance(x.dtype, np.dtypes.StringDType):
        raise TypeError(f"Expected StringDType array, received {x.dtype}.")

    if x.dtype.na_object is not np.nan:
        raise TypeError(
            f"Expected StringDtype with na_object=np.nan, received {x.dtype.na_object}.")


def _to_2d_array(x: npt.NDArray) -> npt.NDArray:
    if x.ndim == 1:
        return x.reshape(-1, 1)

    if x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array. Received shape {x.shape}.")

    return x


def _validate_2d_dict(X: Dict[str, npt.NDArray]) -> tuple[Dict[str, npt.NDArray], int]:
    """
    Minimal validation for dict-based tabular data.
    :return: validated dictionary and number of samples.
    """
    if not isinstance(X, dict):
        raise TypeError(
            f"X must be a dictionary of column arrays, got {type(X)} instead.")
    if len(X) == 0:
        raise ValueError(
            "X must contain at least one feature. It is currently empty.")

    X_out = {}
    n_samples = None

    for key, value in X.items():
        arr = _to_2d_array(value)
        if arr.shape[1] != 1:
            raise ValueError(
                f"Column '{key}' must contain exactly one column. Received shape {arr.shape}.")

        if n_samples is None:
            n_samples = arr.shape[0]

        elif arr.shape[0] != n_samples:
            raise ValueError(
                f"All columns in X must contain the same number of samples. Expected {n_samples}, received {len(arr)} for '{key}'.")

        if not pd.api.types.is_numeric_dtype(arr.dtype):
            _validate_stringdtype_array(arr)

        X_out[key] = arr

    return X_out, n_samples


def _validate_1d_target(y: npt.NDArray, n_samples: int | None) -> npt.NDArray:
    """
    Helper function to validate a 1-dimensional target array. 2D arrays with one column will be reshaped.
    Used in the missing value predictor.
    :param y: the target array.
    :param n_samples: the number of samples in the features (`X`).
    :return: validated and potentially reshaped target array.
    """
    if y.ndim == 2:
        if y.shape[1] != 1:
            raise ValueError(
                f"y must contain exactly one column. Received shape {y.shape}.")
        y = y.reshape(-1)

    elif y.ndim != 1:
        raise ValueError(
            f"Expected 1D or 2D target array with one column. Received shape {y.shape}.")

    if n_samples is not None:
        if y.shape[0] != n_samples:
            raise ValueError(
                f"X and y contain a different number of samples: {n_samples} != {y.shape[0]}.")

    if not pd.api.types.is_numeric_dtype(y.dtype):
        _validate_stringdtype_array(y)

    return y


def _standardise_array_dtypes(X: npt.ArrayLike) -> npt.NDArray:
    """
    Helper to standardise a 1D or 2D array-like object to either:
    - float32 for numeric data
    - `StringDType(na_object = np.nan)` for non-numeric data

    Missing values are normalised to `np.nan`.
    """

    is_numeric = pd.api.types.is_numeric_dtype(np.asanyarray(X))
    # to avoid casting van np.nan to 'nan'
    arr = np.asanyarray(X, dtype=object)

    if arr.ndim not in (1, 2):
        raise TypeError(
            f"Input must be a 1D or 2D array-like object, received {arr.ndim} instead.")

    original_shape = arr.shape
    flat = arr.reshape(-1)

    if is_numeric:
        result = np.array(
            [v if not pd.isna(v) else np.nan for v in flat],
            dtype=np.float32,
        )

    else:
        result = np.array(
            [v if not pd.isna(v) else np.nan for v in flat],
            dtype=str_dtype,
        )

    return result.reshape(original_shape)


def _to_standardised_array_dict(X) -> Dict[str, npt.NDArray]:
    """
    Helper to ensure that X is a dictionary of arrays with dtype `np.float32` or `StringDType(na_object = np.nan)`.
    Input can be a pandas DataFrame or a dictionary.
    """
    if isinstance(X, pd.DataFrame):
        data = {col: X[col].to_numpy() for col in X.columns}
    else:
        data = X

    return {key: _standardise_array_dtypes(value) for key, value in data.items()}


def _raise_on_rare_value(x: npt.NDArray, rare_threshold: int):

    values, count = np.unique(x, return_counts=True)

    if (count <= rare_threshold).any():
        raise ValueError(
            f"found categorical value that occurs less times than {rare_threshold}. This poses a risk of undesirable attribute disclosure. see <LINK>")
    return
