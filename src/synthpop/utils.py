import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Dict

str_dtype = np.dtypes.StringDType(na_object=np.nan)

def to_stringdtype_array(x):
    if isinstance(x,list):
        return np.array(x,dtype=str_dtype)
    return x.astype(str_dtype,copy=False)


def validate_stringdtype_array(x):
    if not isinstance(x.dtype, np.dtypes.StringDType):
        raise TypeError(f"Expected StringDType array, received {x.dtype}.")
    
    if x.dtype.na_object is not np.nan:
        raise TypeError(f"Expected StringDtype with na_object=np.nan, received {x.dtype.na_object}.")

def to_2d_array(x: npt.NDArray) -> npt.NDArray:
    if x.ndim == 1:
        return x.reshape(-1, 1)
   
    if x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array. Received shape {x.shape}.") 

    return x

 
def validate_2d_dict(X: Dict[str, npt.NDArray]) -> tuple[Dict[str, npt.NDArray], int]:
        """
        Minimal validation for dict-based tabular data.
        :return: validated dictionary and number of samples.
        """
        if not isinstance(X, dict):
            raise TypeError(f"X must be a dictionary of column arrays, got {type(X)} instead.")
        if len(X) == 0:
            raise ValueError("X must contain at least one feature. It is currently empty.")

        X_out = {}
        n_samples = None

        for key, value in X.items():
            arr = to_2d_array(value)
            if arr.shape[1] != 1:
                raise ValueError(f"Column '{key}' must contain exactly one column. Received shape {arr.shape}.")           

            if n_samples is None:
                n_samples = arr.shape[0]

            elif arr.shape[0] != n_samples:
                raise ValueError(f"All columns in X must contain the same number of samples. Expected {n_samples}, received {len(arr)} for '{key}'.")

            if not pd.api.types.is_numeric_dtype(arr.dtype):
                validate_stringdtype_array(arr)

            X_out[key] = arr

        return X_out, n_samples


def validate_1d_target(y: npt.NDArray, n_samples: int | None) -> npt.NDArray:
    """
    Helper function to validate a 1-dimensional target array. 2D arrays with one column will be reshaped.
    Used in the missing value predictor.
    :param y: the target array.
    :param n_samples: the number of samples in the features (`X`).
    :return: validated and potentially reshaped target array.
    """
    if y.ndim == 2:
        if y.shape[1] != 1:
            raise ValueError(f"y must contain exactly one column. Received shape {y.shape}.")
        y = y.reshape(-1)
 
    elif y.ndim != 1:
        raise ValueError(f"Expected 1D or 2D target array with one column. Received shape {y.shape}.")

    if n_samples is not None:
        if y.shape[0] != n_samples:
            raise ValueError(f"X and y contain a different number of samples: {n_samples} != {y.shape[0]}.")

    if not pd.api.types.is_numeric_dtype(y.dtype):
        validate_stringdtype_array(y)

    return y
