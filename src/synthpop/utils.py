import numpy as np

str_dtype = np.dtypes.StringDType(na_object=np.nan)

def to_stringdtype_array(x):
    if isinstance(x,list):
        return np.array(x,dtype=str_dtype)
    return x.astype(str_dtype,copy=False)


def validate_dict_x(X):

    if not isinstance(X, dict):
        raise TypeError("X must be a dictionary of column arrays.")
    if len(X) == 0:
        raise ValueError("X must contain at least one feature.")
    first_len = None

    X_out = {}

    for key, col in X.items():
        arr = np.asarray(col)

        if arr.ndim != 1:
            raise ValueError(f"Column '{key}' must be 1-dimensional, got shape {arr.shape}.")
        if len(arr) == 0:
            raise ValueError(f"Column '{key}' is empty.")

        if first_len is None:
            first_len = len(arr)
        elif len(arr) != first_len:
            raise ValueError(
            f"All columns in X must have the same length. "
            f"Expected {first_len}, got {len(arr)} for '{key}'."
            )

        X_out[key] = arr

    n_samples = first_len
    return X_out, n_samples


def validate_y(y,n_samples):
        y_out = np.asarray(y)

        if y_out.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y_out.shape}.")
        if len(y_out) != n_samples:
            raise ValueError(f"X and y must have the same number of samples. Got {n_samples} and {len(y_out)}.")

        return y_out
