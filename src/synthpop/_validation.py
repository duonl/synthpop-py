import numpy as np

def validate_dict_x(X):
    if len(X) == 0:
        raise ValueError("X must contain at least one feature.")

    X_out = {}
    lengths = set()

    for key, col in X.items():
        arr = np.asarray(col)
        if arr.ndim != 1:
            raise ValueError(f"Column '{key}' must be 1-dimensional, got shape {arr.shape}.")
        if len(arr) == 0:
            raise ValueError(f"Column '{key}' is empty.")
        X_out[key] = arr
        lengths.add(len(arr))

    if len(lengths) != 1:
        raise ValueError(f"All columns in X must have the same length, got lengths {lengths}.")
    n_samples = lengths.pop()
    return X_out,n_samples

def validate_y(y,n_samples):
        y_out = np.asarray(y)

        if y_out.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y_out.shape}.")
        if len(y_out) != n_samples:
            raise ValueError(f"X and y must have the same number of samples. Got {n_samples} and {len(y_out)}.")

        return y_out