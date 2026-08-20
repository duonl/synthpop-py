"""
This module contains classes to encode categorical data to numeric data.

"""
from typing import Self, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn import clone
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.utils.validation import check_is_fitted

from synthpop.reproducibility import RandomStateManager
from synthpop.utils import _to_stringdtype_array


class _BaseEncoder(TransformerMixin, BaseEstimator):
    def to_1d(self, arr: npt.NDArray) -> npt.NDArray:
        """
        The expected input of the encoders at run-time is 2D.
        Conceptually, the encoders operate on a single feature column at a time.
        This helper internally converts the 2D input to a 1-dimensional array while allowing both:
          - 1D arrays of shape (n_samples,)
          - 2D single-column arrays of shape (n_samples, 1)

        Arrays with more than one column are rejected.
        """
        if arr.ndim > 1:
            if arr.shape[1] != 1:
                raise ValueError(
                    f"Expected a 1D or a 2D array with exactly one column. Received shape {arr.shape}.")
            return arr.reshape(-1)

        return arr

    def validate_string_array(self, x: npt.NDArray) -> npt.NDArray:
        """
        Transform all missing values in a string array to np.nan. It also converts dtype, reshapes and validates dimensionality.
        :param x: an array of strings.
        :return: the 1-dimensional array of strings with one value for missing.
        """
        arr = _to_stringdtype_array(x)
        return self.to_1d(arr)

    def _check_unseen_values(self, X_val: npt.NDArray) -> None:
        X_missing = np.isnan(X_val)

        # Detect unseen categories
        seen = set(self.mapping_.keys())
        observed = set(X_val[~X_missing])
        unseen = observed - seen
        if unseen:
            raise ValueError(
                f"transform received categories that were not observed during fitting. Unseen values: {sorted(unseen)}. Ensure input was fitted")

    def _apply_mapping(self, X_val: npt.NDArray) -> npt.NDArray[np.float32]:

        if X_val.shape[0] == 0:
            return np.empty(shape=(0, self.n_features_out_), dtype=np.float32)

        result = np.full((len(X_val), self.n_features_out_),
                         np.nan, dtype=np.float32)  # pre-allocation

        X_missing = np.isnan(X_val)

        if X_missing.all():
            return result

        unique_vals, rev_index = np.unique(X_val, return_inverse=True)
        # Note that rev_index does not reconstruct X_val when
        # X_val is a stringDType array with np.nan:
        # >>> x_val = np.array(["a", "a", "b", "b", "c",np.nan, "c"], dtype=str_dtype)
        # >>> unique_vals, rev_index = np.unique(x_val, return_inverse=True)
        # >>> unique_vals[rev_index]
        # >>> array(['a', 'a', 'b', 'b', 'c', 'c', 'c'], dtype=StringDType(na_object=nan))
        # The only differences between the reconstruction and the original are the missing values.
        # So to avoid searching X_val again to create a reverse index, we use this one instead and correct the missing values.

        mapped_vals = np.array([self.mapping_[v]
                               for v in unique_vals], dtype=np.float32)

        result = mapped_vals[rev_index]

        result[X_missing, :] = np.nan    # [np.nan] * self.n_features_out_

        if result.ndim == 1:
            return result.reshape((-1, 1))

        return result


class PCAEncoder(_BaseEncoder):
    """
    Transforms categorical data to one or more numeric columns.
    The user can adjust the number of principal components by passing an instance of :class:`sklearn.decomposition.PCA` to `pca_transform`.
    See `User Guide 4: Data preparation during synthesis <../../user_guides/4_data_preparation.html#pca-encoding>`__ 
    for more in-depth information on PCA encoding.

    :param pca_transform: The pca transform used. The default value is :class:`sklearn.decomposition.PCA`. 
        See `sklearn.decomposition.PCA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_ for the possible parameters. 
        With the default parameters, all principal components are computed and used.

    Examples
    --------

    PCAEncoding is a dependency of the :py:obj:`~synthpop.methods.cart_synth.CartMethod`. The number of components :py:attr:`n_components` in :class:`sklearn.decomposition.PCA` 
    can be changed using the :py:obj:`~synthpop.methods.cart_synth.tune_cart` method.

    **Intended usage in this package is:**
    
    >>> from synthpop.methods import tune_cart
    >>> from synthpop import Synthesiser
    ... 
    >>> default_syn_method = tune_cart(n_components=1)
    >>> syn = Synthesiser(default_syn_method=default_syn_method)

    For more configuration options, see the `configuring CART example <../../examples/configuring_cart_directly.html>`__.

    The encoder can be used directly as follows (note that this is not the intended usage):

    >>> import numpy as np
    >>> from synthpop.data_processing import PCAEncoder
    >>> X = np.array(["a", "a","b","b","c"])
    >>> y = np.array(["x", "x","y","z","w"])
    >>> pca_encoder = PCAEncoder() 
    >>> pca_encoder.fit(X=X, y=y)
    PCAEncoder()
    >>> pca_encoder.transform(X)
    array([[-1.1180340e+00, -1.5000000e+00, -1.2019867e-16],
    [-1.1180340e+00, -1.5000000e+00, -1.2019867e-16],
    [ 2.2360680e+00, -1.2953263e-15, -1.2019867e-16],
    [ 2.2360680e+00, -1.2953263e-15, -1.2019867e-16],
    [-1.1180340e+00,  1.5000000e+00, -1.2019867e-16]], dtype=float32)


    with a different number of principal components (only the first):

    >>> import numpy as np
    >>> from sklearn.decomposition import PCA
    >>> from synthpop.data_processing import PCAEncoder
    >>> pca_encoder = PCAEncoder(pca_transform=PCA(n_components=1))
    >>> X = np.array(["a", "a","b","b","c"])
    >>> y = np.array(["x", "x","y","z","w"])
    >>> pca_encoder.fit_transform(X, y)
    array([[-1.118034],
    [-1.118034],
    [ 2.236068],
    [ 2.236068],
    [-1.118034]], dtype=float32)

    preserving 75% of variance:

    >>> pca_encoder2 = PCAEncoder(pca_transform=PCA(n_components=0.75)) 
    >>> pca_encoder2.fit_transform(X, y) 
    array([[-1.1180340e+00, -1.5000000e+00],
    [-1.1180340e+00, -1.5000000e+00],
    [ 2.2360680e+00, -1.2953263e-15],
    [ 2.2360680e+00, -1.2953263e-15],
    [-1.1180340e+00,  1.5000000e+00]], dtype=float32)

    """

    def __init__(self, pca_transform: PCA | None = None) -> None:
        self.pca_transform = pca_transform

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.target_tags.one_d_labels = True
        tags.target_tags.single_output = True

        tags.input_tags.categorical = True
        tags.input_tags.one_d_array = True
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True

        tags.estimator_type = "transformer"
        return tags

    def fit(self, X: npt.NDArray, y: npt.NDArray) -> Self:
        """
        Calculate the encoding.


        :param X: Feature 1D array of categorical data. This contains the data to be encoded.
        :param y: Target 1D array of categorical data. The encoding is based on this data.
        :return: Fitted encoder.
        """

        self.n_features_in_ = 1

        X_val = self.validate_string_array(X)
        y_val = self.validate_string_array(y)

        if X_val.shape[0] == 0 or y_val.shape[0] == 0:
            raise ValueError("Cannot fit encoder: X and y must be non-empty.")
        if X_val.shape[0] != y_val.shape[0]:
            raise ValueError("Number of observations in X and y do not match")

        # the core of this implementation

        # The alternative to using pandas here is either use scipy or DIY.
        missing_contingency_table = pd.crosstab(X_val, pd.isna(y_val))

        if np.isnan(y_val).all():   # If there are only missing values for y:
            self.mapping_ = {k: [np.nan]
                             for k in missing_contingency_table.index}
            self.n_features_out_ = 1
            return self
        x_such_that_y_is_not_always_missing = missing_contingency_table[
            missing_contingency_table[False] != 0].index

        # the result of pd.crosstab is a pandas DataFrame
        contingency_table = pd.crosstab(
            X_val, y_val, dropna=False).loc[x_such_that_y_is_not_always_missing]

        x_such_that_y_is_always_missing = missing_contingency_table[
            missing_contingency_table[False] == 0].index

        if contingency_table.shape[0] == 1:
            self.mapping_ = {contingency_table.index[0]: np.zeros(
                1)} | {k: [np.nan] for k in x_such_that_y_is_always_missing}
            self.n_features_out_ = 1
            return self

        pca_input = scale(contingency_table, axis=0)

        if self.pca_transform is None:
            self.pca_transform_ = PCA(
                random_state=RandomStateManager.create_instance_seed())
        else:
            # for compatibility with sklearn we need to do this.
            self.pca_transform_ = clone(self.pca_transform)

        pca_result = self.pca_transform_.fit_transform(X=pca_input, y=None)

        # sklearn.decomposition.PCA implements the set_output api
        # that means that the user might configure globally that all transformers return pandas dataframes.
        if isinstance(pca_result, pd.DataFrame):
            pca_result = pca_result.to_numpy()

        self.n_features_out_ = pca_result.shape[1]  # needed for transform

        # The values of X s.t. y is always missing
        mapping_for_missing = {
            k: [np.nan] * self.n_features_out_ for k in x_such_that_y_is_always_missing}

        value_mapping = {contingency_table.index[i]: pca_result[i] for i in range(
            contingency_table.shape[0])}

        self.mapping_ = value_mapping | mapping_for_missing

        return self

    # float32 is optimal for decision trees.
    def transform(self, X: npt.NDArray) -> npt.NDArray[np.float32]:
        """
        replaces each level of `X` with the numeric values determined in :py:meth:`fit`

        :param X: Original column to be encoded.
        :return: Encoded column.
        """

        check_is_fitted(self, 'mapping_')
        X_val = self.validate_string_array(X)
        self._check_unseen_values(X_val)
        return self._apply_mapping(X_val)


class MeanEncoder(_BaseEncoder):
    """
    Transforms categorical data to numeric using mean encoding.
    The feature column `X` is encoded based on a numeric target column `y`.
    See `User Guide 4: Data preparation during synthesis <../../user_guides/4_data_preparation.html#mean-encoding>`__ 
    for more in-depth information on mean encoding

    Examples
    --------
    The :py:obj:`~synthpop.encoders.MeanEncoder` does not have changeable parameters.

    :py:obj:`~synthpop.encoders.MeanEncoder` can be used directly as follows (note that this is not the intended usage):

    >>> from synthpop.data_processing import MeanEncoder
    >>> X = np.array(["a", "a", "b", "b", "c"])
    >>> y = np.array([1, 0, 2, 0, 3])
    >>>
    >>> encoder = MeanEncoder()
    >>> encoder.fit(X, y)
    >>> X_transformed = encoder.transform(X)
    >>> X_transformed
    array([0.5, 0.5, 1.,  1.,  3. ], dtype=float32)
    """

    def __init__(self) -> None:
        pass

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.target_tags.one_d_labels = True
        tags.target_tags.single_output = True

        tags.input_tags.categorical = True
        tags.input_tags.string = True
        tags.input_tags.one_d_array = True
        tags.input_tags.allow_nan = True

        tags.estimator_type = "transformer"
        return tags

    def fit(self, X: npt.NDArray, y: npt.NDArray) -> Self:
        """
        Calculate average y value for each X category.

        :param X: Feature 1D array of categorical data. This contains the data to be encoded.
        :param y: Target 1D array of categorical data. The encoding is based on this data.
        :return: Fitted encoder.
        """

        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(
                f"MeanEncoder requires numeric target array y. Received dtype={y.dtype}")

        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("Cannot fit encoder: X and y must be non-empty.")

        X_val = self.validate_string_array(X)
        y_val = self.to_1d(y)

        if X_val.shape[0] != y_val.shape[0]:
            raise ValueError("Number of observations in X and y do not match")

        self.n_features_in_ = 1
        self.n_features_out_ = 1

        # Identify missing
        X_missing = np.isnan(X_val)
        y_missing = np.isnan(y_val)

        # Fit encoder
        self.mapping_ = {}
        unique_categories = np.unique(X_val[~X_missing])
        for cat in unique_categories:
            mask = (~X_missing) & (X_val == cat)
            valid_targets = y_val[mask & ~y_missing]
            if valid_targets.size == 0:
                self.mapping_[cat] = np.array([np.nan], dtype=np.float32)
            else:
                self.mapping_[cat] = np.mean(
                    valid_targets, dtype=np.float32, keepdims=True)

        return self

    # float32 is optimal for decision trees.
    def transform(self, X: npt.NDArray) -> npt.NDArray[np.float32]:
        """
        Applies mapping from fitting function to X and returns the encoded version ``X_transformed``

        :param X: Original column to be encoded
        :return: Encoded column
        """

        check_is_fitted(self, 'mapping_')
        X_val = self.validate_string_array(X)
        self._check_unseen_values(X_val)
        return self._apply_mapping(X_val)
