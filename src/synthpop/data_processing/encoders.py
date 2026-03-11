"""
This module contains classes to encode categorical data to numeric data. 

"""
from sklearn.base import OneToOneFeatureMixin, TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, validate_data
import pandas as pd
import numpy as np
from typing import Self
import numpy.typing as npt

class PCAEncoder(TransformerMixin, BaseEstimator): 
    """
    Transforms categorical data to one or more numeric columns.


    :param PCA_threshold: maximum number of columns used to encode the feature. explained_variance_threshold has precedence above PCA_threshold.
    :param explained_variance: parameter indicating how much of the total variance should be explained by the principle components. A value of 1 returns all principle components.
    """
    def __init__(self, PCA_threshold:int = 30, explained_variance:float = 0.95):
        pass

    def fit(self,X:npt.ArrayLike, y: npt.ArrayLike) -> Self:
        """
        Determines for each level of ``X`` the corresponding numerical values to encode them with. 

        :param X: The categorical feature that is to be encoded.
        :param y: The target used to encode the feature.

        :return: fitted encoder.
        """
        return self

    def transform(self,X:npt.ArrayLike) -> npt.NDArray[np.float32]:#float32 is optimal for decision trees.
        """
        replaces each level of ``X`` with the numerical values determined in :py:meth:`fit`

        :param X: the feature to be encoded.
        """
        return pd.DataFrame()
    
    def get_feature_names_out(self):
        pass
    
class MeanEncoder(OneToOneFeatureMixin,TransformerMixin, BaseEstimator): 
    """
    Transforms categorical data to numeric using mean encoding. The feature column `X` is encoded based on a numeric target column `y`.

    Examples
        >>> X = np.array(["a", "a", "b", "b", "c"])
        >>> y = np.array([1, 0, 2, 0, 3])

        >>> encoder = MeanEncoder()
        >>> encoder.fit(X, y)
        >>> X_transformed = encoder.transform(X)
        >>> X_transformed
        array([0.5, 0.5, 1.,  1.,  3. ], dtype=float32)
    """
    def __init__(self):
        pass

    def __sklearn_tags__(self):
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

    def fit(self,X:npt.ArrayLike, y: npt.ArrayLike) -> Self:
        """
        Calculate average y value for each X category.
        
        :param X: Feature column.
        :param y: Target column.

        Examples
            >>> X = np.array(["a", "a", "b", "b", "c"])
            >>> y = np.array([1, 0, 2, 0, 3])

            >>> encoder = MeanEncoder()
            >>> encoder.fit(X, y)
        """

        # Required sklearn attributes
        y = np.array([np.nan if (v is pd.NA or v is None) else v for v in y], dtype=float) #for pd.NA compatibility

        X_val, y_val = validate_data(self, X=X, y=y, validate_separately = (
             dict(ensure_2d=False, ensure_min_samples=1, dtype=["str", "object"], ensure_all_finite="allow-nan"),
             dict(ensure_2d=False, ensure_min_samples=1, dtype='numeric', ensure_all_finite="allow-nan")
        ))

        if isinstance(X,pd.Series) and ~pd.isna(X.name):
            self.feature_names_in_ = [X.name]
        self.n_features_in_ = 1

        # Identify missing
        X_missing = np.zeros(len(X_val), dtype=bool)
        X_missing |= pd.isna(X_val)
        y_missing = pd.isna(y_val)
        
        # Fit encoder
        self.mapping_ = {}
        unique_categories = np.unique(X_val[~X_missing].astype(str))
        for cat in unique_categories:
            mask = (~X_missing) & (X_val.astype(str) == cat)
            valid_targets = y_val[mask & ~y_missing]
            if valid_targets.size == 0:
                mean_val = np.nan
            else:
                mean_val = valid_targets.mean()
            
            self.mapping_[cat] = np.float32(mean_val)

        return self

    def transform(self,X:npt.ArrayLike) -> npt.NDArray[np.float32]:#float32 is optimal for decision trees.
        """
        Apply mapping from fitting function to ``X`` and returns the encoded version ``X_transformed``
        
        :param X: Original column to be encoded
        :return: Encoded column

        Examples
            >>> X = np.array(["a", "a", "b", "b", "c"])
            >>> y = np.array([1, 0, 2, 0, 3])

            >>> encoder = MeanEncoder()
            >>> encoder.fit(X, y)
            >>> X_transformed = encoder.transform(X)
            >>> X_transformed
            array([0.5, 0.5, 1.,  1.,  3. ], dtype=float32)
        """

        check_is_fitted(self, 'mapping_')

        # Input validation
        X = np.asarray(X)

        if X.ndim != 1:
            raise ValueError(f"X must be a 1D array, got shape {X.shape}.")
        if len(self.mapping_) == 0:
            return np.full(len(X), np.nan, dtype=np.float32).reshape(-1, 1) #2D output with only nans
        
        # Start transform
        result = np.full(len(X), np.nan, dtype=np.float32)

        # Detect missing
        X_missing = np.zeros(len(X), dtype=bool)
        if X.dtype.kind in ("f", "i"):
            X_missing |= np.isnan(X)
        X_missing |= np.equal(X, None)
        
        # Detect unseen categories
        unseen_categories = set(X[~X_missing]) - set(self.mapping_.keys())
        if unseen_categories:
            raise ValueError(f"Column to be encoded X has unseen categories: {unseen_categories}")
        
        # Apply mapping
        for i, val in enumerate(X):
            if not X_missing[i]:
                result[i] = self.mapping_[val]

        return result
    
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "mapping_")

        if input_features is None:
            if hasattr(self, "feature_names_in_"):
                input_features = self.feature_names_in_
            else:
                input_features = [f"mean_x{i}" for i in range(self.n_features_in_)]

        if hasattr(self, "feature_names_in_"):
            if not np.array_equal(input_features, self.feature_names_in_):
                raise ValueError(f"input_features must match feature_names_in_. Expected {self.feature_names_in_}, got {input_features}")
        base = input_features[0]
        return np.asarray([f"{base}_mean"], dtype=object)

    

