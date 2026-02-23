"""
This module contains classes to encode categorical data to numeric data. 

"""
from typing import Self
from sklearn.base import OneToOneFeatureMixin, TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np



class PCAEncoder(TransformerMixin, BaseEstimator): 
    """
    Transforms categorical data to one or more numeric columns.


    :param PCA_threshold: maximum number of columns used to encode the feature. explained_variance_threshold has precedence above PCA_threshold.
    :param explained_variance: parameter indicating how much of the total variance should be explained by the principle components. A value of 1 returns all principle components.
    """
    def __init__(self, PCA_threshold:int = 30, explained_variance:float = 0.95,pca_transform:PCA = PCA()):
        self._pca_transform = pca_transform #TODO: parameters??
        pass

    def fit(self,X:pd.Series, y: pd.Series) -> Self:
        """
        Determines for each level of ``X`` the corresponding numerical values to encode them with. 

        :param X: The categorical feature that is to be encoded.
        :param y: The target used to encode the feature.

        :return: fitted encoder.
        """

        contingency_table = pd.crosstab(X,y)
        self._pca_transform.fit_transform(X=contingency_table,y=None)
        return self

    def transform(self,X:pd.Series) -> pd.DataFrame:
        """
        replaces each level of ``X`` with the numerical values determined in :py:meth:`fit`

        :param X: the feature to be encoded.
        """
        return pd.DataFrame()
    
    def get_feature_names_out(self):
        pass
    
class MeanEncoder(OneToOneFeatureMixin,TransformerMixin, BaseEstimator): 
    def __init__(self):
        pass

    def fit(self,X:pd.Series, y: pd.Series):
        """
        Calculate average y value for each X category.
        
        :param X: Feature column.
        :param y: Target column.

        Examples
        X = pd.Series(["red", "blue", "red", "blue", "red"], name='color')
        y = pd.Series([1, 0, 2, 0, 3], name='score')

        encoder = MeanEncoder()
        encoder.fit(X, y)
        """
        # Required for get_feature_names_out
        self.feature_names_in_ = np.array([X.name], dtype=object)
        self.n_features_in_ = 1

        # Raises exception if y is not numeric
        if not pd.api.types.is_numeric_dtype(y):
            raise TypeError(f"Column '{y.name}' must be numeric, got {y.dtype}")
        
        # Calculates encoding map
        data = pd.concat([X, y], axis=1)
        self.mapping_ = data.groupby(X.name)[y.name].mean().to_dict()

        return self

    def transform(self,X:pd.Series) -> pd.DataFrame:
        """
        Apply mapping from fitting function to ``X`` and returns the encoded version ``X_transformed``
        
        :param X: Original column to be encoded
        :return: Encoded column

        Examples
        X = pd.Series(["red", "blue", "red", "blue", "red"], name='color')
        y = pd.Series([1, 0, 2, 0, 3], name='score')

        encoder = MeanEncoder()
        encoder.fit(X, y)
        X_transformed = encoder.transform(X)
        """
        check_is_fitted(self, 'mapping_')

        unseen_X_categories = set(X.unique()) - set(self.mapping_.keys())

        if unseen_X_categories:
            # Returns only NaNs if new values are all "missing" 
            if all(pd.isna(val) for val in unseen_X_categories):
                return pd.DataFrame(np.nan, index=X.index, columns=[X.name])
            # Raises error otherwise
            else:
                raise ValueError(f"Column to be encoded has unseen values: {unseen_X_categories}")
        
        # Apply encoding map to X
        X_transformed = X.map(self.mapping_)

        return X_transformed.to_frame()
    

