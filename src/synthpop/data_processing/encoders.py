"""
This module contains classes to encode categorical data to numeric data. 

"""
from typing import Self
from sklearn.base import OneToOneFeatureMixin, TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import numpy.typing as npt



class PCAEncoder(TransformerMixin, BaseEstimator): 
    """
    Transforms categorical data to one or more numeric columns.


    :param PCA_threshold: maximum number of columns used to encode the feature. explained_variance_threshold has precedence above PCA_threshold.
    :param explained_variance: parameter indicating how much of the total variance should be explained by the principle components. A value of 1 returns all principle components.
    """
    def __init__(self, pca_threshold:int = 30, minimum_explained_variance:float = 0.95,_pca_transform:PCA = PCA()):
        self._pca_transform = _pca_transform #TODO: parameters??
        self.pca_threshold= pca_threshold
        self.minimum_explained_variance= minimum_explained_variance
        #self.set_output(transform="pandas")
        pass

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required=True
        tags.target_tags.one_d_labels = True
        tags.target_tags.single_output= True

        tags.input_tags.categorical = True
        tags.input_tags.two_d_array = False
        tags.input_tags.one_d_array = True
        tags.input_tags.allow_nan= True
        tags.input_tags.string = True
        
        tags.estimator_type = "transformer"
        tags.array_api_support = True
        return tags

    def fit(self,X:npt.ArrayLike, y: npt.ArrayLike) -> Self:
        
        self.n_features_in_ = 1

        if X.shape[0] == 0 and y.shape[0]==0:
            self.mapping_ = {}
            if isinstance(X,pd.Series):
                self.feature_names_in_ = [X.name]

            return self

        X_val,y_val = validate_data(self,X=X,y=y, validate_separately = (
            dict(ensure_2d=False,dtype=["str","object"],ensure_all_finite="allow-nan")
            ,dict(ensure_2d=False,dtype=["str","object"],ensure_all_finite="allow-nan") 
            ))
        
        if isinstance(X,pd.Series):
            self.feature_names_in_ = [X.name]
        if X_val.ndim != 1:
            raise ValueError("X should by 1D")
        if y_val.ndim != 1:
            raise ValueError("Y should by 1D")
        
        #The alternative to using pandas here is either use scipy or DIY. 
        contingency_table = pd.crosstab(X_val,y_val,)

        pca_result = self._pca_transform.fit_transform(X=contingency_table.to_numpy(),y=None)
        if isinstance(pca_result,pd.DataFrame):
            pca_result = pca_result.to_numpy()

        self.n_features_out_ = pca_result.shape[1]

        value_mapping = {contingency_table.index[i]: pca_result[i] for i in range(pca_result.shape[0])}

        #The alternative to using pandas here is either use scipy or DIY. 
        missing_contingency_table = pd.crosstab(X_val,[v is None or v is pd.NA or v is np.nan for v in y_val])
        x_such_that_y_is_always_missing = missing_contingency_table[missing_contingency_table[False]==0].index
        mapping_for_missing = {k:[None]*self.n_features_out_ for k in x_such_that_y_is_always_missing}

        self.mapping_ = value_mapping | mapping_for_missing

        return self

    def transform(self,X:npt.ArrayLike) -> npt.ArrayLike:
        """
        replaces each level of ``X`` with the numerical values determined in :py:meth:`fit`

        :param X: the feature to be encoded.
        """
        mapping_including_missing = self.mapping_ | {None:[None]*self.n_features_out_}

        return np.array(
            [
                mapping_including_missing[v] for v in X
            ],
        )
    
    def get_feature_names_out(self,input_features=None):

        if input_features is None:
            return [self.feature_names_in_[0]+f"_pca{i}" for i in range(self.n_features_out_)]
        if input_features != self.feature_names_in_:
            raise ValueError(f"input_features is not feature_names_in_. Expected: {self.feature_names_in_}, actual: {input_features}")
        return [self.feature_names_in_[0]+f"_pca{i}" for i in range(self.n_features_out_)]
    
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
    

