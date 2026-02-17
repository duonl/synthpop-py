from sklearn.base import OneToOneFeatureMixin, TransformerMixin, BaseEstimator
import pandas as pd
from typing import Self

class PCAEncoder(TransformerMixin, BaseEstimator): 
    """
    Transforms categorical data to one or more numeric columns.


    :param PCA_threshold: maximum number of columns used to encode the feature. explained_variance_threshold has precedence above PCA_threshold.
    :param explained_variance: parameter indicating how much of the total variance should be explained by the principle components. A value of 1 returns all principle components.
    """
    def __init__(self, PCA_threshold:int = 30, explained_variance:float = 0.95):
        pass

    def fit(self,X:pd.Series, y: pd.Series) -> Self:
        """
        Determines for each level of ``X`` the corresponding numerical values to encode them with. 

        :param X: The categorical feature that is to be encoded.
        :param y: The target used to encode the feature.

        :return: fitted encoder.
        """
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
        self.mapping_ = {}
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
        data = pd.concat([X, y], axis=1)
        self.mapping_ = data.groupby(X.name)[y.name].mean().to_dict()

        return self

    def transform(self,X:pd.Series) -> pd.DataFrame:
        return pd.DataFrame()
    
    def get_feature_names_out(self):
        pass
    
X = pd.Series(["red", "blue", "red", "blue", "red"], name='color')
y = pd.Series([1, 0, 2, 0, 3], name='score')

encoder = MeanEncoder()
encoder.fit(X, y)