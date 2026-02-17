from sklearn.base import TransformerMixin,BaseEstimator
from abc import ABC, abstractmethod,ABCMeta
import pandas as pd
from typing import Self

class SampleMethod(base_synth.BaseSynthesisStartMethod):
    """
    Synthesis method that samples from the target column. 
    """

    def __init__(self,random_state: RandomState | None | int = None):
        self.random_state = random_state #mandated by scikit-learn developer guide

    def fit_for_first_column(self,y:pd.Series) -> Self:
        """
        Stores the probability distribution of y, including the probability of being None. Also stores the number of rows.
        :param y: the target column of which to store the probability distribution.
        """
        return self
    
    def generate(self,n: int | None = None):
        """
        Takes a sample of size n with replacement from the probability distribution of the target column.
        If n is None, then the same sample size is used as the training data.

        :param n: number of samples to take. 
        """
        return pd.Series()