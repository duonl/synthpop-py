from typing import Self
import pandas as pd

from numpy.random import RandomState

from synthpop.methods.base_synth import BaseSynthMethod

class SampleMethod(BaseSynthMethod):
    """
    Synthesis method that samples from the target column. 
    """

    def __init__(self,random_state: RandomState | None | int = None):
        super().__init__()
        self.random_state = random_state #mandated by scikit-learn developer guide

    def fit(self,X:pd.DataFrame | None, y: pd.Series) -> Self:
        """
        Stores the probability distribution of y, including the probability of being None. Also stores the number of rows.
        :param y: the target column of which to store the probability distribution.
        """
        return self
    
    def transform(self, X: pd.DataFrame| None) -> pd.DataFrame:
        """
        Takes a sample of size n with replacement from the probability distribution of the target column.
        If n is None, then the same sample size is used as the training data.

        :param n: number of samples to take. 
        """
        return pd.Series()