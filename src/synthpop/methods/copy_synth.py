from sklearn.base import TransformerMixin,BaseEstimator
from abc import ABC, abstractmethod,ABCMeta
import pandas as pd
from typing import Self

class CopyMethod(base_synth.BaseSynthesisStartMethod):
    """
    Synthesis method that samples from the target column. 
    """

    def __init__(self,random_state: RandomState | None | int = None):
        self.random_state = random_state #mandated by scikit-learn developer guide

    def fit_for_first_column(self,y:pd.Series) -> Self:
        """
        Stores the entire column in this object

        :param y: The column to be copied.
        """
        return self
    
    def generate(self,n: int | None = None):
        """
        Returns an exact copy of the training data.

        :param n: Must be either None or the number of rows in the column used for training. Raises an exception otherwise. 
        """
        return pd.Series()