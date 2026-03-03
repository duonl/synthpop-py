"""
This module contains classes for different strategies for handling missing (None) values in the target during synthesis. 
"""
from abc import abstractmethod,ABCMeta
import pandas as pd
from sklearn.base import TransformerMixin

from synthpop.data_processing.encoders import MeanEncoder
import numpy.typing as npt

class BaseMissingValueHandler(metaclass=ABCMeta):
    """
    Base class for different strategies to handle missing values in the target variable of a synthesis.
    """

    @abstractmethod
    def prepare_data_for_fit(self,X:pd.DataFrame,y:pd.Series) -> tuple[pd.DataFrame,pd.Series]:
        """
        Prepare the feature and/or target for fitting.
        :param X: the features for the target. Implementers should accept both categoric and numeric data, and should accept missing values here.
        :param y: the target. May contain missing values. Implementers do not need to accept both categorical and numeric targets, but should accept one of them.
        :return: a tuple (X,y) of data ready to be further processed and used for fitting a model. the second item of the tuple (y) may not contain missing values.
        X may contain missing values.
        """
        pass

    @abstractmethod
    def post_synth_transform(self,X:pd.DataFrame,y:pd.Series) -> pd.Series:
        """
        Process synthesised data to include missing values.
        :param X: the features for the target. Implementers should accept both categoric and numeric data, and should accept missing values here.
        :param y: the target, should not contain missing values. 

        :return:  The synthesised target with missing values.
        """ 
        pass


class MissingValuePredictor(BaseMissingValueHandler):
    """
    Use a decision tree to predict which values are missing.
    """

    def __init__(self,encoding: TransformerMixin = MeanEncoder()):
        super().__init__()
    
    def prepare_data_for_fit(self,X:npt.ArrayLike, y:npt.ArrayLike)-> tuple[npt.ArrayLike,npt.ArrayLike]:
        """
        Trains a decision tree to predict when y is missing. Removes rows from both X and y when y is missing.
        :param X: the features for the target. 
        :param y: the target.

        :return: a tuple (X,y). Original data excluding the rows where the y is missing.
        """
        
        return(pd.DataFrame(),pd.Series())

    def post_synth_transform(self,X:npt.ArrayLike,y:npt.ArrayLike) -> npt.ArrayLike:
        """
        Uses a decision tree to determine when y should be missing.
        :param X: the features for the target.
        :param y: the target

        :return:  The synthesised target with missing values.
        """ 
        return pd.Series()
   
class ReplaceNoneWithValue(BaseMissingValueHandler):
    """
    Replace missing values by a specified value, and remove after synthesis.
    """
    
    def prepare_data_for_fit(self,X:npt.ArrayLike, y:npt.ArrayLike)-> tuple[npt.ArrayLike,npt.ArrayLike]:
        """
        Replaces missing values in the target with "N.a.N."
        :param X: the features for the target. 
        :param y: the target.

        :return: a tuple (X,y). Leaves X unchanged. Replaces missing values in the target with "N.a.N."
        """
        
        return(pd.DataFrame(),pd.Series())

    def post_synth_transform(self,X:npt.ArrayLike,y:npt.ArrayLike) -> npt.ArrayLike:
        """
        Replaces "N.a.N." with missing values.
        :param X: the features for the target.
        :param y: the target

        :return:  The synthesised target with missing values.
        """ 
        return pd.Series()