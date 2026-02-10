from abc import ABC, abstractmethod,ABCMeta
import pandas as pd
from sklearn.base import TransformerMixin

from synthpop.data_processing.Encoders import MeanEncoder

class BaseMissingValueHandler(metaclass=ABCMeta):
    """
    Base class for different strategies to handle missing values in the target variable.
    """

    @abstractmethod
    def prepare_data_for_fit(self,X:pd.DataFrame,y:pd.Series) -> tuple[pd.DataFrame,pd.Series]:
        """
        Prepare the feature and/or target for fitting.
        :param X: the features for the target. Implementors should accept both categoric and nummeric data, and should accept missing values here.
        :param y: the target. May contain missing values. Implementors do not need to accept both categorical and nummeric targets, but should accept one of them.

        :return: a tuple (X,y) of data ready to be further processed and used for fitting a model. the second item of the tuple (y) may not contain missing values.
        X may contain missing values.
        """
        pass

    @abstractmethod
    def post_synth_transform(self,x:pd.DataFrame,y:pd.Series) -> pd.Series:
        """
        Process synthesised data to include missing values.
        :param X: the features for the target. Implementors should accept both categoric and nummeric data, and should accept missing values here.
        :param y: the target, should not contain missing values. 

        :return:  The synthesised target with missing values.
        """ 
        pass
    

class MissingValuePredictor(BaseMissingValueHandler):

    def __init__(self,encoding: TransformerMixin = MeanEncoder()):
        super().__init__()
    
    def prepare_data_for_fit(self,X:pd.Series, y:pd.Series)-> tuple[pd.DataFrame,pd.Series]:
        """
        Trains a decision tree to predict when y is missing. Removes rows from both X and y when y is missing.
        :param X: the features for the target. 
        :param y: the target.

        :return: a tuple (X,y). Original data minus the rows where the y is missing.
        """
        
        return(pd.DataFrame(),pd.Series())

    def post_synth_transform(self,X:pd.DataFrame,y:pd.Series) -> pd.Series:
        """
        Uses a decision tree to determine when y should be missing.
        :param X: the features for the target.
        :param y: the target

        :return:  The synthesised target with missing values.
        """ 
        return pd.Series()
    
class ReplaceNoneWithValue(BaseMissingValueHandler):
    
    def prepare_data_for_fit(self,X:pd.Series, y:pd.Series)-> tuple[pd.DataFrame,pd.Series]:
        """
        Replaces missing values in the target with "N.a.N"
        :param X: the features for the target. 
        :param y: the target.

        :return: a tuple (X,y). Leaves X unchanged. Replaces missing values in the target with "N.a.N"
        """
        
        return(pd.DataFrame(),pd.Series())

    def post_synth_transform(self,X:pd.DataFrame,y:pd.Series) -> pd.Series:
        """
        Replaces "N.a.N" with missing values.
        :param X: the features for the target.
        :param y: the target

        :return:  The synthesised target with missing values.
        """ 
        return pd.Series()