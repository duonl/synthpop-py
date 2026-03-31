"""
This module contains classes for different strategies for handling missing (None) values in the target during synthesis. 
"""
from abc import abstractmethod,ABCMeta
import pandas as pd
from sklearn.base import TransformerMixin

from synthpop.data_processing.encoders import MeanEncoder
import numpy.typing as npt
import numpy as np
import copy

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

    def __sklearn_clone__(self):
        return copy.copy(self)


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

    :param missing_marker: The value to replace missing values with.

    Examples
    ========
    >>> import numpy as np
    >>> from synthpop.data_processing.missing_value_handling import ReplaceNoneWithValue
    >>> X = np.array(["a","b","c","c"])
    >>> y = np.array(["x","y",None,"z"])
    >>> replace_missing = ReplaceNoneWithValue()
    >>> x_res,y_res = replace_missing.prepare_data_for_fit(X,y)
    >>> x_res
    array(['a', 'b', 'c', 'c'], dtype='<U1')
    >>> y_res
    array(['x', 'y', 'N.a.N.', 'z'], dtype='<U6')
    >>> replace_missing.post_synth_transform(x_res,y_res)
    array(['x', 'y', None, 'z'], dtype=object)
    """

    def __init__(self,missing_marker:str = "N.a.N."):
        super().__init__()
        self.missing_replacement = missing_marker

    def _copy_y(self,y):
        # The result of np.copy is always a numpy array. If y is a pandas series, the expected output is a pandas series.
        # So if y is a pandas series (or not numpy array), it is better to use copy.copy.
        # If y is a numpy array, it is faster to use np.copy.
        if isinstance(y,pd.Series):
            y_arr = y.copy(deep=True) 
        else:
            y_arr = np.asarray(y,dtype=np.object_,copy=True)

        return y_arr
    
    def prepare_data_for_fit(self,X:npt.ArrayLike, y:npt.ArrayLike)-> tuple[npt.ArrayLike,npt.ArrayLike]:
        """
        Replaces missing values in the target with "N.a.N."

        :param X: the features for the target. 
        :param y: the target.

        :return: a tuple ``(X,y)``. Leaves ``X`` unchanged. Replaces missing values in the target with "N.a.N.". Makes a copy of ``y``. 
        """
        y_arr = self._copy_y(y)
        missing_mask = pd.isna(y_arr)
        if np.any(np.equal(y_arr[~missing_mask], self.missing_replacement)) and missing_mask.any():
            raise ValueError(f"the value {self.missing_replacement} already occurs in y")

        
        y_arr[missing_mask] = self.missing_replacement
        return(X,y_arr.astype(np.str_))

    def post_synth_transform(self,X:npt.ArrayLike,y:npt.ArrayLike) -> npt.ArrayLike:
        """
        Replaces "N.a.N." with missing values.

        :param X: the features for the target.
        :param y: the target

        :return:  The synthesised target with missing values.
        """ 
        y_arr = self._copy_y(y)
        mask = np.equal(y_arr ,self.missing_replacement)
        if not mask.any():
            return np.array(y) if not (isinstance(y,pd.Series) or isinstance(y,np.ndarray)) else y
        
        y_arr = y_arr.astype(np.object_)
        y_arr[mask] = None
        return y_arr
