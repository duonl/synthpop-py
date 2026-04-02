"""
This module contains classes for different strategies for handling missing (None) values in the target during synthesis. 
"""
from abc import abstractmethod,ABCMeta
from sklearn.base import TransformerMixin, clone
from sklearn.tree import DecisionTreeClassifier
from synthpop.data_processing.encoders import MeanEncoder
from synthpop.methods.tree_utils import LeafNodeSampler
import numpy.typing as npt
import numpy as np
import pandas as pd
from typing import Dict
import copy

class BaseMissingValueHandler(metaclass=ABCMeta):
    """
    Base class for different strategies to handle missing values in the target variable of a synthesis.
    """

    @abstractmethod
    def prepare_data_for_fit(self, X: Dict[str, npt.ArrayLike], y: npt.ArrayLike) -> tuple[Dict[str, npt.NDArray], npt.NDArray]:
        """
        Prepare the feature and/or target for fitting.

        :param X: the features of the target. May contain missing values. Implementers should accept both categoric and numeric data, and should accept missing values here.
        :param y: the target column. May contain missing values. Implementers do not need to accept both categorical and numeric targets, but should accept one of them.
        :return: a tuple (X,y) of data ready to be further processed and used for fitting a model. the second item of the tuple (y) may not contain missing values.
        """
        pass

    @abstractmethod
    def post_synth_transform(self, X: Dict[str, npt.ArrayLike], y: npt.ArrayLike) -> npt.NDArray:
        """
        Process synthesised data to include missing values.

        :param X: The features of the target. Implementers should accept both categoric and numeric data, and should accept missing values here.
        :param y: the target column, should not contain missing values. 

        :return:  The synthesised target with missing values.
        """ 
        pass

    @abstractmethod
    def clone(self):
        """
        Create a new instance of the  with the same configuration.

        The method only copies initialisation parameters (from __init__) and does not copy
        any fitted state.
        Similar to sklearn's `clone()`.

        :return: A new, unfitted instance with the same __init__ parameters

        """
        pass

class MissingValuePredictor(BaseMissingValueHandler):
    """
    Use a decision tree to predict which values are missing.

    Learns:
        P(z = 1 | X)

    where:
        z = 1 → missing
        z = 0 → observed
    
    Then samples:
        z ~ Bernoulli(P(z=1|x))

    :param encoding: Default is a :doc: `MeanEncoder` <synthpop.data_processing.encoders.MeanEncoder>.
    :param tree: Decision tree classifier. Default is [DecisionTreeClassifier(min_samples_leaf=5)](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
    :param tree_sampler: Leaf node sampler. Default is :py:meth:LeafNodeSampler
        .
    """

    def __init__(self, encoding: TransformerMixin | None = None, 
                 tree: DecisionTreeClassifier | None = None,
                 tree_sampler: LeafNodeSampler | None = None):
        super().__init__()
        self.encoding = encoding
        self.tree = tree
        self.tree_sampler = tree_sampler

    @classmethod
    def _validate_X_y_dict(cls, X: Dict[str, npt.ArrayLike], y: npt.ArrayLike) -> tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Minimal validation for dict-based tabular data.
        """
        if not isinstance(X, dict):
            raise TypeError(f"X must be a dict[str, array-like], got {type(X)}.")
        if len(X) == 0:
            raise ValueError("X must contain at least one feature.")

        X_out = {}
        lengths = set()

        for key, col in X.items():
            arr = np.asarray(col)
            if arr.ndim != 1:
                raise ValueError(f"Column '{key}' must be 1-dimensional, got shape {arr.shape}.")
            if len(arr) == 0:
                raise ValueError(f"Column '{key}' is empty.")
            X_out[key] = arr
            lengths.add(len(arr))

        if len(lengths) != 1:
            raise ValueError(f"All columns in X must have the same length, got lengths {lengths}.")

        n_samples = lengths.pop()

        y_out = np.asarray(y)

        if y_out.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y_out.shape}.")
        if len(y_out) != n_samples:
            raise ValueError(f"X and y must have the same number of samples. Got {n_samples} and {len(y_out)}.")

        return X_out, y_out
    
    def _build_X_matrix(self, X, fit=False):
        feature_order = getattr(self, "feature_order_", None)
        if feature_order is None:
            feature_order = list(X.keys())
            self.feature_order_ = feature_order

        X_encoded = []
        for col in feature_order:
            values = np.asarray(X[col])
            encoder = self.encoders_.get(col, None)
            if encoder is None:
                encoded = values
            else:
                encoded = encoder.transform(values)
            
            X_encoded.append(encoded.reshape(-1, 1))

        return np.column_stack(X_encoded)
    
    def prepare_data_for_fit(self, X: Dict[str, npt.ArrayLike], y: npt.ArrayLike) -> tuple[Dict[str, npt.NDArray], npt.NDArray]:
        """
        Trains a decision tree to predict when y is missing. Removes rows from both `X` and `y` when `y` is missing.

        First constructs the missingness indicator (`z`) and then applies mean encoding. Finally, a binary decision tree classifier is trained.

        :param X: the features of the target. 
        :param y: the target column.

        :return: a tuple (X, y) of the original data excluding the rows where `y` is missing.
        """
        # input validation
        X_val, y_val = self._validate_X_y_dict(X, y)

        self.feature_order_ = list(X_val.keys())

        self.tree_ = clone(self.tree) if self.tree else DecisionTreeClassifier(min_samples_leaf= 5)
        self.tree_sampler_ = self.tree_sampler.clone() if self.tree_sampler else LeafNodeSampler()
        
        # implementation
        z = pd.isna(y_val)

        self._all_missing = np.all(z)
        self._no_missing = not np.any(z)

        self.encoders_ = {}

        for col in self.feature_order_:
            values = np.asarray(X_val[col])

            if values.dtype.kind in ("O", "U", "S"):
                encoder = clone(self.encoding) if self.encoding else MeanEncoder()
                encoder.fit(values, z)
                self.encoders_[col] = encoder
            else:
                self.encoders_[col] = None
        
        X_matrix = self._build_X_matrix(X_val)    

        if not self._all_missing and not self._no_missing:
            self.tree_.fit(X_matrix, z)
            leaf_ids = self.tree_.apply(X_matrix)
            self.tree_sampler_.fit_sampler(leaf_ids, z)
        else:
            self.tree_ = None
            self.tree_sampler_ = None
        
        # remove the missing value rows for return
        mask = ~pd.isna(y_val)

        X_filtered = {col: X_val[col][mask] for col in X_val}
        y_filtered = y_val[mask]

        return X_filtered, y_filtered

    def post_synth_transform(self, X: Dict[str, npt.ArrayLike], y: npt.ArrayLike) -> npt.NDArray:
        """
        Uses a decision tree to determine when y should be missing.

        :param X: the features for the target.
        :param y: the target column.

        :return:  The synthesised target with missing values.
        """ 

        # input validation
        if not hasattr(self, "tree_") or not hasattr(self, "tree_sampler_") or not hasattr(self, "encoders_") or not hasattr(self, "feature_order_"):
            raise AttributeError("MissingValuePredictor is not fitted. Call `prepare_data_for_fit` first.")

        X, y = self._validate_X_y_dict(X, y)
        n = len(y)

        # implementation
        if self._all_missing:
            return np.full(n, np.nan)
        if self._no_missing:
            return y
        
        X_matrix = self._build_X_matrix(X)

        leaf_ids = self.tree_.apply(X_matrix)
        missing_mask = self.tree_sampler_.sample_from_leaves(leaf_ids)
        missing_mask = np.asarray(missing_mask).astype(bool)

        y_out = y.astype(float).copy()
        y_out[missing_mask] = np.nan

        return y_out
    
    def clone(self):
        """
        Create a new instance of the missing value predictor with the same configuration.

        The method only copies initialisation parameters and does not copy
        any fitted state. Similar to sklearn's `clone()`.

        :return: A new, unfitted instance of `MissingValuePredictor()` with the 
            same `encoding`, `tree` and `tree_sampler` setting.
        
        Examples:
        -----
        >>> MissingValuePredictor().clone()
        """
        return self.__class__(encoding=self.encoding, tree=self.tree, tree_sampler=self.tree_sampler)
   
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

    def __init__(self, missing_marker:str = "N.a.N."):
        super().__init__()
        self.missing_replacement = missing_marker

    def _copy_y(self, y):
        # The result of np.copy is always a numpy array. If y is a pandas series, the expected output is a pandas series.
        # So if y is a pandas series (or not numpy array), it is better to use copy.copy.
        # If y is a numpy array, it is faster to use np.copy.
        if isinstance(y,pd.Series):
            y_arr = y.copy(deep=True) 
        else:
            y_arr = np.asarray(y,dtype=np.object_,copy=True)

        return y_arr
    
    def prepare_data_for_fit(self, X: npt.ArrayLike, y: npt.ArrayLike)-> tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Replaces missing values in the target with "N.a.N."

        :param X: the features of the target. 
        :param y: the target column.

        :return: a tuple `(X,y)`. Leaves `X` unchanged. Replaces missing values in the target with "N.a.N.". Makes a copy of `y`. 
        """
        y_arr = self._copy_y(y)
        missing_mask = pd.isna(y_arr)
        if np.any(np.equal(y_arr[~missing_mask], self.missing_replacement)) and missing_mask.any():
            raise ValueError(f"the value {self.missing_replacement} already occurs in y")

        
        y_arr[missing_mask] = self.missing_replacement
        return(X,y_arr.astype(np.str_))

    def post_synth_transform(self, X: npt.ArrayLike, y: npt.ArrayLike) -> npt.ArrayLike:
        """
        Replaces "N.a.N." with missing values.

        :param X: the features of the target.
        :param y: the target column.

        :return:  The synthesised target with missing values.
        """ 
        y_arr = self._copy_y(y)
        mask = np.equal(y_arr ,self.missing_replacement)
        if not mask.any():
            return np.array(y) if not (isinstance(y,pd.Series) or isinstance(y,np.ndarray)) else y
        
        y_arr = y_arr.astype(np.object_)
        y_arr[mask] = None
        return y_arr
    
    def clone(self):
        """
        Create a new instance of ReplaceNoneWithValue with the same configuration.

        The method only copies initialisation parameters and does not copy
        any fitted state. Similar to sklearn's `clone()`.

        Note: `ReplaceNoneWithValue` does not have learned attributes.

        :return: A new, unfitted instance of `ReplaceNoneWithValue()` with the 
            same `missing_marker` setting.
        
        Examples:
        -----
        >>> ReplaceNoneWithValue().clone()
        """
        return self.__class__(missing_marker = self.missing_replacement)
