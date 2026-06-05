"""
This module contains classes for different strategies for handling missing values (`np.nan`) in the target during synthesis.
"""
from abc import ABCMeta, abstractmethod
from typing import Dict, Self

import pandas as pd
import numpy as np
import numpy.typing as npt
from sklearn.base import TransformerMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import NotFittedError

from synthpop.data_processing.encoders import MeanEncoder
from synthpop.methods.tree_utils import LeafNodeSampler, build_feature_matrix
from synthpop.utils import validate_2d_dict, validate_1d_target

class BaseMissingValueHandler(metaclass=ABCMeta):
    """
    Base class for different strategies to handle missing values in the target variable of a synthesis.
    """

    @abstractmethod
    def prepare_data_for_fit(self, X: Dict[str, npt.NDArray], y: npt.NDArray) -> tuple[Dict[str, npt.NDArray], npt.NDArray]:
        """
        Prepare the feature and/or target for fitting.

        :param X: the features of the target. May contain missing values. Implementers should accept both categoric and numeric data, and should accept missing values here.
        :param y: the target column. May contain missing values. Implementers do not need to accept both categorical and numeric targets, but should accept one of them.
        :return: a tuple (X,y) of data ready to be further processed and used for fitting a model. the second item of the tuple (y) may not contain missing values.
        """
        raise NotImplementedError

    @abstractmethod
    def post_synth_transform(self, X: Dict[str, npt.NDArray], y: npt.NDArray) -> npt.NDArray:
        """
        Process synthesised data to include missing values.

        :param X: The features of the target. Implementers should accept both categoric and numeric data, and should accept missing values here.
        :param y: the target column, should not contain missing values. 

        :return:  The synthesised target with missing values.
        """ 
        raise NotImplementedError

    @abstractmethod
    def clone(self) -> Self:
        """
        Create a new instance of the  with the same configuration.

        The method only copies initialisation parameters (from __init__) and does not copy
        any fitted state.
        Similar to sklearn's `clone()`.

        :return: A new, unfitted instance with the same __init__ parameters

        """
        raise NotImplementedError


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

    :param encoder: Default is a :class:`~synthpop.data_processing.encoders.MeanEncoder`. The encoder must have a `fit_transform` function.
    :param tree: Decision tree classifier. Default is `DecisionTreeClassifier(min_samples_leaf=5) <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html>`_.
    :param tree_sampler: Leaf node sampler. Default is :py:meth:LeafNodeSampler.

    Examples
    --------
    >>> from synthpop.data_processing.missing_value_handling import MissingValuePredictor
    >>> import numpy as np
    >>>
    >>> X = {"num": np.array([25, 30, 35, 40]), "cat": np.array(["A", "B", "A", "B"], dtype=np.dtypes.StringDType(na_object=np.nan))}
    >>> y = np.array([1.0, np.nan, 3.0, np.nan])
    >>>
    >>> mvp = MissingValuePredictor()
    >>> X_clean, y_clean = mvp.prepare_data_for_fit(X, y)
    >>> X_clean
    {'num': array([[25], [35]]), 'cat': array([['A'], ['A']], dtype=StringDType(na_object=nan))}
    >>> y_clean
    array([1., 3.])
    >>>
    >>> # simulate synthetic generation step
    >>> y_synth = np.array([10, 20, 30, 40])
    >>> y_final = mvp.post_synth_transform(X, y_synth)
    >>> y_final
    array([10., nan, nan, 40.])

    """

    def __init__(self, encoder: TransformerMixin | None = None, 
                 tree: DecisionTreeClassifier | None = None,
                 tree_sampler: LeafNodeSampler | None = None) -> None:
        super().__init__()
        self.encoder = encoder
        self.tree = tree
        self.tree_sampler = tree_sampler
    
    def prepare_data_for_fit(self, X: Dict[str, npt.NDArray], y: npt.NDArray) -> tuple[Dict[str, npt.NDArray], npt.NDArray]:
        """
        Trains a decision tree to predict when y is missing. Removes rows from both `X` and `y` when `y` is missing.

        First constructs the missingness indicator (`z`) and then applies mean encoding. Finally, a binary decision tree classifier is trained.

        :param X: the features of the target. 
        :param y: the target column.

        :return: a tuple (X, y) of the original data excluding the rows where `y` is missing.

        Examples
        --------
        >>> from synthpop.data_processing.missing_value_handling import MissingValuePredictor
        >>> import numpy as np
        >>>
        >>> X = {"num": np.array([25, 30, 35, 40]), "cat": np.array(["A", "B", "A", "B"], dtype=np.dtypes.StringDType(na_object=np.nan))}
        >>> y = np.array([1.0, np.nan, 3.0, np.nan])
        >>>
        >>> mvp = MissingValuePredictor()
        >>> X_clean, y_clean = mvp.prepare_data_for_fit(X, y)
        >>> X_clean
        {'num': array([[25], [35]]), 'cat': array([['A'], ['A']], dtype=StringDType(na_object=nan))}
        >>> y_clean
        array([1., 3.])

        """
        # input validation
        X_val, n_samples = validate_2d_dict(X)
        y_val = validate_1d_target(y, n_samples)

        self.feature_order_ = list(X_val.keys())

        self.tree_ = clone(self.tree) if self.tree else DecisionTreeClassifier(min_samples_leaf=5)
        self.tree_sampler_ = self.tree_sampler.clone() if self.tree_sampler else LeafNodeSampler()
        
        # implementation
        z = pd.isna(y_val)

        self._all_missing = np.all(z)
        self._none_missing = not np.any(z)

        self.encoders_ = {}
        X_encoded = {}

        for col in self.feature_order_:
            values = X_val[col]

            if not pd.api.types.is_numeric_dtype(values.dtype):
                encoder = clone(self.encoder) if self.encoder else MeanEncoder()
                transformed = encoder.fit_transform(values, z)
                self.encoders_[col] = encoder
            else:
                transformed = values

            X_encoded[col] = np.asarray(transformed)

        X_matrix = build_feature_matrix(X_encoded, feature_order=self.feature_order_)    

        if not self._all_missing and not self._none_missing:
            self.tree_.fit(X_matrix, z)
            leaf_ids = self.tree_.apply(X_matrix)
            self.tree_sampler_.fit_sampler(leaf_ids, z)
        else:   #leave tree_ and tree_sampler_ unfitted
            pass
        
        # remove the missing value rows for return
        mask = ~pd.isna(y_val)

        X_filtered = {col: values[mask] for col, values in X_val.items()}
        y_filtered = y_val[mask]

        return X_filtered, y_filtered

    def post_synth_transform(self, X: Dict[str, npt.NDArray], y: npt.NDArray) -> npt.NDArray:
        """
        Uses a decision tree to determine when y should be missing.

        :param X: the features for the target.
        :param y: the target column.

        :return:  The synthesised target with missing values.

        Examples
        --------
        >>> from synthpop.data_processing.missing_value_handling import MissingValuePredictor
        >>> import numpy as np
        >>>
        >>> X = {"num": np.array([25, 30, 35, 40]), "cat": np.array(["A", "B", "A", "B"], dtype=np.dtypes.StringDType(na_object=np.nan))}
        >>> y = np.array([1.0, np.nan, 3.0, np.nan])
        >>>
        >>> mvp = MissingValuePredictor()
        >>> X_clean, y_clean = mvp.prepare_data_for_fit(X, y)
        >>>
        >>> #simulate synthetic generation step
        >>> y_synth = np.array([10, 20, 30, 40])
        >>> y_final = mvp.post_synth_transform(X, y_synth)
        >>> y_final
        array([10., nan, nan, 40.])

        """ 
        # input validation
        if (not hasattr(self, "tree_")
            or not hasattr(self, "tree_sampler_")
            or not hasattr(self, "encoders_")
            or not hasattr(self, "feature_order_")):
            raise NotFittedError("MissingValuePredictor is not fitted. Call `prepare_data_for_fit` first.")

        # implementation
        if self._all_missing:
            return np.full(len(y), np.nan)
        if self._none_missing:
            return y
    

        X_val, n_samples = validate_2d_dict(X)
        y_val = validate_1d_target(y, n_samples)
        
        X_encoded = {}
        
        for col in self.feature_order_:
            values = X_val[col]
            
            if col in self.encoders_:
                transformed = self.encoders_[col].transform(values)
            else:
                transformed = values
            
            X_encoded[col] = np.asarray(transformed)

        X_matrix = build_feature_matrix(X_encoded, feature_order=self.feature_order_)  

        leaf_ids = self.tree_.apply(X_matrix)
        missing_mask = self.tree_sampler_.sample_from_leaves(leaf_ids)
        missing_mask = np.asarray(missing_mask).astype(bool)

        y_out = y_val.astype(float).copy()
        y_out[missing_mask] = np.nan

        return y_out
    
    def clone(self) -> Self:
        """
        Create a new instance of the missing value predictor with the same configuration.

        The method only copies initialisation parameters and does not copy
        any fitted state. Similar to sklearn's `clone()`.

        :return: A new, unfitted instance of `MissingValuePredictor()` with the 
            same `encoder`, `tree` and `tree_sampler` setting.
        
        Examples
        --------
        >>> MissingValuePredictor().clone()
        """
        return self.__class__(encoder=self.encoder, tree=self.tree, tree_sampler=self.tree_sampler)
   
class ReplaceNoneWithValue(BaseMissingValueHandler):
    """
    Replace missing values by a specified value, and remove after synthesis.

    :param missing_marker: The value to replace missing values with.

    Examples
    --------
    >>> import numpy as np
    >>> from synthpop.data_processing.missing_value_handling import ReplaceNoneWithValue
    >>> X = np.array(["a","b","c","c"], dtype=np.dtypes.StringDType(na_object=np.nan))
    >>> y = np.array(["x","y",np.nan,"z"], dtype=np.dtypes.StringDType(na_object=np.nan))
    >>> replace_missing = ReplaceNoneWithValue()
    >>> x_res,y_res = replace_missing.prepare_data_for_fit(X,y)
    >>> x_res
    array(['a', 'b', 'c', 'c'], dtype=StringDType(na_object=nan))
    >>> y_res
    array(['x', 'y', 'N.a.N.', 'z'], dtype=StringDType(na_object=nan))
    >>> replace_missing.post_synth_transform(x_res, y_res)
    array(['x', 'y', nan, 'z'], dtype=StringDType(na_object=nan))
    """

    def __init__(self, missing_marker: str = "N.a.N.") -> None:
        super().__init__()
        self.missing_marker = missing_marker
    
    def prepare_data_for_fit(self, X: Dict[str, npt.NDArray], y: npt.NDArray) -> tuple[Dict[str, npt.NDArray], npt.NDArray]:
        """
        Replaces missing values in the target with "N.a.N."

        :param X: the features of the target. 
        :param y: the target column.

        :return: a tuple `(X,y)`. Leaves `X` unchanged. Replaces missing values in the target with "N.a.N.". Makes a copy of `y`. 
        """

        n_samples = X[next(iter(X))].shape[0]
        y_val = validate_1d_target(y.copy(), n_samples)

        missing_mask = pd.isna(y_val)

        if np.any(y_val[~missing_mask] == self.missing_marker):
            raise ValueError(f"the value {self.missing_marker} already occurs in y.")

        y_val[missing_mask] = self.missing_marker

        return X, y_val

    def post_synth_transform(self, X: Dict[str, npt.NDArray], y: npt.NDArray) -> npt.NDArray:
        """
        Replaces "N.a.N." with missing values.

        :param X: the features of the target.
        :param y: the target column.

        :return:  The synthesised target with missing values.
        """ 

        y_val = validate_1d_target(y.copy(), None)

        y_val[y_val == self.missing_marker] = np.nan

        return y_val
    
    def clone(self) -> Self:
        """
        Create a new instance of ReplaceNoneWithValue with the same configuration.

        The method only copies initialisation parameters and does not copy
        any fitted state. Similar to sklearn's `clone()`.

        Note: `ReplaceNoneWithValue` does not have learned attributes.

        :return: A new, unfitted instance of `ReplaceNoneWithValue()` with the 
            same `missing_marker` setting.
        
        Examples
        --------
        >>> ReplaceNoneWithValue().clone()
        """
        return self.__class__(missing_marker=self.missing_marker)
