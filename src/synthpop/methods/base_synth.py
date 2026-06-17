"""
This module contains the base class for all synthesis methods.
""" 
from abc import abstractmethod, ABCMeta
from typing import Self

from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd


class BaseSynthMethod(TransformerMixin, BaseEstimator, metaclass=ABCMeta):
    """
    Base class for all synthesis methods in synthpop. 
    It ensures all child classes implement a fit and a transform method.

    A synthesis method in synthpop is an algorithm to synthesise a column of a dataset, based on already synthesised columns. 
    Specifically, such method learns a conditional distribution of a target column given one or multiple columns. 
    A synthesis method inheriting from this class should work even if there are no predictors (so only a target)

    Both fit and transform should work for numeric and categorical variables.

    Note for child classes: consider cloning behaviour (https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html)
    """

    def __init__(self) -> None:
        super().__init__()
        # If an estimator is given as a parameter, it should be cloned using the clone() method.
    
    @abstractmethod
    def fit(self, X: pd.DataFrame | None, y: pd.Series) -> Self:
        """
        The `fit` method must learn all parameters required to synthesise the target variable from the provided features.
        It does not modify the input data and does not produce any output.

        There should be an implementation of missing values support in case of missing values in feature columns.
        There should be a binary classifier to predict missing values if the target variable includes missing values.
        
        :param X: Dataset of features, may be heterogeneous
        :param y: Target variable
        :return: A fitted model
        """
        # Using sklearn.utils.validation.validate_data, set the attribute feature_names_in_ to X and y.
        # That method sets the attribute. 
        # For example:
        # from sklearn.utils.validation import validate_data
        # ....
        # X, y = validate_data(self, X, y)
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame | None) -> pd.Series:
        """
        The `transform` method must use the fitted model to generate a synthetic version of the target variable and append it as a new column to the input dataset.

        There should be an implementation of missing values support in case of missing values in feature columns.

        Calling `transform` before `fit` raises an error.

        :param X: Input dataset, may be heterogeneous
        :return: Synthetic column.
        """
        return pd.Series()
    
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        return 0.0
    
    # def get_params(self, deep: bool = True) -> dict:
    #     return super().get_params(deep)
    
    # def set_params(self, **params) -> Self:
    #     return super().set_params(**params)
    
    @abstractmethod
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names and category names for transformation. This method is required to support the `set_output(transform="pandas")` API in scikit-learn.
        
        See https://scikit-learn.org/stable/developers/develop.html#developer-api-for-set-output 
        and SLEP018 (https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep018/proposal.html) and SLEP007 (https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep007/proposal.html)
        
        :param input_features: array-like of str or None. Input feature names. If None, the feature names seen during `fit` are used.
        """
        
        pass
