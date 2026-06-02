"""
module for generating synthetic data
"""
from typing import Self
import pandas as pd
from sklearn import clone
from synthpop.methods.base_synth import BaseSynthMethod
from synthpop.methods.cart_synth import CartMethod


class Synthesiser:
    """
    Delegates synthesis tasks to the appropriate synthesis method classes. 

    :param random_seed: seed for randomness.
    :param column_order: list of variable names or list of indexes to define the order in which the columns will be synthesised. Default is the column order of the original dataset.
    :param default_syn_method: BaseSynth object. Synthesis method to apply to each column, except the first one and the ones defined in special_syn_method. Default synthesis method is CartSynth. 
    :param special_syn_method: Dictionary of special synthesis method per variable. If some variables should not follow the default_syn_method, they should be indicated in a dictionary where keys are variable names and values are BaseSynth objects. By default, there is no special synthesis method.
    :param first_column_method: The method for synthesising the first column. This is a special case, since there are no predictors available.
    """
    def __init__(self, random_seed: int,
                 column_order: list[str] | list[int] | None = None,
                 default_syn_method: BaseSynthMethod | None = None,
                 special_syn_method: dict[str, BaseSynthMethod] | None = None,
                 ) -> None:
        
        self.default_syn_method = default_syn_method
        pass

    def fit(self, X: pd.DataFrame, y=None) -> Self:
        """
        Loops through the columns in ``X``, following ``column_order``, and calls the :py:meth:`fit` function of the synthesis method classes given in ``default_syn_method`` and ``special_syn_method``.
        
        For the first column, nothing happens. For the next ones, if the variable name is found in keys of ``special_syn_method``, its corresponding value is the class to be called to fit the synthesiser.
        Otherwise, we use the class defined in ``default_syn_method``.

        :param X: An original dataset on which to fit the synthesiser.
        :param y: Ignored. This parameter exists only for compatibility with sklearn estimators.

        :return: Fitted synthesiser.
        """

        order = X.columns
        self.models_ = {}
        for i,y in enumerate(order):

            if i == 0:
                pred = pd.DataFrame({"init":[0]*X.shape[0]})
            else:
                pred = X[order[0:i]]

            self.models_[order[i]] = clone(self.default_syn_method).fit(pred,X[y])

        return self

    def generate(self, n: int | None = None, random_state: int = 42) -> pd.DataFrame:
        """
        Generate a synthetic dataset of ``n`` rows. 

        This method loops through the columns of ``X``, following ``column_order``, and calls the :py:meth:`transform` function of the synthesis method classes from ``default_syn_method`` and ``special_syn_method``.
        
        For the first column, the method specified in ``first_column_method`` is used. For the next ones, we use their respective fitted functions. Each column is predicted
        using the previously generated columns as features.

        :param n: Number of rows to generate for the synthetic dataset. Default is the same number of rows than the dataset on which the synthesiser was fitted. If one of the synthesis methods copies the original data, this parameter must be None.
        :param random_state: Random seed generator. Default is 42. 
        
        :return: Synthetic dataset
        """
        return pd.DataFrame()
