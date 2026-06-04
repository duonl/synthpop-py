"""
module for generating synthetic data
"""
from typing import Self, Dict
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
                 special_syn_method: Dict[str, BaseSynthMethod] | None = None,
                 ) -> None:
        
        self.default_syn_method = default_syn_method
        self.column_order = column_order
        self.special_syn_method = special_syn_method

    def _get_model(self, y):

        if not (self.default_syn_method is None):
            effective_default_method = clone(self.default_syn_method)
        else:
            effective_default_method = CartMethod()

        if self.special_syn_method is None:
            model = effective_default_method
        elif y in self.special_syn_method:
            model = clone(self.special_syn_method[y])
        else:
            model = effective_default_method

        return model


    def fit(self, X: pd.DataFrame, y=None) -> Self:
        """
        Loops through the columns in ``X``, following ``column_order``, and calls the :py:meth:`fit` function of the synthesis method classes given in ``default_syn_method`` and ``special_syn_method``.
        
        For the first column, nothing happens. For the next ones, if the variable name is found in keys of ``special_syn_method``, its corresponding value is the class to be called to fit the synthesiser.
        Otherwise, we use the class defined in ``default_syn_method``.

        :param X: An original dataset on which to fit the synthesiser.
        :param y: Ignored. This parameter exists only for compatibility with sklearn estimators.

        :return: Fitted synthesiser.
        """

        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"X must be a pandas DataFrame, got {type(X)} instead.")
        if self.column_order is None:
            self.column_order_ = X.columns.to_list()
        elif all([isinstance(item, int) for item in self.column_order]):
            self.column_order_ = X.columns[self.column_order].to_list()
        else:
            self.column_order_ = self.column_order

        self.models_ = {}
        self.n_samples_ = X.shape[0]
        for i, y in enumerate(self.column_order_):

            if i == 0:
                predictors = pd.DataFrame({"init":[0]*X.shape[0]})
            else:
                predictors = X[self.column_order_[0:i]]

            model = self._get_model(y)

            self.models_[self.column_order_[i]] = model.fit(predictors,X[y])

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

        result = pd.DataFrame()

        for i,y in enumerate(self.column_order_):

            if i == 0:
                pred = pd.DataFrame({"init":[0]*self.n_samples_})
            else:
                pred = result

            new_syn_column = self.models_[y].transform(X=pred)
            result[new_syn_column.name] = new_syn_column
        return result
