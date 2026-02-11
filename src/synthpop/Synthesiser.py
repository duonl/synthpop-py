import pandas as pd
from synthpop.methods.base_synth import BaseSynthMethod
from synthpop.methods.cart_synth import TreeRegressorMethod, CartMethod
from collections.abc import Callable
from typing import Self
from sklearn.base import TransformerMixin,BaseEstimator

class Synthesiser(TransformerMixin,BaseEstimator):
    """
    Delegates synthesis tasks to the appropriate synthesis method classes. 

    :param column_order: list of variable names or list of indexes to define the order in which the columns will be synthesised. Default is the column order of the original dataset.
    :param default_syn_method: BaseSynth object. Synthesis method to apply to each column, except the first one and the ones defined in special_syn_method. Default synthesis method is CartSynth. 
    :param special_syn_method: Dictionary of special synthesis method per variable. If some variables should not follow the default_syn_method, they should be indicated in a dictionary where keys are variable names and values are BaseSynth objects. By default, there is no special synthesis method.
    """
    def __init__(self, column_order: list[str] | list[int] | None = None, default_syn_method: BaseSynthMethod = CartMethod(), special_syn_method: dict[str, BaseSynthMethod] | None = None) -> None:
        pass

    def fit(self, X: pd.DataFrame, y=None) -> Self:
        """
        Loops through the columns in ``X``, following ``column_order``, and calls the :py:meth:`fit` function of the synthesis method classes given in ``default_syn_method`` and ``special_syn_method``.
        
        For the first column, nothing happens. For the next ones, if the variable name is found in keys of ``special_syn_method``, its corresponding value is the class to be called to fit the synthetizer.
        Otherwise, we use the class defined in ``default_syn_method``.

        :param X: An original dataset on which to fit the synthesiser.
        :param y: Ignored. This parameter exists only for compatibility with sklearn estimators.

        :return: Fitted synthesiser.
        """
        self.random_state_ = check_random_state(self.random_state)#mandated by scikit-learn developer guide since we need the rng after fitting.

        #Stores the probability distribution of the first column in self.

        # Using sklearn.utils.validation.validate_data, set the attribute feature_names_in_ to X and y.
        # That method sets the attribute. 
        # For example:
        # from sklearn.utils.validation import validate_data
        # ....
        # X, y = validate_data(self, X, y)
        return self

    def transform(self, X_syn:pd.DataFrame |None = None,n: int | None = None) -> pd.DataFrame:
        """
        Generate a synthetic dataset of ``n`` rows if ``n`` is given and ``X_syn`` is None. 

        This method loops through the columns of the data used for fitting, following ``column_order``, and calls the :py:meth:`transform` function of the synthesis method classes from ``default_syn_method`` and ``special_syn_method``.
        
        If ``X_syn`` is ``None``, then for the first column, we simple take a random sample with :py:meth:`pandas.DataFrame.sample`. For the next ones, we use their respective fitted models. Each column is predicted
        using the previously generated columns as features.
        If ``X_syn`` is not ``None``, it is used as features to predict the first column. It will be included as features for all thereafter. 
        Note that the columns in ``X_syn`` should be present in the data used for fitting.

        Setting both ``X_syn`` and ``n`` raises an exception.

        :param X_syn:  Data that should be included in the output and can be included as features for all columns seen in fitting.
        :param n: Number of rows to generate for the synthetic dataset. Default is the same number of rows than the dataset on which the synthetizer was fitted.

        :return: Synthetic dataset
        """

        # should call sklearn.utils.validation.check_is_fitted(self), 
        return pd.DataFrame()
    
    def get_feature_names_out(self,input_features = None):
        pass
