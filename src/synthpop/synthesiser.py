"""
module for generating synthetic data
"""
from typing import Callable, Self, Dict

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.exceptions import NotFittedError

from synthpop.methods.base_synth import BaseSynthMethod
from synthpop.methods.cart_synth import CartMethod
import synthpop.reproducibility


class Synthesiser:
    """
    Delegates synthesis tasks to the appropriate synthesis method classes. 

    :param random_seed: A seed for randomness that makes both model fitting and data generation reproducible.
    :param column_order: list of variable names or list of indexes to define the order in which the columns will be synthesised. Default is the column order of the original dataset.
    :param default_syn_method: Synthesis method to apply to each column, the ones defined in special_syn_method. Default synthesis method is CartMethod. 
    :param special_syn_method: Dictionary of special synthesis method per variable. 
        If some variables should not follow the default_syn_method, they should be indicated in a dictionary where keys are variable names and values are BaseSynthMethod objects. 
        By default, there is no special synthesis method.


    Both ``default_syn_method`` and the values of ``special_syn_method`` can be callables.
    If they are callable, they are expected to take no arguments and produce a child instance of BaseSynthMethod.
    If ``default_syn_method`` is callable, it will be called for each variable that does not have a ``special_syn_method``.
    If a value of ``special_syn_method`` is callable, it will be called once. 

    Examples
    --------
        >>> import pandas as pd
        >>> from synthpop import Synthesiser
        >>> data = pd.DataFrame({
        ...     "first column":[1,2,3,2,1,3],
        ...     "second column":["a","a","b","c","b","c"],
        ...     "third column":[0.2,-3.4,1000.3,33,0,0]
        ... })
        >>> data
        first column second column  third column
        0             1             a           0.2
        1             2             a          -3.4
        2             3             b        1000.3
        3             2             c          33.0
        4             1             b           0.0
        5             3             c           0.0
        >>> synth = Synthesiser(random_seed=963214)
        >>> synth.fit(data)
        <synthpop.synthesiser.Synthesiser object at 0x000001F4BB6B9160>
        >>> synth.generate()
        first column second column  third column
        0           1.0             a      0.200000
        1           3.0             c      0.000000
        2           2.0             b     33.000000
        3           2.0             b   1000.299988
        4           2.0             b   1000.299988
        5           3.0             c      0.000000
        >>> synth.generate(n=10)
        first column second column  third column
        0           1.0             a      0.200000
        1           3.0             c      0.000000
        2           2.0             b     33.000000
        3           2.0             b   1000.299988
        4           2.0             b   1000.299988
        5           3.0             c      0.000000
        6           1.0             a      0.200000
        7           3.0             c      0.000000
        8           1.0             a     -3.400000
        9           1.0             a      0.200000

    """

    def __init__(self, random_seed: int | None = None,
                 column_order: list[str] | list[int] | None = None,
                 default_syn_method: BaseSynthMethod | Callable[[],
                                     BaseSynthMethod] | None = None,
                 special_syn_method: (Dict[str, BaseSynthMethod | Callable[[],  BaseSynthMethod]]
                 | None
                 ) = None
                 ) -> None:

        self.default_syn_method = default_syn_method
        self.column_order = column_order
        self.special_syn_method = special_syn_method
        self.random_seed = random_seed

    def _get_model(self, column_name: str) -> BaseSynthMethod:

        if self.special_syn_method is None:
            use_default = True
        elif column_name in self.special_syn_method:
            use_default = False
        else:
            use_default = True

        if use_default:
            if self.default_syn_method is None:
                effective_default_method = CartMethod()
            elif callable(self.default_syn_method):
                effective_default_method = self.default_syn_method()
                if not isinstance(effective_default_method, BaseSynthMethod):
                    raise ValueError(
                        "If the value of default_syn_method is callable it should return an instance of a child class of BaseSynthMethod")
            else:
                effective_default_method = clone(self.default_syn_method)
            return effective_default_method
        else:
            model = self.special_syn_method[column_name]
            if callable(model):
                new_model = model()
                if not isinstance(new_model, BaseSynthMethod):
                    raise ValueError(
                        f"If the value of special_syn_method is callable for entry '{column_name}' it should return an instance of a child class of BaseSynthMethod")
                return new_model
            else:
                return clone(model)

    def _validate_column_order_unique(self, column_order: list[str] | list[int]):
        unique_column_order = np.unique_counts(column_order)

        if (unique_column_order.counts > 1).any():
            duplicate_list = unique_column_order.values[unique_column_order.counts > 1]
            raise ValueError(
                f"The following columns occur multiple times in Synthesiser.column_order: {duplicate_list}")

    def fit(self, X: pd.DataFrame) -> Self:
        """
        Loops through the columns in ``X``, following ``column_order``, and calls the :py:meth:`fit` function of the synthesis method classes given in ``default_syn_method`` and ``special_syn_method``.

        If the variable name is found in keys of ``special_syn_method``, its corresponding value is the object used to synthesise that variable. The `fit` method of that object will be called.
        Otherwise, we use the object defined in ``default_syn_method``.

        :param X: The original dataset used to fit the synthesiser.

        :return: Fitted synthesiser.
        """

        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                f"X must be a pandas DataFrame, got {type(X)} instead.")

        if len(X) == 0:
            raise ValueError("X cannot be empty.")

        if self.column_order is None:
            self.column_order_ = X.columns.to_list()
        elif all(isinstance(item, int) for item in self.column_order):

            self._validate_column_order_unique(self.column_order)

            array_columns = np.array(self.column_order)
            out_of_bounds = (array_columns >= len(X.columns))
            if out_of_bounds.any():
                raise ValueError(
                    f"The following indices of Synthesiser.column_order are out of bounds: {array_columns[out_of_bounds]}")

            negative_indices = array_columns < 0
            if negative_indices.any():
                raise ValueError(
                    f"The following indices of Synthesiser.column_order are negative: {array_columns[negative_indices]}")

            self.column_order_ = X.columns[self.column_order].to_list()
        elif not all(isinstance(item, str) for item in self.column_order):
            raise ValueError(f"invalid column order: {self.column_order}")
        else:
            self._validate_column_order_unique(self.column_order)
            columns_not_in_df = set(self.column_order) - set(X.columns)
            if len(columns_not_in_df) > 0:
                raise ValueError(
                    f"The following columns of Synthesiser.column_order are not in the dataframe: {sorted(columns_not_in_df)}")
            self.column_order_ = self.column_order

        self.models_ = {}
        self.n_samples_ = X.shape[0]
        with synthpop.reproducibility.RandomStateManager(seed=self.random_seed):
            for i, y in enumerate(self.column_order_):

                if i == 0:
                    predictors = pd.DataFrame(
                        {"init": np.zeros(X.shape[0], dtype=int)})
                else:
                    predictors = X[self.column_order_[0:i]]

                model = self._get_model(y)

                self.models_[self.column_order_[i]] = model.fit(
    predictors,
    X[y],
)

        return self

    def generate(self, n: int | None = None, random_seed: int | None = None) -> pd.DataFrame:
        """
        Generate a synthetic dataset of ``n`` rows. 

        This method loops through the columns of ``X``, following ``column_order``, and calls the :py:meth:`transform` function of the synthesis method objects as used in `fit`.

        :param n: Number of rows to generate for the synthetic dataset. Default is the same number of rows as the dataset on which the synthesiser was fitted. If one of the synthesis methods copies the original data, this parameter must be None.
        :param random_seed: A seed for randomness that overrides the generation seed without refitting the synthesiser.

        :return: Synthetic dataset
        """

        if not hasattr(self, "models_"):
            raise NotFittedError("synthesiser has not been fitted.")

        if n is None:
            n_syn_rows = self.n_samples_
        elif n < 0:
            raise ValueError(
                f"number of rows of the synthetic data must be positive, got {n}")
        else:
            n_syn_rows = n
        
        if random_seed is None:
            seed_to_use = self.random_seed
        else:
            seed_to_use = random_seed

        result = pd.DataFrame()

        with synthpop.reproducibility.RandomStateManager(seed=seed_to_use):
            for i, y in enumerate(self.column_order_):

                if i == 0:
                    pred = pd.DataFrame(
                        {"init": np.zeros(n_syn_rows, dtype=int)})
                else:
                    pred = result

                new_syn_column = self.models_[y].transform(X=pred)
                result = pd.concat([result, new_syn_column], axis=1, join='outer')

        return result
