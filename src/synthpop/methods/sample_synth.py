"""
Synthesis method that samples from the target column.
"""
from typing import Self

import pandas as pd
from sklearn.exceptions import NotFittedError

from synthpop.methods.base_synth import BaseSynthMethod
import synthpop.methods.tree_utils 
from synthpop.reproducibility import RandomStateManager


class SampleMethod(BaseSynthMethod):
    """
    Synthesis method that samples from the target column.

    Examples
    --------

    :class:`SampleMethod` should be given as an argument to the :class:`Synthesiser`'s ``default_syn_method`` or ``special_syn_method``.
    See examples `default synthesis method <../../examples/changing_the_default_method.html>`__ and
    `special synthesis method <../../examples/special_syn_method.html>`__ respectively.

    **Intended usage in the package is thus:**
        
    >>> from synthpop.methods import SampleMethod
    >>> from synthpop import Synthesiser
    ... 
    >>> syn = Synthesiser(special_syn_method={"your_column_name" : SampleMethod()})

    ``SampleMethod`` can be used directly as follows (note that this is not the intended usage):

    >>> from synthpop.methods import SampleMethod
    >>> import pandas as pd
    >>> y = pd.Series(["a", "b", "c"], name="target_column")
    >>>
    >>> model = SampleMethod()
    >>> model.fit(X, y)
    SampleMethod()
    >>> model.transform(X)
    target_column
    0             a
    1             c
    2             b

    As ``SampleMethod`` does not use predictors to sample `y`, it can be called without `X`:
    
    >>> from synthpop.methods import SampleMethod
    >>> import pandas as pd
    >>> y = pd.Series([1, 2, pd.NA], name="new_target_column")
    >>>
    >>> model = SampleMethod(random_state=10)
    >>> model.fit(None, y)
    SampleMethod(random_state=10)
    >>> model.transform(None)
    new_target_column
    0              <NA>
    1              <NA>
    2                 1
    """

    def __init__(self, random_state: int | None = None) -> None:
        super().__init__()
        self.random_state = random_state

    def fit(self, X: pd.DataFrame | None, y: pd.Series) -> Self:
        """
        Stores the empirical distribution of `y`, including missing values.
        Also stores the number of rows.

        :param X: Feature dataset. Can be None. Is not used for learning.
        :param y: Target column.
        """
        if not isinstance(y, pd.Series):
            raise TypeError(f"y must be a pandas Series, got {type(y)} instead.")
        self.target_name_ = y.name
        self.n_samples_ = len(y)
        self.target_dtype_ = y.dtype

        value_counts = y.value_counts(dropna=False)
        self.values_ = value_counts.index.to_numpy()
        self.counts_ = value_counts.to_numpy()

        if self.random_state is None:
            self.random_state_ = RandomStateManager.create_instance_seed()
        else:
            self.random_state_ = self.random_state

        # for sklearn compatibility
        if X is not None:
            if not isinstance(X, pd.DataFrame):
                raise TypeError(
                    f"X must be a pandas DataFrame, got {type(X)} instead."
                )
            self.feature_names_in_ = list(X.columns)
            self.n_features_in_ = X.shape[1]
            
        return self
    
    def transform(self, X: pd.DataFrame | None) -> pd.Series:
        """
        Draws samples with replacement from the empirical distribution.

        :param X: DataFrame of already synthesised features. Determines output size. Can also be `None`. 
            In that case the output size is the same as the size of the fitted `y`.
        :return: Synthetic column sampled from original distribution.
        """
        if (
            not hasattr(self, "values_")
            or not hasattr(self, "counts_")
            or not hasattr(self, "target_name_")
            or not hasattr(self, "n_samples_")
            or not hasattr(self, "random_state_")
            or not hasattr(self, "target_dtype_")
        ):
            raise NotFittedError("SampleMethod is not fitted. Call `fit` first.")
        
        n = len(X) if X is not None else self.n_samples_
        rng = RandomStateManager.create_rng(seed=self.random_state_)
        
        sampled = synthpop.methods.tree_utils._sample_array(rng, self.counts_, self.values_, n)

        return pd.Series(sampled, name=self.target_name_, dtype=self.target_dtype_)
        
    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """
        Return the name of the synthesised target column.

        If the original target column has no name (i.e. `None`), the input feature
        names are returned instead.

        If `input_features` is not provided, fitted feature names are used.
        If fitted feature names are not available, generic feature
        names ``x0, x1, ..., x(n_features_in_ - 1)`` are generated when
        `n_features_in_` is available.

        :param input_features: Names of the input columns. If not provided,
            uses the feature names stored during fitting (`feature_names_in_`).
        :return: Name of the synthesised target column, or the input feature names
            if the target column has no name.
        :raises NotFittedError: If the estimator has not been fitted.
        """
        if not hasattr(self, "target_name_"):
            raise NotFittedError("SampleMethod is not fitted. Call `fit` first.")

        if input_features is None:
            if hasattr(self, "feature_names_in_"):
                input_features = list(self.feature_names_in_)
            elif hasattr(self, "n_features_in_"):
                input_features = [
                    f"x{i}" for i in range(self.n_features_in_)
                ]
            else:
                input_features = []

        if self.target_name_ is None:
            return input_features

        return [self.target_name_]