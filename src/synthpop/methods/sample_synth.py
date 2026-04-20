"""
Synthesis method that samples from the target column. 
"""
from typing import Self
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
from synthpop.methods.base_synth import BaseSynthMethod

class SampleMethod(BaseSynthMethod):
    """
    Synthesis method that samples from the target column. 

    Examples
    --------
    >>> from synthpop.methods.sample_synth import SampleMethod
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


    Without X:
    >>> from synthpop.methods.sample_synth import SampleMethod
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

    def __init__(self, random_state: int | None = None):
        super().__init__()
        self.random_state = random_state

    def fit(self, X: pd.DataFrame | None, y: pd.Series) -> Self:
        """
        Stores the empirical distribution of `y`, including missing values.
        Also stores the number of rows.

        :param X: Feature dataset. Can be None. Is not used for learning.
        :param y: Target column.
        """

        self.target_name_ = y.name if y.name is not None else "target"
        self.n_samples_ = len(y)

        value_counts = y.value_counts(dropna=False)
        self.values_ = value_counts.index.to_numpy()
        self.counts_ = value_counts.to_numpy()

        self._seed = 42 if self.random_state is None else self.random_state
        self.random_state_ = np.random.default_rng(self._seed)

        if X is not None:
            self.feature_names_in_ = getattr(X, "columns", None)

        return self
    
    def transform(self, X: pd.DataFrame | None) -> pd.DataFrame:
        """
        Draws samples with replacement from the empirical distribution.

        :param X: DataFrame of already synthesised features. Determines output size. Can also be `None`. In that case the output size is the same as the size of the fitted `y`.
        :return: Synthetic column sampled from original distribution.
        """
        if (not hasattr(self, "values_")
            or not hasattr(self, "counts_")
            or not hasattr(self, "target_name_")
            or not hasattr(self, "n_samples_")
            or not hasattr(self, "random_state_")):
            raise NotFittedError("SampleMethod is not fitted. Call `fit` first.")
        
        n = len(X) if X is not None else self.n_samples_
        
        cum_counts = np.cumsum(self.counts_)
        r = self.random_state_.integers(0, self.counts_.sum(), size=n)
        idx = np.searchsorted(cum_counts, r, side="right")
        sampled = self.values_[idx]

        return pd.DataFrame({self.target_name_: sampled})
    
    def get_feature_names_out(self, input_features = None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", [])

        if self.target_name_ is None:
            return [input_features]

        return [self.target_name_]