"""
Synthesis method that copies the original data.
"""
from typing import Self

import pandas as pd
from sklearn.exceptions import NotFittedError

from synthpop.methods import base_synth


class CopyMethod(base_synth.BaseSynthMethod):
    """
    Synthesis method that copies from the target column. 

    Examples
    --------

    :class:`CopyMethod` should be given as an argument to the :class:`Synthesiser`'s ``default_syn_method`` or ``special_syn_method``.
    See examples `default synthesis method <../../examples/changing_the_default_method.html>`__ and
    `special synthesis method <../../examples/special_syn_method.html>`__ respectively.

    **Intended usage in the package is thus:**
        
    >>> from synthpop.methods import CopyMethod
    >>> from synthpop import Synthesiser
    ... 
    >>> syn = Synthesiser(special_syn_method={"your_column_name" : CopyMethod()})

    The ``CopyMethod`` can be used directly as follows (note that this is not the intended usage):

    >>> from synthpop.methods import CopyMethod
    >>> import pandas as pd
    >>> 
    >>> X = pd.DataFrame({"X": [1, 2, 3]})
    >>> y = pd.Series(["a", "b", "c"], name="target_column")
    >>>
    >>> model = CopyMethod()
    >>> model.fit(X, y)
    CopyMethod()
    >>> model.transform(X)
    target_column
    0             a
    1             b
    2             c

    As ``CopyMethod`` does not use predictors to copy `y`, it can be called without `X`:

    >>> from synthpop.methods import CopyMethod
    >>> import pandas as pd
    >>> 
    >>> y = pd.Series([1, 2, pd.NA], name="new_target_column")
    >>>
    >>> model = CopyMethod()
    >>> model.fit(None, y)
    CopyMethod()
    >>> model.transform(None)
    new_target_column
    0             1
    1             2
    2             <NA>
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: pd.DataFrame | None, y: pd.Series) -> Self:
        """
        Stores the entire column in this object.

        :param X: Features dataset. Can be `None`. Not used for learning.
        :param y: The column to be copied.
        """
        if not isinstance(y, pd.Series):
            raise TypeError(f"y must be a pandas Series, got {type(y)} instead.")
        
        self.y_ = y.copy()
        self.target_name_ = y.name
        self.n_samples_ = len(y)
        self.target_dtype_ = y.dtype

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
        Returns an exact copy of the fitted target column.

        :param X: DataFrame of already synthesised columns. Can also be `None`. Is used only to validate the number of rows.
        :return: Synthetical column that is copied from the target variable (Original `y`).
        """

        if (
            not hasattr(self, "y_")
            or not hasattr(self, "target_name_")
            or not hasattr(self, "n_samples_")
            or not hasattr(self, "target_dtype_")
        ):
            raise NotFittedError("CopyMethod is not fitted. Call `fit` first.")
        
        if X is not None:
            if len(X) != self.n_samples_:
                raise ValueError(f"Row mismatch: expected {self.n_samples_}, got {len(X)}.")
        
        return pd.Series(self.y_.values, name=self.target_name_, dtype=self.target_dtype_)
    
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
            raise NotFittedError("CopyMethod is not fitted. Call `fit` first.")

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