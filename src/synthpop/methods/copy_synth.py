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
        >>> from synthpop.methods.copy_synth import CopyMethod
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

    Without X:
        >>> from synthpop.methods.copy_synth import CopyMethod
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

        # for sklearn compatibility
        if X is not None:
            self.feature_names_in_ = getattr(X, "columns", None)

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
        ):
            raise NotFittedError("CopyMethod is not fitted. Call `fit` first.")
        
        if X is not None:
            if len(X) != self.n_samples_:
                raise ValueError(f"Row mismatch: expected {self.n_samples_}, got {len(X)}.")
        
        return pd.Series(self.y_.values, name=self.target_name_)
    
    def get_feature_names_out(self, input_features=None) -> list[str]:
        if not hasattr(self, "target_name_"):
            raise NotFittedError("CopyMethod is not fitted. Call `fit` first.")

        if input_features is None:
            input_features = getattr(self, "feature_names_in_", [])

        if self.target_name_ is None:
            return [input_features]

        return [self.target_name_]