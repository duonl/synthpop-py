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
    """

    def __init__(self):
        super().__init__()

    def fit(self, X: pd.DataFrame | None, y: pd.Series) -> Self:
        """
        Stores the entire column in this object

        :param X: Optional: Features dataset. Not used for learning.
        :param y: The column to be copied.
        """
        self.y_ = y.copy()
        self.target_name_ = y.name if y.name is not None else "target"
        self.n_samples_ = len(y)

        # for sklearn compatibility
        if X is not None:
            self.feature_names_in_ = getattr(X, "columns", None)

        return self
    
    def transform(self, X: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Returns an exact copy of the fitted target column.

        :param X: Optional: DataFrame of already synthesised columns. Is used only to validate the number of rows.
        :return: One column of synthetic data that is identical to the target variable (Original `y`).
        """

        if (not hasattr(self, "y_")
            or not hasattr(self, "target_name_")
            or not hasattr(self, "n_samples_")):
            raise NotFittedError("CopyMethod is not fitted. Call `fit` first.")
        
        if X is not None:
            if len(X) != self.n_samples_:
                raise ValueError(f"Row mismatch: expected {self.n_samples_}, got {len(X)}.")
        
        return pd.DataFrame({self.target_name_: self.y_.values})
    
    def get_feature_names_out(self, input_features = None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", [])

        if self.target_name_ is None:
            return input_features

        return [self.target_name_]