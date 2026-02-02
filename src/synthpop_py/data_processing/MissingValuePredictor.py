from sklearn.base import BaseEstimator
import pandas as pd


class MissingValuePredictor:
    
    def fit(self,X:pd.Series, y:pd.Series):
        pass

    def predict(self,X:pd.DataFrame) -> list[bool]:
        return [False]