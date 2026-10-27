"""
classes for synthesising data with XGBoost (not implemented yet)
"""
import pandas as pd
from synthpop.methods import base_synth


class XGBRegressorSynth(base_synth.BaseSynthMethod):
    
    def __init__(self):
        super().__init__()
        raise NotImplementedError
    
    def fit(self,X:pd.DataFrame, y: pd.Series):
        raise NotImplementedError
        return self
    
    def transform(self,X:pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
        return pd.DataFrame()

class XGBClassifierSynth(base_synth.BaseSynthMethod):

    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def fit(self,X:pd.DataFrame, y: pd.Series):
        raise NotImplementedError
        return self
    
    def transform(self,X:pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
        return pd.DataFrame()

