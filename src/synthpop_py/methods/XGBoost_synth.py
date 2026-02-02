from synthpop_mq.methods import base_synth
import pandas as pd

class XGBRegressorSynth(base_synth.BaseSynthMethod):
    
    def fit(self,X:pd.DataFrame, y: pd.Series):
        return self
    
    def transform(self,X:pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()

class XGBClassifierSynth(base_synth.BaseSynthMethod):

    def fit(self,X:pd.DataFrame, y: pd.Series):
        return self
    
    def transform(self,X:pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()

