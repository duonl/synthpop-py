# Custom encoder

```python

class CustomEncoder(TransformerMixin, BaseEstimator): 
# This class can also inherit from OneToOneFeatureMixin if input- and output features are the same

    def __init__(self):
        pass
    def fit(self,X:pd.Series, y: pd.Series) -> Self:
        pass
    def transform(self,X:pd.Series) -> pd.DataFrame:
        return pd.DataFrame()
    
```

