# Examples


## Custom synth method
An example of a custom synth method. It can be defined as follows:
```python
class CustomSynth(BaseSynth):
    def __init__(self, some_param) -> None:
        super().__init__()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        return super().fit(X, y)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return super().transform(X)
```

Using it can be done like this:

```python


synth = Synthesiser(
    special_syn_method={
        "B": CustomSynth(some_param=42)
        }
    ,default_syn_method=CartSynth(
        regressor=CartRegressorSynth(min_samples_leaf=5,encoder= Encoder.Encoder(a=3))
       )
    )

pipeline = Pipeline[StandardScaler(), synth]

data = pd.DataFrame()
syn_data = synth.fit(data).generate()
```

## custom pipeline

```python

pipeline = Pipeline([("customEncoder",MyEncoder()), CartRegressorSynth(min_samples_leaf=10)])

synth = Synthesiser(
    special_syn_method={
        "B": CustomSynth(some_param=42)
        }
    ,default_syn_method=CartSynth(
        regressor=pipeline
       )
    )

data = pd.DataFrame()
syn_data = synth.fit(data).generate()
```

## custom encoder

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