# Custom synth method
In this example we will explain how to create your own custom synthesiser method! Whether it might be for your own use case, or to extend synthpop's capabilities, synthpop has built a framework that helps you create your own method.

An example of a custom synth method. It can be defined as follows:
```python
class CustomSynth(BaseSynth):
    def __init__(self, some_param) -> None:
        super().__init__()

    def fit(self, X: pd.DataFrame| None, y: pd.Series) -> Self:
        return super().fit(X, y)
    
    def transform(self, X: pd.DataFrame| None) -> pd.DataFrame:
        return super().transform(X)
```

Using it can be done like this:

```python

synth = Synthesiser(
    special_syn_method={
        "B": CustomSynth(some_param=42)
        }
    )

data = pd.DataFrame()
syn_data = synth.fit(data).generate()
```