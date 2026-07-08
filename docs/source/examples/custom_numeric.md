# Custom pipeline for numeric targets

```python
pipeline = Pipeline([("customEncoder",MyEncoder()), CartRegressorSynth(min_samples_leaf=10)])

synth = Synthesiser(
    ,default_syn_method=CartSynth(
        regressor=pipeline
       )
    )

data = pd.DataFrame()
syn_data = synth.fit(data).generate()
```