# Introduction

Here there would be some introductory text.

Basic usage:
```python
from Synthpop.Synthesiser import Synthesiser

data = load_data()
synth = Synthesiser()


synth.fit(data)

synth_data = synth.generate()
```

Setting parameters can be done as follows:

```python
from Synthpop.Synthesiser import Synthesiser

data = load_data()
synth = Synthesiser()

synth.fit(data)

synth_data = synth.generate(default_syn_method=CartSynth(
        regressor=CartRegressorSynth(min_samples_leaf=5)
       ))
```
As seen above, you need to create a [CartRegressorSynth object.](CartRegressorSynth) This is because these kind of parameters are specific to synt methods.

The default synthpop method includes a default encoding method (target-mean-encoding for a regressor and PCA-encoding for a classifier), but those can be changed in that manner:

```python
from Synthpop.Synthesiser import Synthesiser
from Synthpop.cart_synth import CartSynth, CartRegressorSynth
from sklearn.preprocessing import OneHotEncoder

data = load_data()
synth = Synthesiser(
                default_syn_method=CartSynth(
                        regressor = CartRegressorSynth(
                                encoder = OneHotEncoder()
                )
        )
)

synth.fit(data)
synth_data = synth.generate()

```