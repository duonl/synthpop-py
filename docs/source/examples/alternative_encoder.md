# Alternative encoder

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
