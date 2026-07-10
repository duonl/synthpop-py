# Adjusting parameters
If you want to adjust some parameters, there are more than one way to do it.
One possible way is this:
```python
from synthpop import Synthesiser
import pandas as pd
data = pd.read_csv("path/to/your/data.csv")
synthetic_data = Synthesiser(
    default_syn_method=CartMethod(#Use Cart for all columns
        regressor=TreeRegressorMethod(ccp_alpha=0.001),#Set the method and parameters for numeric targets
        classifier=TreeClassifierMethod(class_weight="balanced")#Set the method and parameters for categoric targets.
        )
    ).fit(data).generate()
print(synthetic_data)
```