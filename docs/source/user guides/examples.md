# Examples

## regular usage
Your first synthetic dataset can be made something like this:

```python
from synthpop import Synthesiser
import pandas as pd
data = pd.read_csv("path/to/your/data.csv")
synthetic_data = Synthesiser().fit(data).generate()
print(synthetic_data)
```
## adjusting the column order
The order in which the columns are synthesised matters a lot for the quality of the synthetic data.
The order can be adjusted like this:

```python
from synthpop import Synthesiser
import pandas as pd
data = pd.DataFrame(
    np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=["a", "b", "c"]
)
synthetic_data = Synthesiser(column_order=["b","a","c"]).fit(data).generate()
print(synthetic_data)
```

## adjusting parameters
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
## Custom synth method
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

## custom pipeline for numeric targets

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

## Copy the first column

If you want to copy the first column instead of sampling, it would look like this:
```python
from synthpop.methods.copy_synth import CopyMethod

data = pd.Dataframe()
synth = Synthesiser(special_syn_method={
        "name_of_first_column": CopyMethod()
        })
syn_data = synth.fit(data).generate()
```

## Copy the first column in combination with other column order
If you do not want to sample the first column and copy it instead, you need to specify the `CopyMethod` for the column that is first in the given order of columns.
```python
from synthpop import Synthesiser
import pandas as pd
data = pd.DataFrame(
    np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=["a", "b", "c"]
)
synthetic_data = Synthesiser(
    column_order=["b","a","c"],
    special_syn_method = {
        "b":CopyMethod()
    }
    ).fit(data).generate()
print(synthetic_data)
```

## Copy an other column than the first

If you want to copy the any other column than the first, it would look like this:
```python
from synthpop.methods.copy_synth import CopyMethod

data = pd.Dataframe()
synth = Synthesiser(
    special_syn_method={
        "other_column_name": CopyMethod()
        }
    )
syn_data = synth.fit(data).generate()
```

Note that ``synth.fit(data).generate(10)`` would raise an exception, unless the column being copied has exactly 10 rows. 
