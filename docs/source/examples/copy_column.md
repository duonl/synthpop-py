# Copy the first column

If you want to copy the first column instead of sampling, it would look like this:
```python
from synthpop.methods.copy_synth import CopyMethod

data = pd.Dataframe()
synth = Synthesiser(special_syn_method={
        "name_of_first_column": CopyMethod()
        })
syn_data = synth.fit(data).generate()
```

# Copy the first column in combination with other column order
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

# Copy an other column than the first

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