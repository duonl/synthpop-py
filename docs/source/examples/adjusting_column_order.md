# Adjusting the column order
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