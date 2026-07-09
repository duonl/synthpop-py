# Regular usage
Your first synthetic dataset can be made something like this:

```python
from synthpop import Synthesiser
import pandas as pd
data = pd.read_csv("path/to/your/data.csv")
synthetic_data = Synthesiser().fit(data).generate()
print(synthetic_data)
```