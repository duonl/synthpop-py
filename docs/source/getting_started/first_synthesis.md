# Regular usage
Your first synthetic dataset can be made something like this:

```python
from synthpop import Synthesiser
import pandas as pd
orig_df = pd.read_csv("path/to/your/data.csv")
syn_df = Synthesiser().fit(orig_df).generate(n=len(orig_df.index))
print(synthetic_data)
```

To check the quality of your synthesis you can make a univariate plot:
```python
from synthpop.plotting.plot import plot_univariate_distributions
plots = plot_univariate_distributions(
    orig_df=orig_df,
    syn_df=syn_df,
    save_path=None,
    interactive=False,
    )
for fig in plots:
    fig.show()
```

or a multivariate Standardized Propensity Score Mean Squared Error:
```python
from synthpop.utility_metrics.spmse import pairwise_spmse
from synthpop.plotting import plot_spmse

spmse = pairwise_spmse(orig_df, syn_df)

fig = plot_spmse(
    spmse=spmse,
    save_path=None,
    show_plot=True,
    )
```
