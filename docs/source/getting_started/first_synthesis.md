# Your first synthesis
This section explains the basic and general workflow for generating and evaluating a simple synthetic dataset with synthpop. Please see the {doc}`../user_guides/user_guides_index` for more in depth usecases.
## Generate synthetic data

To create your first synthetic dataset, start by loading your original data into a `pandas.DataFrame`. The {doc}`../api_reference/synthesiser_class/synthesiser` standard method learns the statistical patterns in the original data and uses this information to generate new synthetic records with similar characteristics.

```python
from synthpop import Synthesiser
import pandas as pd
orig_df = pd.read_csv("path/to/your/data.csv")
syn_df = Synthesiser().fit(orig_df).generate(n=len(orig_df.index))
print(synthetic_data)
```

The `fit()` step learns the relationships and distributions present in the original dataset. The `generate()` step uses this learned information to create synthetic records. In this example, the number of synthetic records is set equal to the number of records in the original dataset.

## Evaluate the synthetic data

Synthetic data should be evaluated before it is used for analysis or shared with others. Synthpop provides tools to compare the synthetic dataset with the original data and assess how well important characteristics have been preserved.

### Univariate distributions

A first check is to compare univariate distributions of individual variables in the original and synthetic dataset. 

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

### Multivariate Standardized Propensity Mean Squared Error (SPMSE)
Univariate comparisons evaluate variables individually, but they do not capture relationships between variables. 
To assess whether multivariate patterns are preserved, synthpop provides metrics such as the Standardized Propensity Mean Squared Error (SPMSE).

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

Lower SPMSE values indicate that the synthetic data has similar multivariate properties to the original data. 
Combining univariate and multivariate evaluations provides a more complete understanding of the quality and usefulness of the generated synthetic dataset.