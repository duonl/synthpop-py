# Generate a larger synthetic dataset
In the previous examples, we generated a synthetic dataset with the same number of rows as the original dataset. However, synthetic datasets do not need to have the same size as the original data.

One advantage of synthetic data is that we can generate more records than were available in the original dataset. This can be useful when testing analysis pipelines, developing software or creating datasets for simulation studies.

## Load the data
For this example, we use the [Titanic dataset](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv) from `seaborn`. Unlike the diabetes dataset used in the previous example, this dataset contains both numerical and categorical variables making it a realistic example for demonstrating synthetic data generation.
```python
import seaborn as sns

data = sns.load_dataset("titanic")

data.head(3)
```
The first three rows of the dataset are:
|    |   survived |   pclass | sex    |   age |   sibsp |   parch |    fare | embarked   | class   | who   | adult_male   | deck   | embark_town   | alive   | alone   |
|---:|-----------:|---------:|:-------|------:|--------:|--------:|--------:|:-----------|:--------|:------|:-------------|:-------|:--------------|:--------|:--------|
|  0 |          0 |        3 | male   |    22 |       1 |       0 |  7.25   | S          | Third   | man   | True         | nan    | Southampton   | no      | False   |
|  1 |          1 |        1 | female |    38 |       1 |       0 | 71.2833 | C          | First   | woman | False        | C      | Cherbourg     | yes     | False   |
|  2 |          1 |        3 | female |    26 |       0 |       0 |  7.925  | S          | Third   | woman | False        | nan    | Southampton   | yes     | True    |   

The dataset contains information about passengers aboard the Titanic, including demographic characteristics, ticket information, and whether each passenger survived.

## Create and fit the Synthesiser
First, we create and fit the synthesiser as in the previous examples.
```python
from synthpop import Synthesiser

synthesiser = Synthesiser(random_seed=1)

synthesiser.fit(data)
```
During fitting, synthpop-py learns the relationships between variables and estimates the distributions needed to generate new observations. The number of rows in the original dataset does not limit the number of synthetic observations that can be generated.

## Generate a larger synthetic dataset
The number of synthetic records can be specified using the `n` parameter of {func}`~synthpop.synthesiser.Synthesiser.generate`.
```python
synthetic_data = synthesiser.generate(n=5000)
```
The generated dataset now contains 5000 observations:
```python
synthetic_data.shape
```
```text
(5000, 15)
```
The generated data has the same columns as the original dataset, but contains many more rows.
```python
data.shape
```
```text
(891, 15)
```

## Evaluate the larger synthetic dataset
Even when generating a larger dataset, it is important to verify that the synthetic data still represents the original data well.

For example, we can compare the univariate distributions:
```python
from synthpop.plotting import plot_univariate_distributions

plot_univariate_distributions(
    orig_df=data,
    syn_df=synthetic_data,
    interactive=True
)
```
Because the synthetic dataset contains more observations, the absolute counts will naturally differ. Therefore, these plots compare **relative frequencies and densities** rather than raw counts.

For categorical variables, the proportion of each category should be similar. For numerical variables, the shape of the distribution should be preserved.

We can also evaluate pairwise relationships using S_pMSE:
```python
from synthpop.utility_metrics import pairwise_spmse
from synthpop.plotting import plot_spmse

spmse = pairwise_spmse(
    orig_df=data,
    syn_df=synthetic_data,
)

plot = plot_spmse(spmse, show_plot=True)
```
The S_pMSE calculation also accounts for the different dataset sizes when comparing the pairwise relationship between variables.
```{note}
The S_pMSE is influenced by the number of observations in the original and synthetic datasets. Generating more synthetic rows may result in larger S_pMSE values, even if the underlying quality of the synthesis has not changed, because small differences can be estimated more precisely. Therefore, S_pMSE values are most meaningful when comparing synthesis methods or parameter settings on datasets of the same size.
```

More information about evaluating utility can be found in {ref}`User Guide 5: Evaluating utility <5-evaluating-utility>`. 

Generating larger synthetic datasets is useful when a specific application requires more observations than are available in the original dataset. However, generating more data does not automatically improve the quality of the synthetic data. The quality depends on how well the synthesis process captures the important patterns and relationships present in the original data.

When evaluating the synthetic dataset above, you may have noticed that some relationships between variables were not perfectly preserved. In particular, the S_pMSE values can indicate that some pairwise relationships differ between the original and synthetic datasets. This is expected: the default synthesis settings provide a good starting point, but they may not always be optimal for every dataset.

## Next steps
The next example explains how the **synthesis order** influences the generated data and how changing this order can improve the preservation of relationships between variables.

After learning how to control the synthesis order, later examples will explore further ways to customise the synthesis process, including changing synthesis methods for individual columns.

```{warning}
Generating larger synthetic datasets is not compatible with synthesis methods that directly copy observations from the original data, such as {class}`~synthpop.methods.copy_synth.CopyMethod`. These methods rely on existing records and therefore cannot generate additional synthetic observations.

The use of `CopyMethod` and how to select synthesis methods for individual columns is explained in [Module 2](changing_the_default_method.md).
```