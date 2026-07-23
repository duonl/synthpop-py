# Generating a larger synthetic dataset
In the previous examples, we generated a synthetic dataset with the same number of rows as the original dataset. However, synthetic datasets do not need to have the same size as the original data.

One advantage of synthetic data is that we can generate more records than were available in the original dataset. This can be useful when testing analysis pipelines, developing software or creating datasets for simulation studies.

## Loading the data
For this example, we use the Iris dataset from `scikit-learn`. Unlike the diabetes dataset used in the previous example, this dataset contains both numerical variables and a categorical variable. Initially, the categorical variable is represented as integers. In the code below, we map them to the related names.
```python
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)

data = iris.frame

# use the names instead of integer representation
data["target"] = data["target"].map(
    dict(enumerate(iris.target_names))
)

data.head(3)
```
The first three rows of the dataset are:
|    |   sepal length (cm) |   sepal width (cm) |   petal length (cm) |   petal width (cm) | target   |
|---:|--------------------:|-------------------:|--------------------:|-------------------:|:---------|
|  0 |                 5.1 |                3.5 |                 1.4 |                0.2 | setosa   |
|  1 |                 4.9 |                3   |                 1.4 |                0.2 | setosa   |
|  2 |                 4.7 |                3.2 |                 1.3 |                0.2 | setosa   |     

The dataset contains measurements of iris flowers. The target column describes the flower species and is a categorical variable. In total, the dataset has 150 rows.
```warning
The Iris dataset contains only 150 observation, which is small for training and evaluating a synthetic data model. 

We use the Iris dataset because it is a simple, well-known, and readily available dataset that makes it easy to demonstrate the effect of changing the synthesis order without introducing unnecessary complexity.
```

## Creating and fitting the Synthesiser
First, we create and fit the synthesiser like we've done in previous examples.
```python
from synthpop.synthesiser import Synthesiser

synthesiser = Synthesiser(random_seed=1)

synthesiser.fit(data)
```
During fitting, synthpop-py learns the relationships between variables and estimates the distributions needed to generate new observations. The number of rows in the original dataset does not limit the number of synthetic observations that can be generated.

## Generating a larger synthetic dataset
The number of synthetic records can be specified using the `n` parameter of {func}`~synthpop.synthesiser.Synthesiser.generate`.
```python
synthetic_data = synthesiser.generate(n=5000)
```
The generated dataset now contains 5000 observations:
```python
synthetic_data.shape
```
```text
(5000, 5)
```
The generated data has the same columns as the original dataset, but contains many more rows.
```python
data.shape
```
```text
(150, 5)
```

## Evaluating the larger synthetic dataset
Even when generating a larger dataset, it is important to verify that the synthetic data still represents the original data well.

For example, we can compare the univariate distributions:
```python
from synthpop.plotting.plot_univariate import plot_univariate_distributions

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
from synthpop.utility_metrics.spmse import pairwise_spmse
from synthpop.plotting.plot_spmse import plot_spmse

spmse = pairwise_spmse(
    orig_df=data,
    syn_df=synthetic_data,
)

plot = plot_spmse(spmse, show_plot=True)
```
The S_pMSE calculation also accounts for the different dataset sizes when comparing the pairwise relationship between variables.

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