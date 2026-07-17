# Your first synthetic dataset

Welcome to **synthpop-py**. In this example, we will walk through the complete workflow of creating a synthetic dataset. We will start with an existing dataset. For this example, we use the diabetes dataset from `scikit-learn`. This dataset contains measurements from diabetes patients and a target variable representing disease progression. The dataset only contains continuous variables for this example.

## Loading the data
First, we load the dataset and convert it to a pandas DataFrame.
```python
from sklearn.datasets import load_diabetes

diabetes = load_diabetes(as_frame=True)
data = diabetes.frame

data.head(3)
```
The first three rows of the dataset are:
|    |         age |        sex |        bmi |          bp |          s1 |         s2 |         s3 |          s4 |          s5 |         s6 |   target |
|---:|------------:|-----------:|-----------:|------------:|------------:|-----------:|-----------:|------------:|------------:|-----------:|---------:|
|  0 |  0.0380759  |  0.0506801 |  0.0616962 |  0.0218724  | -0.0442235  | -0.0348208 | -0.0434008 | -0.00259226 |  0.0199075  | -0.0176461 |      151 |
|  1 | -0.00188202 | -0.0446416 | -0.0514741 | -0.0263275  | -0.00844872 | -0.0191633 |  0.0744116 | -0.0394934  | -0.0683315  | -0.092204  |       75 |
|  2 |  0.0852989  |  0.0506801 |  0.0444512 | -0.00567042 | -0.0455995  | -0.0341945 | -0.0323559 | -0.00259226 |  0.00286131 | -0.0259303 |      141 |

Each row represents one observation in the original dataset. Our goal is generate a new dataset with the same structure, but containing synthetic records instead of the original records.

## Creating a synthesiser
Now we create a {class}`~synthpop.synthesiser.Synthesiser` object.
```python
from synthpop.synthesiser import Synthesiser

synthesiser = Synthesiser(random_seed=1)
```
The synthesiser controls the complete synthesis process. During fitting, it learns patterns from the original data, such as relationships between variables and the distributions of individual columns.

At this stage, no synthetic data has been created yet. We have only configured the object that will perform the synthesis.

More information about initialising the synthesiser can be found in {ref}`User Guide 2.2: The Synthesiser class <22-synthesiser-class>`.

## Fitting the synthesiser
Next, we fit the synthesiser on the original dataset.
```python
synthesiser.fit(data)
```
During fitting, synthpop-py analyses the variables in the dataset and estimates the relationships required for generating synthetic data.

The original data is not modified. Instead, the synthesiser stores the information needed to create new observations.

More information about fitting the synthesiser can be found in {ref}`User Guide 2.4: Fitting the synthesiser <24-fitting-synthesiser>`.

## Generating synthetic data
We can now generate a synthetic dataset.
```python
synthetic_data = synthesiser.generate()

synthetic_data.head(3)
```

The generated dataset has the same columns as the original dataset:

|    |         age |        sex |         bmi |          bp |         s1 |         s2 |         s3 |          s4 |         s5 |         s6 |   target |
|---:|------------:|-----------:|------------:|------------:|-----------:|-----------:|-----------:|------------:|-----------:|-----------:|---------:|
|  0 |  0.00175052 | -0.0446416 | -0.00405033 |  0.00810098 |  0.0218222 |  0.0412743 | -0.0434008 |  0.039106   | -0.0545396 | -0.0176461 |      128 |
|  1 |  0.0598711  |  0.0506801 | -0.0212953  |  0.00465813 |  0.0644768 |  0.0494162 |  0.0302319 | -0.00259226 |  0.0383939 | -0.0052198 |      196 |
|  2 | -0.0817979  | -0.0446416 | -0.0816528  | -0.0400989  | -0.0455994 | -0.0370128 |  0.0339135 | -0.0394934  | -0.0891334 | -0.092204  |       72 |

****Congratulations, you have created your first synthetic dataset!****
<img src="../images/tada_emoji.gif" width="25" style="vertical-align: middle;">

The generated records are not copies of the original observations. Instead, they are newly generated values that aim to preserve the statistical properties and relationships present in the original data.

More information about generating synthetic data can be found in {ref}`User Guide 2.5: Generating synthetic data <25-generating-synthetic-data>`.

## Evaluating the synthetic data
### Comparing individual variable distributions
Creating synthetic data is only the first step. A synthetic dataset should also be evaluated to determine whether it is useful for its intended purpose.

An important part of utility evaluation is checking whether individual variables have similar distributions in the original and synthetic datasets. The function {func}`~synthpop.plotting.plot_univariate.plot_univariate_distributions` creates histograms and bar plots that show you the marginal distributions of individual variables.
```python
from synthpop.plotting.plot_univariate import plot_univariate_distributions

figures = plot_univariate_distributions(
    orig_df=data,
    syn_df=synthetic_data,
    interactive=True # this automatically opens the plots in your default browser to scroll through
)
```
Running the code above opens the plots interactively. You will see that the first plot looks like:
![Univariate distributions of "age"](../images/age_distribution.png)

As you can see, the marginal distributions overlap mostly. There are some small differences, but generally speaking, the synthesiser reproduced the variable correctly.

More information about plotting the univariate distributions can be found in {ref}`User Guide 7.1: Univariate distribution visualisation <71-univariate-distribution-visualisation>`.

### Evaluating pairwise relationships with S_pMSE
Another important aspect of utility is whether relationships between variables are preserved. `synthpop-py` provides the pairwise Standardised propensity Mean Squared Error (S_pMSE) metric ({func}`~synthpop.utility_metrics.spmse.pairwise_spmse`) for this purpose.

```python
from synthpop.utility_metrics.spmse import pairwise_spmse

spmse = pairwise_spmse(
    orig_df=data,
    syn_df=synthetic_data
)

spmse.head(3)
```

The output contains the S_pMSE values for each pair of variables:

|    | column1   | column2   |   S_pMSE |
|---:|:----------|:----------|---------:|
|  0 | age       | age       |  1.02676 |
|  1 | age       | sex       |  0.92953 |
|  2 | age       | bmi       |  1.38183 |

Lower S_pMSE values indicate that the relationship between two variables is better preserved in the synthetic data. The interpretation of S_pMSE values and its limitations are explained in {ref}`User Guide 5.3.1: S_pMSE <531-spmse>`.

To make the results easier to inspect, we can visualise the S_pMSE values as a heatmap:
```python
from synthpop.plotting.plot_spmse import plot_spmse

plot_spmse(spmse)
```
![Heatmap of S_pMSE values](../images/spmse.png)
These visualisations allow us to compare variables one by one. As seen in the plot, all S_pMSE values are below 3 which shows us that the synthetic and original data are not statistically distinguishable with respect to the relationship between two variables.

More information about this visualisation and the interpretation can be found in {ref}`User Guide 7.2: S_pMSE heatmap <72-spmse-heatmap>`. 

## Next steps
Congratulations, you have created and evaluated your first synthetic dataset. Now it is time to find out how to make even better synthetic datasets. The next examples in this module explain how to make your synthesis reproducible, how to generate larger datasets and the importance of the synthesis order. The next module will go into more detail on more parameters to make your synthesis better.
