# Your first synthetic dataset

In this guide, you'll create your first synthetic dataset with synthpop-py. We'll use the [diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) from `scikit-learn` and walk through the basic workflow:
1. Load the original data.
2. Create a `Synthesiser`.
3. Fit the Synthesiser to the original data.
4. Generate synthetic data.
5. Perform a basic comparison of the original and synthetic data.

This example introduces the core workflow. For a more detailed example, including utility evaluation with univariate distributions and S_pMSE, see the [**Your first synthetic dataset**](../examples/your_first_synthetic_dataset.md) example in the [Examples module](../examples/examples_index.md). The example there follows the same workflow as this guide, but explores the evaluation of the synthetic data in more detail.

## Loading the data
We begin with an existing dataset. The diabetes dataset contains measurements from diabetes patients and a target variable representing disease progression. The dataset contains only continuous variables.

First, load the dataset and convert it to a pandas DataFrame:
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

Each row represents one observation in the original dataset. Our goal is generate a new dataset with the same structure, but containing synthetic data instead of the original data.

## Creating a synthesiser
Now we create a {class}`~synthpop.synthesiser.Synthesiser` object.
```python
from synthpop.synthesiser import Synthesiser

synthesiser = Synthesiser(random_seed=1)
```
The Synthesiser controls the complete synthesis process. During fitting, it learns patterns from the original data, such as relationships between variables and the distributions of individual columns.

At this stage, no synthetic data has been created yet. We have only configured the object that will perform the synthesis.

More information about initialising the Synthesiser can be found in {ref}`User Guide 2.2: The Synthesiser class <22-synthesiser-class>`.

## Fitting the Synthesiser
Next, we fit the Synthesiser on the original dataset.
```python
synthesiser.fit(data)
```
During fitting, synthpop-py analyses the variables and estimates the internal relationships between the data.

The default synthesis method is Classification and Regression Trees (CART). synthpop-py supports several synthesis methods, which can be configured and combined in different ways. These methods are covered in more detail in the [Synthesis methods examples](../examples/changing_the_default_method.md) and [User Guide 3: Synthesis methods](../user_guides/3_synthesis_methods.md).

The original data is not modified. Instead, the Synthesiser stores the information needed to create new observations.

More information about fitting the Synthesiser can be found in {ref}`User Guide 2.4: Fitting the synthesiser <24-fitting-synthesiser>`.

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
![Tada](../images/tada_emoji.gif){width=25px}

The generated data are not copies of the original observations. Instead, they are newly generated values that aim to preserve the statistical properties and relationships present in the original data.

More information about generating synthetic data can be found in {ref}`User Guide 2.5: Generating synthetic data <25-generating-synthetic_data>`.

## A first look at the synthetic data
Creating synthetic data is only the first step. It is important to evaluate whether the synthetic data are suitable for their intended purpose.

A simple first check is to compare the original and synthetic datasets. For example, you can compare summary statistics:
```python
data.describe()
```
and:
```
synthetic_data.describe()
```

You can also compare individual variable distributions. For example, the following function creates visualisations of the  marginal distributions of the variables:
```python
from synthpop.plotting.plot_univariate import plot_univariate_distributions

figures = plot_univariate_distributions(
    orig_df=data,
    syn_df=synthetic_data,
    interactive=True # this automatically opens the plots in your default browser to scroll through
)
```
These plots allow you to visually inspect whether the distributions of individual variables are similar between the original and synthetic datasets. You can also expand your evaluation to reviewing the pairwise relationships of variables using the {ref}`S_pMSE metric <531-spmse>`.

For a complete introduction to evaluating synthetic data, including univariate distribution visualisation and the pairwise S_pMSE metric, continue with the [Your first synthetic data example](../examples/your_first_synthetic_dataset.md) in the Examples module. This is the same workflow you have just followed, but the example goes further into evaluating the quality and utility of the generated data.

## Next steps
Congratulations, you have created and evaluated your first synthetic dataset. The next step is to explore the [Examples module](../examples/examples_index.md). The first example, [**Your first synthetic dataset**](../examples/your_first_synthetic_dataset.md) covers the same workflow as this guide and expands on it with a more detailed evaluation of the synthetic data.

After that, you can explore examples covering:
- [making your synthesis reproducible](../examples/reproducible_synthesis.md);
- [generating larger datasets](../examples/generating_a_larger_dataset.md);
- [changing the synthesis order](../examples/changing_the_synthesis_order.md); and
- [using different synthesis methods](../examples/changing_the_default_method.md).

For a broader overview of the available examples, see the [Examples module](../examples/examples_index.md).