# Use the `tune_cart` function
In the previous examples, we used the synthesis methods with their default configuration. This works well as a starting point, but the characteristics of a dataset may call for some additional tuning.

For example, a dataset may contain a large number of observations, in which case using larger leaf nodes can produce a more generalised model. Similarly, categorical predictors may contain many levels, making them computationally expensive to process. In such cases, reducing the dimensionality of the data before passing it to the classifier may be beneficial.

The CART synthesis method can be configured by constructing a {class}`~synthpop.methods.cart_synth.CartMethod` instance and changing its individual components manually. However, this can involve configuring several underlying estimators separately from each other. How to do this will be explained in the following example: [Configure the CART components directly](./configure_cart_directly.md).

For common CART tuning options, synthpop-py provides the {func}`~synthpop.methods.cart_synth.tune_cart` convenience function. It lets us change the most commonly tuned parameters in one place and applies those settings consistently to the relevant CART components.

In this example, we will introduce how to use the {func}`~synthpop.methods.cart_synth.tune_cart` convenience function. More precisely, we will show how to adjust:
1. the number of observations in a leaf node `n_leaves`;
2. the number of PCA Components `n_components`; and
3. the consideration threshold for rare categories `rare_categories_threshold`.

## Implement `tune_cart`
Suppose we are working with the Titanic dataset and want every variable to use CART synthesis, but we want each decision tree to require at least 10 observations in a leaf.

Instead of constructing a `CartMethod` and configuring each decision tree individually, we can use `tune_cart`:
```python
from synthpop import Synthesiser
from synthpop.methods import tune_cart

tuned_cart = tune_cart(n_leaves=10)

synthesiser = Synthesiser(
    random_seed=1,
    default_syn_method=tuned_cart,
)

synthesiser.fit(data)

synthetic_data = synthesiser.generate()
```
Here, `tune_cart(n_leaves=10)` returns a [factory](https://en.wikipedia.org/wiki/Factory_(object-oriented_programming)) that creates a `CartMethod` configured with `n_leaves=10`.

## Control the size of the decision tree leaves
The `n_leaves` parameter controls the minimum number of observations in each leaf of the decision tree used by CART. For example, defining the minimum number to be 20:
```python
synthesiser = Synthesiser(
    random_seed=1,
    default_syn_method=tune_cart(n_leaves=20),
)
```
The value `20` is passed to the `min_samples_leaf` parameter of the decision trees used for:
- :any:`classification <~synthpop.methods.cart_synth.TreeClassifierMethod>`;
- :any:`regression <~synthpop.methods.cart_synth.TreeRegressorMethod>`; and
- [predicting missing values](../api_reference/data_processing/Missing_value.rst).

The parameter corresponds to `min_samples_leaf` in [scikit-learn's decision tree estimators](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).

A decision tree works by repeatedly splitting the training data into smaller groups. With `n_leaves=20`, the tree cannot create a leaf containing fewer than 20 observations. This limits how specifically the tree can model the training data.

This affects the synthetic data because the tree's predictions are based on these groups. With a **smaller** value, the tree can create smaller and more specific groups, allowing it to capture fine-grained relationships in the data. However, this also makes it easier for the tree to learn patterns that are specific to a small number of observations. This can result in the tree [overfitting](https://en.wikipedia.org/wiki/Overfitting) the original data, which results in less privacy protection.

With a **larger** value, the tree must base its predictions on larger groups of observations. This generally produces a more generalised and less complex model, reducing the influence of individual or rare observations and potentially lowering computational costs. The trade-off is that the tree may no longer be able to capture genuine complex relationships in the data, resulting in [underfitting](https://en.wikipedia.org/wiki/Overfitting#Underfitting:~:text=training%20and%20inference.-,Underfitting,-edit).

## Change the number of principal components
CART uses a {class}`~synthpop.data_processing.encoders.PCAEncoder` to encode categorical predictors before passing them to the classifier. This is necessary because the classifier operates on numeric features. Additionally, categorical variables can contain many levels, which can substantially increase the computational cost of the model.

The `n_components` parameter controls the number of principal components and thus how much of this encoded information is retained by PCA.

For example:
```python
tuned_cart = tune_cart(n_components=2)
```
With `n_components=2`, PCA reduces the encoded categorical predictors to two principal components before they are passed to the classifier.

In other words, instead of giving the classifier the full set of encoded categorical information, we give it a smaller representation containing the two dimensions that capture the most variance in the data.

This can be useful when categorical predictors have many categories. Reducing the number of components can make the classifier work with a simpler representation of the predictors and can prevent it from modelling very small differences in the encoded data. Less dimensions also require less computing power.

There is a trade-off, however. Fewer components mean less information is available to the classifier. If important relationships are contained in components that are discarded, the classifier may produce less accurate predictions and therefore synthetic data that captures fewer of the relationships present in the original data.

Conversely, retaining more components preserves more information but gives the classifier a more complex representation to work with.

You can specify either the number of components or the proportion of the variance that should be retained:
```python
tune_cart(n_components=2)
```
keeps two components that explain the most variance, while:
```python
tune_cart(n_components=0.9)
```
keeps enough components to explain 90% of its variance.

`n_components` accepts the same types of values as the {class}`underlying scikit-learn PCA <sklearn:sklearn.decomposition.PCA>`:
- an integer greater than or equal to 1 specifies the number of components;
- a float between 0 and 1 specifies the proportion of variance to retain;
- `None` uses the default PCA behaviour. No additional reduction is requested then.

For more information about the role of PCA in CART, see {ref}`Guide 4.1.1: PCA encoding <411-pca-encoding>`.

## Configure the rare-category check
CART can check categorical predictors for values that occur only a small number of times. This is a privacy safeguard: a very rare category can make it easier for information about an individual to be inferred from the synthetic data. See the example on [risk of privacy loss due to rare categories](rare_categories.md) for more information about this specific privacy caveat.

The `rare_categories_threshold` parameter determines which categories are considered rare. A category is considered rare when  it occurs fewer than the specified threshold.

For example,
```python
tune_cart(rare_categories_threshold=5)
```
means that a category occurring fewer than 5 times is considered rare.

The check does not change how the CART model is fitted or how categories are synthesised. Instead, it provides a warning when rare categories make up a substantial proportion of the observations in a predictor. Synthesis continues after the warning.

Specifically, a warning is raised when observations belonging to rare categories account for at least 25% of the observations.

For example, suppose a categorical predictor contains these frequencies:
**Category** | **Frequency**
--- | ---
A | 60
B | 20
C | 15
D | 5

With:
```python
tune_cart(rare_categories_threshold=20)
```
categories `C` and `D` are rare because they occur fewer than 20 times. Together, they account for 20 observations out of 100, or 20% of the dataset. Therefore, the 25% warning threshold is not reached. If instead the rare categories accounted for 25 or more observations, the check would raise a warning:
**Category** | **Frequency**
--- | ---
A | 55
B | 20
C | 15
D | 5
E | 5

The warning is intended to draw attention to a potential privacy risk rather than prevent synthesis from proceeding. For more information about the reason for this check and the privacy implications of rare categorical values, see {ref}`User guide 6.1.2: Attribute disclosure <612-attribute-disclosure>`.

A lower `rare_categories_threshold` means that fewer categories are considered rare. A higher threshold is more conservative because more categories can be classified as rare.

By default, `rare_categories_threshold` is `None`. In that case, `tune_cart` uses the value of `n_leaves` as the threshold. This means:
```python
tune_cart()
```
has an effective rare-category threshold of `5`, because `n_leaves` defaults to `5`.

Similarly:
```python
tune_cart(n_leaves=10)
```
has an effective threshold of `10`, unless `rare_categories_threshold` is explicitly specified. This means that changing `n_leaves` can also change which categories are considered rare unless you provide an explicit threshold.

If you want to disable the check entirely, set the threshold to `0`:
```python
synthesiser=Synthesiser(
    default_syn_method=tune_cart(
        rare_categories_threshold=0,
    ),
)
```
With the check disabled, synthesis will no longer warn about rare categorical values. You therefore lose this safeguard against potential attribute disclosure.

## Tune several parameters together
The tree parameters control different aspects of CART:
**Parameter** | **What changes?** | **Potential effect**
--------------|-------------------|---------------------
`n_leaves` | How small a group a decision tree can base a prediction on | Larger values produce more generalised trees; smaller values allow more specific patterns
`n_components` | How much encoded categorical information is given to the classifier | Fewer components simplify the representation but may discard useful information
`rare_categories_threshold` | Which rare categorical values are considered rare | Higher values classify more values as rare and can therefore trigger the privacy warning more easily

For example, suppose we want to:
- require at least 10 observations in each leaf;
- retain enough principal components to explain 90% of its variance; and
- consider categorical values occurring fewer than 3 times to be rare.

We can configure all three behaviours in one place:
```python
tuned_cart = tune_cart(
    n_leaves=10,
    n_components=0.9,
    rare_categories_threshold=3,
)

synthesiser = Synthesiser(
    random_seed=1,
    default_syn_method=tuned_cart,
)

synthesiser.fit(data)

synthetic_data = synthesiser.generate()
```
This gives us a CART model that is less likely to make predictions from very small groups, uses a reduced representation of categorical predictors, and applies an explicit check for rare categorical values. If rare categories account for at least 25% of the observations, the check raises a warning, but synthesis can continue.

## Use different CART configurations for different columns
A tuned CART factory can also be used with [`special_syn_method`](./using_different_methods_for_different_columns.md). This is useful when most columns should use one configuration, but a particular column needs a different one. For example, suppose we want most variables to use a minimum leaf size of 10, but `fare` needs a more conservative configuration with a minimum leaf size of 20:
```python
synthesiser = Synthesiser(
    random_seed=1,
    default_syn_method=tune_cart(n_leaves=10),
    special_syn_method={
        "fare": tune_cart(n_leaves=20),
    },
)

synthesiser.fit(data)

synthetic_data = synthesiser.generate()
```

The resulting configuration is:

**Column**              | **CART configuration**
---                     | ---
`fare`                  | `n_leaves=20`
all remaining columns   | `n_leaves=10`

As with other synthesis methods, the method specified in `special_syn_method` overrides the default for that column.

## When should you use `tune_cart`?
Constructing `CartMethod` manually remains the most flexible option when you need to customise individual components of the CART method. For example, if you want to replace the PCA encoder or use a different missing-value handling strategy, construct `CartMethod` and the underlying components directly:
```python
CartMethod(
    regressor=TreeRegressorMethod(...),
    classifier=TreeClassifierMethod(...),
)
```
If you only need to adjust the common CART parameters exposed by `tune_cart`, however, the convenience function is usually the simpler choice:
```python
tune_cart(
    n_leaves=5,
    n_components=None,
    rare_categories_threshold=None,
)
```
This keeps the configuration concise while ensuring that the corresponding CART components receive the same settings.

## Summary
`tune_cart` provides a convenient way to configure the most commonly tuned CART parameters without manually constructing the underlying components.

The available parameters are:
**Parameter**  | **Purpose**
----------------------------| ---
`n_leaves`                  | Sets the minimum number of observations in the leaf nodes of the decision trees.
`n_components`              | Controls the number of principal components retained by the PCA encoder.
`rare_categories_threshold` | Determines which categorical predictor values are considered rare. A warning is raised when rare categories account for at least 25% of the observations.

For more advanced customisation, see the [next examples](configure_cart_directly.md) in this module or {ref}`User Guide 3.1 CART synthesis <31-cart-synthesis>` to learn how to configure individual CART components directly.

## Next steps
We now have seen how to customise the most common CART parameters using `tune_cart`. In the next example, we will look at a more advances CART configuration, where individual components such as the decision trees, categorical encoders, and missing-value handler can be replaced or configured independently.