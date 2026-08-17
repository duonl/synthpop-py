# Tune the CART synthesis method
In the previous examples, we used the synthesis methods with their default configuration. This works well as a starting point, but the characteristics of a dataset may call for some additional tuning.

For example, a dataset might contain many observations, where using larger leaf nodes could produce a more generalised model. Or categorical predictors might contain many levels, making dimensionality reduction useful before they are passed to the classifier.

The CART synthesis method can be configured by constructing a {class}`~synthpop.methods.cart_synth.CartMethod` and changing its individual components. However, this can involve configuring several underlying estimators separately. For common CART tuning options, synthpop-py provides the {func}`~synthpop.methods.cart_synth.tune_cart` convenience function. It lets us change the most commonly tuned parameters in one place and applies those settings consistently to the relevant CART components.

In this example, we will introduce how to use the convenience function. In the next example, we will show how to configure the underlying components directly.

## Start with a tuned CART method
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
The `n_leaves` parameter controls the minimum number of observations in each leaf of the decision tree used by CART. For example:
```python
synthesiser = Synthesiser(
    random_seed=1,
    default_syn_method=tune_cart(n_leaves=20),
)
```
The value `20` is passed to the `min_samples_leaf` parameter of the decision trees used for:
- classification;
- regression; and
- predicting missing values.

A decision tree works by repeatedly splitting the training data into smaller groups. With `n_leaves=20`, the tree cannot create a leaf containing fewer than 20 observations. This limits how specifically the tree can model the training data.

This affects the synthetic data because the tree's predictions are based on these groups. With a **smaller** value, the tree can create smaller and more specific groups, allowing it to capture fine-grained relationships in the data. However, this also makes it easier for the tree to learn patterns that are specific to a small number of observations. This can result in the tree [overfitting](https://en.wikipedia.org/wiki/Overfitting) the original data, which results in less privacy protection.

With a **larger** value, the tree must base its predictions on larger groups of observations. This generally produces a more generalised model and reduces the influence of individual or rare observations. The trade-off is that the tree may no longer be able to capture genuine complex relationships in the data, resulting in [underfitting](https://en.wikipedia.org/wiki/Overfitting#Underfitting:~:text=training%20and%20inference.-,Underfitting,-edit).

For example, compared with the default:
```python
tune_cart(n_leaves=5)
```
using:
```python
tune_cart(n_leaves=50)
```
makes the CART models less specific: every prediction must be based on at least 50 training observations reaching the relevant leaf.

The parameter corresponds to `min_samples_leaf` in [scikit-learn's decision tree estimators](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).

## Change the number of principal components
CART uses a {class}`~synthpop.data_processing.encoders.PCAEncoder` to encode categorical predictors before they are passed to the classifier. This is necessary because the classifier operators on numerical features, while categorical variables may contain many different categories.

The `n_components` parameter controls how much of this encoded information is retained by PCA.

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

The `rare_categories_treshold` controls this check. For example,
```python
tune_cart(rare_categories_threshold=20)
```
means that a categorical predictor containing a value that occurs fewer than 20 times causes synthesis to stop with an exception. This does not change how the CART model is fitted or how categories are synthesised. Instead, it acts as a safeguard before synthesis proceeds: it prevents the model from being fitted when a categorical predictor contains a potentially risky rare category.
```warning
the implementation of the rare category check will change in the future
```
A lower threshold allows rarer categories to pass the check, while a higher threshold is more conservative and rejects more categories. By default, `rare_categories_threshold` is `None`. In that case, `tune_cart` uses the value of `n_leaves` as the threshold. This means:
```python
tune_cart()
```
has an effective rare-category threshold of `5`, because `n_leaves` defaults to `5`. Similarly:
```python
tune_cart(n_leaves=10)
```
has an effective threshold of `10`, unless `rare_categories_threshold` is explicitly specified. This means that changing `n_leaves` can also change the rare-category check unless you provide an explicit threshold.

If you want to disable the check entirely, set the threshold to `0`:
```python
synthesiser(
    default_syn_method=tune_cart(
        rare_categories_threshold=0,
    ),
)
```
Disabling the check means that synthesis will no longer stop when a categorical predictor contains rare values. You therefore lose this safeguard against potential attribute disclosure. For more information about the reason for this check and the privacy implications of rare categorical values, see {ref}`User guide 6.1.2: Attribute disclosure <612-attribute-disclosure>`.

## Tune several parameters together
The tree parameters control different aspects of CART:
**Parameter** | **What changes?** | **Potential effect**
--------------|-------------------|---------------------
`n_leaves` | How small a group a decision tree can base a prediction on | Larger values produce more generalised trees; smaller values allow more specific patterns
`n_components` | How much encoded categorical information is given to the classifier | Fewer components simplify the representation but may discard useful information
`rare_categories_threshold` | Which rare categorical values are allowed to proceed to synthesis | Higher values provide a more conservative privacy safeguard

For example, suppose we want to:
- require at least 10 observations in each leaf;
- retain enough principal components to explain 90% of its variance; and
- reject categorical values occurring fewer than 3 times.

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
This gives us a CART model that is less likely to make predictions from very small groups, uses a reduced representation of categorical predictors, and applies an explicit check for rare categorical values.

## Use different CART configurations for different columns
A tuned CART factory can also be used with `special_syn_method`. This is useful when most columns should use one configuration, but a particular column needs a different one. For example, suppose we want most variables to use a minimum leaf size of 10, but `fare` needs a more conservative configuration with a minimum leaf size of 20:
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
`CartMethod` remains the most flexible option when you need to customise individual components of the CART method. For example, if you want to replace the PCA encoder or use a different missing-value handling strategy, construct `CartMethod` and the underlying components directly:
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
`rare_categories_threshold` | Sets the minimum frequency for categorical predictor values, or disables the check when set to `0`.

For more advanced customisation, see the next examples in this module or {ref}`User Guide 3.1 CART synthesis <31-cart-synthesis>` to learn how to configure individual CART components directly.

## Next steps
We now have seen how to customise the most common CART parameters using `tune_cart`. In the next example, we will look at a more advances CART configuration, where individual components such as the decision trees, categorical encoders, and missing-value handler can be replaced or configured independently.