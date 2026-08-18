# Configure the CART components directly
In the [previous example](./tune_cart_function.md), we used {func}`~synthpop.methods.cart_synth.tune_cart` to adjust the most common CART settings. This is the most convenient approach when we want to change parameters such as the minimum leaf size or the number of PCA components.

Sometimes, however, we need more control. For example, we might want to:
- change a parameter of the underlying decision tree;
- configure an encoder in more detail;
- use a different categorical encoder;
- change how missing target values are handled; or
- replace one of the components entirely.

In these cases, we can construct a {class}`~synthpop.methods.cart_synth.CartMethod` class directly and provide the components we want to customise.

This example builds on the previous example by looking inside the `CartMethod` and showing what can be configured.

## How CART is constructed
`CartMethod` is a wrapper that chooses between two different tree-based methods depending on the type of variable being synthesised:
- {class}`~synthpop.methods.cart_synth.TreeClassifierMethod` for categorical targets;
- {class}`~synthpop.methods.cart_synth.TreeRegressorMethod` for numeric targets.

Each of these methods is itself composed of several components:
**Component**           | **Purpose**
------------------------|-------------
`tree`                  | Builds the decision tree used to model the target.
`encoder`               | Converts categorical predictor values into numeric features for the tree.
`missing_handler`       | Determines how missing values in the target are handled.
`tree_sampler`          | Samples values from the leaves of the fitted tree.
`rare_categories_threshold` | Checks categorical predictors for potentially risky rare categories

The defaults are slightly different for classification and regression:
**Component** | **Classification** | **Regression**
--------------|--------------------|-------------------
`tree` | {class}`sklearn.tree.DecisionTreeClassifier` | {class}`sklearn.tree.DecisionTreeRegressor`
`encoder` | {class}`~synthpop.data_processing.encoders.PCAEncoder` | {class}`~synthpop.data_processing.encoders.MeanEncoder`
`missing_handler` | {class}`~synthpop.data_processing.missing_value_handling.ReplaceMissingWithValue` | {class}`~synthpop.data_processing.missing_value_handling.MissingValuePredictor`
`tree_sampler` | `LeafNodeSampler` | `LeafNodeSampler`

for example, the following constructs a CART method using all of these defaults:
```python
from synthpop import Synthesiser
from synthpop.methods import CartMethod

cart = CartMethod()

synthesiser = Synthesiser(default_syn_method=cart)
```
This would be the same as just calling:
```python
synthesiser = Synthesiser()
```
The important point is that these components are not fixed parts of CART. They can be supplied explicitly without constructing the method.

This is the more flexible, but more complicated, alternative to the `tune_cart` convenience function we used in the [previous example](./tune_cart_function.md).

## Configure the underlying decision trees
The decision trees are responsible for dividing observations intro groups with similar target values. Their configuration therefore has a direct effect on how specific the resulting model can become.

With `tune_cart`, we can change the minimum leaf size:
```python
tune_cart(n_leaves=10)
```
When configuring the components directly, we can instead construct the appropriate `scikit-learn` tree ourselves.

For example, suppose we want the trees to have at least 20 observations in each leaf, while also changing another tree parameter:
```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from synthpop.methods import TreeClassifierMethod, TreeRegressorMethod

cart = CartMethod(
    classifier=TreeClassifierMethod(
        tree=DecisionTreeClassifier(
            min_samples_leaf=20,
            min_impurity_decrease=1e-6,
        ),
    ),
    regressor=TreeRegressorMethod(
        tree=DecisionTreeRegressor(
            min_samples_leaf=20,
            min_impurity_decrease=1e-6,
        ),
    ),
)

synthesiser = Synthesiser(default_syn_method=cart)
```
Here, the same tree configuration is applied to both categorical and numeric targets, but they may also receive different configurations. This gives us access to the full set of parameters supported by the corresponding `scikit-learn` estimator, rather than only the parameters exposed by `tune_cart`.

For example, we could configure parameters such as:
- `max_depth`, to limit the depth of the tree;
- `max_features`, to limit the number of features considered at each split;
- `min_samples_split`, to control when a node may be split; or
- `min_impurity_decrease`, to require a minimum improvement before making a split.

See {class}`sklearn.tree.DecisionTreeClassifier` and {class}`sklearn.tree.DecisionTreeRegressor` for the available parameters.

```note
When configuring the trees directly, remember that classification and regression use different `scikit-learn` estimators. If both types of variables occur in the dataset, configure both `classifier` and `regressor` if you want the same customisation to apply to both.
```

## Configure the categorical encoders
CART cannot pass categorical predictor values such as `"male"` or `"female"` directly to a scikit-learn decision tree. Categorical predictors are therefore transformed into numeric representations before they are passed to the tree.

The default encoder depends on the target:
- categorical targets use {class}`~synthpop.data_processing.encoders.PCAEncoder`;
- numeric targets use {class}`~synthpop.data_processing.encoders.MeanEncoder`.

We can configure the existing `PCAEncoder` by supplying a configured {class}`sklearn.decomposition.PCA` instance.

For example, in the previous example we used:
```python
tune_cart(n_components=2)
```
The equivalent using direct configuration is:
```python
from sklearn.decomposition import PCA
from synthpop.data_processing import PCAEncoder

cart = CartMethod(
        classifier=TreeClassifierMethod(
                encoder = PCAEncoder(
                        pca_transform=PCA(n_components=2)
                ),
        ),
)

synthesiser = Synthesiser(default_syn_method=cart)
```
The effect is the same: categorical predictors used by the classifier are encoded using two principal components.

Direct configuration becomes more useful when we want to configure other PCA options as well. For example:
```python
PCAEncoder(
        pca_transform=PCA(
                n_components=0.9,
                whiten=True,
                svd_solver="full",
        ),
)
```
This is something `tune_cart` deliberately does not expose. The convenience function provides a simple interface for the most common setting, while direct configuration gives us access to the underlying estimator. For the full list of available parameters, see [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

For more information about the role of PCA in CART, see {ref}`User Guide 4.1.1: PCA encoding <411-pca-encoding>`.

## Use a different encoder
We are not limited to the encoders provided by synthpop-py. The `encoder` parameter accepts a transformer that follows the expected `scikit-learn` transformer interface.

For example, we could use `scikit-learn's` {class}`sklearn.preprocessing.OneHotEncoder` instead of `MeanEncoder`:
```python
from sklearn.preprocessing import OneHotEncoder

cart = CartMethod(
        regressor=TreeRegressorMethod(
                encoder=OneHotEncoder(
                        sparse_output=False,
                        handle_unknown="ignore",
                ),
        ),
)

synthesiser = Synthesiser(default_syn_method=cart)
```
This changes how categorical predictors are represented when a numeric target is synthesised.

With the default `MeanEncoder`, each category is represented by the mean value of the numeric target for that category. For example, if `profession` is a categorical predictor and `age` is the numeric target, the encoder might produce a representation such as:
**profession** | **encoded value**
---|---
`doctor` | `42.3`
`teacher`| `38.7`
`engineer`| `45.1`

With `OneHotEncoder`, each category instead becomes a separate binary feature:
**profession** | **doctor** | **teacher** | **engineer**
---|---|---|---
`doctor`   | 1 | 0 | 0
`teacher`  | 0 | 1 | 0
`engineer` | 0 | 0 | 1

The decision tree can therefore make splits based directly on the individual categories rather than on the target-dependent mean encoding.

Other `scikit-learn` encoders can also be used, provided that they are compatible with the input and output expected by the CART implementation. For example, depending on the data and modelling goal, alternatives could include:
- {class}`sklearn.preprocessing.OneHotEncoder`, for explicit binary indicators for each category;
- {class}`sklearn.preprocessing.OrdinalEncoder`, when an ordinal or integer representation is appropriate; or
- a custom `scikit-learn` compatible transformer implementing a project-specific encoding strategy. An example of this can also be found in the example module (TO DO ADD LINK).

## Configure the missing value handling


