# 3. Synthesis methods

This section describes the synthesis methods available in synthpop-py. A synthesis method defines how individual columns are generated within a synthetic dataset. In synthpop-py, each column is generated sequentially using a user-specified or default synthesis method. This means that the generation of a given column may depend on previously synthesised columns, which act as predictors.

The synthesis process is orchestrated by the {class}`~synthpop.synthesiser.Synthesiser`, which delegates the task of synthesising each column to an appropriate method. Each method implements a common interface and exposes two key operations:

- {class}`~synthesiser.Synthesiser.fit`: learns from the observed (original) data
- {class}`~synthesiser.Synthesiser.generate`: synthesises a column using previously generated synthetic data

The default method is {class}`~synthpop.methods.cart_synth.CartMethod`, which models conditional distributions using decision trees and a leaf-node sampling strategy inspired by the [synthpop R package](https://www.synthpop.org.uk/).

Available synthesis methods are:
- {class}`~synthpop.methods.cart_synth.CartMethod`
- {class}`~synthpop.methods.sample_synth.SampleMethod`
- {class}`~synthpop.methods.copy_synth.CopyMethod`

---

(31-cart-synthesis)=
## 3.1. CART synthesis (default method)

CART (Classification And Regression Trees) is the default synthesis method in synthpop-py. It models the conditional distribution of each target column given the columns that have already been synthesised.

<!-- omit in toc -->
### Intended usage

CART is normally used through the {class}`~synthpop.synthesiser.Synthesiser`, which orchestrates the sequential synthesis of the complete dataset:
```python
>>> from synthpop import Synthesiser

>>> synth = Synthesiser()
>>> synthetic_data = synth.fit(original_data).generate()
```

Because CART is the default method, this is equivalent to explicitly configuring it as the default:
```python
>>> from synthpop.methods import CartMethod

>>> synth = Synthesiser(default_syn_method=CartMethod())
>>> synthetic_data = synth.fit(original_data).generate()
```

Both approaches use the same synthesis method. Explicitly passing `CartMethod()` is therefore only necessary when you want to make the choice of synthesis method explicit or customise its configuration. See {ref}`Guide 3.1.4: Configuring CART <314-configuring-cart` and the [Configure CART directly example](../examples/configure_cart_directly.md) for more information.

The method can also be configured for a specific column using `special_syn_method`. See {ref}`Guide 2.2.3: Column-level control <223-column-level-control>`.

<!-- omit in toc -->
### How CART works

For a target column $Y$, CART leans an approximation of its conditional distribution given previously synthesised columns:
```{math}
P(Y \mid X_1, \dots, X_k)
```
where $X_1, \dots, X_k$ are the available predictor columns.

CART recursively partitions the prediction space into smaller regions using a decision tree. Each region corresponds to a terminal (leaf) node. During synthesis, an observation is assigned to a leaf according to its predictor values, and the synthetic target value is sampled from the observed target values associated with that leaf.

Unlike standard decision-tree prediction, synthpop-py does not replace the target values in a leaf with a single prediction such as the mean or most likely category. Instead, it samples from the empirical distribution of observed values in that leaf. This preserves local variability and allows the method to reproduce complex, non-Gaussian conditional distributions.

The following diagram illustrates this process:

```{mermaid}
---
zoom:
---
flowchart TD
    A["Root node<br/>Age < 50?"] -->|Yes| B["Profession = Cook?"]
    A -->|No| C["Leaf 3<br/><br/>Observed target values:<br/>[72, 75, 80]"]

    B -->|Yes| D["Leaf 1<br/><br/>Observed target values:<br/>[50, 55, 60]"]
    B -->|No| E["Leaf 2<br/><br/>Observed target values:<br/>[62, 65, 68]"]

    D --> F["Synthetic sample:<br/>draw randomly from<br/>[50, 55, 60]"]
    E --> G["Synthetic sample:<br/>draw randomly from<br/>[62, 65, 68]"]
    C --> H["Synthetic sample:<br/>draw randomly from<br/>[72, 75, 80]"]
```

<!-- omit in toc -->
### When to use CART

CART is recommended when:
- preserving relationships between variables is important;
- realistic conditional distributions are required;
- non-linear relationships need to be captured;
- preserving local variability is important;
- interpretability of the learned local structure is useful; or
- limiting the influence of individual observations through minimum leaf sizes is important for privacy (see {ref}`Guide 6: Evaluating and improving privacy <6122-rare-categories` for more information).

A minimum number of observations per terminal leaf can be specified to prevent the tree from creating leaves based on very small groups of observations. Increasing the minimum leaf size generally reduces the risk of highly specific splits and limits the influence of individual observations, but can also reduce the model's ability to capture detailed patterns. The default value in synthpop-py is 5 observations per leaf.

<!-- omit in toc -->
### Direct use of `CartMethod`

{class}`~synthpop.methods.cart_synth.CartMethod` implements the synthesis-method interface used by {class}`~synthpop.synthesiser.Synthesiser`. It can therefore also be fitted and used directly:

```python
>>> from synthpop.methods import CartMethod

>>> X = pd.DataFrame({'age': [20, 40, 60], 'profession': ['butler', 'cook', 'cook']})
>>> y = pd.Series([50.0, 60.0, 70.0], dtype='float32', name='length')
>>> method = CartMethod().fit(X, y)
>>> method.transform(X)
0    50.0
1    70.0
2    60.0
Name: length, dtype: float32
```    

Here, `X` contains the predictor columns that have already been synthesised, while `y` is the observed target column that the method learns to synthesise. After fitting, `transform(X)` generates values for that target based on the supplied predictors.

This illustrates how an individual synthesis method operates, but it is not normally how a complete dataset is synthesised. `CartMethod` operates on one target column at a time, whereas {class}`~synthpop.synthesiser.Synthesiser` orchestrates the synthesis of the complete dataset. In particular, `Synthesiser` determines the column synthesis order and supplies previously generated synthetic columns as predictors.

Most users should therefore use `Synthesiser` rather than {ref}`Constructing a CART method directly <314-configuring-cart>`. Direct construction is primarily intended for advanced use cases where the underlying [components need to be customised](../examples/configure_cart_directly.md).

(311-algorithm)=
### 3.1.1. Algorithm

CART is applied sequentially to the columns of the dataset.

For the first column, no predictors are available. In that case, CART samples directly from the empirical distribution of that column. For each subsequent target column $Y$ assigned to `CartMethod`, the following steps are performed:

1. **Feature construction**
   
   Use all previously synthesised columns as predictors:
   ```{math}
   X = (X_1, \dots, X_k)
   ```
   Where $X_1, \dots X_k$ are the columns that have already been synthesised. The available predictors therefore depend on the column synthesis order.

2. **Preprocess the data**

   The predictors and target are prepared before fitting the decision tree:
   - Categorical predictors are encoded using the appropriate encoder By default, {ref}`PCA encoding <411-pca-encoding>` is used for categorical targets and {ref}`mean encoding <412-mean-encoding>` for numeric targets. This preprocessing is required because `scikit-learn` decision trees operate on numeric predictors. For more details, see {ref}`Guide 4.1: Encoding categorical predictors <41-encoding-categorical-predictors>`;
   - Missing values in the target variable are handled according to the target type because `scikit-learn` cannot be fitted when the target contains missing values. For more details, see {ref}`Guide 4.2: Handling missing values <42-handling-missing-values>`.

3. **Fit the decision tree**
   
   Fit a decision tree to learn the relationship between the predictors and target:
   ```{math}
   T = \text{Tree}(X, Y)
   ```
   The fitted tree partitions the predictor space into regions represented by leaf nodes.

4. **Assign training observations to leaves**
   
   Each observed training observation is passed through the fitted tree:
   ```{math}
   \ell_i = T(X_i)
   ```
   Where $\ell_i$ is the leaf assigned to observation $i$.
   

5. **Store the empirical distribution for each leaf**

   For each leaf node, synthpop-py stores the observed target values associated with that leaf.

   Let $Y_i$ denote the target value of observation $i$ and let $\ell_i$ denote the leaf node assigned to that observation. For a given leaf $\ell$, the probability of sampling a target value $y$ is:
   ```{math}
   P(Y=y \mid \ell)
   =
   \frac{
   \operatorname{count}(Y_i=y \text{ and } \ell_i=\ell)
   }{
   \operatorname{count}(\ell_i=\ell)
   }
   ```
   This preserves the empirical variability of observations within each leaf instead of reducing the target to a single predicted value such as a mean or most likely category.

6. **Generate synthetic values**
   
   For each synthetic observation:
   - pass its predictor values through the fitted tree to determine the corresponding leaf;
   - sample a target value from the empirical distribution stored for that leaf.

   This leaf-node sampling strategy allows synthpop-py to reproduce local relationships and complex empirical distributions while preserving variations within regions of the predictor space.

### 3.1.2. Properties

CART has several useful properties for synthetic data generation:
- **Conditional modelling**: captures dependencies between a target and its available predictors.
- **Non-linear relationships**: decision trees can represent non-linear and interaction effects without requiring them to be specified explicitly.
- **Mixed data types**: categorical and missing values can be handled through synthpop-py's preprocessing components.
- **Flexible distributions**: leaf-node sampling can reproduce complex and highly non-Gaussian empirical distributions.
- **Local variability**: sampling from observed values within leaves preserves variation that would be lost if the tree returned only a mean or mode.
- **Reproducible**: the method supports a `random_state` for reproducible sampling.

### 3.1.3. Limitations

CART also has several limitations:
- **Tree configuration**: synthesis quality depends on parameters controlling tree complexity, particularly the minimum leaf size.
- **Available predictors**: CART can only reproduce relationships that can be learned from the predictors available when a target column is synthesised.
- **Column ordering**: because synthesis is sequential, the synthesis order determines which predictors are available for each target. The ordering of predictors within a single CART model does not affect the fitted tree; it is the selection of available predictors through the synthesis order that matters.

(314-configuring-cart)=
### 3.1.4. Configuring CART
The behaviour of the CART synthesis method can be customised by replacing or configuring its underlying components. For example, users can change decision-tree hyperparameters, the categorical encoder, or the strategy used to handle missing values.

For common tuning requirements, synthpop-py provides the convenience function {func}`~synthpop.methods.cart_synth.tune_cart`, which applies the same configuration consistently to all tree-based components used by CART:
```python
>>> from synthpop.methods import tune_cart

>>> tuned_cart = tune_cart(n_leaves=10, n_components=1)
```

The resulting method can be passed to {class}`~synthpop.synthesiser.Synthesiser`:
```python
>>> Synthesiser(default_syn_method=tuned_cart)
```

Currently, `tune_cart` supports:
- `n_leaves`: sets the minimum number of observations in each leaf node of the decision trees used during synthesis. It is passed to `min_samples_leaf` in each [`scikit-learn` tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html). Increasing this value can improve privacy by limiting the influence of individual components, but may also reduce the model's ability to capture fine-grained patterns.
- `n_components`: sets the number of principal components retained by the {class}`~synthpop.data_processing.encoders.PCAEncoder` used for categorical predictors. More information can be found in {ref}`Guide 4.1.1 <411-pca-encoding>`. 

For more detailed customisation, {class}`~synthpop.methods.cart_synth.CartMethod` and its underlying components can be constructed directly:
```python
>>> CartMethod(
...    regressor=TreeRegressorMethod(
...       tree=DecisionTreeRegressor(
...          min_samples_leaf=10,    # equivalent to minbucket in synthpop-r
...          min_impurity_decrease=1e-08,   # equivalent to cp in synthpop-r
...       ),
...       missing_handler=MissingValuePredictor(
...          tree=DecisionTreeClassifier(min_samples_leaf=10)
...       )
...    ),
...    classifier=TreeClassifierMethod(
...       tree=DecisionTreeClassifier(
...          min_samples_leaf=10,    # equivalent to minbucket in synthpop-r
...          min_impurity_decrease=1e-08,   # equivalent to cp in synthpop-r
...       ),
...       encoder=PCAEncoder(
...          pca_transform=PCA(n_components=1)
...       )
...    )
... )
```

This approach provides full control over the individual components of the CART synthesis method and is intended primarily for advanced users. More information can be found in the [Configure CART directly example](../examples/configure_cart_directly.md).

---

(32-sample-synthesis)=
## 3.2. Sample synthesis method

{class}`~synthpop.methods.sample_synth.SampleMethod` is an unconditional synthesis method that generates a target column by drawing values from its empirical marginal distribution observed in the original data.

Unlike {class}`~synthpop.methods.cart_synth.CartMethod`, `SampleMethod` does not use predictors. It therefore does not model conditional relationships between variables. Each synthetic value is sampled independently from the distribution of observed values in the target column.

<!-- omit in toc -->
### Intended usage

For synthesising a complete dataset, specify `SampleMethod` when creating the {class}`~synthpop.synthesiser.Synthesiser`:
```python
>>> from synthpop import Synthesiser

>>> synth = Synthesiser(default_syn_method=SampleMethod())
>>> synthetic_data = synth.fit(original_data).generate()
```  

This configures `SampleMethod` as the default synthesis method for the dataset. {class}`~synthpop.synthesiser.Synthesiser` is responsible for orchestrating the synthesis of the complete dataset and determining which columns are passed to each synthesis method. The method can also be configured for a specific column using `special_syn_method`. See {ref}`Guide 2.2.3: Column-level control <223-column-level-control>`.

`SampleMethod` is useful when a column should be synthesised independently of the other columns, or when preserving its marginal distribution is more important than preserving relationships with other variables.

<!-- omit in toc -->
## How Sample synthesis works

For a target column $Y$, `SampleMethod` approximates its marginal distribution:
```{math}
P(Y) \approx \hat{P}(Y)
```

where $\hat{P}(Y)$ is the empirical distribution of the observed target column.

During synthesis, values are sampled independently from this empirical distribution. Values that occur frequently in the original data have a correspondingly higher probability of being sampled. Missing values are treated as valid outcomes and are sampled according to their observed frequency.

For example, if a column contains:
```python
[1, 1, 1, 2, 2, 3]
```
Then its empirical distribution is:
P(Y = 1) = 3/6
P(Y = 2) = 2/6
P(Y = 3) = 1/6

A synthetic column is generated by independently drawing values according to these probabilities. With a sufficiently large synthetic sample, its marginal distribution will approach the empirical distribution of the original column.

<!-- omit in toc -->
## When to use Sample synthesis
`SampleMethod` is recommended when:
- preserving the marginal distribution of a variable is the primary objective;
- relationships between the target variable and other variables are not important;
- the target variable should be synthesised independently of other columns;
- a simple, transparent synthesis method is preferred;
- reproducing the exact set of observed values is desirable; or
- a variable does not have meaningful predictors that can be used to model its conditional distribution.

Because `SampleMethod` samples directly from the empirical distribution, it preserves the observed frequencies of values in expectation. This makes it particularly suitable for variables where the marginal distribution is more important than relationships with other variables.

However, sampling independently does not preserve associations between variables. For example, if income varies systematically with age in the original data, applying `SampleMethod` independently to both columns will reproduce their individual marginal distributions but not their relationship. When preserving such conditional relationships is important, a method such as {ref}`CART synthesis <31-cart-synthesis>` is generally more appropriate.

<!-- omit in toc -->
## Direct use of `SampleMethod`

{class}`~synthpop.methods.sample_synth.SampleMethod` implements the synthesis-method interface used by {class}`~synthpop.synthesiser.Synthesiser`. It can therefore also be fitted and used directly:
```python
>>> from synthpop.methods import SampleMethod

>>> y = pd.Series([1, 2, pd.NA], name="new_target_column")
>>> model = SampleMethod(random_state=10).fit(None, y)
>>> model.transform(None)
new_target_column
0              <NA>
1              <NA>
2                 1
```

Unlike methods such as `CartMethod`, `SampleMethod` does not require predictor data, so both `fit` and `transform` receive `None` for the predictors. Here, `y` is the observed target column used to estimate the empirical distribution. After fitting, `transform(None)` generates synthetic values by sampling from that distribution.

This illustrates how an individual synthesis method operates, but it is not normally how a complete dataset is synthesised. `SampleMethod` operates on one target column at a time, whereas {class}`~synthpop.synthesiser.Synthesiser` orchestrates the synthesis of the complete dataset.

Most users should therefore use Synthesiser rather than constructing `SampleMethod` directly. Direct use is primarily useful when working with an individual synthesis method or when building a custom synthesis workflow.

### 3.2.1. Algorithm
For each target column $Y$ assigned to `SampleMethod`, the following steps are performed:

1. **Observe the target column**

   The method receives the original target column and does not require any predictor columns:
   ```{math}
   Y = (Y_1, \dots, Y_N)
   ```

2. **Compute empirical frequencies**

   For each distinct observed value $y_i$, calculate its empirical probability:
   ```{math}
   \hat{P}(Y = y_i) = \frac{n_i}{N}
   ```
   Where:
   - $\hat{P}(Y)$ is the empirical distribution of the observed column;
   - $y_i$ is the $i$-th distinct observed value;
   - $n_i$ is the number of times $y_i$ occurs in the original column;
   - $N$ is the total number of observed values in the original column.

   Missing values are included in the empirical distribution and therefore receive a probability based on their observed frequency.

3. **Store value–frequency pairs**

   The method stores the observed values together with their frequencies:
   ```{math}
   \{(y_i, n_i)\}
   ```

4. **Generate synthetic values**

   For each synthetic observation, sample one value independently from the empirical distribution
   ```{math}
   Y^{syn} \sim \hat{P}(Y)
   ```
   Sampling is performed with replacement, so the same observed value can be generated multiple times. The resulting synthetic column therefore has the same expected marginal distribution as the original column, but generation is not conditioned on any other variables.

### 3.2.2. Properties

`SampleMethod` has te following properties:
- **Unconditional**: does not use predictor variables.
- **Marginal distribution preservation**: reproduces the empirical marginal distribution of the target column in expectation.
- **Independent row sampling**: each synthetic value is sampled independently of the other synthetic rows.
- **Observed-value sampling**: synthetic values are drawn from values observed in the original column.
- **Missing-value preservation**: missing values can be generated with a probability corresponding to their observed frequency.
- **Simple and reproducible**: the method estimates only the empirical distribution and supports a `random_state` for reproducible sampling.

### 3.2.3. Limitations

The main limitation of `SampleMethod` is that it does not preserve relationships between variables.
- **No dependency modelling**: the distribution of $Y$ is the same regardless of the values of other variables.
- **No conditional distributions**: the method models $P(Y)$ rather than $P(Y \mid X)$.
- **No interactions**: relationships or interactions between the target and other columns cannot be reproduced.
- **Observed values only**: the method can only generate values that occur in the original column. It cannot generate new numeric values between or outside the observed values.
- **Potentially unrealistic combinations**: when several columns are generated independently using `SampleMethod`, combinations of values that are rare or absent in the original data can occur.
- **No dependency modelling through sequential synthesis**: using `SampleMethod` within the sequential {class}`~synthpop.synthesiser.Synthesiser` framework does not make the method conditional. It continues to ignore previously synthesised columns when generating its target.

For datasets where preserving relationships between variables is important, a conditional method such as {ref}`CART synthesis <31-cart-synthesis>` is generally more appropriate.

---

(33-copy-synthesis)=
## 3.3. Copy synthesis method

{class}`~synthpop.methods.copy_synth.CopyMethod` deterministically reproduces the observed values of a target column without modification. Unlike the other synthesis methods, it does not estimate a statistical model or generate new values. Instead, each synthetic row receives the corresponding value from the original column.

`CopyMethod` is intended for variables where direct reproduction of the original values is explicitly acceptable.

<!-- omit in toc-->
### Intended usage

`CopyMethod` is normally assigned to a specific column using the `special_syn_method` parameter of {class}`~synthpop.synthesiser.Synthesiser`:
```python
>>> from synthpop import Synthesiser
>>> from synthpop.methods import CopyMethod

>>> synth = Synthesiser(
...     special_syn_method={"a": CopyMethod()}
... )
>>> synthetic_data = synth.fit(original_data).generate()
```

In this example, column `a` is copied directly from the original dataset, while the other columns are synthesised using the configured default synthesis method.

```{warning}
{class}`~synthpop.methods.copy_synth.CopyMethod` reproduces observed values directly and therefore does not provide any privacy protection for the copied variable. Any sensitive, confidential or identifying information can be present in the synthetic dataset. As a result, this method should only be used for variables where direct reproduction of the original values is acceptable.
```

<!-- omit in toc -->
### How Copy synthesis works
For a target column $Y$, `CopyMethod` does not estimate a distribution or conditional relationship. Instead, it preserves the original value associated with each row:
```{math}
Y_i^{syn} = Y_i^{obs}
```
for every row $i$.

This means that the synthetic column has exactly the same values, ordering, data type, and missing-value pattern as the observed column.

Because there is a one-to-one correspondence between synthetic and observed rows, the synthetic dataset must contain the same number of rows as the original dataset. `CopyMethod` therefore does not support generating a different number of synthetic observations.

<!-- omit in toc -->
### When to use Copy synthesis
`CopyMethod` is appropriate when:
- a variable must remain unchanged in the synthetic dataset;
- the original values are intentionally allowed to be disclosed;
- a column contains information that should be retained exactly rather than statistically synthesised;
- deterministic preservation of row-level values is required; or
- the column is needed unchanged to maintain a specific data structure or downstream workflow.

`CopyMethod` may be particularly useful for non-sensitive variables that act as fixed identifiers or structural fields, provided that retaining them does not itself create a disclosure risk.

Because the method reproduces values exactly, it should **not** be used for sensitive or identifying variables unless their direct disclosure has been explicitly assessed and accepted. For guidance on assessing the privacy implications of retaining original values, see {ref}`Guide 6: Evaluating and improving privacy <6122-rare-categories>`.

<!-- omit in toc -->
### Direct use of `CopyMethod`

{class}`~synthpop.methods.copy_synth.CopyMethod` implements the synthesis-method interface used by {class}`~synthpop.synthesiser.Synthesiser`. It can therefore also be fitted and used directly:
```python
>>> from synthpop.methods import CopyMethod

>>> y = pd.Series([1, 2, pd.NA], name="new_target_column")
>>> model = CopyMethod().fit(None, y)
>>> model.transform(None)
new_target_column
0             1
1             2
2             <NA>
```

Here, `y` is the observed target column. `CopyMethod` does not require predictor data, so both `fit` and `transform` receive `None` for the predictors.

After fitting, `transform(None)` returns the stored observed values unchanged.

This illustrates how an individual synthesis method operates, but it is not normally necessary to use `CopyMethod` directly. For a complete dataset, {class}`~synthpop.synthesiser.Synthesiser` should generally be used so that `CopyMethod` can be assigned to the appropriate columns through `special_syn_method`.

### 3.3.1. Algorithm

For each target column $Y$ assigned to `CopyMethod`, the following steps are performed:

1. **Store the observed column**

   The observed target column is retained:
   ```{math}
   Y \leftarrow Y^{obs}
   ```

2. **Enforce row consistency**

   The synthetic dataset must contain the same number of rows as the original dataset. This is required because `CopyMethod` assigns each synthetic row the value from the corresponding observed row. If the number of rows were changed, there would no longer be a one-to-one correspondence between the original and synthetic observations.

3. **Generate synthetic values**
   During generation, return the stored values without modification:
   ```{math}
   Y^{syn} = Y^{obs}
   ```

### 3.3.2. Properties

`CopyMethod` has the following properties:
- **Deterministic**: repeated generation produces the same values.
- **No randomness**: no random sampling or stochastic modelling is performed.
- **Exact value preservation**: every observed value is reproduced unchanged.
- **Exact ordering**: values occur in the same row order as in the original column.
- **Exact missing-value preservation**: missing values are reproduced in their original positions.
- **No statistical modelling**: the method does not estimate a distribution or learn relationships between variables.
- **One-to-one correspondence**: each synthetic row corresponds directly to the same row in the original dataset.

### 3.3.3. Limitations

The main limitation of `CopyMethod` is that it provides no synthesis or privacy protection for the copied variable.
- **No privacy protection**: observed values are reproduced directly and may contain sensitive, confidential, or identifying information.
- **No statistical modelling**: the method does not generalise from the observed data or model its distribution.
- **Fixed number of rows**: the synthetic dataset must contain the same number of rows as the original dataset.
- **No new values**: the method cannot generate values that were not present in the original column.
- **No independence**: because values are copied row by row, the copied column retains its exact relationship with the original row structure.
- **Potential row-linkage risk**: if the copied column is combined with other information that allows rows to be linked to the original dataset, the exact values may enable disclosure.
- **Relationships can become invalid after independent row operations**: if the copied column is subsequently combined with data from another source, or if rows are reordered, filtered, sampled, or otherwise manipulated independently, the one-to-one correspondence may be broken. The copied values may then no longer correspond to the intended records or preserve their original relationships with other variables.

---

## 3.4. Method comparison

| Method | Models | Uses predictors | Randomness | Relationship modelling | Distribution preserved | Typical use |
|---|---|---|---|---|---|---|
| {class}`~synthpop.methods.cart_synth.CartMethod` | $P(Y \mid X)$ | Yes | Yes | Conditional relationships | Conditional distribution | General-purpose synthesis |
| {class}`~synthpop.methods.sample_synth.SampleMethod` | $P(Y)$ | No | Yes | No | Marginal distribution | Fast baseline, simple synthesis |
| {class}`~synthpop.methods.copy_synth.CopyMethod` | $Y^{syn} = Y^{obs}$ | No | No | No | Exact observed values | Identifiers, structural columns |


---

## 3.5. Practical guidance

In most cases, **CART should be used as the default synthesis method** because it can model conditional relationships between variables while preserving local variability and complex empirical distributions, and its tree-based modelling can provide better privacy protection than methods that directly reproduce observed values.

Choose a different method when the requirements for a particular variable are different:
- **Use CART** when relationships between variables, conditional distributions, or non-linear patterns are important.
- **Use Sample** when preserving the marginal distribution is sufficient and the variable does not need to retain relationships with other variables.
- **Use Copy** when a variable must remain exactly unchanged and direct reproduction of its observed values is acceptable.
- Use mixed configurations via `special_syn_method` when different variables require different treatment.

All methods can be combined within a single {class}`~synthpop.synthesiser.Synthesiser` instance to support hybrid synthesis workflows.