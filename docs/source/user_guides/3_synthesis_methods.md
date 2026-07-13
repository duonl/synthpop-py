# 3. Synthesis methods

This section describes the synthesis methods available in **synthpop-py**. A synthesis method defines how individual columns are generated within a synthetic dataset. In `synthpop-py`, each column is generated sequentially using a user-specified or default synthesis method. This means that the generation of a given column may depend on previously synthesised columns, which act as predictors.

The synthesis process is orchestrated by the {class}`~synthpop.synthesiser.Synthesiser`, which delegates the task of synthesising each column to an appropriate method. Each method implements a common interface and exposes two key operations:

- {class}`~synthesiser.Synthesiser.fit`: learns from the observed (original) data
- {class}`~synthesiser.Synthesiser.generate`: synthesises a column using previously generated synthetic data

The default method is {class}`~synthpop.methods.cart_synth.CartMethod`, which models conditional distributions using decision trees and a leaf-node sampling strategy inspired by the R package synthpop.

Available synthesis methods are:
- {class}`~synthpop.methods.cart_synth.CartMethod`
- {class}`~synthpop.methods.sample_synth.SampleMethod`
- {class}`~synthpop.methods.copy_synth.CopyMethod`

---

## 3.1. CART synthesis (default method)

CART (Classification And Regression Trees) is the default synthesis method. 
```python
>>> X = pd.DataFrame({'age': [20, 40, 60], 'profession': ['butler', 'cook', 'cook']})
>>> y = pd.Series([50, 60, 70], name='length')
>>> method = CartMethod().fit(X, y)
>>> method.transform(X)
0    50.0
1    70.0
2    60.0
Name: length, dtype: float32
```                

CART generates a column by learning an approximation of the conditional distribution:
```{math}
P(Y \mid X_1, \dots, X_k)
```
where $Y$ is the target column and $X_1, \dots, X_k$ are previously synthesised columns.

CART (Classification And Regression Trees) models the relationship between a target column and previously synthesised columns by recursively partitioning the predictor space into smaller regions. Each region corresponds to a leaf node in the fitted decision tree.

During synthesis, a synthetic observation is first assigned to a leaf node based on its predictor values. Instead of returning the average value or most likely category predicted by the tree, synthpop-py samples from the observed target values that were present in that leaf during training. This leaf-node sampling strategy allows the synthetic data to preserve local variability and reproduce complex empirical distributions.

The following diagram illustrates a simplified CART model. The internal nodes represent decision rules applied to predictor variables, while the leaf nodes contain the empirical distribution of observed target values from which synthetic values are sampled.

```{mermaid}
flowchart TD
    A["Root node<br/>Age < 50?"] -->|Yes| B["Profession = Cook?"]
    A -->|No| C["Leaf 3<br/><br/>Observed target values:<br/>[72, 75, 80]"]

    B -->|Yes| D["Leaf 1<br/><br/>Observed target values:<br/>[50, 55, 60]"]
    B -->|No| E["Leaf 2<br/><br/>Observed target values:<br/>[62, 65, 68]"]

    D --> F["Synthetic sample:<br/>draw randomly from<br/>[50, 55, 60]"]
    E --> G["Synthetic sample:<br/>draw randomly from<br/>[62, 65, 68]"]
    C --> H["Synthetic sample:<br/>draw randomly from<br/>[72, 75, 80]"]
```

CART is recommended when:
- preserving relationships between variables is important;
- realistic conditional distributions are required;
- interpretability of local structure matters.

(311-algorithm)=
### 3.1.1. Algorithm

For the first column, no predictors are available. In that case, CART samples directly from the empirical distribution of the target column. For each subsequent column $Y$, CART performs the following steps:

1. **Feature construction**
   
   Use all previously synthesised columns as predictors:
   ```{math}
   X = (X_1, \dots, X_k)
   ```
   Where $X_1, \dots X_k$ are the columns that have already been synthesised.

2. **Preprocessing**

   The variables are prepared before fitting the decision tree:
   - Categorical predictors are encoded using the appropriate encoder (defaults: {ref}`PCA encoding <411-pca-encoding>` for categorical targets and {ref}`mean encoding <412-mean-encoding>` for numeric targets). This step is added because `scikit-learn` decision trees only work with numeric predictors. For more details, see {ref}`Guide 4.1 <41-encoding-categorical-predictors>`;
   - Missing values in the target variable are handled according to the target type. This step is added because `scikit-learn` cannot fit on missing targets. For more details, see {ref}`Guide 4.2 <42-handling-missing-values>`.

3. **Model fitting**
   
   A decision tree is fitted to learn the relationship between the predictors and target:
   ```{math}
   T = \text{Tree}(X, Y)
   ```
   The fitted tree partitions the predictor space into regions represented by leaf nodes.

4. **Leaf assignment**
   
   Each observed training observation is assigned to a leaf node:
   ```{math}
   \ell_i = T(X_i)
   ```
   Where $\ell_i$ is the leaf node reached by observation $i$.
   

5. **Store empirical leaf distributions**

   For each leaf node, synthpop-py stores the empirical distribution of target values observed during training.

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
   This distribution preserves the variability of observations within each leaf instead of reducing the target to a single predicted value such as a mean or most likely category.

6. **Synthesis**
   
   For each synthetic observation:
   - assign the observation to a leaf using the fitted decision tree;
   - sample the target value from the empirical distribution associated with that leaf.

   This leaf-node sampling strategy allows synthpop-py to reproduce local relationships and complex empirical distributions while preserving variations within regions of the predictor space.

### 3.1.2. Properties

- Captures conditional dependencies between variables.
- Preserves non-linear relationships between predictors and targets.
- Supports mixed data types through preprocessing.
- Reproduces complex empirical distributions, including highly non-Gaussian distributions, through leaf-node sampling.
- Preserves local variability rather than only modelling conditional averages or modes.

### 3.1.3. Limitations

- The quality of synthesis depends on the selected tree parameters, such as leaf size and tree complexity.
- CART models only relationships captured by the available predictors. Important dependencies cannot be reproduced if relevant predictors are unavailable.
- As part of the sequential synthesis framework, the available predictors depend on the column synthesis order. The ordering of predictors within a single CART model does not affect the fitted tree; only the selection of available predictors through the synthesis order matters.

### 3.1.4. Configuring CART
The behaviour of the CART synthesis method can be customised by replacing or configuring its individual components. For example, users can modify the underlying decision trees, change the categorical encoder or select a different missing value handling strategy.

The most flexible approach is to construct a {class}`~synthpop.methods.cart_synth.CartMethod` directly:
```python
CartMethod(
   regressor=TreeRegressorMethod(
      tree=DecisionTreeRegressor(
         min_samples_leaf=10,    # equivalent to minbucket in synthpop-r
         min_impurity_decrease=1e-08,   # equivalent to cp in synthpop-r
      ),
      missing_handler=MissingValuePredictor(
         tree=DecisionTreeClassifier(min_samples_leaf=10)
      )
   ),
   classifier=TreeClassifierMethod(
      tree=DecisionTreeClassifier(
         min_samples_leaf=10,    # equivalent to minbucket in synthpop-r
         min_impurity_decrease=1e-08,   # equivalent to cp in synthpop-r
      ),
      encoder=PCAEncoder(
         pca_transform=PCA(n_components=1)
      )
   )
)
```
For common tuning options, synthpop-py provides the convenience function {func}`~synthpop.methods.cart_synth.tune_cart`, which applies the same configuration consistently to all tree-based components:
```python
tune_cart(n_leaves=10, n_components=1)
```
Or directly in the {class}`~synthpop.synthesiser.Synthesiser`:
```python
Synthesiser(default_syn_method=tune_cart(n_leaves=10, n_components=1))
```
Currently `tune_cart` supports the following parameters:
- `n_leaves`: sets the minimum number of observations in each leaf node of the decision trees used during synthesis. Passed to `min_samples_leaf` in each `scikit-learn` tree.
- `n_components`: configures the number of principal components retained by the {class}`~synthpop.data_processing.encoders.PCAEncoder` used for categorical predictors. More information can be found in {ref}`Guide 4.1.1 <411-pca-encoding>`. 

---

## 3.2. Sample synthesis method

The {class}`~synthpop.methods.sample_synth.SampleMethod` generates a column by drawing values from its empirical marginal distribution observed in the original data. It does not use any predictors and therefore does not model relationships between variables.
```python
>>> y = pd.Series([1, 2, pd.NA], name="new_target_column")
>>> model = SampleMethod(random_state=10).fit(None, y)
>>> model.transform(None)
new_target_column
0              <NA>
1              <NA>
2                 1
```

It approximates:
```{math}
P(Y) \approx \hat{P}(Y)
```

where $\hat{P}(Y)$ is the empirical distribution of the observed column.

### 3.2.1. Algorithm

1. Compute empirical frequencies:
   ```{math}
   \hat{P}(Y = y_i) = \frac{n_i}{N}
   ```

2. Store value–frequency pairs:
   ```{math}
   \{(y_i, n_i)\}
   ```

3. During generation:
   - sample values with replacement according to $\hat{P}(Y)$.

### 3.2.2. Properties

- Preserves marginal distribution exactly in expectation
- Independent sampling per row
- Includes missing values as valid outcomes
- No dependency modelling

### 3.2.3. Limitations

- Does not preserve relationships between variables
- Equivalent to an unconditional bootstrap sampler

---

## 3.3. Copy synthesis method

The {class}`~synthpop.methods.copy_synth.CopyMethod` deterministically reproduces the observed column without any modification. It is used when a variable must remain unchanged in the synthetic dataset.
```python
>>> y = pd.Series([1, 2, pd.NA], name="new_target_column")
>>> model = CopyMethod().fit(None, y)
>>> model.transform(None)
new_target_column
0             1
1             2
2             <NA>
```

It implements:
```{math}
Y^{syn} = Y^{obs}
```

```{warning}
{class}`~synthpop.methods.copy_synth.CopyMethod` reproduces observed values directly and does not provide statistical protection for the copied variable. Any sensitive, confidential or identifying information copied using this method remains present in the synthetic dataset.
```

### 3.3.1. Algorithm

1. Store observed column:
   ```{math}
   Y \leftarrow Y^{obs}
   ```

2. During generation:
   - return stored values unchanged

3. Enforce row consistency:
   - synthetic dataset must have same number of rows as original

### 3.3.2. Properties

- Fully deterministic
- No randomness
- Preserves ordering exactly
- Preserves missing values exactly

### 3.3.3. Limitations

- No statistical modelling
- No generalisation beyond observed data
- Does not preserve relationships under resampling or permutation

---

## 3.4. Method comparison

| Method | Uses predictors | Randomness | Preserves relationships | Marginal distribution | Typical use |
|--------|----------------|-------------|------------------------|------------------------|--------------|
| {class}`~synthpop.methods.cart_synth.CartMethod`   | Yes            | Yes         | Yes                    | Approximate            | General-purpose synthesis      |
| {class}`~synthpop.methods.sample_synth.SampleMethod` | No             | Yes         | No (marginal only)                    | Exact (empirical)      | Fast baseline, simple synthesis     |
| {class}`~synthpop.methods.copy_synth.CopyMethod`   | No             | No          | No                     | Exact                  | Identifiers, structural columns  |

---

## 3.5. Practical guidance

In most cases, **CART should be used as the default synthesis method** due to its ability to model conditional relationships between variables.

However:

- Use **Copy** when a variable must remain unchanged.
- Use **Sample** when a simple marginal model is sufficient or when computational simplicity is preferred.
- Use mixed configurations via `special_syn_method` when different variables require different treatment.

All methods can be combined within a single {class}`~synthpop.synthesiser.Synthesiser` instance to support hybrid synthesis workflows.