# 3. Synthesis methods

This section describes the synthesis methods available in **synthpop-py**. A synthesis method  defines how individual columns are generated within a synthetic dataset. In `synthpop-py`, each column is generated sequentially using a user-specified or default synthesis method. This means that the generation of a given column may depend on previously synthesised columns, which act as predictors.

The synthesis process is orchestrated by the `Synthesiser`, which delegates the task of synthesising each column to an appropriate method. Each method implements a common interface and exposes two key operations:

- `fit`: learns from the observed (original) data
- `generate`: synthesises a column using previously generated synthetic data

The default method is CART, which models conditional distributions using decision trees and a leaf-node sampling strategy inspired by the R package synthpop.

Available synthesis methods are:
- CART
- Sample
- Copy

---

## 3.1. CART synthesis (default method)

CART (Classification And Regression Trees) is the default synthesis method. 
```python
>>> X = pd.DataFrame({'age': [20, 40, 60], 'profession': ['butler', 'cook', 'cook']})
>>> y_num = pd.Series([50, 60, 70], name='length')
>>> method = CartMethod().fit(X, y)
>>> method.transform(X)
0    50.0
1    70.0
2    60.0
Name: length, dtype: float32
```                

CART generates a column by learning a conditional model:
```{math}
P(Y \mid X_1, \dots, X_k)
```

where $Y$ is the target column and $X_1, \dots, X_k$ are previously synthesised columns.

Instead of directly sampling from a parametric distribution, CART partitions the feature space into regions (tree leaves) and approximates the conditional distribution within each region using empirical data.

CART is recommended when:
- relationships between variables are important;
- high-quality synthetic data is required;
- interpretability of local structure matters.

#### 3.1.1. Algorithm

For the first column, no predictors are available. In that case, CART samples directly from the empirical distribution of the target column. For each subsequent column $Y$, CART performs:

1. **Feature construction**
   - Use all previously synthesised columns as predictors:
     ```{math}
     X = (X_1, \dots, X_{k})
     ```

2. **Preprocessing**
   - Categorical variables are encoded (defaults: PCA and mean encoding)
   - Missing values are handled depending on target type

3. **Model fitting**
   - Fit a decision tree:
     ```{math}
     T = \text{Tree}(X, Y)
     ```

4. **Leaf assignment**
   - For each observation:
     ```{math}
     \ell_i = T(X_i)
     ```

5. **Leaf-based sampling (key mechanism)**
   - For each leaf $\ell$, construct empirical distribution:
     ```{math}
     P(Y \mid \ell) = \frac{\text{count}(Y)}{\sum \text{count}(Y)}
     ```

6. **Synthesis**
   - For synthetic rows:
     - assign leaf
     - sample $Y$ from empirical leaf distribution

### 3.1.1. Leaf node sampling

A key feature of CART synthesis is **explicit sampling from leaf nodes**.

Instead of predicting a single value:
```{math} 
\hat{y} = f(X)
```

the model samples:
```{math}
Y \sim P(Y \mid \ell(X))
```

This preserves intra-leaf variability and allows the synthetic data to reproduce realistic local distributions rather than only conditional means or modes.

### 3.1.3. Properties

- Captures conditional dependencies between variables
- Preserves non-linear relationships
- Supports mixed data types
- Reproduces local empirical distributions via leaf sampling

### 3.1.4. Limitations

- No support for explicit predictor selection (no prediction matrix as in synthpop-R)
- Sensitive to column ordering due to sequential synthesis
- Approximation quality depends on tree depth and leaf structure

### 3.1.5. Configuring CART
The behaviour of the CART synthesis method can be customised by replacing or configuring its individual components. For example, users can modify the underlying decision trees, change the categorical encoder or select a different missing value handling strategy.

The most flexible approach is to construct a `CartMethod` directly:
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
For common tuning options, synthpop-py provides the convenience function `tune_cart`, which applies the same configuration consistently to all tree-based components:
```python
tune_cart(n_leaves=10, n_components=1)
```
Or directly in the `Synthesiser`:
```python
Synthesiser(default_syn_method=tune_cart(n_leaves=10, n_components=1))
```
Currently `tune_cart` supports the following parameters:
- `n_leaves`: sets the minimum number of observations in each leaf node of the decision trees used during synthesis. Passed to `min_samples_leaf` in each tree.
- `n_components`: configures the number of principal components retained by the `PCAEncoder` used for categorical predictors. More information can be found in 4.3.1. (ADD LINK)

---

## 3.2. Sample synthesis method

The Sample method generates a column by drawing values from its empirical marginal distribution observed in the original data. It does not use any predictors and therefore does not model relationships between variables.
```python
>>> y = pd.Series([1, 2, pd.NA], name="new_target_column")
model = SampleMethod(random_state=10).fit(None, y)
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

The Copy method deterministically reproduces the observed column without any modification. It is used when a variable must remain unchanged in the synthetic dataset.
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

## 4. Method comparison

| Method | Uses predictors | Randomness | Preserves relationships | Marginal distribution | Typical use |
|--------|----------------|-------------|------------------------|------------------------|--------------|
| CART   | Yes            | Yes         | Yes                    | Approximate            | General-purpose synthesis      |
| Sample | No             | Yes         | No (marginal only)                    | Exact (empirical)      | Fast baseline, simple synthesis     |
| Copy   | No             | No          | No                     | Exact                  | Identifiers, structural columns  |

---

## 5. Practical guidance

In most cases, **CART should be used as the default synthesis method** due to its ability to model conditional relationships between variables.

However:

- Use **Copy** when a variable must remain unchanged.
- Use **Sample** when a simple marginal model is sufficient or when computational simplicity is preferred.
- Use mixed configurations via `special_syn_method` when different variables require different treatment.

All methods can be combined within a single `Synthesiser` instance to support hybrid synthesis workflows.