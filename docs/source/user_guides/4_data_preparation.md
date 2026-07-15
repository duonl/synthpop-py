# 4. Data preparation for synthesis
Many synthesis methods in `synthpop-py`, most notably {class}`~synthpop.methods.cart_synth.CartMethod`, rely on machine learning models from `scikit-learn`. Because these models only accept numerical input, synthpop-py automatically prepares the data before fitting and generation.

Before a synthesis method such as CART is fitted, synthpop-py prepares the predictor variables according to their data type.

For each target column:
- numeric predictors (features) are used directly;
- categorical predictors are converted into numerical representations using an encoder; and
- missing values in the target are handled differently depending on target type:

For fitting a categorical target:
1. Replace missing target values with temporary category (e.g. "N.a.N."). (See {ref}`section 4.2.1: Missing as category <421-missing-as-category>`)
2. Encode categorical predictors using the PCA encoder. (See {ref}`section 4.1.1: PCA encoding <411-pca-encoding>`)
3. Fit a classification tree.

For fitting a numeric target:
1. Fit a Missing Value Predictor using the original predictors and target. (See {ref}`section 4.2.2: Predicting missing values <422-predicting-missing-values>`)
2. Remove rows where the target is missing.
3. Encode categorical predictors using the Mean encoder. (See {ref}`section 4.1.2: Mean encoding <412-mean-encoding>`)
4. Fit a regression tree.

All preprocessing components are stored with the fitted synthesis model and reused during generation to ensure consistent transformations.

This preprocessing is performed internally and usually requires no user intervention. In most cases the default behaviour is appropriate. Experienced users may replace individual preprocessing components when custom behaviour is required.

This section explains how categorical predictors are encoded, how missing values are handled, and how these preprocessing steps fit into the synthesis pipeline.

---

(41-encoding-categorical-predictors)=
## 4.1 Encoding categorical predictors
[`Scikit-learn` decision trees](https://scikit-learn.org/stable/modules/tree.html) only accept numerical input. Therefore, categorical variables such as:
```text
Colour
------
Red
Blue
Green
```
must be transformed into numeric representations before fitting a tree-based synthesis model.

Unlike naive label encoding, synthpop-py uses target-informed encodings that incorporate the relationship between predictor categories and the target variable. This allows categories with similar relationships to the target to receive similar numeric representations, which can improve the quality of the tree-based synthesis model.[^1]

[^1]: This is also a key difference from the original synthpop R implementation, which uses different approaches for handling categorical predictors. The encoding strategy used in synthpop-py can substantially improve computational performance and model fitting efficiency for some synthesis tasks.

The encoder used depends on the target type:
| Target type | Default encoder |
|-------------|-----------------|
| Categorical | PCA Encoder |
| Numeric | Mean Encoder |

(411-pca-encoding)=
### 4.1.1. PCA encoding
The {class}`~synthpop.data_processing.encoders.PCAEncoder` method is used for categorical targets and produces a numerical representation of categorical levels based on their relationship with the target. By default, all principal components are retained, although users can reduce the dimensionality by configuring the underlying {class}`sklearn PCA <sklearn:sklearn.decomposition.PCA>` transformation, see the example in {ref}`section 4.3.1: Choosing the number of principal components. <431_choosing_pca>`.
```python
>>> X = np.array(["a", "a", "b", "b", "c"])
>>> y = np.array(["x", "x", "y", "z", "w"])
>>> pca_encoder = PCAEncoder().fit(X, y)
>>> pca_encoder.transform(X)
array([[-1.1180340e+00, -1.5000000e+00, -1.2019867e-16],
[-1.1180340e+00, -1.5000000e+00, -1.2019867e-16],
[ 2.2360680e+00, -1.2953263e-15, -1.2019867e-16],
[ 2.2360680e+00, -1.2953263e-15, -1.2019867e-16],
[-1.1180340e+00,  1.5000000e+00, -1.2019867e-16]], dtype=float32)
```

Computation works as follows:

#### 4.1.1.1. Contingency table

Let:
- $m$ be the number of feature categories,
- $q$ be the number of target categories.

A contingency table $C \in \mathbb{R}^{m \times q}$ is constructed where:

```{math}
C_{ij} = \operatorname{count}(X = i, Y = j)
```

Each row corresponds to a feature level and each column corresponds to a target level.

#### 4.1.1.2. Centring

Each column of the contingency table is centred by subtracting its mean:
```{math}
C'_{ij} = C_{ij} - \mu_j
```
where $\mu_j$ is the mean of column $j$.

#### 4.1.1.3. Scaling

Each column is scaled by its standard deviation:
```{math}
C''_{ij} = \frac{C'_{ij}}{\sigma_j}
```
If $\sigma_j = 0$, no scaling is applied for that column.

#### 4.1.1.4. PCA projection

PCA is applied to the scaled matrix $C''$. This produces a rotation matrix defined by the eigenvectors of:
```{math}
(C'')^\top C''
```

Each category is then represented by its coordinates in the principal component space. The resulting encoding may be multi-dimensional. To select components, let $\sigma_i$ denote the singular values. The proportion of explained variance for the first $k$ components is:
```{math}
\frac{\sum_{i=1}^{k} \sigma_i}{\sum_{j=1}^{q} \sigma_j}
```

The number of components $k$ is chosen such that a desired fraction of variance is retained. By default, synthpop-py retains all components. The component selection described above is only relevant when users configure PCA dimensionality reduction manually, as seen in {ref}`section 4.3.1: Choosing the number of principal components. <431_choosing_pca>`.

(412-mean-encoding)=
### 4.1.2. Mean encoding
The {class}`~synthpop.data_processing.encoders.MeanEncoder` method is used for numeric targets. Each category is replaced by the average target value observed for that category.
```python
>>> X = np.array(["a", "a", "b", "b", "c"])
>>> y = np.array([1, 0, 2, 0, 3])
>>> encoder = MeanEncoder().fit(X, y)
>>> encoder.transform(X)
array([0.5, 0.5, 1.,  1.,  3. ], dtype=float32)
```

#### 4.1.2.1. Computation

Let $G_k$ be the set of indices where feature $X$ equals category $c_k$. The encoded value is:
```{math}
\mu_k = \frac{1}{|G_k|} \sum_{i \in G_k} y_i
```

Missing values in $y$ are ignored. Each observation is mapped as:
```{math}
\tilde{x}_i = \mu_{x_i}
```

This method produces a single numeric feature that captures the average relationship between category and target.

---

(42-handling-missing-values)=
## 4.2. Handling missing values
Missing values are handled explicitly during synthesis because standard decision tree implementations cannot train on missing target values. If missing target values are passed directly to `scikit-learn` trees, model fitting will fail with an exception. Therefore, missing targets must be transformed before fitting and reconstructed after synthesis.

It is implemented via a missing value handling interface, which transforms data before and after synthesis. Synthpop-py uses two complementary strategies for missing value handling, depending on the data type of the target:
- Categorical targets: missing values are treated as an additional category during synthesis. The synthesis model can therefore learn missingness as one of the possible target outcomes.
- Numeric targets: missingness is modelled separately from the target value. A dedicated missing value predictor learns the probability that the target is missing, while the regression model is trained only on observed target values.

This separation allows synthpop-py to reproduce both the generated values and the missingness patterns present in the original data. Missing value handling is integrated into the synthesis methods in synthpop-py.

(421-missing-as-category)=
### 4.2.1. Treating missing as category
For categorical synthesis, missing values in the target are transformed into a valid categorical state before tree training using {class}`~synthpop.data_processing.missing_value_handling.ReplaceMissingWithValue`.
```python
>>> X = np.array(["a","b","c","c"], dtype=np.dtypes.StringDType(na_object=np.nan))
>>> y = np.array(["x","y",np.nan,"z"], dtype=np.dtypes.StringDType(na_object=np.nan))
>>> replace_missing = ReplaceMissingWithValue()
>>> x_res,y_res = replace_missing.prepare_data_for_fit(X,y)
>>> y_res
array(['x', 'y', 'N.a.N.', 'z'], dtype=StringDType(na_object=nan))
>>> replace_missing.post_synth_transform(x_res, y_res)
array(['x', 'y', nan, 'z'], dtype=StringDType(na_object=nan))
```

#### 4.2.1.1. Process
Let:
```{math}
y \in C \cup \{\mathrm{NaN}\}
```
We define a transformation:
```{math}
T(y_i) = 
\begin{cases}
\text{"N.a.N." } & \text{if } y_i=\mathrm{NaN} \\
y_i & \text{otherwise}
\end{cases}
```
After synthesis, this mapping is inverted:
```{math}
\text{"N.a.N."} \to \mathrm{NaN}
```

This method ensures that trees never see missing values in targets which they cannot handle, missingness is preserved exactly and missing values are structurally reproducible without imputation.

(422-predicting-missing-values)=
### 4.2.2. Predicting missing values
For numeric targets, missing values are treated probabilistically using {class}`~synthpop.data_processing.missing_value_handling.MissingValuePredictor`.
```python
>>> X = {"num": np.array([25, 30, 35, 40])}
>>> y = np.array([1.0, np.nan, 3.0, np.nan])
>>> mvp = MissingValuePredictor()
>>> X_clean, y_clean = mvp.prepare_data_for_fit(X, y)
>>> X_clean
{'num': array([[25], [35]])}
>>> y_clean
array([1., 3.])
>>> # simulate synthetic generation step
>>> y_synth = np.array([10, 20, 30, 40])
>>> y_final = mvp.post_synth_transform(X, y_synth)
>>> y_final
array([10., nan, nan, 40.])
```

We introduce a binary missingness variable $z$ such that:
```{math}
z_i =
\begin{cases}
1, & \text{if } y_i \text{ is NaN}, \\
0, & \text{if } y_i \text{ is not NaN}.
\end{cases}
```

The probability of missingness is learned using a classification tree, in the same way that other categorical synthesis models learn target distributions:
```{math}
P(z = 1 \mid x), \quad P(z = 0 \mid x).
```

#### 4.2.2.1. Process
1. Construct the missingness target vector.
2. Apply mean encoding to the categorical predictors. Mean encoding is used as the boolean missingness vector is considered numerical.
3. Fit a classifier for missingness:
```{math}
P(z \mid x)
```
4. During synthesis:
- Pass the predictors through the missing value predictor
- Sample missingness using the probability distribution associated with the corresponding leaf node
- If $z = 1$, output NaN
- else keep the numeric value generated by the regression tree

With this {class}`~synthpop.data_processing.missing_value_handling.MissingValuePredictor`, the synthetic data can reproduce missingness structures, not just frequencies as missingness is conditional on the features.

---

## 4.3. Customising preprocessing components
The default preprocessing components are suitable for most synthesis tasks. Users can customise individual components by constructing a {class}`~synthpop.methods.cart_synth.CartMethod` manually. For convenience, some common customisations are also available through helper functions such as {func}`~synthpop.methods.cart_synth.tune_cart`.

(431_choosing_pca)=
### 4.3.1. Choosing the number of principal components
By default, {class}`~synthpop.data_processing.encoders.PCAEncoder` retains all principal components (or the number determined by the configured {class}`PCA <sklearn:sklearn.decomposition.PCA>` object). The dimensionality of the encoding can be reduced by supplying a custom PCA transformation.

For example, to retain only a single principal component:
```python
method = CartMethod(
    classifier=TreeClassifierMethod(
        encoder=PCAEncoder(
            pca_transform=PCA(n_components=1)
        )
    )
)
```
For convenience, the same PCA configuration can also be passed through {func}`~synthpop.methods.cart_synth.tune_cart`:
```python
tune_cart(n_components=1)
```
Or directly in the {class}`~synthpop.synthesiser.Synthesiser`:
```python
Synthesiser(default_syn_method=tune_cart(n_components=1))
```  
The `n_components` parameter is passed directly to {class}`PCA <sklearn:sklearn.decomposition.PCA>` and therefore accepts the same values:
- an integer specifying the exact number of principal components;
- a float between 0 and 1 specifying the fraction of explained variance to retain;
- `None` to reduce all components.

Reducing the number of components decreases the dimensionality of the encoded predictors, while increasing it preserves more of the association structure between predictors and targets. While reducing the number of components can decrease the quality of the synthetic data, it is faster to compute.

### 4.3.2. Changing the missing value handling strategy
The missing value handling strategy is determined by the `missing_handler` argument of the internal tree synthesis methods of {class}`~synthpop.methods.cart_synth.CartMethod`.

By default:
- {class}`~synthpop.methods.cart_synth.TreeRegressorMethod` uses {class}`~synthpop.data_processing.missing_value_handling.MissingValuePredictor`;
- {class}`~synthpop.methods.cart_synth.TreeClassifierMethod` uses {class}`~synthpop.data_processing.missing_value_handling.ReplaceMissingWithValue`.

The default for the {class}`~synthpop.methods.cart_synth.TreeClassifierMethod` can be overridden to use the probabilistic {class}`~synthpop.data_processing.missing_value_handling.MissingValuePredictor` instead of {class}`~synthpop.data_processing.missing_value_handling.ReplaceMissingWithValue`:
```python
CartMethod(
    classifier=TreeClassifierMethod(
        missing_handler=MissingValuePredictor()
    )
)
```
This change is not implemented in {func}`~synthpop.methods.cart_synth.tune_cart` as it is not considered a common customisation.










