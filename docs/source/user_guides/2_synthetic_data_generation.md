# Synthetic Data Generation

This section describes how synthetic data is generated using **synthpop-py**, including the role of the `Synthesiser`, synthesis methods, preprocessing, and the sequential modelling procedure.

---

## 2.1 Overview of the synthesis workflow

Synthetic data generation in synthpop-py follows a **sequential modelling approach**, where variables are generated one after another according to a specified column order.

At a high level, the workflow is:

```
Original data
    │
    ▼
Synthesiser.fit()
    │
    ├── Preprocessing (encoding + missing value handling)
    ├── Fit synthesis models per variable
    └── Store fitted models
    │
    ▼
Synthesiser.generate()
    │
    ├── Sequential prediction (column-by-column)
    ├── Leaf node sampling
    └── Post-processing
    │
    ▼
Synthetic data
```

Each generated variable is conditioned on previously generated variables, making the **column order a critical modelling assumption**.

---

## 2.2 The Synthesiser class

The central interface in synthpop-py is the `Synthesiser` class. It provides a unified API for fitting synthesis models and generating synthetic datasets.

```python
Synthesiser(
    random_seed=None,
    column_order=None,
    default_syn_method=None,
    special_syn_method=None
)
```

### 2.2.1 Key parameters

- **`random_seed`**  
  Controls reproducibility of both fitting and generation.

- **`column_order`**  
  Defines the order in which variables are synthesised.  
  This order is structurally important: each variable is generated conditional on previously generated variables.

- **`default_syn_method`**  
  The synthesis method applied to all variables unless explicitly overridden.

- **`special_syn_method`**  
  Dictionary mapping variable names to custom synthesis methods.

---

## 2.3 Fitting the synthesiser

The `fit` method learns a sequence of predictive models from the original dataset.

```python
synth.fit(X)
```

### 2.3.1 Behaviour

During fitting:

1. The column order is determined (either user-defined or from the DataFrame).
2. Preprocessing is applied automatically:
   - categorical variables are encoded
   - numeric variables are used directly or transformed as required by the synthesis method
3. A separate model is fitted for each variable in sequence.
4. Each model is trained using all previously processed variables as predictors.

Formally, for a variable $(X_j)$, the model learns:

$$ P(X_j \mid X_1, \dots, X_{j-1}) $$

All preceding variables are used as predictors by default.

---

## 2.4 Synthesis methods

A **synthesis method** defines how each variable is modelled conditionally.

The default method is **CART-based synthesis**, implemented via `CartMethod`, which automatically selects either:

- a regression tree (for numeric variables), or  
- a classification tree (for categorical variables)

### 2.4.1 Available methods

- CART method (default)
- Copy method
- Sample method

More information about these methods can be found in User Guide: ADD

### 2.4.2 Column-level control

Different variables can use different synthesis methods:

```python
Synthesiser(
    default_syn_method=CartMethod(),
    special_syn_method={
        "income": SampleMethod(),
        "age": CartMethod()
    }
)
```

---

## 2.5 Preprocessing

Preprocessing is handled internally during `fit()` and is not typically configured by the user.

It includes:

- encoding of non-numeric variables
- handling of missing values in numeric variables
- fitting auxiliary models required by the synthesis methods

Missing value handling is integrated into the synthesis methods and is applied automatically.

---

## 2.6 Leaf node sampling

A key component of CART-based synthesis is **leaf node sampling**.

After a decision tree is fitted for a variable:

1. Each training observation is assigned to a leaf node.
2. The empirical distribution of the target variable is stored per leaf.
3. During generation, new observations are routed through the tree.
4. Values are sampled from the empirical distribution of the corresponding leaf node.

This step ensures that synthetic values are not deterministic tree predictions but are instead **drawn from observed local distributions**, which improves variability and realism.

Although this mechanism is central to the performance of CART synthesis, it is largely internal to the implementation.

---

## 2.7 Generating synthetic data

Once fitted, synthetic data can be generated using:

```python
synthetic = synth.generate(n=1000)
```

### 2.7.1 Behaviour

The `generate` method:

1. Initializes an empty synthetic dataset.
2. Iterates through variables in `column_order`.
3. For each variable:
   - uses previously generated synthetic columns as predictors
   - applies the fitted model
   - samples values using leaf node sampling
4. Returns a fully synthetic dataset.

Each variable is generated conditionally:

$$ \tilde{X}_j \sim P(X_j \mid \tilde{X}_1, \dots, \tilde{X}_{j-1}) $$

where $(\tilde{X})$ denotes synthetic variables.

---

## 2.8 Reproducibility

Reproducibility is controlled through the `random_seed` parameter.

- Setting `random_seed` in the `Synthesiser` ensures reproducible model fitting.
- The `generate(random_seed=...)` argument overrides only the randomness of the generation step.

This allows users to:
- reuse fitted models, and
- generate multiple synthetic datasets from the same fitted synthesiser.

---

## 2.9 Minimal example

```python
from synthpop.synthesiser import Synthesiser

synth = Synthesiser(random_seed=42)
synth.fit(df)

synthetic_df = synth.generate(n=1000)
```

---

## 2.10 Key design principle

The synthesis procedure is intentionally sequential and autoregressive:

- each variable depends on previously generated variables
- structure is defined by `column_order`
- variability is introduced via leaf node sampling

This design closely follows the original **synthpop** methodology while providing a modular and extensible Python implementation.