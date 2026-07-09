# 2. Synthetic Data Generation

This section describes how synthetic data is generated using **synthpop-py**, including the role of the `Synthesiser`, synthesis methods, preprocessing, and the sequential modelling procedure.

---

## 2.1. Overview of the synthesis workflow

Rather than modelling the joint distribution of all variables simultaneously, synthpop-py models **one variable at a time according to a specified column order**. Each variable is synthesised using previously generated variables as predictors, allowing the synthetic dataset to preserve complex dependencies while keeping the modelling procedure modular. Preprocessing, such as encoding categorical variables and handling missing values, is performed internally by the selected synthesis methods.

At a high level, the workflow is:
```{mermaid}
flowchart LR

    A[Original data]

    subgraph FIT["Synthesiser.fit()"]
        direction TB
        B[Determine column order]
        C[Fit synthesis for each variable]
        D[Apply preprocessing within synthesis methods]
        E[Store fitted models]
        B --> C --> D --> E
    end

    subgraph GEN["Synthesiser.generate()"]
        direction TB
        F[Use previously generated variables as predictors]
        G[Apply synthesis method to one column]
        H[Generate columns sequentially]
        I[Post-processing]
        F --> G --> H --> I
    end

    J[Synthetic data]

    A --> FIT
    FIT --> GEN
    GEN --> J
```

The synthesis procedure is intentionally sequential and autoregressive:
- Each generated variable is conditioned on previously generated variables
- the **column order is a critical modelling assumption** as it defines structure

This design closely follows the original **synthpop** methodology while providing a modular and extensible Python implementation.

---

## 2.2. The Synthesiser class

The central interface in synthpop-py is the `Synthesiser` class. It provides the main interface for configuring synthesis methods, fitting models on observed data and generating synthetic datasets.

```python
synth = Synthesiser(random_seed=42)
synth.fit(df)

synthetic_df = synth.generate(n=1000)
```

### 2.2.1. Key parameters

During initialisation (`Synthesiser()`):
- **`random_seed`**  
  A seed for randomness that makes both model fitting and data generation reproducible.

- **`column_order`**  
  Defines the order in which variables are synthesised.  
  This order is structurally important: each variable is generated conditional on previously generated variables.  
  If not specified, the column order of the original dataset will be used.

- **`default_syn_method`**  
  The synthesis method applied to all variables unless explicitly overridden.  
  If not specified, `CartMethod` is used as the default synthesis method.

- **`special_syn_method`**  
  Dictionary mapping variable names to custom synthesis methods.

During `fit()`:
- **`X`**  
  The original dataset used to fit the synthesiser.

During `generate()`:
- **`n`**  
  The number of rows to generate for the synthetic dataset.  
  The default is the same number of rows as the original dataset.
- **``random_seed`**  
  A seed for randomness that overrides the generation seed without refitting the synthesiser.

---

### 2.2.2. Synthesis methods

A **synthesis method** defines how each variable is modelled conditionally.

The default method is **CART-based synthesis**, implemented via `CartMethod`, which automatically selects either:

- a regression tree (for numeric variables), or  
- a classification tree (for categorical variables)

Other available methods are:
- Copy method
- Sample method

More information about these methods can be found in User Guide 3 (ADD LINK)

### 2.2.3. Column-level control

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

## 2.3. Preprocessing

Preprocessing is handled internally during `fit()` and is not typically configured by the user.

It includes:

- encoding of non-numeric variables
- handling of missing values in numeric variables
- fitting auxiliary models required by the synthesis methods

Missing value handling is integrated into the synthesis methods and is applied automatically.

Preprocessing is described in more detail in User Guide 4. (ADD LINK)

---

## 2.4. Fitting the synthesiser

The `fit` method learns a sequence of predictive models from the original dataset.

```python
synth.fit(X)
```

### 2.4.1. Behaviour

During fitting:

1. The column order is determined (either user-defined or from the DataFrame).
2. Preprocessing is applied automatically:
   - categorical variables are encoded
   - numeric variables are used directly or transformed as required by the synthesis method
3. A separate model is fitted for each variable in sequence.
4. Each model receives all previously synthesised variables in the specified column order as predictors.

Formally, for a variable $(X_j)$, the model learns:
```{math}
P(X_j \mid X_1, \dots, X_{j-1})
```

All preceding variables are used as predictors by default.

---

## 2.5. Generating synthetic data

Once fitted, synthetic data can be generated using:

```python
synthetic = synth.generate(n=1000)
```

### 2.5.1. Behaviour

The `generate` method:

1. Initializes an empty synthetic dataset.
2. Iterates through variables in `column_order`.
3. For each variable:
   - uses previously generated synthetic columns as predictors
   - applies the fitted model
   - samples values according to the fitted synthesis method. For the default CART method, this is done using leaf-node sampling (see Guide 3.1.1 ADD LINK).
4. Returns a fully synthetic dataset.

Each variable is generated conditionally:
```{math}
\tilde{X}_j \sim P(X_j \mid \tilde{X}_1, \dots, \tilde{X}_{j-1})
```

where $(\tilde{X})$ denotes synthetic variables.

---

## 2.6. Reproducibility

Reproducibility is controlled through the `random_seed` parameter.

- Setting `random_seed` when constructing the `Synthesiser` makes both model fitting and data generation reproducible.
- Passing `random_seed` to `generate()` overrides the generation seed without refitting the synthesiser.

This allows users to:
- reuse fitted models, and
- generate multiple synthetic datasets from the same fitted synthesiser.

---