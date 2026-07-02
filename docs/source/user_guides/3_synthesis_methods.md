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

### 3.1.1. Overview

CART (Classification And Regression Trees) is the default synthesis method. It generates a column by learning a conditional model:

$$ P(Y \mid X_1, \dots, X_k) $$

where $Y$ is the target column and $X_1, \dots, X_k$ are previously synthesised columns.

Instead of directly sampling from a parametric distribution, CART partitions the feature space into regions (tree leaves) and approximates the conditional distribution within each region using empirical data.

---

#### 3.1.2. Algorithm

For each column $Y$, CART performs:

1. **Feature construction**
   - Use all previously synthesised columns as predictors:
     $$ X = (X_1, \dots, X_{k}) $$

2. **Preprocessing**
   - Categorical variables are encoded (defaults: PCA and mean encoding)
   - Missing values are handled depending on target type

3. **Model fitting**
   - Fit a decision tree:
     $$ T = \text{Tree}(X, Y) $$

4. **Leaf assignment**
   - For each observation:
     $$ \ell_i = T(X_i) $$

5. **Leaf-based sampling (key mechanism)**
   - For each leaf $\ell$, construct empirical distribution:
     $$ P(Y \mid \ell) = \frac{\text{count}(Y)}{\sum \text{count}(Y)} $$

6. **Synthesis**
   - For synthetic rows:
     - assign leaf
     - sample $Y$ from empirical leaf distribution

---

### 3.1.3. Leaf node sampling

A key feature of CART synthesis is **explicit sampling from leaf nodes**.

Instead of predicting a single value:
$$ \hat{y} = f(X) $$

the model samples:
$$ Y \sim P(Y \mid \ell(X)) $$

This preserves intra-leaf variability and allows the synthetic data to reproduce realistic local distributions rather than only conditional means or modes.

---

### 3.1.4. Properties

- Captures conditional dependencies between variables
- Preserves non-linear relationships
- Supports mixed data types
- Reproduces local empirical distributions via leaf sampling

---

### 3.1.5. Limitations

- No support for explicit predictor selection (no prediction matrix as in synthpop-R)
- Sensitive to column ordering due to sequential synthesis
- Approximation quality depends on tree depth and leaf structure

---

### 3.1.6. Use cases

CART is recommended when:

- relationships between variables are important;
- high-quality synthetic data is required;
- interpretability of local structure matters.

---

## 3.2. Sample synthesis method

### 3.2.1. Overview

The Sample method generates a column by drawing values from its empirical marginal distribution observed in the original data. It does not use any predictors and therefore does not model relationships between variables.

It approximates:

$$ P(Y) \approx \hat{P}(Y) $$

where $\hat{P}(Y)$ is the empirical distribution of the observed column.

---

### 3.2.2. Algorithm

1. Compute empirical frequencies:
   $$ \hat{P}(Y = y_i) = \frac{n_i}{N} $$

2. Store value–frequency pairs:
   $$ \{(y_i, n_i)\} $$

3. During generation:
   - sample values with replacement according to $\hat{P}(Y)$.

---

### 3.2.3. Properties

- Preserves marginal distribution exactly in expectation
- Independent sampling per row
- Includes missing values as valid outcomes
- No dependency modelling

---

### 3.2.4. Limitations

- Does not preserve relationships between variables
- Equivalent to an unconditional bootstrap sampler

---

## 3.3. Copy synthesis method

### 3.3.1. Overview

The Copy method deterministically reproduces the observed column without any modification. It is used when a variable must remain unchanged in the synthetic dataset.

It implements:

$$ Y^{syn} = Y^{obs} $$

---

### 3.3.2. Algorithm

1. Store observed column:
   $$ Y \leftarrow Y^{obs} $$

2. During generation:
   - return stored values unchanged

3. Enforce row consistency:
   - synthetic dataset must have same number of rows as original

---

### 3.3.3. Properties

- Fully deterministic
- No randomness
- Preserves ordering exactly
- Preserves missing values exactly

---

### 3.3.4. Limitations

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

## 5. Practical guidance

In most cases, **CART should be used as the default synthesis method** due to its ability to model conditional relationships between variables.

However:

- Use **Copy** when a variable must remain unchanged.
- Use **Sample** when a simple marginal model is sufficient or when computational simplicity is preferred.
- Use mixed configurations via `special_syn_method` when different variables require different treatment.

All methods can be combined within a single `Synthesiser` instance to support hybrid synthesis workflows.