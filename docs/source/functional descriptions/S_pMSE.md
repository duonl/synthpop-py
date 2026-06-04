# Standardised propensity Mean Squared Error (S_pMSE)

## 1. Introduction

The Standardised Propensity Mean Squared Error (S_pMSE) is a statistical utility measure used to quantify the similarity between an original dataset and a synthetic dataset. It evaluates how well the joint distributions of variable pairs are preserved in the synthetic data relative to the original data [1](references)

## 2. Input and output

The computation of S_pMSE requires the following inputs:
- An original dataset X with $n_o$ rows and $p$ columns.
- A synthetic dataset with $n_s$ rows and the same $p$ columns. The order of the columns is not relevant, but there needs to be a check that the column names match between both datasets.
- A maximum number of groups $\text{max\_bins} \in \mathbb{N}$ used to discretise numeric columns.

Let $X$ and $Y$ be any variables (=columns) of the original dataset. The output of the S_pMSE function is a dataset containing all pairs of variables $(X, Y)$ with their corresponding S_pMSE value. Because $\text{S\_pMSE}(X, Y)$ and $\text{S\_pMSE}(Y, X)$ are equal, $\text{S\_pMSE}(Y, X)$ is neither calculated or included in the output dataset.

## 3. Detailed process

The computation of S_pMSE for a given pair of variables $(X, Y)$ (from the original dataset) consists of the following steps:
1. Preprocessing 
2. Construction of joint frequency tables
3. Calculation of rescaled differences
4. Computation of expected frequencies
5. Calculation of S_pMSE

### 3.1 Preprocessing 

For each variable pair $(X, Y)$:
- If a variable is numeric, it is discretised into at most $\text{max\_bins}$ bins.
- Missing values (`pandas.NA`, `NumPy.nan`, `None` ) are replaced with `NumPy.nan`$.

After preprocessing, both variables are treated as categorical variables with a finite number of levels.

### 3.2 Construction of joint frequency tables

Let X and Y be any variables of the original dataset. 

For the original and synthetic datasets, joint frequency tables are constructed. Let:
- $f_{orig}(x, y)$ denote the number of observations in the original dataset for which $X=x$ and $Y=y$.
- $f_{syn}(x, y)$ denote the corresponding number of observations in the synthetic dataset.

### 3.3 Calculation of rescaled differences

For each joint category pair $(x, y)$, the difference between the synthetic and original counts is computed after rescaling the original counts to the synthetic sample size. Let:

$$ \Delta(x, y) = f_{syn}(x, y) - \frac{n_s}{n_o} * f_{orig}(x, y) $$

### 3.4 Computation of expected frequencies

For each category pair $(x, y)$, the expected frequency is defined as:

$$ \mathbb{E}(x, y) = (f_{\text{orig}}(x, y) + f_{\text{syn}}(x, y)) * \frac{n_s}{n_s+n_o}$$

Only category pairs with strictly positive expected frequency are retained for subsequent calculations. Let $k$ denote the number of such category pairs.

### 3.5 Calculation of the S_pMSE

Let $k$ be the number of unique category pair $(x, y)$, for which $f_{orig}(x, y)$ or $f_{syn}(x, y)$ is not 0.

The S_pMSE for the variable pair $(X, Y)$ is computed, for $k>1$:

$$ \text{S\_pMSE}(X, Y) = \frac{1}{k - 1} \sum_{x, y} \frac{\Delta(x, y)^2}{\mathbb{E}(x, y)} $$

The statistic is normalised by $k-1$ to account of the number of contributing category pairs.

## 4. Mathematical properties and constraints

### 4.1 Support restriction

Only category pairs with $\mathbb{E}(x, y) > 0$ contribute to the statistic. Category pairs absent from both datasets are excluded from the computation.

### 4.2 Symmetry

The statistic is invariant under swapping the original and synthetic datasets, since both $E(x, y)$ and the squared difference are symmetric.

## 5. Edge cases and special situations

### 5.1 Missing values

Missing values are treated as an explicit category (N.a.N.). While this preserves information about missingness, it may inflate the influence of missing data patterns on the S_pMSE.

If one dataset only has missing values, the S_pMSE is still defined.

### 5.2 Constant variables

If both variables are constant and equal, $k=1$ so the statistic is undefined due to division by zero. The function sends a warning and returns 0 for the variable pair.

### 5.3 Different unique categories

If some combinations of categories exist in the original dataset, but not in the synthetic dataset, the synthetic frequency is 0. And vice-versa.

## 6. Limitations and considerations

- This metric evaluates similarity at the level of pairwise joint distributions and does not capture higher-order dependencies among more than two variables. The measure is sensitive to discretisation choices for numeric variables. 
- The S_pMSE metric uses a test of significance, therefore interpreting the results is dependent on the size of the dataset: utility of a dataset can better be measured if the number of observations is large enough. 


## 7. References
(references)=
[1] Joshua Snoke, Gillian M. Raab, Beata Nowok, Chris Dibben, Aleksandra Slavković, General and Specific Utility Measures for Synthetic Data,
Journal of the Royal Statistical Society: Series A (Statistics in Society), Volume 181, Issue 3, 2018, Pages 663–688.
