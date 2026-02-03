# Mean Encoding
## 1. Introduction
Mean encoding is a supervised encoding technique that transforms a categorical feature into a numeric representation by exploiting its relationship with a numerical target variable. Each category is replaced by the average value of the target for observations belonging to that category. The method preserves target-related structure in the categorical predictor while producing an one-dimensional numeric encoding. This encoding is a required step in our synthesis pipeline because the underlying decision tree models do not support categorical input variables.

## 2. Input and output
The input for mean encoding consists of two columns:
- A categorical feature with $m$ distinct levels to be encoded
- A numerical target variable

The output is a single numeric column that represents the encoded categorical feature.

## 3. Detailed process
The mean encoding process consists of the following steps:
1. Construction of the feature-target table
2. Grouping by feature level
3. Computation of target means
4. Application of the encoding map

### 3.1 Construction of the feature-target table
A two-column table is constructed from the data, here the first column contains the categorical feature values and the second column contains the corresponding numerical target values. Each row represents one observation.

### 3.2 Grouping by feature level
Let the categorical feature $X$ have distinct levels:
```{math}
C = \{c_1,c_2,\dots,c_m\}
```
For each category $c_k \in C$, define the group of observations with that category as
```{math}
G_k = \{i | x_i = c_k\}
```
where $i$ is the index of the observation in the dataset. $G_k$ is the group of all indices $i$ where the feature $x_i$ is equal to the category $c_k$

### 3.3 Computation of target means
For each feature level (category) $c_k$, the mean of the target variable is computed as

```{math}
\mu_k = \frac{1}{|G_k|} \sum_{i \in G_k} y_i,
```

where:
- $|G_k|$ is the length of the set of indices of observations with feature value $c_k$,
- $y_i$ is the target value of observation $i$.

Missing target values are ignored in the computation. The result is an encoding map:
```{math}
E = \{ (c_k, \mu_k) \mid k \in \{1, \dots, m\} \}.
```

### 3.4 Application of the encoding map
The encoding map $E$ can be interpreted as a function that maps  category to its mean target value:
```{math}
E : \mathcal{C} \to \mathbb{R}, \quad E(c_k) = \mu_k.
```
Each observation is then encoded by applying $E$ to its feature value. For observation $i$ with feature value $x_i \in C$, the encoding of $x$ will be:
```{math}
\tilde{x}_i = E(x_i)
```

## 4. Mathematical properties and constraints
### 4.1 Dimensionality
Mean encoding produces exactly one numeric value per feature level. Therefore, the encoded feature has dimension 1, regardless of the number of original categories.

## 5. Edge cases and special situations
### 5.1 Missing values
If the feature is missing, the output of the encoding should be missing.
If there is a non-missing value of the feature for which the target is always missing, the encoding should produce a missing value as well.
If there is a non-missing value of the feature for which the target is sometimes but not always missing, the encoding should exlude the missing target values when calculating the mean.


## 6. Limitations and considerations
Mean encoding assumes that the expected value of the target variable is informative for distinguishing between categories. The method is sensitive to outliers, rare categories, and data leakage.
