# PCA encoding

## 1. Introduction
PCA encoding is a method to transform categorical features into numeric representations by exploiting the relationship between a feature and a categorical target variable. This method constructs a contingency table and applies Principal Component Analysis (PCA) to obtain a low-dimensional, variance-preserving numerical encoding of the feature levels. This encoding is a required step in our synthesis pipeline because the underlying decision tree models do not support categorical input variables.

## 2. Input and output
The input for PCA encoding consists of two columns:
- A categorical feature with $m$ distinct levels to be encoded.
- A categorical target variable with $q$ distinct levels.
The output is one or more numeric columns that represent the feature.

## 3. Detailed process
The PCA encoding process consists of the following steps:
1. Construction of the contingency table
2. Centring of the contingency table
3. Scaling
4. Principal Component Analysis

### 3.1 Construction of the contingency table
A contingency table $C$ is constructed with rows corresponding to the $m$ levels of the feature and columns corresponding to the $q$ levels of the target. Each entry in $C_{ij}$ represents the number of observations for which feature level $i$ co-occurs with target level $j$. As a result, $C$ is an $m \times q$ matrix.

### 3.2 Centring of the contingency table
The contingency table is centred column-wise. For each column, the mean across all rows is computed, resulting in an $m$ dimensional vector. This mean vector is subtracted from the corresponding column so that each column has zero mean.

### 3.3 Scaling
Each column of the centred contingency table is scaled by its standard deviation. For every column, the standard deviation of the entries (the numbers) is computed and the column is divided by this standard deviation. If the standard deviation of a column is zero, no scaling is applied to that column.

### 3.4 Principal Component Analysis
Principal Component Analysis is applied to the centred and scaled contingency table. PCA computes a rotation matrix that aligns the principal components with the coordinate axes and produces the associated singular values $sigma_i$, with $i \in \{1, \dots \min(m,q)\}$. 

This rotation matrix is a $q \times q$ matrix. It consists of the eigenvectors of the covariance matrix \( C^\top C\). Each eigenvector represents a principal component. The columns of this rotation matrix form the numeric encoding of the categorical feature.

## 4. Selection of principal components
Eah principal component explains a proportion of the total variance in the contingency table. This proportion is computed by dividing the eigenvalue corresponding to a component by the sum of all eigenvalues.

If at least a fraction $f\in (0,1]$ of the total variance must be preserved, the number of components $k$ is chosen as the smallest integer such that:
```{math}
\sum_{i=0}^k \frac{\sigma_i}{\sum_{j=1}^q \sigma_j} > f.
```
Only the first $k$ principal components are kept as part of the encoding.

## 5. Mathematical properties and constraints
### 5.1 Rank of the contingency table
[The number of principle components with non-zero singular values is equal to the rank of the contingency table $C$](https://en.wikipedia.org/wiki/Singular_value_decomposition). [The rank of a matrix is the dimension of the vector space spanned by its columns](https://en.wikipedia.org/wiki/Rank_(linear_algebra)). Since the columns of $C$ are $m$-dimensional, the rank of $C$ is at most $m$. Because $C$ has $q$ columns, the rank is also at most $q$. Therefore, the number of non-zero singular values is less than or equal to $\min(m,q)$.

## 6. Edge cases and special situations

|feature column | target column| output of encoding/expected behaviour|
|---------------|--------------|-------------------|
| missing |  not missing| missing|
|missing | missing | missing|
|any specific non-missing value| always missing for that specific feature value | missing|
|any specific non-missing value| sometimes but not always missing for that specific feature value|the encoding treats the missing target value as a normal value and the default PCA is applied
|many different non-missing values| the same constant over all values of the feature (any non-missing value)| The rotation matrix of PCA becomes an identity matrix. The contingency table becomes a 1D vector of the number of occurrences of the feature level. |
|constant (one non-missing value for all rows)| many different non-missing values or constant (any non-missing value) for that specific feature value| The contingency table becomes a constant vector. After centring, this vector is the zero vector. The feature should be encoded with 0.|

### 6.1 Missing values
As seen in the table above, there are different strategies for missing values depending on the context. When the feature is missing, the output of the encoding should always be missing (rows 1 and 2). If there is a non-missing value of the feature for which the target is always missing, the encoding should produce a missing value as well (row 3).
If there is a non-missing value of the feature for which the target is sometimes but not always missing, the encoding should treat a missing value as a value (row 4).

### 6.2 Constant non-missing values
Constant non-missing values create zero-variance columns. If a column in the contingency table has zero variance, scaling cannot be applied. Such columns do not contribute to variance-based component selection and effectively do not influence the PCA result. When the feature column is non-missing non-constant, but the target is constant, the PCA encoding decays to count encoding (row 5). If the feature column is constant and the target column is not missing, then the encoding becomes the total number of rows (row 6).

## 7. Limitations and considerations
PCA encoding assumes that relationships between feature levels and target classes can be meaningfully captured by linear combinations of contingency table columns. The method may be sensitive to rare categories, strong class imbalance, and the choice of variance preservation threshold. Additionally, the resulting components may be less interpretable than simpler encoding schemes.