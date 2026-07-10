# Missing Value Predictor
## 1. Introduction
The Missing Value Predictor is a probabilistic component that models the presence of missing values (Null/NaNs) in a target variable as a function of the feature space. Its purpose is to reproduce realistic missingness patterns in synthetic data.

Standard decision tree regressors and classifiers in common machine learning libraries (e.g. CART in scikit-learn) can handle missing values in features but not in targets. As a result, the algorithm throws an exception and the predictors never generate NaN outputs, even when the original data contain missing targets.

The Missing Value Predictor addresses this limitation by explicitly learning the probability that a target variable is missing. It generates a Boolean outcome indicating whether the synthesised target value should be missing. This allows the synthesiser to reproduce both the frequency and the conditional structure of missingness.

## 2. Input and output
The Missing Value Predictor operates on the same inputs as the standard target predictor:
- The original dataset $X$.
- A target vector $y$ that may contain missing values.

The output is a Boolean vector $z \in \{0,1\}^n$, where  
```{math}
  z_i =
  \begin{cases}
  1 & \text{if the target is missing (NaN)} \\
  0 & \text{if the target is original}
  \end{cases}
```

This output is later used to decide whether a generated target value should be replaced by NaN.

## 3. Detailed process
For each target variable, the synthesis system trains two conceptually separate models. First, a value predictor that generates the target value assuming it is not missing. Then, a Missing Value Predictor that models the probability that the target is missing.

The Missing Value Predictor is trained and used in 4 steps:
1. Construction of the missingness target vector
2. Apply [mean encoding](Mean-encoding.md) to the categorical variables in the original data.
3. Training of the null classifier
4. Probabilistic sampling

### 3.1 Construction of the missingness target vector
The target vector $y$ is transformed into a binary indicator vector $z$ such that
```{math}
z_i =
\begin{cases}
1, & \text{if } y_i \text{ is NaN}, \\
0, & \text{if } y_i \text{ is not NaN}.
\end{cases}
```
This new vector becomes the training target for the decision tree of the Missing Value Predictor.

### 3.2 Apply mean encoding to the categorical original data.
After construction of the missingness target vector, mean encoding is applied to the categorical features. See [mean encoding](Mean-encoding.md). The newly constructed boolean target value is interpreted here as a numeric value that is either 0 or 1. 

### 3.3 Training of the null classifier
A binary decision tree classifier is trained on the original encoded dataset $X$ and the binary missingness indicator $z$. For each observation $x$, the classifier estimates a conditional distribution. This is the empirical missingness rate of the training observations that ended up in that leaf:
```{math}
P(z = 1 \mid x), \quad P(z = 0 \mid x).
```

### 3.4 Probabilistic sampling
During synthesis, the trained classifier outputs a probability vector for each input observation $x$ based on the leaf node in which $x$ falls. A Bernoulli sample is drawn from this probability vector to decide whether the target should be set to NaN. Note that the same encoding as used during fitting needs to be applied to the synthetic data before using the decision tree classifier. The mapping is learned from the original data, and not relearned from the synthetic data.

## 4. Mathematical properties and constraints
### 4.1 Separation from value prediction
The Missing Value Predictor does not generate target values. It only determines whether a value is missing. The final synthetic target is obtained as
```{math}
\tilde{y}_i =
\begin{cases}
\text{NaN}, & \text{if } z_i = 1, \\
\hat{y}_i, & \text{if } z_i = 0,
\end{cases}
```
where $\hat{y}_i$ is produced by the standard predictor.


## 5. Edge cases and special situations
### 5.1 No missing values
If the training target contains no NaNs, the missingness indicator $z$ is identically zero. In this case, the Missing Value Predictor degenerates to a constant model that always predictors "not missing".

### 5.2 All values missing
If all target values are NaN, the Missing Value Predictor degenerates to a constant model that always predicts "missing".

### 5.3 Rare missingness
When the missing values are extremely rare, many candidate splits of the tree isolate only a handful of positive samples. Enforcing a minimum leaf size prevents the tree from creating such degenerate leaves, because any split that would produce a very small child node is rejected. As a result, the model avoids divisions by zero, empty-class estimates, and other ill-defined operations, which in turn may eliminate the generation of NaN values during training or prediction.



