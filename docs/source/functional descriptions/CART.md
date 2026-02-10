# CART synthesis model

## 1. Introduction

Classification And Regregression Trees (CART) can be used to generate synthetic data.
This model can generate one column of data based on multiple other columns.
The column that is to be generated is called the target. The columns used to generate the target are the features.

## 2. Input and output

The inputs for fitting a CART model are:
- The features (a tabular dataset of original data with numeric and/or categorical columns used to predict the target)
- The target (a one column table of original data)

The inputs for generating synthetic data with a CART model are:
- A synthetic version of the features used for fitting.

The output is one column of synthetic data that is simular to the target.

## 3. Detailed process

The algorithm consists of two phases: 

1. Fitting the synthesiser
2. Generating a synthetic dataset

### 3.1 Fitting the synthesiser

A - Encoding categorical features

Decision trees from scikit-learn do not run with categorical features, therefore those variables must be transformed into numeric variables. The default encoder in the synthpop synthesiser is based on the target variable: a mean encoder when the target is numeric and a PCA encoder when the target is categorical. A custom encoder can be specified when defining a specific regressor or a specific classifier. 

For details about the default encoders, please refer to those pages: [mean encoding](Mean-encoding.md) or [PCA encoding](PCA-encoding.md). The encoders are fitted on all data, including rows where the target and/or feature are missing. 

The next steps depend on wheter the target is numeric or categorical.

#### 3.1.1 Fitting with a categorical target 

f the target contains missing values, then the missing values are replaced by the value "N.a.N". If that value already occurs, an error should be raised and the process should stop. This happens because if a variable contains both an existing "N.a.N" value and 'regular' missing values, we assume a data quality problem.

#### 3.1.2 Fitting with a numeric target
The [Missing Value Predictor](Null predictor.md) is fitted with the features (without encodings) and target, without any modifications.
After, any rows where the target is missing are filtered out.
A Decision tree classifier is fitted with the (possibly encoded) features and the transformed categorical target.


### 3.2 Generating a synthetic dataset

#### 3.2.1 Generating a numerical column
- Use the missing value predictor on the already synthesised variables to generate the rows where the synthetic target variable is missing.
For the rows for which the synthetic target variable is not missing:
- Apply the same encoding used when fitting to the already synthesised categorical data.
- Apply the fitted decision tree to the (encoded) already synthesised data to determine the leaf node that each row corresponds to.
- Draw a random sample from the data associated with the leaf node.

#### 3.2.2 Generating a categorical column
- take the PCA encoding from the real data and apply it to any already synthesised categorical variable
- Apply the fitted decision tree to the (encoded) already synthesised data to determine the leaf node that each row corresponds to.
- Draw a random sample from the data associated with the leaf node.
- Replace any "N.a.N" values with missing value. 


## 4. Mathematical properties and constraints

Only numeric and categorical variables are supported. Other data types must be converted or removed before synthesis.

## 5. Edge cases and special situations

## 6. Limitations and considerations

### 6.1 No prediction matrix

In synthpop-R, users can configure a prediction matrix which specifies, for each variable, which previously synthesised variables should be included or excluded as predictors in the synthesis method. In the current version of synthpop-py, this level of control is not available: all variables synthesised earlier are automatically used as features in the model.
