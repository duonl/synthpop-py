# CART synthesis model

## 1. Introduction

Classification And Regression Trees (CART) can be used to generate synthetic data.
This model can generate one column of data based on multiple other columns.
The column that is to be generated is called the target. The columns used to generate the target are the features.

## 2. Input and output

The inputs for fitting a CART model are:
- Features as a tabular dataset of original data with numeric and/or categorical columns used to predict the target
- A target as a one column table of original data

The input for generating synthetic data with a CART model is:
- A synthetic version of the features used for fitting.

The output is one column of synthetic data that is similar to the target.

## 3. Detailed process

The algorithm consists of two phases: 

1. Fitting the synthesiser
2. Generating a synthetic dataset

### 3.1 Fitting the synthesiser

How the synthesiser is fit depends on whether the target column contains a numeric or categorical variable. Below we will explain the different approaches.

#### 3.1.1 Fitting with a categorical target 
Decision trees from scikit-learn cannot run with categorical features, therefore those variables must first be transformed into numeric variables. The default encoder in the synthpop synthesiser for categorical targets is a [PCA encoder](PCA-encoding.md). A custom encoder can be specified when defining a specific classifier.

Once the categorical features are encoded, we check for missing values in the target. If the target contains missing values, then the missing values are replaced by the value "N.a.N". If that value already occurs, an error should be raised and the process should stop. This happens when a variable contains both an existing "N.a.N." value and 'regular' missing values. We then assume a problem in data quality.

When these steps are completed, a classification tree can be fit. For details about the fitting, see the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).

#### 3.1.2 Fitting with a numeric target
As mentioned, the categorical features must first be encoded. The default encoder for numeric targets is a [Mean encoder](Mean-encoding.md). Again, a custom encoder can be specified when defining a specific classifier.

Here, if the target contains missing values, the [Missing Value Predictor](MissingValuePredictor.md) is fitted with the original unencoded features and target. After, any rows where the target is missing are filtered out. Then a regression tree can be fit using the filtered features and target. For details about the fitting, we refer you again to the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html).

### 3.2 Generating a synthetic dataset
To generate a synthetic dataset, per column is checked to see whether it will be numeric or categorical. For both types, a different scheme is followed.

For the generation of a categorical column, we apply the following steps:
1. Take the PCA encoding from the original data and apply it to any already synthesised categorical variable (previous columns).
2. Apply the fitted decision tree to the (encoded) already synthesised data to determine the leaf node that each row corresponds to.
3. Draw a random sample from the data associated with the leaf node.
4. Replace any "N.a.N." values with missing value. 

For the generation of a numeric column, we apply the following steps:

1. Apply the same mean encoding used when fitting (from the original data) to the already synthesised categorical features.
2. Apply the fitted decision tree to the (encoded) already synthesised data to determine the leaf node that each row corresponds to.
3. Draw a random sample from the data associated with the leaf node.
4. Apply the Missing Value Predictor on the newly generated column

## 4. Mathematical properties and constraints

Only numeric and categorical variables are supported. Other data types must be converted or removed before synthesis.

## 5. Edge cases and special situations

## 6. Limitations and considerations

### 6.1 No prediction matrix

In synthpop-R, users can configure a prediction matrix which specifies, for each variable, which previously synthesised variables should be included or excluded as predictors in the synthesis method. In the current version of synthpop-python, this level of control is not available; all variables synthesised earlier are automatically used as features in the model.
