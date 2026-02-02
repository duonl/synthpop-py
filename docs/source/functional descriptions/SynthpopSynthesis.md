# Synthpop synthesis

## 1. Introduction

The synthpop synthesis algorithm is designed to generate a synthetic version of tabular data. It can handle numeric and categorical variables but does not support other data types such as dates or complex objects. This algorithm proceeds sequentially across columns, synthesising each target variable using all preceding columns as predictors.

## 2. Input and output

The synthesiser requires:
- A tabular dataset with numeric and/or categorical columns.
- A column order specifying which columns are synthesised first. By default, it is the order given in the dataset.
- A default synthesising method which will be applied by to all columns. A classification or regression tree is used by default.
- An optional list of special synthesising method associated with variable names, to indicate which alternative methods should be used for which variable(s).
- A number of rows to generate for the synthetic dataset. By default, it is the same number of rows as the inputted original dataset.

The output from the synthesiser is a fully synthetic dataset with the specified number of rows and the same columns as the original dataset.

## 3. Detailed process

The algorithm consists of two phases: 

1. Fitting the synthesiser
2. Generating a synthetic dataset

### 3.1 Fitting the synthesiser

A - Encoding categorical features

Decision trees from scikit-learn do not run with categorical features, therefore those variables must be transformed into numeric variables. The default encoder in the synthpop synthesiser is based on the target variable: a mean encoder when the target is numeric and a PCA encoder when the target is categorical. A custom encoder can be specified when defining a specific regressor or a specific classifier. 

For details about the default encoders, please refer to those pages: [mean encoding](Mean-encoding.md) or [PCA encoding](PCA-encoding.md)

B - Fitting decision tree

A decision tree is fitted for the target variable, with the feature variables as predictors. Rows where the target column is empty are ignored. The decision tree is applied to the original data. The combination of the leaf node and the value of the target column is stored for every row in the original data.

C - Fitting binary classifier

A decision tree classifier is fitted. The predictors are all the features of the target column. The prediction target of this classifier is a Boolean indicating whether the target is empty or not.

### 3.2 Generating a synthetic dataset

The synthetic dataset is generated incrementally, column by column:
1. The first column is synthesised by taking a sample with replacement of the user specified number of rows. 
2. Subsequent target columns:
- Apply the fitted decision tree to the already synthesised data to determine the leaf node that it corresponds to.
- Draw a random sample from the data associated with the leaf node.
- Use the fitted binary classifier to predict the probability of the target being empty.
- Sample from the probability calculated in step 3. If true, the target value is set to missing, else the value from step 2 is used.

## 4. Mathematical properties and constraints

Only numeric and categorical variables are supported. Other data types must be converted or removed before synthesis.

## 5. Edge cases and special situations

## 6. Limitations and considerations

### 6.1 No prediction matrix

In synthpop-R, users can configure a prediction matrix which specifies, for each variable, which previously synthesised variables should be included or excluded as predictors in the synthesis method. In the current version of synthpop-py, this level of control is not available: all variables synthesised earlier are automatically used as features in the model.
