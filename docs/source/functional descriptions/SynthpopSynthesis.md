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
For each column, a model is fitted in the order specified above. All previous columns are the features, the current column is the target.
The default model is [CART](CART.md)

### 3.2 Generating a synthetic dataset

The synthetic dataset is generated incrementally, column by column:
1. The first column is synthesised by taking a sample with replacement of the user specified number of rows. 
2. For subsequent models, the fitted model is used to generate synthetic data using the data that has already be synthesised. 

## 4. Mathematical properties and constraints

Only numeric and categorical variables are supported. Other data types must be converted or removed before synthesis.

## 5. Edge cases and special situations

### 5.1 Original dataframe is empty
If the user attempts to make a synthetic version of a empty dataframe, en exception should be raised. 

### 5.2 One column in the dataframe
If there is only one column in the dataframe, that one column should be sampled.

## 6. Limitations and considerations

### 6.1 No prediction matrix

In synthpop-R, users can configure a prediction matrix which specifies, for each variable, which previously synthesised variables should be included or excluded as predictors in the synthesis method. In the current version of synthpop-py, this level of control is not available: all variables synthesised earlier are automatically used as features in the model.
