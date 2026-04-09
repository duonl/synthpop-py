# Copy synthesis method

## 1. Introduction
The Copy synthesis method is a deterministic data generation method that reproduces an observed target variable without learning a statistical relationship to any feature set.

This method is primarily used in a synthesis pipelines where a variable must be included for structural or logical completeness, but should not be influenced by other variables in the dataset. Typical use cases include ensuring a column remains part of a dataset while explicitly excluding it from predictive modelling or preserving original values for reference variables or identifiers.

## 2. Input and output
The inputs for fitting a Copy synthesis method are:
- Features as a tabular dataset of original data with numeric and/or categorical columns (optional and not used for learning)
- A target as a one column table of original data

The input for generating synthetic data with a CART model is:
- A synthetic version of the features used for fitting with the same number of rows

The output is one column of synthetic data that is identical to the target variable.

## 3. Detailed process
The Copy synthesis method consists of two phases:

1. Fitting the synthesiser
2. Generating a synthetic dataset

### 3.1 Fitting the synthesiser
During fitting, no predictive model is trained and no relationship between features and target is learned. Instead the method identifies the target variable to ensure the same variable is reproduced during generation. The observed values of the target variable are stored internally so that they can be reproduced during generation.

Any feature variables provided during fitting are ignored for modelling purposes. They are only accepted for interface consistency with other synthesis methods.

### 3.2 Generating a synthetic dataset
To generate a synthetic column, the method produces values based solely on the stored target variable from the fitting phase. The number of rows in the input synthetic dataset must match the number of values stored during fitting. The values generated for the target variable are not influenced by any values in the synthetic feature set.

## 4. Edge cases and special situations

### 4.1 Missing values
Any missing values in the original target are copied into the synthetic target.

## 5. Limitations and considerations
The Copy method does not learn relationships between variables. As a result, it cannot capture or reproduce dependencies between the target and feature variables. The method is intended as a structural or baseline component within a larger synthesis framework and is not designed to function as a standalone generative model.