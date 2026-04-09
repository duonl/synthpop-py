# Sample synthesis method

## 1. Introduction
The Sample synthesis method is a probabilistic data generation approach that reproduces a target variable by drawing values from its empirical distribution observed in the original dataset. The method does not model relationships between variables and is therefore independent of any feature set. Typical use cases include generating synthetic baseline distributions for a variable, introducing controlled randomness into synthesis pipelines or serving as a simple probabilistic alternative to deterministic copying.

Within a Synthpop synthesis, the default method for the first column is a sample.

## 2. Input and output
The inputs for fitting a Sample synthesis method are:
- Features as a tabular dataset of original data with numeric and/or categorical columns (optional and not used for learning)
- A target as a one column table of the original data

The input for generating synthetic data with a Sample method is:
- A synthetic version of the features used for fitting. The number of rows determines the required output size.

The output is one column of synthetic data that is similar to the target.

## 3. Detailed process
The Sample synthesis method consists of two phases:

1. Fitting the synthesiser
2. Generating a synthetic dataset

### 3.1 Fitting the synthesiser
During fitting, the method does not learn a predictive model. Instead, it constructs an empirical representation of the target variable's distribution.

### 3.2 Generating a synthetic dataset
To generate a synthetic column, the values are drawn with replacement from the observed target distribution. Each value is selected according to its empirical probability learned during fitting. The number of samples is the same as the number of rows of the already synthesised features. The expected distribution of the synthetic output matches the empirical distribution of the original data.

## 4. Mathematical properties and constraints
The Sample synthesis method treats the target variable as an empirical discrete distribution. Sampling is performed independently for each row.

## 5. Edge cases and special situations
### 5.1 Missing values
Missing values are treated as valid outcomes in the empirical distribution and may be sampled accordingly, preserving their observed frequency.

## 6. Limitations and considerations
The Sample method does not capture any relationships between the target and feature variables.