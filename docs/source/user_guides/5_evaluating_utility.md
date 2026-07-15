# 5. Evaluating utility
Synthetic data utility describes how well synthetic data preserve the statistical properties and analytical usefulness of the original data.

A synthetic dataset with high utility should resemble the original dataset in important aspects, such as marginal distributions, relationships between variables, and patterns relevant for downstream analyses. However, utility is application dependent: a dataset intended for exploratory analysis may require different utility properties than a dataset intended for training machine
learning models or estimating statistical quantities.

Utility evaluation should therefore be performed with respect to the intended use case. No single metric captures all aspects of synthetic data quality, and multiple complementary evaluations are recommended.

The current utility evaluation tools in synthpop-py focus on distributional similarity:
- Univariate comparisons to evaluate whether individual variables have similar distributions.
- Bivariate (pairwise) comparisons to evaluate whether relationships between pairs of variables are preserved.
- The current quantitative metric for bivariate utility evaluation is the Standardised Propensity Mean Squared Error (S_pMSE).

---

## 5.1. Workflow
A typical workflow for evaluating utility is:
1. Generate a synthetic dataset.
2. Compare univariate distributions between the original and synthetic data.
3. Evaluate how pairwise (bivariate) relationships between variables.
4. Interpret the results in relation to the intended use case.

A recommended evaluation process starts with univariate distributions because these provide a basic check that individual variables are reproduces correctly. If variables have substantially different marginal distributions, further relationship-based evaluations are unlikely to be meaningful.

After validating individual variables, pairwise relationships can be assessed. Preserving relationships between variables is important because many analyses depend on associations rather than individual distributions alone.

Evaluating utility can also extent to trivariate and multivariate distributions and relationships.

---

## 5.2. Univariate distributions
Univariate utility evaluation compares the distribution of each variable individually between the original and synthetic datasets.

Examples of aspects that can be inspected include:
- Frequency or density distributions of categorical variables.
- Histograms or density distributions of numeric variables.
- Proportions of missing values.
- Range and summary statistics of numeric variables.

Univariate distributions can typically be inspected visually. Synthpop-py provides the {func}`~synthpop.utility_metrics.plot_univariate.plot_univariate_distributions` visualisation function for comparing distributions (see {ref}`Guide 7.1: Univariate distribution visualisations <71-univariate-distribution-visualisation>`). These visualisations allow users to inspect whether synthetic variables reproduce important characteristics of the original data.
```python
from synthpop.plotting import plot_univariate_distributions
plot_univariate_distributions(original_data, synthetic_data)
```

See
{func}`~synthpop.plotting.plot_univariate.plot_univariate_distributions`
for available parameters and configuration options.

A synthetic dataset may have similar marginal distributions while still failing to preserve relationships between variables. Therefore, univariate evaluation should be complemented with relationship-based evaluation.

---

## 5.3. Bivariate distributions and relationships
Bivariate utility evaluation measures whether relationships between pairs of variables are preserved.

Examples of relationships that may be important include:
- Associations between categorical variables.
- Correlations between numeric variables.
- Relationships between categorical and numeric variables.
- Differences in missingness patterns between variables.

Pairwise comparisons provide more information than univariate comparisons, because they evaluate whether the synthetic data maintain dependencies present in the original dataset.

Synthpop-py currently evaluates pairwise relationships using the Standardised Propensity Mean Squared Error (S_pMSE), implemented through {func}`~synthpop.utility_metrics.spmse.pairwise_spmse`.

(531-spmse)=
### 5.3.1. Standardised Propensity Mean Squared Error (S_pMSE)
The pairwise Standardised Propensity Mean Squared Error (S_pMSE) is a statistical measure that quantifies differences between pairwise joint distributions in an original and synthetic dataset[^1]. This metric is also used in the original `synthpop` R implementation.

[^1]: Joshua Snoke, Gillian M. Raab, Beata Nowok, Chris Dibben, Aleksandra Slavković (2018), *General and Specific Utility Measures for Synthetic Data*, in Journal of the Royal Statistical Society: Series A (Statistics in Society), Volume 181, Issue 3, Pages 663–688.

For each pair of variables, S_pMSE compares the observed frequencies in the original dataset with those in the synthetic dataset. Missing values are included in the computation and are treated as an additional category. Consequently, S_pMSE evaluates both the preservation of observed values and the preservation of missingness patterns.
```python
from synthpop.utility_metrics.spmse import pairwise_spmse

scores = pairwise_spmse(original_data, synthetic_data)

scores.head()
```
The function returns a pandas DataFrame containing one row for every pair of variables:
column 1 | column 2 | S_pMSE
---------|----------|-------
age | age | 1.03
age | income | 15.01
income | income | 2.18

A low S_pMSE indicates that the synthetic dataset preserves the pairwise (bivariate) distributions well. A higher S_pMSE indicates larger differences between the original and synthetic datasets.

The complete function signature and available parameters are documented in
{func}`~synthpop.utility_metrics.spmse.pairwise_spmse`.

#### 5.3.1.1. Definition
For two variables $X$ and $Y$, the original and synthetic datasets are first converted into joint frequency tables.
Let:
```{math}
f_{orig}(x,y)
```
be the number of observations in the original dataset with values $x$ and $y$, and:
```{math}
f_{syn}(x,y)
```
the corresponding count in the synthetic dataset.

The synthetic frequencies are compared against the original frequencies after rescaling for differences in dataset size:
```{math}
\Delta (x,y)=f_{syn}(x,y)-\frac{n_s}{n_o}f_{orig}(x,y)
```
where:
- $n_o$ is the number of observations in the original dataset.
- $n_s$ is the number of observations in the synthetic dataset.

The expected frequency is:
```{math}
\mathbb{E}(x,y)=(f_{orig}(x,y) + f_{syn}(x,y))\frac{n_s}{n_o + n_s}
```
The S_pMSE is then:
```{math}
\text{S\_pMSE}(X,Y)=\frac{1}{k-1}\sum_{x,y}\frac{\Delta(x,y)^2}{\mathbb{E}(x,y)}
```
where $k$ is the number of unique value combinations with a non-zero expected frequency.

#### 5.3.1.2. Interpretation
S_pMSE measures the difference between the pairwise distributions of the original and synthetic datasets.
- Lower values indicate that the synthetic data better preserve the pairwise distribution.
- Higher values indicate larger deviations from the original relationships.

Synthpop-py provides the {func}`~synthpop.plotting.plot_spmse.plot_spmse` visualisation function that plots the values in a heatmap (see {ref}`Guide 7.2: S_pMSE heatmap <72-spmse-heatmap>`). This allows for fast identification of poorly synthesised variable pairs.

The absolute value of S_pMSE depends on factors such as:
- Dataset size;
- Number of unique values or categories in the variables;
- The distribution of frequencies across value combinations.

S_pMSE values are most useful when comparing synthesis approaches or identifying variable pairs where relationships are less maintained.

#### 5.3.1.3. Discretisation of numeric variables
S_pMSE operates on frequency tables, which require variables to have a finite number of levels.

Numeric variables are therefore discretised into bins before calculating the metric. The bin boundaries are determined jointly from the original and synthetic datasets.

The maximum number of bins can be controlled using the `max_bins` parameter.
```python
pairwise_spmse(original_data, synthetic_data, max_bins=25)
```
Increasing the number of bins provides a more detailed comparison but may make the metric more sensitive to small frequency differences.

Missing values are discretised in bin number `max_bins`+1.

#### 5.3.1.4. Properties
Important properties of S_pMSE are:
- **Symmetry**
```{math}
\text{S\_pMSE}(X,Y)=\text{S\_pMSE}(Y,X)
```
- **Pairwise evaluation**  
The metric evaluates relationships between two variables at a time and does not capture dependencies involving three or more variables.

#### 5.3.1.5. Limitations
S_pMSE has several limitations:
- It only evaluates pairwise relationships and does not capture multivariate dependencies.
- Numeric variables are affected by the chosen discretisation.
- Interpretation depends on dataset size and variable cardinality.

---







