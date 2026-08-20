# Workflow, structure and key concepts

synthpop-py follows a workflow for generating and evaluating synthetic data:

## 1. Prepare the data

Start with an original dataset containing the variables you want to synthesize. synthpop-py uses [pandas](https://pandas.pydata.org/docs/index.html) as its primary data interface.

Datasets should be provided as a `pandas.DataFrame`, where rows represent observations and columns represent variables. Data cleaning and most preprocessing steps are outside the scope of synthpop-py and should be performed by the user before synthesis.

## 2. Configure the synthesis process

Define the synthesis methods and optional parameters that control how synthetic data is generated.

synthpop-py supports multiple synthesis approaches, allowing users to select methods that are appropriate for their dataset and application. See [User Guide 2: Synthetic data generation](../user_guides/2_synthetic_data_generation.md) or [User Guide 3: Synthesis methods](../user_guides/3_synthesis_methods.md) for more information.

## 3. Generate synthetic data

Run the synthesis process to generate one or more synthetic datasets.

The generated datasets consist of artificial records created by learning patterns from the original data. They are designed to preserve important statistical properties without reproducing the original observations.

## 4. Evaluate synthetic data

Synthetic data should be evaluated based on both **utility** and **privacy**.

Utility evaluation measures whether the synthetic data preserves important characteristics of the original dataset. Privacy evaluation considers whether the synthetic data could reveal information about individuals in the original dataset.

Common evaluation approaches include comparing univariate distributions and calculating multivariate measures such as Standardized Propensity Mean Squared Error (S_pMSE).

## 5. Use and share synthetic data

After evaluation, synthetic data can be used for development, research, testing, and collaboration without requiring direct access to the original sensitive dataset.

# Key concepts

## Synthetic data

Synthetic data is artificially generated data designed to reproduce the statistical properties of an observed dataset. Unlike anonymized versions of real data, synthetic datasets contain newly generated records rather than modified copies of original records.

## Utility

Utility describes how useful a synthetic dataset is for analyses that would otherwise be performed on the original data. In synthpop-py, the term utility is used broadly to describe how well the synthetic data preserves the statistical properties and relationships of the original data. A synthetic dataset with high utility should therefore support analyses that lead to conclusions similar to those obtained from the original data.

## Privacy

Privacy refers to the extent to which information about individuals in the original dataset can be inferred from the synthetic data. Although synthetic data can reduce disclosure risks, generating synthetic data does not automatically guarantee privacy. The level of privacy protection required depends strongly on the intended use of the synthetic data, the characteristics of the original data, and the context in which the data will be used. Privacy should therefore be considered when evaluating synthetic datasets.

## Variables and relationships

Relationships between variables are essential for realistic synthetic data.

For example, age, education level, and income may be related in the original dataset. synthpop-py's CART-based synthesis methods learn such relationships and use them when generating synthetic records.

## Reproducibility

Synthetic data generation involves randomness. By controlling random seeds and documenting synthesis settings, users can reproduce synthetic datasets and analyses.