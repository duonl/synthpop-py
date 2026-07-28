# User Guide
Welcome to the **synthpop-py User Guide**. This guide provides a detailed introduction to the concepts, algorithms, and workflows used throughout the package. It explains how synthetic data are generated, how the available synthesis methods and preprocessing components work, and how to evaluate the utility and privacy of synthetic datasets.

If you are new to synthpop-py, we recommended starting with the [Getting Started Guide](../getting_started/getting_started.md). It introduces the package, explains how to install it and walks through the generation of your first synthetic dataset.

The guides below provide more detailed explanations of specific topics:

- **{doc}`1_introduction`** introduces synthetic data and its role in statistical disclosure control. It discusses the motivation for generating synthetic data, common use cases, and the terminology used throughout the package.

- **{doc}`2_synthetic_data_generation`** describes the synthetic data generation process in synthpop-py. It introduces the synthesis workflow, the `Synthesiser` class, sequential synthesis, and creating synthetic data.

- **{doc}`3_synthesis_methods`** provides an overview of the available synthesis methods, explains when they are appropriate, and describes how they can be customised.

- **{doc}`4_data_preparation`** explains how data are prepared during synthesis, including preprocessing, encoding categorical variables, and handling missing values.

- **{doc}`5_evaluating_utility`** discusses how to assess whether synthetic data preserve the statistical properties of the original dataset. It introduces the utility measures currently implemented in synthpop-py and provides guidance on interpreting the results.

- **{doc}`6_evaluating_privacy`** introduces the main types of disclosure risk associated with synthetic data and explains how privacy should be considered when releasing synthetic datasets.

- **{doc}`7_visualisations`** describes the visualisation tools included in synthpop-py for comparing original and synthetic datasets and interpreting utility metrics.

- **{doc}`8_custom_synthesis_methods`** explains how to extend synthpop-py by implementing custom synthesis methods, preprocessing components, and other extensible parts of the package.

## Guides
```{toctree}
:maxdepth: 2
1_introduction.md
```
```{toctree}
:maxdepth: 2
2_synthetic_data_generation.md
```
```{toctree}
:maxdepth: 2
3_synthesis_methods.md
```
```{toctree}
:maxdepth: 3
4_data_preparation.md
```
```{toctree}
:maxdepth: 3
5_evaluating_utility.md
```
```{toctree}
:maxdepth: 3
6_evaluating_privacy.md
```
```{toctree}
:maxdepth: 2
7_visualisations.md
```
```{toctree}
:maxdepth: 2
8_custom_synthesis_methods.md
```
```{toctree}
maxdepth: 2
privacy_issue.ipynb
```
