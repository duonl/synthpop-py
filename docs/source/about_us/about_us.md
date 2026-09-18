# About Us

**synthpop-py** is an open-source Python package for generating high-quality synthetic data, with a primary focus on tabular data. 
It is based on the established [synthpop R package](https://www.synthpop.org.uk/), but provides a modern python interface and software architecture specifically designed for usability, reproducibility, maintainability and extensibility.

synthpop-py is developed at [**DUO (Dienst Uitvoering Onderwijs)**](https://duo.nl/particulier/), 
the executive organisation of the Dutch Ministry of Education, Culture and Science.

## Why synthetic data?

Synthetic data can help researchers, data scientist, statisticians and organisations work with realistic datasets when access to original or sensitive data is restricted. synthpop-py learns statistical relationships from an original dataset and uses these relationships to generate new synthetic observations. Properly synthesised data does not have any row-to-row relationships to the original data. As such, it may be considered non-personal data in the concept of privacy laws such as the GDPR.

However, synthetic data does not automatically eliminate privacy risks. Sensitive information or distinctive patterns from the original data may still be reproduced or inferred from synthetic data. Users should therefore always carefully consider the characteristics and sensitivity of the original data, the synthesis methods, the settings used, and the intended use and sharing context of the synthetic data.

## Supported data and synthesis methods
synthpop-py is designed for structured tabular datasets. It currently supports:

- numeric variables;
- categorical variables;
- boolean variables;
- ordinal variables; and
- missing values.
The package is not currently designed for unstructured data such as free text, images, audio, or time-series data. Synthesis is performed on one tabular dataset at a time. synthpop-py does not synthesise an entire relational database as a single unit, and it does not provide explicit support for relational structures such as primary key/foreign key relationships.

Synthesis methods
synthpop-py currently provides the following synthesis methods:

- {class}`~synthpop.methods.cart_synth.CartMethod` — A Classification and Regression Trees method, the default sequential modelling method;
- {class}`~synthpop.methods.sample_synth.SampleMethod` — a sampling-based synthesis method; and
- {class}`~synthpop.methods.copy_synth.CopyMethod` — a copying-based method.
The package is designed with extensibility in mind. Additional synthesis methods can be developed and integrated through the package's architecture, allowing users and contributors to extend the available synthesis functionality.

For more information about the methods and their implementation, please see the [User Guides](../user_guides/user_guides_index.md).

## Intended users

synthpop-py is primarily intended for:

- Government organisations working with sensitive administrative data;
- Researchers conducting analyses where access to real data may be restricted;
- Data scientists and developers who need realistic data for testing and development; and
- Organisations interested in open-source approaches to synthetic data generation.

synthpop-py is designed to be flexible and accessible to users with different levels of technical experience and a wide range of requirements. The [Examples](../examples/examples_index.md) provide practical guidance for different use cases and may help you find an approach that best suits your needs.

## Limitations and responsible use

Synthetic data should not automatically be considered anonymous or free from privacy risks.

The privacy and utility of a synthetic dataset depend on factors including:

- the characteristics of the original dataset;
- the synthesis methods and configuration used;
- the variables included in the data; and
- the intended use and sharing context.
Users are responsible for evaluating whether generated synthetic data is suitable for their intended purpose and whether it meets applicable privacy, security, legal, and regulatory requirements.

In particular, synthpop-py does not currently provide formal privacy guarantees such as differential privacy. Users should therefore not assume that the use of synthetic data alone guarantees that individuals in the original data cannot be identified or that sensitive information cannot be inferred.

Synthetic data should also be evaluated for utility. A synthetic dataset that provides strong privacy protection may not preserve the statistical characteristics required for a particular analysis, while a highly useful synthetic dataset may retain patterns that increase disclosure risk.

## Project goals and roadmap

synthpop-py is developed as an open-source project. Making the source code available allows users and researchers to inspect the implementation, understand the methods used, reproduce results, and contribute to further development. Please see our [Github](https://github.com/duonl/synthpop-py) or [Developers Guide](../developer/developer_index.md) if you're interested in developing synthpop-py.

The initial release focuses on providing a robust Python implementation of the core sequential modelling approach, with an emphasis on:

- usability;
- reproducibility;
- maintainability;
- extensibility; and
- integration with the Python data science ecosystem.

Future development may include:

- additional synthesis methods;
- privacy evaluation functionality; and
- performance and scalability improvements.

The project roadmap will continue to evolve based on user feedback and the needs of the synthetic data community.

## Relationship to the synthpop R package

synthpop-py is based on the core concepts of the established [synthpop R package](https://www.synthpop.org.uk/), which provides a widely used framework for generating synthetic tabular data through sequential modelling.

The sequential modelling approach synthesises variables one at a time, using previously synthesised variables as predictors for subsequent variables. This allows statistical relationships between variables in the original data to be incorporated into the generated synthetic data.

synthpop-py brings this approach to the Python ecosystem through a native Python interface and a modular software architecture designed to support usability, testing, reproducibility, maintainability, and future extension.

The two packages are related but are not currently feature-equivalent. The original R package provides a broader collection of synthesis methods and evaluation functionality, while synthpop-py currently focuses on a smaller set of synthesis methods with an emphasis on providing a robust and extensible Python implementation.

The development of synthpop-py has been undertaken with approval from, and in cooperation with, the authors of the original synthpop R package.

For background on the methodology and the use of synthpop-py, see the references below[^1][^2][^3].

[^1]: Ji, E., Ohn, J.H., Jo, H. et al. (2025). *Evaluating the utility of data integration with synthetic data and statistical matching*. Scientific Reports, 15, 19627. https://doi.org/10.1038/s41598-025-01514-0
[^2]: Khan, M.S.N., Reje, N., Buchegger, S. (2022). *Utility Assessment of Synthetic Data Generation Methods*. https://doi.org/10.48550/arXiv.2211.14428
[^3]: Drechsler, J. (2022). *Challenges in Measuring Utility for Fully Synthetic Data*. In Privacy in Statistical Databases: International Conference, PSD 2022, Paris, France, 220–233. https://doi.org/10.1007/978-3-031-13945-1_16


```{include} ../../../GOVERNANCE.md
```

# License 

```{literalinclude} ../../../LICENSE.md
```

# Acknowledgements

synthpop-py is developed by a team of employees of the Dutch Ministry of Education, Culture and Science, within [**DUO (Dienst Uitvoering Onderwijs)**](https://duo.nl/particulier/), the executive organisation of the Ministry. The project is intended to be developed and maintained as open-source software.

The project builds on the methodology and ideas of the original [synthpop R package](https://www.synthpop.org.uk/) and is developed with approval from, and in cooperation with, its original authors.