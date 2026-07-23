# Synthpop-py

**synthpop-py** is a Python package for generating synthetic tabular data using sequential modelling methods. It is based on the methodology and ideas of the established [`synthpop` R package](https://www.synthpop.org.uk/), while providing a modern python interface and software architecture designed for usability, reproducibility, maintainability and extensibility.

Synthetic data can help researchers, data scientist, statisticians and organisations work with realistic datasets when access to original or sensitive data is restricted. synthpop-py learns statistical relationships from an original dataset and uses these relationships to generate new synthetic observations. Properly synthesised data does not have any row-to-row relationships to the original data. As such, it may be considered non-personal data in the concept of privacy laws such as the GDPR.

> **Project status**: synthpop-py is currently under active development and the project is working towards its `1.0.0` release

## Quick start

The following example demonstrates the basic synthpop-py workflow:

```python
import pandas as pd
from synthpop import Synthesiser

# Load original data
orig_df = pd.read_csv("path/to/your/data.csv")

# Create and fit the synthesiser
synth = Synthesiser(random_seed=42)
synth.fit(orig_df)

# Generate synthetic data
syn_df = synth.generate()
```

The original data is provided as a `pandas.DataFrame`. The `Synthesiser` learns the required synthesis models when `fit()` is called, after which `generate()` creates a new synthetic dataset. It is possible to generate a dataset with a different number of observations than the original data.

For a complete introduction to preparing data, configuring synthesis, generating synthetic datasets, and evaluating the results, see the [Getting Started guide](https://synthpop-py.readthedocs.io/en/develop/getting_started/getting_started.html).

## Installation
### Requirements

synthpop-py requires:
- **Python 3.13 or later**
- **pandas 3.0.2 or later**
- **scikit-learn 1.8.0 or later**
- **plotly 6.5.2 or later**

The runtime dependencies are installed when installing synthpop-py from PyPI

### Install from PyPi
The recommended installation method is through PyPI:

```bash
pip install synthpop-py
```

You can verify the installation by importing the package:

```python
import synthpop
```

For installation in a virtual environment, installation from source, and troubleshooting, see the [Installation documentation](https://synthpop-py.readthedocs.io/en/latest/getting_started/installation.html).

## Documentation

The full documentation is available on [Read the Docs](https://synthpop-py.readthedocs.io/en/latest/).

It includes:

* **[Getting Started](https://synthpop-py.readthedocs.io/en/develop/getting_started/getting_started.html)** — an introduction to generating synthetic data with `synthpop-py`.
* **[User Guide](https://synthpop-py.readthedocs.io/en/latest/user_guides/user_guides_index.html)** — detailed guidance on data preparation, synthetic data generation, synthesis methods, and evaluation.
* **[Examples](https://synthpop-py.readthedocs.io/en/latest/examples/examples_index.html)** - detailed examples on creating a synthesis, evaluating the data, adjusting parameters and creating custom synthesis methods.
* **[API Reference](https://synthpop-py.readthedocs.io/en/develop/api_reference/synthpop.html)** — documentation of the Python API.
* **[Developer documentation](https://synthpop-py.readthedocs.io/en/latest/developer/developer_index.html)** — information about the project architecture, development workflow, and contributing to the package.

## Supported data and synthesis methods

synthpop-py is designed for structured tabular datasets. It currently supports:

* numerical variables;
* categorical variables;
* boolean variables;
* ordinal variables; and
* missing values.

The package is not currently designed for unstructured data such as free text, images, or audio, or for time-series data.

### Synthesis methods

synthpop-py currently provides the following synthesis methods:

* **CART** — the default sequential modelling method;
* **Sampling** — a sampling-based synthesis method; and
* **Copying** — a copying-based method.

The package is designed with extensibility in mind. Additional synthesis methods can be developed and integrated through the package's architecture, allowing users and contributors to extend the available synthesis functionality.

See the [User Guide](https://synthpop-py.readthedocs.io/en/latest/user_guides/3_synthesis_methods.html) for detailed information about synthesis methods and their configuration.

## Evaluating synthetic data

Synthetic data should be evaluated before it is used or shared. In particular, users should consider both **utility** and **privacy**.

Utility describes how well the synthetic data preserves the statistical properties and relationships that are relevant to the intended use. synthpop-py provides functionality for evaluating synthetic data utility.

Privacy concerns the extent to which information about individuals in the original data could be inferred from the synthetic data. Synthetic data can reduce the need to share sensitive original data, but generating synthetic data does **not** automatically guarantee privacy or anonymity.

Privacy evaluation functionality is planned for future development. Until then, users should independently assess the privacy risks associated with their datasets, synthesis settings, and intended use.

For more information about synthetic data evaluation, see the documentation.

## Relationship to `synthpop`

synthpop-py is based on the ideas and methodology of the established [`synthpop` R package](https://www.synthpop.org.uk/), which provides a widely used framework for generating synthetic tabular data through sequential modelling.

The sequential modelling approach synthesises variables one at a time, using previously synthesised variables as predictors for subsequent variables. This allows statistical relationships between variables in the original data to be incorporated into the generated synthetic data.

synthpop-py brings this approach to the Python ecosystem through a native Python interface and a modular software architecture designed to support usability, testing, reproducibility, maintainability, and future extension.

The two packages are related but are not currently feature-equivalent. The original R package provides a broader collection of synthesis methods and evaluation functionality, while synthpop-py currently focuses on a smaller set of synthesis methods with an emphasis on providing a robust and extensible Python implementation.

The development of synthpop-py has been undertaken with approval from, and in cooperation with, the authors of the original `synthpop` R package.

For background on the methodology and the use of `synthpop`, see the references below.

## Limitations and responsible use

Synthetic data should not automatically be considered anonymous or free from privacy risks.

The privacy and utility of a synthetic dataset depend on factors including:

* the characteristics of the original dataset;
* the synthesis methods and configuration used;
* the variables included in the data; and
* the intended use and sharing context.

Users are responsible for evaluating whether generated synthetic data is suitable for their intended purpose and whether it meets applicable privacy, security, legal, and regulatory requirements.

In particular, synthpop-py does not currently provide formal privacy guarantees such as differential privacy. Users should therefore not interpret the use of synthetic data as, by itself, providing a guarantee that individuals in the original data cannot be identified or that sensitive information cannot be inferred.

Synthetic data should also be evaluated for utility. A synthetic dataset that provides strong privacy protection may not preserve the statistical characteristics required for a particular analysis, while a highly useful synthetic dataset may retain patterns that increase disclosure risk.

## Project status and roadmap

synthpop-py is currently under active development, with the project working towards its `1.0.0` release.

The initial release focuses on providing a robust Python implementation of the core sequential modelling approach, with an emphasis on:

* usability;
* reproducibility;
* maintainability;
* extensibility; and
* integration with the Python data science ecosystem.

Future development may include:

* additional synthesis methods;
* privacy evaluation functionality; and
* performance and scalability improvements.

The project roadmap will continue to evolve based on user feedback and the needs of the synthetic data community.

## Contributing

synthpop-py is intended to be an open-source project, and contributions are welcome.

For information about contributing to the project, setting up a development environment, and the development workflow, see the project's [CONTRIBUTING.md](CONTRIBUTING.md) and the [developer documentation](https://synthpop-py.readthedocs.io/en/latest/developer/developer_index.html).

## License

synthpop-py is licensed under the **European Union Public Licence (EUPL), version 1.2**.

See the [LICENSE](LICENSE.md) file for the full licence text.

## Acknowledgements

synthpop-py is currently developed by a team of employees of the **Dutch Ministry of Education, Culture and Science**. The project is intended to be developed and maintained as open-source software.

The project builds on the methodology and ideas of the original [`synthpop` R package](https://www.synthpop.org.uk/) and is developed with approval from, and in cooperation with, its original authors.

### References

For background on the utility and application of synthetic data and the `synthpop` methodology, see:

* Ji, E., Ohn, J.H., Jo, H. et al. (2025). *Evaluating the utility of data integration with synthetic data and statistical matching*. Scientific Reports, 15, 19627. https://doi.org/10.1038/s41598-025-01514-0
* Khan, M.S.N., Reje, N., Buchegger, S. (2022). *Utility Assessment of Synthetic Data Generation Methods*. https://doi.org/10.48550/arXiv.2211.14428
* Drechsler, J. (2022). *Challenges in Measuring Utility for Fully Synthetic Data*. In Privacy in Statistical Databases: International Conference, PSD 2022, Paris, France, 220–233. https://doi.org/10.1007/978-3-031-13945-1_16

