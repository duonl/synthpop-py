# About Us

**synthpop-py** is an open-source Python package for generating high-quality synthetic data, with a primary focus on tabular data. 
It is developed at [**DUO (Dienst Uitvoering Onderwijs)**](https://duo.nl/particulier/), 
the executive organisation of the Dutch Ministry of Education.

synthpop-py is a package for generating high-quality synthetic data, with a primary focus on tabular data. 
It is a Python redesign of the established [Synthpop R package](https://www.synthpop.org.uk/), 
developed with approval from, and in cooperation with, the original authors.

The project is currently developed and funded by DUO.

## Why synthetic data?

Government organisations, research institutions, and other organisations often work with data that cannot easily be shared or made available for development, testing, or research purposes.

Synthetic data can provide a solution by creating artificial datasets that reproduce relevant statistical characteristics and relationships found in real data. This can enable organisations to work with realistic data while reducing the risk of exposing sensitive information.

## Statistical modelling

synthpop-py uses a statistical modelling technique (specifically: Classification and Regression Trees) to capture patterns and relationships within the input data. Whereafter it uses the learnt model to generate synthetic data.

For more information about the methods and implementation, see the [User Guides](../user_guides/user_guides_index.md).

## Intended users

synthpop-py is primarily intended for:

- Government organisations working with sensitive administrative data
- Researchers conducting analyses where access to real data may be restricted
- Data scientists and developers who need realistic data for testing and development
- Organisations interested in open-source approaches to synthetic data generation

Generally synthpop is built very flexible, such that it proves to be usefull for different levels of skills and technical requirements. Please see the [Examples](../examples/examples_index.md) for an example that might suit your desired usecase.

## Open source

synthpop-py is developed as an open-source project. Making the source code available allows users and researchers to inspect the implementation, understand the methods used, reproduce results, and contribute to further development. Please see our [Github](https://github.com/duonl/synthpop-py) or [Developers Guide](../developer/developer_index.md) if you're interested in developing synthpop-py.

```{include} ../../../GOVERNANCE.md
```

# License 

```{literalinclude} ../../../LICENSE.md
```
