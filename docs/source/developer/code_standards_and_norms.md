# Code Standards, Norms and Conventions
This document defines the coding standards and conventions used throughout the Synthpop codebase. It establishes what is considered acceptable and maintainable code, ensuring consistency across all contributions.

It covers guidelines for code structure, naming, readability, testing practices and general design principles. In addition, it describes tooling and checks that help enforce or validate these standards in practice.

These standards are not intended to prescribe a single "correct" solution in every case. Instead, they provide a shared baseline for quality and serve as a tie-breaker when multiple valid implementation approaches exist. In such cases, consistency with the codebase takes priority over personal preference.

For developers, this documentation should be used during implementation and when preparing code for review. For reviewers, it provides the criteria for evaluating whether a contribution meets the project's quality expectations.

## Language and form
- British English
- [PEP8](https://peps.python.org/pep-0008/)
- Use the term "original data" to refer to the data that is not synthetic. 
- Use typehints where appropriate.
- Code and documentation must use clear, unambiguous naming.
- Avoid abbreviations unless they are widely established in the domain (e.g. "df" for DataFrame in local scope only)
- Function and variable names should describe intent, not implementation.
- Avoid overly generic names such as `data`, `process`, `handle`, unless context makes meaning obvious.
- Error messages should be actionable (explain what went wrong and why).

## Design and develop principles
- Test Driven Development, with a code coverage of at least 80%. 
- [SOLID](https://en.wikipedia.org/wiki/SOLID) principles.
- It should always be possible to generate synthetic data within 5 lines of code. 
- All classes that inherit from BaseEstimator should follow [the developer guides of scikit-learn](https://scikit-learn.org/stable/developers/develop.html)
- Use numpy arrays where possible. 
- The user of this package should be able to use pandas.
- Prefer explicit over implicit behaviour.
- Avoid premature abstraction: prefer simple solutions until complexity justifies generalisation.

A synthesis method should accept pandas dataframes/series and heterogeneous data.
Making the data homogeneous should happen within the synthesis method.
It is best to use numpy within a synthesis method. See [#45](https://github.com/Synthpop-data/synthpop-py/issues/45).

### Informative issues and pull requests
| issue/pull request number | topic |
| ------------------------- | ----- |
| [#1](https://github.com/Synthpop-data/synthpop-py/issues/1), [#2](https://github.com/Synthpop-data/synthpop-py/pull/2)| Missing values |
| [#13](https://github.com/Synthpop-data/synthpop-py/issues/13), [#17](https://github.com/Synthpop-data/synthpop-py/issues/17), [#19](https://github.com/Synthpop-data/synthpop-py/pull/19), [#26](https://github.com/Synthpop-data/synthpop-py/pull/26)| user expectations|
| [#45](https://github.com/Synthpop-data/synthpop-py/issues/45) | relation to scikit-learn |

## Testing philosophy
Tests should primarily validate behaviour, not implementation details. The goal is to ensure correctness under expected usage and robustness under edge cases. Tests should:

- Describe real usage scenarios where possible.
- Be deterministic and independent.
- Avoid reliance on external systems unless explicitly mocked (exceptions in integration tests).
- Prefer readability over cleverness.

### Standard topics in testing
What should be tested is dependent on what is implemented and follow from the requirements. However, there are standard tests that apply to most situations. The following list should be treated as inspiration, not as an exhaustive or prescriptive set of rules:

#### 1. Output correctness (type, shape, structure)
Verify that otput conform to expected data contracts.
- Correct return type (e.g. `list`, `numpy.ndarray`, `pandas.DataFrame`, `pandas.Series`)
- Correct shape and dimensionality
- Correct schema:
    - Column names/keys
    - Column order (if relevant)
    - Data types per column or field
- No unexpected nulls or missing fields unless explicitly allowed.

#### 2. Functional correctness (behaviour)
Verify that the implementation produces correct results for known cases.
- Simple deterministic examples with known outputs
- Obvious sanity cases, for example:
    - taking the mean of a set with one value
    - sorting an already sorted list
    - metric of two identical datasets
- Identity cases (input equals output where expected)
- Symmetry/invariance properties where applicable
- Tests that reflect how the code is supposed to be used
- Functions behave according to documented assumptions
- Violations produce predictable errors.

#### 3. Edge cases and boundary conditions
Ensure robustness under extreme or unusual inputs.
- Empty inputs (empty DataFrame, empty array, empty series)
- Single-element inputs
- Zero, negative, and extreme numeric values
- Max/min values for numeric types
- Degenerate cases (constant columns, identical rows, etc.)
- For every occurrence of `==`, `<`, `<=`, `>`, `>=` there are edge cases
- For every numeric division there is a possibility to have a zero division
- Whenever there is a `==` between numeric values, there could be floating point precision issues ((e.g. `0.1 + 0.2 ≠ 0.3`)

#### 4. Input validation and consistency
Ensure the function behaves correctly when inputs are invalid or inconsistent.
- Missing or optional arguments (e.g. one input specifies the output file, a second input specifies output format. What happens if an output format is given but no output file?)
- Mismatched input dimensions
- Incompatible types between inputs
- Conflicting parameters (e.g. two mutually exclusive settings such as one input that specifies the number of principle components and a second input that specifies a threshold to select principle components. What happens when both are given?)
- Assumptions between inputs being violated

#### 5. Integration and interaction behaviour
Verify correctness when components interact.
- End-to-end pipeline correctness
- Compatibility between fit/transform/generate stages
- Interaction between multiple parameters or modules
- Backward compatibility with previous versions of outputs

#### 5. Scikit-learn estimator compliance
For estimators inheriting from `BaseEstimator` or similar:
- Compatibility with [`check_estimator`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html#sklearn.utils.estimator_checks.check_estimator) or [`parametrize_with_checks`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html#sklearn.utils.estimator_checks.parametrize_with_checks), which is a pre-made set of tests for compatibility with scikit-learn. More general tools can be found [here](https://scikit-learn.org/stable/api/sklearn.utils.html).
- Proper implementation of `fit`, `transform`, `predict`, etc.
- Parameter consistency (`get_params`/`set_params`)
- Stateless behaviour of `transform` where expected

#### 6. Performance and scalability (when relevant)
Not always required, but important for data-heavy operations:
- Runtime behaviour on large datasets
- Avoidance of unnecessary copies
- Stability under repeated execution

 #### 7. Reproducability
 - Same seed gives the same output
 - Controlled randomness
 - No hidden global state affecting results
 
## Documentation expectations
- Public classes and functions must include docstrings.
- Docstrings should describe purpose, inputs, outputs, assumptions and limitations.
- Complex logic should be explained inline with comments where necessary.
- Examples in docstrings are encouraged.
- Documentation should be updated alongside code changes.
 
## Useful tools for testing and code quality

- Code Spell Checker extension for vscode.
- autopep8 extension for vscode. 
- pytest-cov for code coverage of the unit tests. (installed if you installed this package with the dev dependency group)
- pylint for code feedback and code analysis. 
    
