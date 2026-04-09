# Checklist for Review
This checklist is intended to guide reviewers in performing consistent, thorough and efficient code reviews within the Synthpop project. Its goal is to ensure that all contributions meet the agreed standards for correctness, readability, maintainability and alignment with project conventions.

For first-time reviewers, it is recommended to briefly consult the [Developing Synthpop](developing.md) and [Code Standards, Norms and Conventions](code_standards_and_norms.md) documents. These provide the necessary context on workflows and quality expectations, which the checklist operationalises.

## Before you review
- [ ] Understand the purpose of the change (linked issue or feature request)
- [ ] Understand intended user interaction and usage context
- [ ] Understand expected inputs, outputs and behaviour
- [ ] Understand the scope of the pull request.
- [ ] Identify related issues, dependencies or constraints
- [ ] Confirm which parts of the codebase are in scope for your review

The following sub-list should be checked in order. If one sub list is not complete, request changes before reviewing any further. 
After changes have been submitted, start with the first sublist again. 

## 1. Scope and global scan
- [ ] Only relevant files are included in the pull request (automatically generated files or files that should not be public)
- [ ] The pull request only includes the requested feature (no unrelated changes)
- [ ] All merge conflicts have been resolved.
- [ ] No debug code, prints or commented-out blocks are present
- [ ] Documentation builds successfully without errors
- [ ] You understand which files you are responsible for reviewing
- [ ] The implementation matches the intended feature scope
- [ ] All tests pass without warnings
- [ ] Example code from the issue or docstrings runs and behaves as expected

## 2. Behaviour and requirements validation
- [ ] The implementation matches the feature specification (functional description)
- [ ] All required functionality described in the issue is present
- [ ] Behaviour is correct in normal usage scenarios
- [ ] Behaviour is correct in expected real-world usage
- [ ] Assumptins made by the implementation are valid and documented
- [ ] Violations of assumptions result in predictable and informative errors

## 3. Reviewing the tests
### Coverage and completeness
- [ ] All new logic paths are covered by tests
- [ ] Both success and failure cases are tested
- [ ] Edge cases from the specification are explicitly tested
- [ ] Regression tests are added where relevant (previous bugs stay fixed)
### Standard test dimensions
- [ ] Correct output type, shape, and schema are tested
- [ ] Behaviour in trivial cases is tested (e.g. empty inputs, single values)
- [ ] Behaviour in realistic use cases is tested
- [ ] Edge cases are covered (boundary values, extreme inputs, degenerate cases)
Find more standard topics in testing in the [Code Standards, Norms and Conventions](code_standards_and_norms.md)
### Obvious / sanity cases
- [ ] Where relevant, sanity cases are tested, such as:
    - operations on single-element inputs
    - identity operations (e.g. sorting already sorted data)
    - comparison of identical datasets
    - no-op transformations
### Test quality
- [ ] Each test has a clear behavioural assertion
- [ ] Tests are easy to read and logically structured
- [ ] Tests validate behaviour, not implementation details
- [ ] Tests are determinisitc and reproducible
- [ ] Randomness is controlled via seeds or mocks where needed
- [ ] Tests are isolated and independent (no shared state or order dependence)
- [ ] Tests clean up all side effects, even when the test fails (files, globals, environment changes)
- [ ] Unit tests are  lightweight (run quickly and do not use a lot of computer resources) and fully automatic. This is to enable automatic checks.

## 4. Reviewing the code
### Correctness and design
- [ ] Implementation correctly satisfies the specification
- [ ] Code is not more complex than necessary
- [ ] No duplicate code is unnecessarily introduced
- [ ] Each component has a clear and single responsibility
- [ ] SOLID principles are respected where applicable
- [ ] Any edge case you see is covered by unit tests
### Data and numerical integrity
- [ ] Input data is not unintentionally mutated
- [ ] No data leakage occurs between stages (fit/transform/generate) (e.g. refitting in transform)
- [ ] Floating point comparisons are handled safely
### API consistency
- [ ] Function and method names are consistent with existing naming conventions in the codebase
- [ ] Parameter names are intuitive and consistent with similar functions and modules (same concept → same name)
- [ ] Parameter ordering follows existing conventions:
    - [ ] required parameters come before optional parameters
    - [ ] commonly used parameters appear first
    - [ ] ordering is consistent with similar functions
- [ ] Optional parameters have sensible and safe defaults
- [ ] Default values are consistent with similar functions and reflect typical use cases

## 5. Edge cases, robustness and error handling
- [ ] Empty inputs are handled correctly
- [ ] Single-element inputs behave correctly
- [ ] Extreme numeric values are handled safely
- [ ] Division by zero is handled or prevented
- [ ] Invalid inputs fail in a predictable and informative way
- [ ] Behaviour is stable across repeated executions

## 6. Scikit-learn estimator compliance (if applicable)
- [ ] Estimator passes `check_estimator` or `parametrize_with_checks`
- [ ] `fit`, `transform`, etc. behave correctly and consistenly
- [ ] `get_params` and `set_params` work correctly
- [ ] Estimator is stateless where required (e.g. `transform`)
- [ ] There are no mutable default arguments
- [ ] Implementation follows scikit-learn developer guidelines

## 7. Performance and scalability (when relevant)
- [ ] Implementation scales reasonably with dataset size
    - [ ] There are no `for`-loops that could be vectorised
- [ ] No avoidable copies of large datasets are made
- [ ] Complexity is acceptable for intended use cases

## 8. Documentation and usability
- [ ] Docstrings are accurate and reflect actual behaviour
- [ ] Examples are included for non-trivial functionality
- [ ] Examples are correct and executable
- [ ] Terminology is consistent with project standards

## 9. Form and style
- [ ] Code follows PEP8 standards
- [ ] British English is used
- [ ] Type hints are used where appropriate
- [ ] No unnecessary or unclear comments are present

## 10. Final check
- [ ] You can clearly explain what the change does and why it is correct
- [ ] You would be comfortable maintaining this code in the future
- [ ] The change improves or maintains overall codebase quality
