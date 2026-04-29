# Checklist for Review
This checklist is intended to guide reviewers in performing consistent, thorough and efficient code reviews within the Synthpop project. Its goal is to ensure that all contributions meet the agreed standards for correctness, readability, maintainability and alignment with project conventions.

For first-time reviewers, it is recommended to briefly consult the [Developing Synthpop](developing.md) and [Code Standards, Norms and Conventions](code_standards_and_norms.md) documents. These provide the necessary context on workflows and quality expectations, which the checklist operationalises.

A review is expected to be complete. However, if during early stages of review (e.g. scope, behaviour or method design) it becomes clear that the implementation requires major changes (such as fundamental design issues, incorrect architecture or invalid core assumptions) then:
- You may stop the review early
- You must explicitly communicate this in the review
- You should describe what blocks further review, why continued review is not useful at this stage and what must be addressed before continuation.

## Before you review
- [ ] Understand the purpose of the change (linked issue or feature request)
- [ ] Understand intended user interaction and usage context
- [ ] Understand expected inputs, outputs and behaviour
- [ ] Understand the scope of the pull request
- [ ] Identify related issues, dependencies or constraints
- [ ] Confirm which parts of the codebase are in scope for your review. A developer can ask for a partial review only

## 1. Scope and global scan
This checks whether the pull request is self-contained, relevant and structurally valid before deeper review begins.
- [ ] Only relevant files are included in the pull request (automatically generated files or files that should not be public)
- [ ] The pull request only includes the requested feature (no unrelated changes)
- [ ] All merge conflicts have been resolved.
- [ ] You understand which files you are responsible for reviewing
- [ ] The implementation matches the intended feature scope

## 2. Reviewing the method (architecture and design foundation)
This checks whether the system design is fundamentally sound before correctness or tests.
- [ ] Core design follows OOP principles (encapsulation, responsibility separation, abstraction)
- [ ] TDD principles are respected where applicable (test-first structure, incremental implementation)
- [ ] Method structure is logical and maintainable
- [ ] No fundamental design issues that would invalidate the implementation
- [ ] Randomness is controlled via seeds or mocks in the test where needed

## 3. Reviewing the code completeness (implementation correctness)
This checks whether all required functionality from the specification is fully implemented in the code.
- [ ] The implementation matches the feature specification (functional description)
- [ ] All required functionality described in the issue is implemented
- [ ] The implementation correctly reflects the intended behaviour in real-world usage
- [ ] Invalid assumptions fail safely with informative errors
- [ ] The code produces expected outputs for intended inputs

## 4. Reviewing the test completeness
This checks whether the implementation is fully and meaningfully validated across all functional paths and data flows.
- [ ] All implemented functionality is covered by tests
- [ ] Both success and failure cases are tested
- [ ] Edge cases from the functional description are explicitly tested
- [ ] Correct stubs, mocks and spies are used where applicable in the tests
- [ ] Real-world usage scenarios are represented in tests
- [ ] Regression tests are added where relevant
- [ ] Critical paths are explicitly tested, not only indirectly covered
- [ ] Data flow through the implementation is tested end-to-end
    - [ ] Output of one component matches input expectations of the next
    - [ ] No implicit mismatches exist between stages (e.g. shape, schema, dtype, structure)
    - [ ] Mocked interfaces respect real implementation contracts
- [ ] Integration-style tests validate that mocked assumptions match real system behaviour

## 5. Reviewing the tests (quality and robustness)
This checks whether tests are reliable, maintainable and validate behaviour rather than implementation.
### Coverage dimensions
This checks whether tests ocover the full behavioural space of the system from a user and specification perspective.
- [ ] Tests reflect expected user-facing behaviour, not internal implementation details
- [ ] Correct output type, shape, and schema are tested
- [ ] Behaviour in trivial cases is tested (e.g. empty inputs, single values)
- [ ] Edge cases are covered (boundary values, extreme inputs, degenerate cases)
- [ ] Where relevant, sanity cases are tested, such as:
    - operations on single-element inputs
    - identity operations (e.g. sorting already sorted data)
    - comparison of identical datasets
    - no-op transformations
Find more standard topics in testing in the [Code Standards, Norms and Conventions](code_standards_and_norms.md)
### Test quality
This checks whether tests are well-structured, deterministic and suitable for long-term maintenance and automation.
- [ ] Each test has a clear behavioural assertion
- [ ] Tests are easy to read and logically structured
- [ ] Tests are deterministic and reproducible
- [ ] Tests are isolated and independent (no shared state or order dependence)
- [ ] Tests clean up all side effects, even when the test fails (files, globals, environment changes)
- [ ] Unit tests are  lightweight (run quickly and do not use a lot of computer resources) and fully automatic. This is to enable automatic checks.

## 6. Reviewing the code (implementation quality)
This checks whether the implementation is simple, correct and maintainable without hidden side effects or structural issues.
### Structure
This checks whether the solution is logically decomposed and easy to reason about.
- [ ] Code is not more complex than necessary
- [ ] No duplicate code is unnecessarily introduced
- [ ] Each component has a clear and single responsibility
- [ ] SOLID principles are respected where applicable
- [ ] No hidden coupling between unrelated components
### Integrity and side effects
This checks whether the implementation is correctly manages data flow and avoids unintended state changes
- [ ] Input data is not unintentionally mutated
- [ ] No data leakage occurs between stages (fit/transform/generate) (e.g. refitting in transform)
- [ ] Floating point comparisons are handled safely
- [ ] Division by zero is handled or prevented
- [ ] Optional parameters have sensible and safe defaults
- [ ] Behaviour is stable across repeated executions

## 7. Scikit-learn estimator compliance (if applicable)
This checks whether the implementation conforms to scikit-learn's API and behavioural contract, if applicable.
- [ ] Estimator passes `check_estimator` or `parametrize_with_checks`
- [ ] `fit`, `transform`, etc. behave correctly and consistenly
- [ ] `get_params` and `set_params` work correctly
- [ ] Estimator is stateless where required (e.g. `transform`)
- [ ] There are no mutable default arguments
- [ ] Implementation follows scikit-learn developer guidelines

## 8. Performance and scalability (when relevant)
This checks whether the implementation is computationally efficient and suitable for expected data sizes.
- [ ] Implementation scales reasonably with dataset size
    - [ ] There are no `for`-loops that could be vectorised
- [ ] No avoidable copies of large datasets are made
- [ ] Complexity is acceptable for intended use cases

## 9. Form and style
This checks whether the code is readable, consistent and aligned with project-wide conventoins.
- [ ] Code follows PEP8 standards
- [ ] British English is used
- [ ] Type hints are used where appropriate
- [ ] No unnecessary or unclear comments are present
- [ ] No debug code, prints or commented-out blocks are present
### API consistency
This checks whether the code and language is predictable, consistent and behaves similarly to comparable parts of the codebase.
- [ ] Function and method names are consistent with existing naming conventions in the codebase
- [ ] Parameter names are intuitive and consistent with similar functions and modules (same concept → same name)
- [ ] Parameter ordering follows existing conventions:
    - [ ] required parameters come before optional parameters
    - [ ] commonly used parameters appear first
    - [ ] ordering is consistent with similar functions
- [ ] Default values are consistent with similar functions and reflect typical use cases

## 10. Documentation and usability
This checks whether the code is understandable and usable without reading implementation details.
- [ ] Docstrings are accurate and reflect actual behaviour
- [ ] Assumptions made by the implementation are valid and documented
- [ ] Examples are included for non-trivial functionality
- [ ] Examples are correct and executable
- [ ] Terminology is consistent with project standards
- [ ] Documentation builds successfully without errors

## 11. Final check
This validates overall reviewer confidence in correctness, maintainability and long-term ownership.
- [ ] You can clearly explain what the change does and why it is correct
- [ ] You would be comfortable maintaining this code in the future
- [ ] The change improves or maintains overall codebase quality
