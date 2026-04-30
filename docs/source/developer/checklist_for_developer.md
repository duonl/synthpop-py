# Checklist for developers

This checklist defines the standard set of actions required when contributing code to the Synthpop project. It is designed to help developers maintain a consistent workflow and ensure that every change meets the project's expectations for quality, completeness and review readiness.

The checklist follows the full development lifecycle: from preparing your work before starting, through implementation, to preparing a pull request and responding to review feedback. Each section reflects a different stage of development and helps prevent common issues such as incomplete implementations, missing tests or unclear pull requests.

For first-time contributors, it is recommended to read [Developing Synthpop](developing.md) first, as it explains the overall workflow and principles that this checklist operationalises in practice. The [Code Standards, Norms and Conventions](code_standards_and_norms.md) document is also relevant when making decisions about implementation details.

Use this checklist as a practical working guide during development. It is not meant to slow you down, but to ensure that the work is structured, reviewable and aligned with the project's expectations before submission.

## Before you begin
- [ ] Fully understand the feature request and its intended behaviour. Read the functional descriptions.
- [ ] Fully understand the expected inputs, outputs and user interaction
- [ ] Fully understand the existing codebase context relevant to the feature
- [ ] Review any proof-of-concept or reference implementation (if provided)
- [ ] Identify edge cases and failure modes before implementation
- [ ] Confirm that the feature scope is clear and well-defined

## Design & preparation
- [ ] Decide how the feature fits into the existing architecture
- [ ] Ensure the solution aligns with project design principles (SOLID, reproducibility, etc.)
- [ ] Identify required changes to tests and documentation
- [ ] Ensure no unnecessary scope expansion is introduced
- [ ] Plan test cases before writing the implementation (Test Driven Development mindset)

## During development
- [ ] Work on a feature branch based on  `develop`
- [ ] Only implement the scoped feature. Found other things that need to be done first? Open a new issue and make a different branch
- [ ] Apply Test Driven Development where possible
- [ ] Write unit tests using [Given-When-Then structure](https://martinfowler.com/bliki/GivenWhenThen.html)
- [ ] Apply the PEP8 standard when useful. 
- [ ] Use British English in code and documentation.
- [ ] Ensure code is deterministic unless explicitly designed otherwise
- [ ] Handle edge cases explicitly in implementation
- [ ] Keep functions and classes focused and minimal in responsibility

## Before making a pull request
- [ ] All tests pass locally
- [ ] No warnings or errors in test or runtime execution
- [ ] Full test suite has been run (not just selected tests)
- [ ] Code behaves as expected in realistic usage scenarios
- [ ] Feature works in a clean environment (not relying on local state)
- [ ] No debug code, prints or temporary artifacts remain
- [ ] Documentation and docstrings are up to date and accurate
- [ ] All relevant tests and documentation changes are included
- [ ] No unrelated files are included in the branch

## Creating the pull request
- [ ] The pull request merges into develop.
- [ ] The pull request title is descriptive and includes relevant issue references
- [ ] The pull request description clearly explains what changed, why it changed, the scope of the change and any known limitations or areas requiring attention
- [ ] If no scope is specified, the reviewer assumes a full review is requested. If requesting a partial review, the scope is explicitly defined in the pull request description. It is clearly stated:
  - [ ] which parts should be reviewed
  - [ ] which parts are not ready for review
  - [ ] what kind of feedback is expected (e.g. design, approach, adherence to conventions, usability)
- [ ] Any merge conflicts have been resolved
- [ ] All tests are re-run after resolving conflicts
- [ ] Commit history is clean and ready for squash merge

## After review feedback
- [ ] All review comments have been addressed or responded to
- [ ] All "Before making a pull request" checks still pass after changes
- [ ] Any merge conflicts have been resolved and tests re-run
- [ ] No additional unrelated changes were introduced during fixes

## Done condition
A feature is considered complete when:
- All reviewer feedback has been resolved
- The pull request is approved by a reviewer with write access
- All tests pass in the final state of the branch
- All checks pass

The pull request must be merged using **squash and merge** to maintain a clean commit history. This can be done by clicking on *Squash and merge* on Github at the bottom of the pull request. Make sure to click on the dropdown menu if you only see a button with *Merge pull request*. To squash merge in the command line you first run:
```{bash}
git checkout target_branch
git merge --squash feature_branch
git commit -m "Commit message"
```


