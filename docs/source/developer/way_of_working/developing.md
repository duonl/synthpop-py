# Developing Synthpop

This document describes how development within the Synthpop project is structured and executed. It provides the foundational context needed to understand how work is planned, implemented, reviewed and released. It covers the process that guides development.

For first-time contributors, this page should be read first. It provides the mental model of how the project operates and how individual tasks fit into the broader system. The other documentation, such as the developer checklist, review checklist and code standards, builds on the processes defined here.

## Assumed context
The development team is currently small. These guidelines assume that only a few developers will be working in parallel at any given time. Bug fixes and new features are expected to be developed concurrently, while the pace of introducing new features remains modest.

## Contributing and Development Setup
If you wish to contribute to Synthpop, start by forking the repository to your own account. Direct commit access to the main repository is restricted.

Contributions can be done in your fork and are submitted via pull requests from your fork to the upstream repository.

Recommended setup after forking:
```bash
git clone <your-fork-url>
cd synthpop
git remote add upstream <original-repo-url>
```
This allows you to keep your fork up to date:
```bash
git fetch upstream
git checkout develop
git merge upstream/develop
```

### Prerequisites
The project uses `pipx` and `Poetry` for dependency management. Below are Windows-specific instructions to install both.

1. Open a terminal in the root of the repository.
2. Install pipx: `py -m pip install --user pipx`
3. Navigate to the Scripts folder (path shown in the warning): `cd <USER folder>\AppData\Roaming\Python\Python<VERSION>\Scripts`
4. Add pipx to your PATH: `.\pipx.exe ensurepath`
5. Restart your terminal.
6. Verify installation: `pipx --version`
7. Install Poetry: `pipx install poetry`
8. Restart your terminal again.
9. Verify installation: `poetry --version`

### Installing Project Dependencies
1. Open a terminal in the root of the repository.
2. Install all dependencies: `poetry install --with=docs`
3. Navigate to the docs folder: `cd docs`
4. Build the documentation: `poetry run sphinx-build source build`

### Updating the Documentation
After modifying documentation, open a terminal in the root of the repository. Run the following commands to update the rendered documentation:
```bash
cd docs
poetry run sphinx-build -M clean source build
poetry run sphinx-build source build
```

## Branching model
The project follows the [classic Git branching model](https://nvie.com/posts/a-successful-git-branching-model/). This ensures clearly defined and stable versions.

### Main Branch
The **main branch** should at all times be suitable for general use. This is the version user obtain when running ```pip install synthpop```. Code in this branch is fully tested, documented and validated. No direct commits can be made to this branch. The branch is never closed.

### Develop Branch
The **develop branch** contains the latest features integrated features. It may include more functionality that is not yet present in the **main** branch. No direct commits can be made to this branch. Each feature should be tested by the developer and reviewed by other developers before being squash merged. This branch is never closed.

### Feature Branches
For each new feature, a branch is created from **develop**. The branch is named `{ticket}-{description}`. Code in this branch may be incomplete or unstable. After implementation, testing and review, the branch is squash merged into **develop** and then closed.

### Release Branches
There is a deliberate difference in code quality between the **develop** and **main** branch. To bridge this gap, a **release branch** is created from **develop**. The branch is named `release-v{version}`. Only bug fixes and testing is allowed in this branch. After approval, the branch is squash merged into both **main** and **develop** and closed afterwards.

### Hotfix Branches
In the event of a critical issue in the main branch, a **hotfix branch** is created directly from **main**. The branch is named `hotfix-{ticket}`. Only critical fixes are allowed. After approval, the branch is squash merged into both **main** and **develop**. It is closed after merge. After such incident, the existing code testing procedures should be reviewed and evaluated.

|branch type| example name| created from| When created | When closed | merge targets| activities|
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| main | main | nothing | codebase created| never|None| making code available for general users, no direct commits|
| develop| develop| nothing| codebase created| never| None, unless to prevent merge conflicts| collecting new features, no direct commits|
|feature branches|  12-implement-xgboost|develop| starting to develop a new feature| feature tested by developer and code reviewed| develop| developing new features, writing automatic tests, writing documentation, incorporating feedback|
| release branches| release-1.2.0| develop | all the features for a new version have been merged in develop| code is stable and approved by (a representative group of) users| main, develop | testing by users, finishing touches, bugfixes|
|hotfix| hotfix-3| main | a critical bug has been found in main| the bug has been fixed and merged| main, develop| fixing the critical bug|


## Workflow

This section describes several common scenarios for tasks that a developer may need to perform

### Developing a feature
Before a feature can be developed, the process described in [Defining a new feature](Defining_a_new_feature.md) needs to be completed.
The first step is to read the functional documentation. Is it clear what needs to be built? Can you locate the feature request ticket(s)? Do you know which users will use this feature? Can you identify any edge cases or exceptions?

1. Sync your fork with the upstream **develop** branch.
2. Create a new branch in your fork from **develop**, named ```{ticket number}-{descriptive name}```.
3. Apply test-driven development when implementing the feature. 
4. Test the feature from a user perspective.
5. Update the documentation. Add example code demonstrating how to use the feature. Verify that the examples run as expected.
6. Push your branch to your fork.
7. Create a pull request from your fork into the upstream **develop** branch.
8. Resolve any merge conflict in your feature branch. In general, the development branch takes precedence.
9. Re-test thoroughly. Verify that no functionality was lost, the example code still runs, and all automated tests pass.
10. Request a code review and incorporate any feedback. Test again, as changes may introduce new issues. Request review again. One reviewer is enough for small things. For more complex or high-impact features it is recommended to have your code reviewed by multiple people. 
11. If everything is in order, the pull request is squash merged into **develop** by a maintainer. Your branch can then be deleted.

**Tip:** Regularly sync your fork with upstream **develop** to reduce merge conflicts.

### Releasing a new version

1. Identify the commit on the **develop** branch that contains all features intended for the next release. 
2. Determine the version number.
3. Create a release branch named ```release-v{version number}```.
4. Update the package metadata where necessary. 
4. Run all automated tests and checks. Fix any bugs in the release branch.
5. Use the code as a user would and fix any bugs found in the release branch.
6. Carefully review the documentation for the new features. Ensure that it is clear and usable.
7. Ask selected users to test the release and address any reported issues.
8. Obtain merge approval *TODO: who needs to approve?*
9. Squash merge the release branch into both **main** and **develop**, then close the release branch.
10. Upload the new version to PyPi. 

## Bug Handling
A behaviour is considered a bug when it contradicts documentation or it violates reasonable user expectations.

A valid bug report must include:
- Observed behaviour
- Expected behaviour
- Reproducible conditions

The report allows us to determine whether the expected behaviour is justified. If not, the issue may indicate that documentation needs improvement rather than code.

If tests pass but behaviour is wrong, the test suite is incomplete. New tests should be added to reproduce the observed behaviour. Ask yourself whether the situation that triggered the bug is realistic, and whether the impact of the bug is proportionate and logical. 

For example, suppose that attempting to generate a synthetic version of a 1000x1000 table containing only the value 0 causes the system to crash. If the documentation does not specify any limits and a user encounters this scenario, there is a clear discrepancy between expected and observed behaviour. The actual behaviour is a crash, while the expected behaviour is a valid 1000x1000 table filled with zeros. This therefore constitutes a bug.

### Bug Resolution Principles
- Add tests reproducing the issues.
- Identify and fix the root cause.
- Improve error handling if necessary.
- Evaluate whether behaviour is logically sound. If not, alternative logic may need to be introduced to handle the case more gracefully.

### Bug Triaging
The exact workflow to fix bugs depends on where the bug is found.

```{mermaid}
---
zoom:
caption: flowchart of when and how to fix bugs.
---
flowchart LR
    A0[Reporter claims a bug] -->A1{Is the clear what the expected and observed outcomes are?}
    A1 -->|no| A2[Clear this up with the reporter]
    A1 -->|yes| A3{Is it clear what triggers these outcomes?}
    A3 -->|no| A2
    A3 --> |yes| A
    A[Actual behaviour is not expected behaviour] --> B{Does the actual behaviour contradict the documentation?}
    
    B --> |no, but expectations are common sense| C[Proceed as bug]
    B --> |no, but the expectations are reasonable| D[Treat it like a feature request]
    B --> |no, misunderstanding| D1{Can documentation be improved?}

    D1 --> |yes| D
    D1 --> |no| Z[There is nothing to be done]

    B --> |yes| C2{On what type of branch has the bug been found?}

    C2 --> |main| E{Significant user impact?}
    E --> |yes| F[Critical bug → use hotfix process]
    E --> |no| Z[Fill in a bug report in the issues on Github]

    C2 -->|feature branch| G{Is it in code of this branch?}
    G -->|yes| H[Fix in the feature branch]
    G -->|No| D

    C2 --> |release branch| I[Fix in release branch → merge back to develop]
    C2 -->|develop| D
```

## Versioning
This project follows [Semantic Versioning](https://semver.org/) where possible. However, at the time of writing, the project is still in an early stage and cannot yet be considered a mature open-source project. As a result, no guarantees can be made. Furthermore, since there is no structural funding, we cannot commit to providing long-term support for any specific version.

### Pre-release Versions
Use suffix `-alpha.x`. Increment `x` after each testing cycle.

### Initial State
- The `main` branch is empty.
- The `develop` branch contains the initial structure and documentation.
- No formal versions exist yet.

## General Development Principles
- Prefer clarity over cleverness.
- Write tests alongside code.
- Keep changes small and focused.
- Document all user-facing functionality.
- Avoid breaking changes unless justified.
- Continuously evaluate test coverage.

A complete list can be found in [Code Standards, Norms and Conventions](code_standards_and_norms.md).

## Final Notes
This document is intended to evolve alongside the project. Developers are encouraged to propose improvements where processes become unclear or inefficient.
