# Developing Synthpop

This document provides guidelines for developing Synthpop. It outlines the principles and practices to follow when working on the project.

## Assumed context
The development team is currently very small. These guidelines are written with the assumption that only a few developers will be working in parallel at any given time. Bug fixes and new features are expected to be developed concurrently, although the overall pace of introducing new features is anticipated to be modest.

## The branching model
The branching model is follows the classic Git branching model as described [here](https://nvie.com/posts/a-successful-git-branching-model/).
This approach is used to ensure clearly defined and stable versions.

- The **main branch** should at all times be suitable for general use. This is the version user obtain when running ```pip install synthpop```.
    Code in this branch must be complete for the given version, properly documented, and thorougly tested. This includes testing by the original developer, peer review by other developers, and validation by selected users. This branch is never closed.
- The **develop branch** contains the latest features under development. It may include more functionality that is not yet present in the **main** branch. Each feature should be tested by the developer and reviewed by other developers before being merged. This branch is never closed.
- For each new feature, a dedicated **feature branch** is created. The branch name should include a ticket number and a descriptive title. Code in a feature branch may be incomplete or unstable. When a developer starts working on a new feature, these branches are created from the **develop** branch. Once the feature is implemented, tested, and reviewed, it is merged back into **develop** and then closed.
- There is a deliberate difference in code quality between the **develop** and **main** branches. To bridge this gap, a **release branch** is created for each version. Only bug fixes and testing are permitted in a release branch. This branch is created once al planned features for a version have been merged into **develop**. Any bug fixes made in a release branch must also be merged back into **develop**, after which the branch should not be closed. After approval of the release by (a representative group of) users, it is merged into the **main** and closed. 
- In the event of a critical issue in the main branch, a **hotfix branch** may be created firectly from **main**. The branch name should referenace the relevant bug ticket. Only the critical fix may be implemented in this branch. Once resolved, the fix is merged into both **main** and **develop**, and the hotfix branch is closed. After such an incident, the existing code testing procedures should be reviewed and evaluated.

|branch type| example name| created from| When created | When closed | merge targets| activities|
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| main | main | nothing | codebase created| never|None| making code available for general users, no direct commits|
| develop| dev| nothing| codebase created| never| None, unless to prevent merge conflicts| collecting new features, no direct commits|
|feature branches|  12-implement-xgboost|develop| starting to develop a new feature| feature tested by developer and code reviewed| develop| developing new features, writing automatic tests, writing documentation, incorporating feedback|
| release branches| release-1.2.0| develop | all the features for a new version have been merged in develop| code is stable and approved by (a representative group of) users| main, develop | testing by users, finishing touches, bugfixes|
|hotfix| hotfix-3| main | a critical bug has been found in main| the bug has been fixed and merged| main, develop| fixing the critical bug|


## Workflows

This section describes several common scenarios for tasks that a developer may nee to perform

### Defining/designing a feature

### Developing a feature

The first step is to read the functional documentation. Is it clear what needs to be built? Can you locate the feature request ticket(s)? Do you know which users will use this feature? Can you identify any edge cases or exceptions?

1. Ensure that you have the latest version of the **develop** branch in your local repository (```git pull```).
2. Create a new branch from **develop**, named ```{ticket number}-{descriptive name}```.
3. Apply test-driven development when implementing the feature. 
4. Test the feature as you expect a user would use it.
5. Update the documentation. Add example code demonstrating how to ue the feature and verify that the examples works as expected. 
6. Create a merge request to merge the feature branch into **develop**. 
7. Check for merge conflicts and resolve them in your feature branch. In general, the develop branch takes precedence.
8. Test again after resolving conflicts. Verify that no functionality was lost, that the example code still works, and that all automated tests are present and passing.
9. Request a code review and incorporate any feedback. Test again, as changed may introduce new issues. Request review again. One reviewer is enough for small things. For more complex or high-impact features it might be a good idea to have it reviewed by multiple people. 
10. If everything is in order, merge the feature branch into **develop** and close it. 

**Tip:** regularly merge **develop** into your feature branch. This reduces the risk of complex conflicts towards the end of development.

### Releasing a new version

1. Identify the commit on the **develop** branch that contains all features intended for the next release. 
2. Determine the version number.
3. Create a release branch named ```release-v{version number}```.
4. Update the package metadata where necessary. 
4. Run all automated tests and checks. Fix any bugs in the release branch.
5. Use the code as a user would and fix any bugs found in the release branch.
6. Carefully review the documentation for the new features. Ensure that it is clear and usable.
7. If any bug fixes are relevant for ongoing development, merge the release branch back into **develop** without closing it.
8. Ask selected users to test the release and address any reported issues.
9. Obtain approval to merge into **main**. *TODO: who needs to approve?*
10. Merge the release branch into **main** and **develop**, then close the release branch.
11. Upload the new version to PyPi. 

## About bugs
The behaviour of this package is considered a bug when:
1. It contradicts the documentation, or
2. The documentation does not describe the behaviour and it is what a user would reasonably expect.

To identify a bug, both the observed behaviour and the expected behaviour must be defined. This allows us to determine whether the user's expectations where justified. If not, the issue may indicate that the documentation needs to be improved rather than the code.

When the behaviour is incorrect but all automated tests pass, this indicates that the test suite is incomplete. In such cases, new tests should be added to reproduce the observed behaviour. Developers should also consider whether additional related tests are required. Ask yourself whether the situation that triggered the bug is realistic, and whether the impact of the bug is proportionate and logical. 

For example, suppose that attempting to generate a synthetic version of a 1000x1000 table containing only the value 0 causes the system to crash. If the documentation does not specify any limits and a user encounters this scenario, there is a clear discrepancy between expected and observed behaviour. The actual behaviour is a crash, while the expected behaviour is a valid 1000x1000 table filled with zeros. This therefore constitutes a bug.

The next step is to identify the root cause of the crash. If, for instance, it is caused by a division by zero, unit tests should be added to cover this case. Moreover, a division by zero should not cause the system to crash, which implies that error handling within the package needs to be improved.

Finally, consider whether the underlying operation makes sense from a user perspective. It may be necessary to introduce alternative logic, such as a no-op synthesis method, to handle such cases more gracefully.

### Triaging bugs
The exact workflow to fix bugs depends on where the bug is found.

```{mermaid}
---
zoom:
caption: flowchart of when and how to fix bugs.
---
flowchart LR
    A0[The reporter claims there is a bug] -->A1{Is the clear what the expected and observed outcomes are?}
    A1 -->|no| A2[Clear this up with the reporter]
    A1 -->|yes| A3{Is it clear what triggers these outcomes?}
    A3 -->|no| A2
    A3 --> |yes| A
    A[Actual behaviour is not expected behaviour] --> B{Does the actual behaviour contradict the documentation?}
    
    B --> |no, but the expectations of the reporter are common sense.| C
    B --> |no, but the expectations of the reporter sound reasonable.| D[Treat it like a feature request.]
    B --> |no, the reporter has a misunderstanding of the package| D1{Review the documentation and communications about the package. Is there any room for improvement?}
    D1 --> |yes| D
    D1 --> |no| Z[There is nothing to be done]
    B --> |yes| C{On what type of branch has the bug been found? }
    C --> |main| E{Does it have a significant impact on the users?}
    E --> |yes| F[This is a critical bug, use the hotfix method]
    C -->|On a feature branch| G{Is it in the code being developed on that branch}
    G -->|yes| H[Fix it in that branch]
    G -->|No| D
    C --> |On a release branch| I[Fix it in that branch, merge to develop, but do not close the release branch.]
    C -->|develop| D
```

## Versioning
In this project we aim to use [Semantic Versioning](https://semver.org/). However, at the time of writing, the project is still in an early stage and cannot yet be considered a mature open-source project. As a result, no guarantees can be made. Furthermore, since there is no structural funding, we cannot commit to providing long-term support for any specific version.

Semantic versioning does not fully cover certain cases, such as the initial development and pre-release versions. The repository will initially contain an empty **main** branch and a **develop** branch with the project structure and documentation.  At this stage, the project does not yet have a formal version number.

Pre-release versions will use the suffix ```-alpha.x``` where x is incremented after each cycle of testing and bug fixing. 