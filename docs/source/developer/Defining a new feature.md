# Defining a new feature

This guide describes the process of defining a feature, from initial idea to the start of development.

## What kind of new features we are looking for
The aim of this package is to facilitate the creation of synthetic data. The primary intended audience is government institutions.
Reliability and trustworthiness are more important than performance. See also CONTRIBUTING.md (ADD LINK). We cannot accept untested and/or unproven technology.
On the other hand, new features that improve usability are very welcome. 

More features do not always make for a better package. Adding a new feature has several consequences:
- users need to understand the feature
- it must be maintained
- it may introduce bugs

## Step 1: Feature Request
The first step is to let us know that you have an idea for an improvement. This can be done by submitting a feature request via GitHub. You do not need to provide code or a detailed plan at this stage.
The purpose of this step is to establish whether (and roughly when) we want this functionality in the package.
Some important things to mention in the initial feature request are:
- What problem does this solve?
- Who will use it?
- To what extent is this a proven concept? A proof of concept or additional information may be helpful at this stage.

In the discussion that follows, the following points may be addressed:
- How many people will use this feature? Does it generalise to other user groups?
- What is the impact on users who benefit from this feature?
- What is the impact on other users?
- Who do you expect will develop this feature?

During this discussion, choices will be made regarding the required functionality.
These choices are consolidated in a document called the functional description.
This document describes the functional requirements for this feature. It should be independent of programming language.
This document sets the goal for the next phase.

Based on this and the follow-up discussion, the product owner decides whether the feature will be adopted into the package.
If the feature is accepted, it moves on to the next step.

## Step 2: Design Consensus
What exactly happens in this step depends on the feature. 
The aim of this step is to reach consensus on how the feature should be implemented. 
While it may be tempting to skip this step and immediately begin development, there is a risk that the feature will need to be completely rewritten to fit within the package.
Some things to establish in this step are:
- expected usage (for example, sample coding showing how to use the feature)
- suggested code structure or mock-up (empty classes, methods and functions), including function signatures. Adding documentation to the mock-up can be very helpful in clarifying the plan
- additional dependencies. We need to be carefull with external packages, since it may cause compatibility or versioning conflicts.
- accepted user inputs
- (if applicable) notable edge cases
- which parts of the codebase need to change
- related development work. There might be other features being developed that needs to work together with the feature of might have some overlap.
There are multiple effective ways to communicate this. Describing it in natural language in the feature request may be a good starting point.
As the plan becomes more concrete, it may be helpful to implement a code mock-up in a branch. This can provide a head start for development.
By the end of this stage, a developer should have all the information needed to implement the feature.

## Step 3: Planning
The next step is to decide when to develop the feature. The product owner is responsible for this planning.
The following points are taken into account when scheduling development of a new feature:
- A release containing breaking changes (changes that require users to alter their code) places a burden on users. It may be best to bundle breaking changes into a single major release.
- A feature may have synergies with other features
- Developer capacity
- Urgency of the feature
- Overall priorities


